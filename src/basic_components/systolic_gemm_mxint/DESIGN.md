# MXINT Systolic GEMM — Design

MXINT (microscaling integer) systolic GEMM compute unit. Mirrors the MXFP design in
`../systolic_gemm_mx/` but the operands are integers with a shared per-block scale, and
the cross-K reduction stays in the integer domain (one INT→FP convert at the very end)
to save area.

`value = (-1)^sign · (magnitude / 2^(W-1)) · 2^(scale - bias)`, where a block of
elements shares one `scale`. **Elements are TWO'S COMPLEMENT** (the PE multiplies them
with `$signed`), NOT sign-magnitude — `pack_mxint_to_bin` from `plena_quant` emits
sign-magnitude and must not be used as-is for this datapath.

## Dimensions

`C[M][N] = A[M][K] · B[K][N]` (only `M == N` supported).

| name          | meaning                                             | default |
|---------------|-----------------------------------------------------|---------|
| `M` / `N`     | output tile edge (= mini-array edge, `COMPUTE_DIM`) | 4       |
| `K`           | reduction depth per instruction                     | 8       |
| `BLOCK_DIM`   | MXINT scale block size (= per-mini accumulation depth) | 4    |
| `ROW_BLOCK_NUM` = `K / BLOCK_DIM` | parallel mini systolic arrays   | 2       |

The K reduction is split **spatially** (`ROW_BLOCK_NUM` parallel mini-arrays) and
**temporally** (`BLOCK_DIM` MACs per PE). One MXINT scale block = one mini-array's depth.

## Module hierarchy

```
mxint_systolic_mcu                  interface-aligned with mx_systolic_mcu (GEMM-only)
├─ ROW_BLOCK_NUM × mxint_mini_systolic_array   one per K-segment → INT partial tiles
│   ├─ 2·COMPUTE_DIM × roller_buffer           PISO: parallel load, serial (skewed) out
│   └─ COMPUTE_DIM² × mxint_default_pe         output-stationary INT MAC + scale-sum
├─ mxint_sum_across                 cross-K INT-domain reduce (align scales, integer add)
├─ COMPUTE_DIM² × mxint_acc_2_fp    INT accumulator → storage FP (a SINGLE conversion)
├─ block_data_buffer                assemble [M][N] tiles into [M][K] rows, unroll to v_result
└─ register_slice_wo_hs (result_buffer) → v_result
```

### mxint_default_pe
Output-stationary PE. `mac_fire = in_top_valid && in_left_valid`; product
`$signed(top)·$signed(left)` accumulates into a lossless `fix_accumulator`
(`MX_T+MX_L+clog2(ACC_DEPTH)` bits). `out_scale = top_scale + left_scale - bias`.
`out_valid` latches when `acc_counter == ACC_DEPTH` and HOLDS until reset. Requires a
1-cycle idle gap between consecutive dot products (load-on-clear semantics) — see
`doc/design-decisions/pe-gap-cycle.md`.

### mxint_mini_systolic_array
`roller_buffer[t]` loaded at cycle `t` (`load_cnt == t`) emits its `ACC_DEPTH` K-values
from cycle `t+1`; adjacent rollers are one cycle apart, producing the systolic
wavefront. PE(i,j) sees both valids high for `ACC_DEPTH` cycles. **One tile per reset**
— `load_cnt` and the PE state only clear on `rst`.

### mxint_sum_across
Reduces `ROW_BLOCK_NUM` partial tiles in the INTEGER domain. Aligns the per-segment
scales to a common low scale (`eff_min = max(min_scale, max_scale - MAX_SHIFT)`),
left-shifts (lossless) / right-shifts (cap overflow) each segment, integer-adds them.
1-cycle pipeline. Lossless when all segment scales are within `MAX_SHIFT`.

### mxint_acc_2_fp
Converts the reduced 2's-complement integer accumulator + biased scale **directly to
storage FP** (`FP_EXP_WIDTH`/`FP_MANT_WIDTH`). Only one conversion in the whole datapath
— there is no intermediate "accumulator-FP" precision.

## MCU interface (aligned with `mx_systolic_mcu`)

GEMM-only; GEMV deferred. `control` (`M_OP`) handles `MM_IC` (load) and `MM_WO` (write
out). Operand mapping mirrors the MX MCU: **v1 = TOP** (multiplicand, per-element scale),
**v2 = LEFT** (multiplier, per-block scale).

| port | dir | width | notes |
|------|-----|-------|-------|
| `control` | in | `[3:0]` | `M_OP` (MM_IC=0x5, MM_WO=0x7, STALL_M=0x0), inlined as localparams |
| `acc_waddr` / `fetch_next_acc_waddr_valid` / `fetch_next_acc_waddr_ready` | in/in/out | — | column-block write address handshake to `block_data_buffer` |
| `wait_for_output` | in | — | gates the unroll |
| `v1_element` / `v1_scale` / `v1_in_valid` | in | `[K][MX_T_INT]` / `[K][SCALE]` / 1 | TOP, per-element scale (block-replicated) |
| `v2_element` / `v2_scale` / `v2_in_valid` | in | `[K][MX_L_INT]` / `[ROW_BLOCK_NUM][SCALE]` / 1 | LEFT, per-block scale |
| `v_result` / `v_result_write_req` | out | `[K][FP]` / 1 | K-wide output row, M rows over M cycles |
| `empty_in_progress` | out | 1 | high while writing out (busy) |

## Control / dataflow

1. **MM_IC** — `load_active = v1_in_valid && v2_in_valid` drives all mini-arrays for M
   cycles (cycle t = A row t / B col t). NOTE: gate on the valids, NOT the registered
   `control_in_exe` (that would drop the first load cycle and misalign the PE valids).
   Mini p gets `v2_element[p*BLOCK_DIM +: BLOCK_DIM]` (LEFT), `v1_element[...]` (TOP),
   `v2_scale[p]`, `v1_scale[p*BLOCK_DIM]`.
2. Mini-arrays compute → `mxint_sum_across` → `mxint_acc_2_fp` → `gebm_result` [M][N] in
   storage FP. `reduced_valid`/`gebm_result` HOLD stable once computed (no result latch).
3. **MM_WO** — `start_writeback = MM_WO && reduced_valid && !wrote` pushes the held tile
   once into `block_data_buffer` (guarded by `wrote`, which re-arms when control leaves
   MM_WO). The buffer writes the tile at column block `acc_waddr*N` and, when
   `acc_waddr == K/M-1 && wait_for_output`, unrolls M rows of `v_result`.
   `empty_in_progress` (= `writeback_busy`) spans the M output rows, tracked by
   `out_row_counter`.

### Back-to-back tiles — internal soft reset
The mini-arrays / PE are one-tile-per-reset, so the MCU soft-resets them between tiles
(no external `rst` between instructions needed):

```
writeback_done = writeback_busy && unrolled_data_out_valid && (out_row_counter == M-1);
mini_clear     = mini_clear_pending && !load_active;            // mini .rst = rst || mini_clear
mini_clear_pending: set on writeback_done, cleared on load_active, reset on rst.
```

`mini_clear` holds the arrays in reset through the idle gap after a tile and drops the
instant the next load begins (`!load_active` prevents it from wiping the fresh load). The
result is already in `block_data_buffer` by `writeback_done`, so clearing is safe. Needs
a ≥1-cycle gap between MM_WO and the next MM_IC.

## Tests
`test/mxint_default_pe_tb.py`, `test/mxint_mini_systolic_array_tb.py`,
`test/mxint_systolic_mcu_tb.py`. All read RTL params back from the DUT, pack elements as
two's complement, and use real-value (`qdata`) golden references. The MCU TB runs 3
GEMM tiles back-to-back (single reset at start, only an idle gap between tiles) to
exercise the internal soft-reset.

## Status / TODO
- Done: PE, mini-array, sum-across, acc→FP, MCU (GEMM, interface-aligned), back-to-back.
- TODO: multi-pass `acc_waddr` accumulation test (only single column-block tested so far);
  plug the MCU into `matrix_machine.sv`; GEMV core (deferred).
