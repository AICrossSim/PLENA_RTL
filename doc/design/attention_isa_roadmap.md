# Attention & ISA Roadmap (RTL bring-up)

_Last updated: 2026-09-28_

This note explains how attention runs on the current RTL, which ISA features
the hand-written attention workloads deliberately avoid and why, where the
batched-matmul instructions stand, and the RTL bugs that had to be fixed to get
softmax/attention through. For the verified workloads and match rates see
`workload_matrix.md`.

## 1. Current state

The RTL is built at the **downscaled** tile config `MLEN=8, VLEN=8, HLEN=8,
BLEN=4` (`src/definitions/configuration.svh`; the comments there mark the
production target as `MLEN=VLEN=64`). Activations are MXINT8
(`precision.svh: ACT_MX_INT_ENABLE=1`, `WT_MX_INT_ENABLE=1`), the vector SRAM
is FP12 (e6m5). `linear`, `rms_norm`, `silu`, `ffn`, `softmax`, single-head
`attention`, `mha`, `rope` and a full `llama_layer` pass at this config
(`workload_matrix.md`).

RTL mechanisms that were needed for the attention family and are now in the tree:

- **Vector-address drift lock** (`pipeline_control.sv`, "Vector-address drift
  lock"): corrects a vector element op's port-A address at the matrix → vector
  boundary.
- **Reduction-after-broadcast fixes** in `addr_monitor.sv` and
  `vector_machine.sv` that unblocked softmax (§6).
- **MXINT negligible-block guard** in `fp_2_mx_int_block.sv` (§5), needed for
  causal masking.
- **Systolic cold-start primer** (`data_flow_control.sv`, `matrix_pipe_warm`,
  `COLD_START_PRIME_DEPTH = 2*MLEN`): holds the first matmul's permit until the
  sequentially-filled weight rows are warm, so the first BLEN×BLEN output tile is
  not computed from partial weights.

## 2. No standalone `bmm` workload

There is no batched-matmul generator in `tools/testworkloads/`, on purpose:

- The `batched_matmul_asm` template in `PLENA_Compiler/asm_templates` is an
  **orphan**: the ATen compiler does not use it. The compiler tiles a generic
  matmul into `MLEN×MLEN` blocks via `vram_sub_projection_asm` (same family as
  the proven `projection_asm`) and does *attention's* batched matmul with
  `M_BTMM` — never a generic bmm.
- `batched_matmul_asm` tiles N by **BLEN** with column-masked sub-tiles
  (`col_offset=(i%tiles_per_mlen)*blen`, masked `M_MM_WO`). That sub-tiling is
  only validated at the compiler's `MLEN=64` target and produces a period-BLEN
  repeat at small MLEN.
- A "projection-per-batch" bmm would just be `linear` run B times — no new
  capability over `linear`.

## 3. Transposed / batched matmul instructions

The ISA spec (https://aicrosssim.github.io/PLENA_Doc/isa_spec.html#m_btmm) defines:

```
M_TMM   0, rs1, rs2 : SystolicArray += VSRAM[rs2] @ MSRAM[rs1]ᵀ   (transposed read)
M_BMM   0, rs1, rs2 : per-head VSRAM[rs2] @ MSRAM[rs1]
M_BTMM  0, rs1, rs2 : per-head, MSRAM transposed (Q@Kᵀ)  — M_TMM's batched form
   dims: [MLEN/HLEN, MLEN, HLEN] @ [HLEN, MLEN] = [MLEN/HLEN, MLEN, MLEN]
   → runs independent matmuls in parallel (multi-head attention)
M_BMM_WO rd, imm    : write systolic array → VSRAM[rd+imm], per-head stride
```

**What the RTL implements today**

- `decoder.sv` (matrix-op decode, around lines 446-471) maps `M_MM`, `M_TMM`,
  `M_BMM`, `M_BTMM` → `MM_IC` and `M_MM_WO`, `M_BMM_WO` → `MM_WO`; it sets
  `m_transposed_read` for `M_TMM`/`M_BTMM` and `m_per_head` for
  `M_BMM`/`M_BTMM`/`M_BMM_WO` (`OP_BUNDLE.m_per_head`, `operation.svh`).
- `matrix_machine.sv` passes `m_per_head` to `mxint_systolic_mcu.sv`. In
  per-head mode the MCU keeps the `ROW_BLOCK_NUM = MLEN/KLEN` mini-array partial
  sums **separate** (one head each, K broadcast to all minis) instead of summing
  them across K, and drains `ROW_BLOCK_NUM × BLEN` rows instead of `BLEN`. With
  `KLEN=4` (the MXINT scale block) and `MLEN=8` that is at most **2 heads of 4
  lanes**; at `MLEN=64` it would be 16 heads of 4 lanes. The per-head mode exists
  only in the MXINT MCU, not in the MXFP systolic path.
- `M_BMV` / `M_BMV_WO` (batched matrix-vector) are defined in `operation.svh`
  but not decoded; the `M_OP` enum slots `MM_BIC`, `BMM_WO`, `BMV_WO` are unused.

**Verification status.** The per-head path is exercised only by the
ATen-compiled `gqa` workload (packed grouped-query attention), which currently
**fails** at `MLEN=8` (73% match, see `workload_matrix.md` → Open issues). The
transposed read (`m_transposed_read`, `matrix_sram/subsram.sv` +
`subtile_transpose.sv`) is likewise not covered by any passing workload. All
hand-written attention workloads (`attention`, `mha`, `rope`, `llama_layer`)
therefore avoid `M_TMM`/`M_BTMM` entirely (§4).

## 4. Single-head attention needs no transposed/batched matmul

For **one head** with `kv_len == VLEN` and `head_dim == MLEN`, Q@Kᵀ runs on the
proven `M_MM` projection path and still lands softmax-friendly:

- K is stored **transposed** in HBM as a `(d, kv_len)` weight, so a plain
  `projection_asm` (`act @ weight`) with `act = Q (q_len, d)` computes exactly
  `Q @ Kᵀ = S (q_len, kv_len)`.
- Because `kv_len == VLEN`, `S` lands in **one out-tile**, row-major /
  query-major — each query's logits are one contiguous VRAM row and softmax
  reduces **within** the row (`V_RED_MAX`/`V_RED_SUM`), no cross-tile reduction.
- `P @ V` is then a plain `projection_asm` (`M_MM`) with `act = P` (already in
  VRAM) and `weight = V (kv_len, d)`.
- Causal masking adds a pre-staged `(q_len, kv_len)` mask (0 for k≤q, `MASK_NEG`
  for k>q) to the scaled logits with `V_ADD_VV` before the softmax.
- Multi-head (`mha`, `llama_layer`) is this pipeline looped over heads, with
  each head's Kᵀ/V as its own HBM tensor (see the layout notes in
  `workload_matrix.md`).

So the whole single-head pipeline stays on instructions that `linear`/`ffn`/
`silu` already prove. `M_TMM`/`M_BTMM` become necessary only for `kv_len > VLEN`
(multi-tile softmax) and for **head-packing**, which is a performance feature
that pays off at the `MLEN=64` target, not a correctness requirement here.

`projection_T_asm` (the template that would compute `act @ weight.T` directly)
has a **period-BLEN weight-select bug** for `out_features <= MLEN`: its
within-MLEN-group weight offset is `(weight_row % tiles) * blen` where
`projection_asm` uses `* blen * mlen`. Storing K transposed and using
`projection_asm` sidesteps it.

## 5. MXINT sparse-block quantizer guard (`fp_2_mx_int_block.sv`)

The causal `MASK_NEG` sentinel is squeezed between two FP12 limits: too negative
and the exp SFU saturates (uniform P); too small and masked P stays ≈1e-6
instead of flushing to 0. `-12` is the usable optimum, so masked P is ≈1e-6.

Activations are MXINT, so P is re-quantized to MXINT by `fp_2_mx_int_block.sv`
(in the vector-SRAM read path) for the P@V matmul. That converter is faithful:
an all-masked block encodes a very negative shared exponent with large integer
mantissas. Combined with a normal-scale block in the **fixed-point MXINT
systolic accumulate**, the extreme scale gap overflows and produces garbage
output columns on heavily-masked rows. (The MXFP path never hits this — its
per-element exponents absorb the scale.)

**Guard:** the module's all-*zero* block guard is extended to also collapse an
all-*negligible* block (`signed_exp_max < MIN_SHARED_EXP = -8`, i.e. max element
< 2⁻⁸) to a clean 0. Naively flooring the shared exponent (as the MXFP path
does) would be **wrong** for MXINT — its integer mantissas carry no per-element
exponent, so flooring would round every normal sub-1 activation (softmax
P ≈ 0.06 → 0.094). The guard only touches blocks far below any real activation
(normal ≥ 2⁻⁴; masked ≤ 2⁻¹⁶), so it is loss-free on real data.

## 6. RTL bug found during bring-up: broadcast-op → `V_RED_SUM` deadlock

Bisected with `--skip-asm-gen` on the built RTL (each line = one sim):

| ASM pattern | Result |
|---|---|
| `V_MUL_VV gp3,gp1,gp1` then `V_RED_SUM f3,gp3` (repeat) | `C_BREAK` reached |
| `V_MUL_VF gp3,gp1,f1` then `V_RED_SUM f3,gp3` (repeat) | **hang** |
| `V_MUL_VF` + `V_EXP_V` in a loop, no reduction | `C_BREAK` reached |
| single bare `V_RED_SUM` on preloaded data | `C_BREAK` reached |
| broadcast op + 4 scalar NOPs + `V_RED_SUM` | **hang** (not a latency issue) |

A `V_RED_SUM` that consumes data produced by a **broadcast** vector op
(`V_MUL_VF`/`V_EXP_V`) hung; the same reduction consuming `V_MUL_VV` output
completed. rms/silu/ffn never hit this (rms's reductions consume `V_MUL_VV`
output; silu/ffn have no reduction after a broadcast op). softmax was the first
workload to exercise reduction-after-broadcast and exposed **two** distinct RTL
bugs (found with in-RTL `$display` probes — static analysis missed both):

**Bug 1 — phantom addr-0 write-tracker lock (`addr_monitor.sv`).** The
decoder's deferred write-address mechanism carries a broadcast element op's
`v_update_waddr` onto the *following* op's `update_v_waddr`, but not its
address — the carried op's `addr_2` is 0. `addr_monitor` therefore inserted a
write-tracker entry at **address 0** that nothing ever cleared, and a
`V_RED_SUM` reading row 0 locked on it forever. **Fix:** the `update_v_waddr`
insert branch was removed (it never tracked a real address; the
element → reduction RAW hazard is already serialized by `v_elem_busy`).

**Bug 2 — reduction-prepare starvation (`vector_machine.sv`).** With bug 1 gone
the reduction issued but never produced a result. `recorded_element_v_control`
and `recorded_broadcast_en` were sticky (only updated on a non-STALL *element*
op); after `V_EXP_V` they stayed `EXP`/`1`, so the prepare mux fired
`complete_element_prepare` on the broadcast branch for the FOLLOWING reduction
and starved `complete_reduct_prepare`. **Fix:** clear
`recorded_element_v_control <= STALL_V_ELEMENT` and `recorded_broadcast_en <= 0`
when a reduction is recorded.

Both fixes only touch the reduction-after-broadcast path; `softmax` passes at
100% and rms/silu/ffn/linear were regression-checked.

## 7. Roadmap

1. **[done] Single-head attention, MHA, RoPE and a full decoder layer on the
   hand-written template path** — see `workload_matrix.md`.
2. **Verify the per-head (`M_BTMM`/`M_BMM_WO`) and transposed-read paths.**
   The RTL is in place (§3) but its only workload, `gqa`, fails at `MLEN=8`.
3. **Fix the vector-SRAM >4096-element failure** (`silu_down` at its default
   shape, `workload_matrix.md` → Open issues); any larger workload will hit it.
4. **ATen compiler path** (`--use-aten-compiler` on `linear`, `rms_norm`, `ffn`,
   `softmax`, `rope`; default for `gqa`) — not part of the verified matrix at
   this config. Getting it green lets workloads be driven from PyTorch instead
   of hand-written templates.
5. **Build and test at the `MLEN=64` target** (parameterize the test infra over
   `MLEN/VLEN/BLEN`), where head-packing pays off and the compiler's packed GQA
   attention is meant to run.
