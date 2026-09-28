# Workload Matrix

What the RTL is verified to run today, at the configuration it is built at.

**Hardware tile config:** `MLEN=8, VLEN=8, HLEN=8, BLEN=4`
(`src/definitions/configuration.svh`). Activations are **MXINT8**
(`precision.svh: ACT_MX_INT_ENABLE=1`); the vector SRAM holds **FP12** (e6m5).

Run any row with `just rtl-sim <workload> [rebuild=true|false] [args...]`
(see `doc/testing.md`). **Pass bar:** an element matches when
`|err| <= 0.1 + 0.1 * |gold|`; a run passes at >= 90% match.

## Verified workloads

| Workload | File | Args (if non-default) | Shape | Match | Notes |
|---|---|---|---|---|---|
| linear | `linear.py` | — | batch 8, 128 → 256 | 95.46% | Y = X @ W on the `projection_asm` (`M_MM`) path. `in/out_features` multiples of MLEN, `batch` multiple of BLEN. |
| rms_norm | `rms_norm.py` | — | batch 8, hidden 128 | 100% | `V_MUL_VV` + `V_RED_SUM` reductions, rsqrt via the scalar SFU. |
| silu | `silu.py` | — | batch 8, hidden 128 | 100% | Vector non-linearity chain (`V_SUB_VF`/`V_EXP_V`/`V_ADD_VF`/`V_RECI_V`/`V_MUL_VV`). |
| linear_silu | `linear_silu.py` | — | batch 8, 128 → 256 | 100% | FFN up-projection + SiLU. |
| silu_down | `silu_down.py` | — | batch 8, 128 → 256 → 128 | 100% | SiLU + down-projection. Also 100% at 64→64, 128→128 and 64→256. |
| ffn | `ffn.py` | — | batch 8, hidden 64, inter 64 | 100% | Full SwiGLU MLP (up/gate → SiLU → mul → down). |
| softmax | `softmax.py` | `--hidden-size 8` | batch 16, hidden 8 | 100% | Row-wise softmax; `hidden_size == VLEN` so each row is one VRAM row and the reductions stay in-row. |
| attention | `attention.py` | `--q-len 8 --kv-len 8 --head-dim 8` | q 8, kv 8, d 8, causal | 100% | Single-head SDPA (Q@Kᵀ → softmax → P@V) on the `M_MM` path with K stored transposed. Also passes at `--q-len 16`. Single-tile constraints: `head_dim == MLEN`, `kv_len == VLEN`, `q_len % BLEN == 0`. `--no-causal` for bidirectional. |
| mha | `mha.py` | `--q-len 8 --kv-len 8 --head-dim 8` | 4 heads, q 8, kv 8, d 8 | 100% | Multi-head = single-head attention looped over `--num-heads`; each head's Kᵀ/V is its own HBM tensor with a per-head base address register (see below). Same single-tile constraints per head. |
| rope | `rope.py` | `--q-len 8 --kv-len 8 --head-dim 8` | q 8, kv 8, d 8, causal | 100% | Single-head attention + RoPE. Q is rotated on-chip (`rope_asm`, pre-staged cos/sin/rotation tables), K is rotated offline into the transposed weight. HF/NeoX `rotate_half` convention, base 10000. Requires `head_dim == VLEN` as well. |
| llama_layer | `llama_layer.py` | `--head-dim 8 --seq-len 8` | S 8, 4 heads, d 8 (hidden 32), inter 64 | 99.22% | Full decoder layer (`--stage full`): RMSNorm → Q-proj → attention → O-proj → residual → RMSNorm → SwiGLU FFN → residual. `head_dim == MLEN`, `seq_len % BLEN == 0`, `inter_size % MLEN == 0`. Residual error is MX quantization across seven chained matmuls. |
| prefetch | `prefetch.py` | — | — | pass | Infrastructure test: HBM → SRAM prefetch; not a numeric workload. |
| loop | `loop.py` | — | — | pass | Infrastructure test: `C_LOOP_START`/`C_LOOP_END` nesting; not a numeric workload. |

`just rtl-suite` runs `linear_8x128x256`, `rms_norm_8x128` and `silu_8x128`
against a single Verilator build and passes.

## Open issues

Fixed on the way here: any workload whose `hbm.mem` exceeded 128 KiB (e.g. `silu_down`
8x128x256, `linear --out-features 512`) read garbage because the simulation HBM model's
per-port word-address width `simulation_pkg::FAKE_HBM_ADDR_WIDTH` was 16; it is now 20.

| Workload | Status | Detail |
|---|---|---|
| gqa | **FAILS** (`--seq-len 8 --kv-seq-len 8 --hq 2`, 73% match) | Packed grouped-query attention generated through the ATen compiler path (`--use-aten-compiler` is its default). Constraints: `hkv == 1`, `seq_len`/`kv_seq_len <= MLEN`, `hq * h_qkv <= MLEN`. Under investigation. |
| bmm | **Not supported** | There is no standalone batched-matmul generator: the `batched_matmul_asm` template is not used by the compiler and is not validated at this tile size. The ISA's per-head matmuls (`M_BMM`/`M_BTMM`/`M_BMM_WO`) are decoded and wired to the MXINT systolic MCU but are exercised only by the failing `gqa` path. See `attention_isa_roadmap.md` §2-3. |

## Layout notes that any multi-tensor workload must respect

- **Multi-head weight packing (`mha`, `llama_layer`).** MXINT weights store
  element and scale data in separate per-tensor regions, and the HBM controller
  finds a weight's scale region from the weight's *base address register*
  (`base + C_SET_SCALE_REG`, where `projection_asm` sets
  `C_SET_SCALE_REG = in_features * out_features`). Each head's Kᵀ/V must
  therefore be packed as its own HBM tensor, with the head's base in the address
  register itself rather than in a GP-register offset. Packing all heads into one
  `(H*D, KV)` tensor makes every head but the last read garbage scales and emit
  zeros.
- **Multi-tile outputs are stored tiled, not row-major.** When the output width
  exceeds VLEN, the VRAM golden must be laid out as
  `reshape(S, width // VLEN, VLEN).permute(1, 0, 2)` (as `linear`/`ffn` do); a
  plain reshape reports a meaningless match rate.
- **Instruction offset is per shape.** Each generator writes
  `INSTRUCTION_STORAGE_OFFSET` for its own `hbm.mem` layout into
  `configuration.svh`, which is compiled into the Verilator model. Reusing a
  cached build (`rebuild=false`) for a different shape loads the program at the
  wrong address; rebuild when the shape changes.
- **General shape rules.** `in_features`/`out_features`/`hidden` multiples of
  MLEN (8); `batch`/`q_len`/`seq_len` multiples of BLEN (4); attention-family
  workloads are single-tile (`head_dim == MLEN`, `kv_len == VLEN`).
