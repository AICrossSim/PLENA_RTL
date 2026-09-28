"""Single-head attention workload generator for PLENA.

Computes one head of scaled-dot-product attention

    S = (Q @ K^T) * (1/sqrt(d))          (q_len, kv_len)
    P = softmax(S, dim=-1)               (q_len, kv_len)
    O = P @ V                            (q_len, d)

using ONLY instructions that are implemented and proven on the current RTL
(MLEN=16, VLEN=16, BLEN=4):

  * Q @ K^T   -> the proven ``projection_T_asm`` (M_MM, act @ weight.T).
                K is the "weight" stored naturally as (kv_len, d).
  * softmax   -> the proven vector non-linearity chain
                (V_MUL_VF scale, V_RED_MAX, V_SUB_VF, V_EXP_V, V_RED_SUM,
                 S_RECI_FP, V_MUL_VF), the same ops silu/rms exercise.
  * P @ V     -> the proven ``projection_asm`` (M_MM).
                V is the "weight" stored naturally as (kv_len, d).

Why no M_TMM / M_BTMM:
  The flash-attention templates compute Q@K^T with M_TMM (per-head) or M_BTMM
  (head-packed), but in this RTL M_TMM decodes to STALL_M (decoder.sv:462 --
  m_transposed_read is set but the m_op mapping omits M_TMM) and M_BTMM is
  unimplemented entirely.  For a SINGLE head with ``kv_len <= VLEN`` and
  ``d == MLEN`` the projection output already lands as (q_len, kv_len)
  row-major / query-major in one out-tile, so softmax reduces WITHIN a VRAM row
  and the whole pipeline stays on the proven M_MM path.  See
  ``doc/design/attention_isa_roadmap.md``.

VRAM element layout (single tile each):
    Q  : [0            .. q_len*d)             staged from HBM by preload_act
    S/P: [q_len*d      .. q_len*d + q_len*kv_len)   Q@K^T, softmaxed in place
    O  : [above end    .. + q_len*d)            P@V, verified

HBM data order: [Q, K, V].  Q is prefetched to VRAM; K and V are matrix-
prefetched by the two projections (addr regs a1=K base, a2=V base).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

# Add necessary paths
_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"
if str(_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_TOOLS_PATH))

_PLENA_TOOLS_PATH = _PROJECT_PATH / "PLENA_Tools"
if str(_PLENA_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_PLENA_TOOLS_PATH))

_COMPILER_PATH = _PROJECT_PATH / "PLENA_Compiler"
if str(_COMPILER_PATH) not in sys.path:
    sys.path.insert(0, str(_COMPILER_PATH))

from .base import WorkloadGenerator

from asm_templates import (
    preload_act_asm,
    preload_addr_reg_asm,
    reset_reg_asm,
    projection_asm,
    projection_T_asm,
)
from asm_templates._imm import load_large_int_str as _load_large_int
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")
# Defaults follow the tile sizes (head_dim == MLEN, kv_len == VLEN are hard constraints).
_MLEN, _VLEN = _HW_TILE_SIZES["MLEN"], _HW_TILE_SIZES["VLEN"]

DEFAULT_MX_FORMAT = None  # None means use precision.svh setting


def attention_cpu(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """CPU reference: O = softmax(Q@K^T * scale) @ V for a single head."""
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v.float())


def softmax_rows_asm(
    s_base_address: int,
    q_len: int,
    kv_len: int,
    vlen: int,
    scale_fp_address: int,
    alive_registers: list[int],
    mask_base_address: int = None,
) -> str:
    """Row-wise softmax over the QK^T scores held in VRAM.

    S is (q_len, kv_len) row-major with kv_len <= vlen, so each query's logits
    occupy one contiguous vlen-wide VRAM row at ``s_base + q*vlen``.  We loop
    over the q_len rows and, per row, compute

        s   = s * scale                 (V_MUL_VF, broadcast)
        s   = s + mask[q]               (V_ADD_VV, causal, if mask_base_address)
        m   = max(s)                    (V_RED_MAX -> f2)
        s   = s - m                     (V_SUB_VF, broadcast, reverse=0)
        s   = exp(s)                    (V_EXP_V)
        l   = sum(s)                    (V_RED_SUM -> f3)
        l   = 1 / l                     (S_RECI_FP)
        s   = s * l                     (V_MUL_VF, broadcast)  = softmax row

    Causal mask (optional): a (q_len, kv_len) matrix pre-staged in VRAM at
    ``mask_base_address`` (row q at ``mask_base + q*vlen``) holding 0 for k<=q and
    a large negative sentinel for k>q. Added AFTER the scale via ``V_ADD_VV``
    (vector-vector, so the following ``V_RED_MAX`` reads clean, non-broadcast
    data). fp regs used: f0 (hw zero), f1 (scale), f2 (max), f3 (sum).
    """
    addr = alive_registers[0]
    loop = alive_registers[1]
    maddr = alive_registers[2] if mask_base_address is not None else None

    code = "; Row-wise Softmax (single head)\n"
    # Load the QK scale (1/sqrt(d)) into f1.
    code += f"S_LD_FP f1, gp0, {scale_fp_address}\n"
    code += _load_large_int(addr, s_base_address)
    if mask_base_address is not None:
        code += _load_large_int(maddr, mask_base_address)
    code += f"C_LOOP_START gp{loop}, {q_len}\n"
    # Scale logits by 1/sqrt(d).
    code += f"V_MUL_VF gp{addr}, gp{addr}, f1, 0\n"
    if mask_base_address is not None:
        # s = s + causal_mask[q]  (vector-vector add of the pre-staged mask row).
        code += f"V_ADD_VV gp{addr}, gp{addr}, gp{maddr}, 0\n"
    # Running max for numerical stability.
    code += f"V_RED_MAX f2, gp{addr}, 0\n"
    # s = s - max  (reverse=0 -> operand - scalar).
    code += f"V_SUB_VF gp{addr}, gp{addr}, f2, 0, 0\n"
    # s = exp(s - max).
    code += f"V_EXP_V gp{addr}, gp{addr}, 0\n"
    # l = sum(exp).  Reset the accumulator first (S_RED_SUM accumulates into fp).
    code += "S_ADD_FP f3, f0, f0\n"
    code += f"V_RED_SUM f3, gp{addr}\n"
    # l = 1 / sum.
    code += "S_RECI_FP f3, f3\n"
    # Spacer so multi-cycle S_RECI_FP retires before the V_MUL_VF reads f3
    # (mirrors the rms_norm reciprocal spacer).
    for _ in range(4):
        code += "S_ADDI_INT gp0, gp0, 0\n"
    # s = exp(s - max) / sum  = softmax row.
    code += f"V_MUL_VF gp{addr}, gp{addr}, f3, 0\n"
    # Next query row (and mask row).
    code += f"S_ADDI_INT gp{addr}, gp{addr}, {vlen}\n"
    if mask_base_address is not None:
        code += f"S_ADDI_INT gp{maddr}, gp{maddr}, {vlen}\n"
    code += f"C_LOOP_END gp{loop}\n"
    return code


class AttentionWorkload(WorkloadGenerator):
    """Single-head scaled-dot-product attention workload (prefill).

    Causal by default (Llama-style; ``--no-causal`` for bidirectional). The causal
    mask is pre-staged in VRAM and added to the scaled logits before softmax.

    Constraints for this first bring-up (single-tile, proven M_MM path):
        d       == MLEN            (head dim = matrix tile)
        kv_len  == VLEN            (keys fit one VRAM row -> in-row softmax)
        q_len   %  BLEN == 0       (queries tile by BLEN)
    """

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    # Negative sentinel added to masked logits so exp() -> ~0 AFTER the row-max
    # subtraction. The value is squeezed between two hardware limits:
    #  - |x| too big: the FP12 exp SFU saturates to garbage (per silu) -> a huge
    #    constant swamps the row -> uniform P (seen at -30000 and at -20/input~-23).
    #  - |x| too small: masked P stays ~1e-6 (nonzero).
    # There is no value that makes masked P a *clean* 0: to flush exp to 0 needs input
    # < ~-21.7 (FP12 2^-31), but the exp SFU already saturates by ~-20. So -12 is the
    # best usable point (P is perfectly causal). The residual ~1e-6 masked-P entries
    # then form all-tiny MXFP blocks whose shared exponent overflows when the P@V
    # systolic array re-quantizes P -> garbage output columns on the heavily-masked
    # early-query rows (q<8, whose entire second MXFP block is masked). Rows q>=8 are
    # exact. This is an MXFP-quantizer robustness limitation, NOT a masking bug (the
    # masking itself is verified correct). See doc/design/attention_isa_roadmap.md.
    MASK_NEG = -12.0

    def __init__(
        self,
        q_len: int = _VLEN,
        kv_len: int = _VLEN,
        head_dim: int = _MLEN,
        mx_format: str = None,
        causal: bool = True,
        skip_asm_gen: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert q_len % self.BLEN == 0, f"q_len must be divisible by BLEN={self.BLEN}"
        assert head_dim == self.MLEN, (
            f"head_dim must equal MLEN={self.MLEN} for the single-tile projection path "
            f"(got {head_dim}); larger head_dim needs K-splitting."
        )
        assert kv_len == self.VLEN, (
            f"kv_len must equal VLEN={self.VLEN} so a query's logits fit one VRAM row "
            f"for in-row softmax (got {kv_len}); larger kv_len needs multi-tile softmax "
            f"or M_TMM (see doc/design/attention_isa_roadmap.md)."
        )
        self.q_len = q_len
        self.kv_len = kv_len
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.causal = causal
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.skip_asm_gen = skip_asm_gen

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        actual_quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        self.quant_config = actual_quant_config
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # 1. Random Q, K, V for a single head. Keep magnitudes modest so the
        #    scaled logits stay in range of the FP12 exp unit.
        q = torch.randn(self.q_len, self.head_dim, dtype=torch.bfloat16)
        k = torch.randn(self.kv_len, self.head_dim, dtype=torch.bfloat16)
        v = torch.randn(self.kv_len, self.head_dim, dtype=torch.bfloat16)

        # 2. Quantize each to the hardware MX format.
        q_q = self._quantize(q)
        q_k = self._quantize(k)
        q_v = self._quantize(v)

        # 2b. Causal mask (q_len, kv_len): 0 where k<=q, MASK_NEG where k>q. Added
        #     to the SCALED logits before softmax (matches the ASM: scale then add).
        if self.causal:
            qi = torch.arange(self.q_len).unsqueeze(1)
            ki = torch.arange(self.kv_len).unsqueeze(0)
            mask = torch.where(ki <= qi, 0.0, self.MASK_NEG).to(torch.float32)
        else:
            mask = torch.zeros(self.q_len, self.kv_len, dtype=torch.float32)

        # 3. Golden attention modelling the hardware datapath precision: S from the
        #    quantized Q,K; + causal mask; row-wise softmax; then QUANTIZE P (the RTL
        #    re-quantizes the softmax output to MXFP when it reads it as the P@V
        #    activation, same as linear quantizes its activation) before P@V with V.
        s = torch.matmul(q_q.float(), q_k.float().transpose(-1, -2)) * self.scale + mask
        p = torch.softmax(s, dim=-1)
        p_q = self._quantize(p.to(torch.bfloat16))
        golden_result = torch.matmul(p_q.float(), q_v.float())  # (q_len, head_dim)

        # 4. Memory layout inputs: HBM holds [Q, K^T, V, mask].
        #    K is stored TRANSPOSED as (d, kv_len) so Q@K^T can use the proven
        #    projection_asm (act @ weight, weight stored (in=d, out=kv_len))
        #    instead of projection_T_asm (which has a period-BLEN weight-select
        #    bug for out_features <= mlen). Golden still uses the natural q_k.
        q_k_hbm = q_k.transpose(0, 1).contiguous()  # (d, kv_len)
        input_tensors = {
            "q_tensor": q_q.to(torch.bfloat16),
            "k_tensor": q_k_hbm.to(torch.bfloat16),
            "v_tensor": q_v.to(torch.bfloat16),
        }
        specified_data_order = ["q_tensor", "k_tensor", "v_tensor"]
        tensor_shapes = [
            (self.q_len, self.head_dim),
            (self.head_dim, self.kv_len),   # K stored transposed
            (self.kv_len, self.head_dim),
        ]
        if self.causal:
            # 4th tensor: the causal mask, pre-staged in VRAM and added to S.
            input_tensors["mask_tensor"] = mask.to(torch.bfloat16)
            specified_data_order.append("mask_tensor")
            tensor_shapes.append((self.q_len, self.kv_len))

        # 5. Instruction storage offset (suite mode may force a shared offset).
        auto_offset = calculate_instr_storage_offset_from_shapes(
            tensor_shapes, precision_settings, hbm_row_width
        )
        forced = getattr(self, "force_instr_offset", None)
        if forced is not None:
            assert forced >= auto_offset, (
                f"force_instr_offset {forced} < data size {auto_offset} for this case"
            )
            instr_offset = forced
        else:
            instr_offset = auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        # 6. Save tensors + debug txt.
        tensor_paths = self._save_tensors(input_tensors)
        paths["tensors"] = tensor_paths

        # 7. Assembly.
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(
                    f"--skip-asm-gen was set but no existing assembly was found at {asm_path}."
                )
            print(f"Skipping assembly generation; reusing existing {asm_path}")
        else:
            asm_path.write_text(self._generate_assembly_with_template())
        paths["asm"] = asm_path

        # 8. FP SRAM preload: slot 0 = 0.0, slot 1 = 1/sqrt(d) (QK scale).
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        fp_preload = [0.0, self.scale]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        # 9. HBM mem files.
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="attention",
            data=None,
            specified_data_order=specified_data_order,
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=self.mx_format,
            instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 10. Golden VRAM for O (strided layout, same convention as linear).
        #     O is (q_len, head_dim); head_dim == VLEN so a single out-tile.
        o_base_row = (self.q_len * self.head_dim + self.q_len * self.kv_len) // self.VLEN
        golden_2d = golden_result.reshape(self.q_len, self.head_dim).to(torch.float32)
        golden_3d = golden_2d.reshape(self.q_len, self.head_dim // self.VLEN, self.VLEN)
        golden_transposed = golden_3d.permute(1, 0, 2)
        total_vram_rows = self.q_len * (self.head_dim // self.VLEN)
        golden_vram = golden_transposed.reshape(total_vram_rows, self.VLEN)
        vram_golden_paths = save_golden_vram(
            golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN
        )
        paths["golden_vram"] = vram_golden_paths

        golden_paths = self._save_golden(golden_result, filename="golden_result.pt")
        paths["golden"] = golden_paths

        # 11. Verification params: check the O region (all q_len rows).
        actual_format = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False,
            "check_vram": True,
            "vram_start_row_idx": o_base_row,
            "vram_num_rows": total_vram_rows,
            "row_dim": self.VLEN,
            "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_vram_rows,
            "vram_total_rows": total_vram_rows,
            "golden_vram_file": "golden_vram_result.pt",
            "workload_type": "attention",
            "q_len": self.q_len,
            "kv_len": self.kv_len,
            "head_dim": self.head_dim,
            "output_shape": list(golden_result.shape),
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        # 12. Test params.
        test_params = {
            "workload_type": "attention",
            "q_len": self.q_len,
            "kv_len": self.kv_len,
            "head_dim": self.head_dim,
            "scale": self.scale,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
            "seed": self.seed,
        }
        with open(self.build_dir / "test_params.json", "w") as f:
            json.dump(test_params, f, indent=2)
        paths["test_params"] = self.build_dir / "test_params.json"

        # 13. Env vars.
        self._set_env_vars(paths)
        return paths

    def _generate_assembly_with_template(self) -> str:
        vlen = self.VLEN
        mlen = self.MLEN
        blen = self.BLEN
        preload_len = self.HBM_V_Prefetch_Amount

        # HBM element offsets for K, V, mask (Q lives at offset 0, reg a0).
        # Each tensor's base is the hbm_row_width-aligned cumulative size of the
        # tensors PRECEDING it (element and scale regions land on separate aligned
        # rows), matching the stager. The old packed real_data_ratio estimate
        # under-counted the scale-row padding at MLEN=8, landing bases too low so
        # the reader fetched zeros. Order matches specified_data_order: Q, K^T, V.
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        q_shape = (self.q_len, self.head_dim)
        k_shape = (self.head_dim, self.kv_len)   # K stored transposed
        v_shape = (self.kv_len, self.head_dim)
        k_hbm_offset = calculate_instr_storage_offset_from_shapes(
            [q_shape], precision_settings, hbm_row_width
        )
        v_hbm_offset = calculate_instr_storage_offset_from_shapes(
            [q_shape, k_shape], precision_settings, hbm_row_width
        )
        m_hbm_offset = calculate_instr_storage_offset_from_shapes(
            [q_shape, k_shape, v_shape], precision_settings, hbm_row_width
        )

        # VRAM element bases.
        q_base = 0
        s_base = self.q_len * self.head_dim
        o_base = s_base + self.q_len * self.kv_len
        m_base = o_base + self.q_len * self.head_dim   # causal mask staging

        code = "; Single-Head Attention (Q@K^T -> softmax -> P@V)\n"
        code += (
            f"; q_len={self.q_len}, kv_len={self.kv_len}, d={self.head_dim}, "
            f"scale=1/sqrt(d)={self.scale:.6f}\n"
        )
        code += f"; MLEN={mlen}, BLEN={blen}, VLEN={vlen}\n"
        code += f"; VRAM: Q@{q_base}  S/P@{s_base}  O@{o_base}\n\n"

        # a1 = K HBM base, a2 = V HBM base (a3 = mask HBM base when causal).
        addr_regs = [1, 2] + ([3] if self.causal else [])
        addr_vals = [k_hbm_offset, v_hbm_offset] + ([m_hbm_offset] if self.causal else [])
        code += preload_addr_reg_asm(
            addr_reg_to_set=addr_regs,
            available_registers=addr_regs,
            addr_reg_val=addr_vals,
        )
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

        # Stage Q (q_len, d) HBM -> VRAM at q_base, layout (d//mlen, q_len, mlen).
        code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=self.q_len,
            hidden_size=self.head_dim,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=q_base,
            activation_offset_reg=0,
            stride_size=self.head_dim,
        )

        # Stage the causal mask (q_len, kv_len) HBM -> VRAM at m_base (a3 = base).
        if self.causal:
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
            code += "; --- preload causal mask ---\n"
            code += preload_act_asm(
                vlen=vlen,
                preload_len=preload_len,
                batch=self.q_len,
                hidden_size=self.kv_len,
                alive_registers=[1, 2, 3, 4, 5],
                act_vram_offset=m_base,
                activation_offset_reg=3,
                stride_size=self.kv_len,
            )

        # S = Q @ K^T  via the proven projection_asm.  K is stored TRANSPOSED in
        # HBM as (d, kv_len), so as a projection weight (in=d, out=kv_len) it yields
        # act @ weight = Q @ K^T = S (q_len, kv_len).  (projection_T_asm has a
        # period-BLEN weight-select bug for out_features <= mlen; projection_asm is
        # the same path linear uses and passes.)
        code += "; --- Q @ K^T -> S (projection_asm, K stored transposed) ---\n"
        code += projection_asm(
            mlen=mlen,
            blen=blen,
            batch=self.q_len,
            hidden_size=self.head_dim,       # in_features = d
            alive_registers=[1, 2, 3, 4, 5, 6],
            w_base_hbm_offset_reg=1,          # a1 -> K^T base
            activation_base_address=q_base,
            result_base_address=s_base,
            out_features=self.kv_len,
        )

        # Reset scratch regs before the vector softmax phase.
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

        # softmax(S) in place, row by row (+ causal mask add when enabled).
        code += "; --- softmax(S)%s -> P ---\n" % (" + causal mask" if self.causal else "")
        code += softmax_rows_asm(
            s_base_address=s_base,
            q_len=self.q_len,
            kv_len=self.kv_len,
            vlen=vlen,
            scale_fp_address=1,               # fp_sram slot 1 = 1/sqrt(d)
            alive_registers=[1, 2, 3],
            mask_base_address=(m_base if self.causal else None),
        )

        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

        # O = P @ V  via projection (act=P at s_base, weight=V stored (kv_len,d)).
        code += "; --- P @ V -> O ---\n"
        code += projection_asm(
            mlen=mlen,
            blen=blen,
            batch=self.q_len,
            hidden_size=self.kv_len,          # in_features = kv_len
            alive_registers=[1, 2, 3, 4, 5, 6],
            w_base_hbm_offset_reg=2,          # a2 -> V base
            activation_base_address=s_base,
            result_base_address=o_base,
            out_features=self.head_dim,
        )

        code += "\n; End of attention test\n"
        code += "C_BREAK\n"
        return code

    def get_config(self) -> dict:
        return {
            "workload_type": "attention",
            "q_len": self.q_len,
            "kv_len": self.kv_len,
            "head_dim": self.head_dim,
            "scale": self.scale,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "causal": self.causal,
            "quant_config": self.quant_config,
            "seed": self.seed,
            "mx_format": self.mx_format,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-head attention workload for PLENA RTL simulation"
    )
    parser.add_argument("--q-len", type=int, default=_VLEN, help="Query sequence length (divisible by BLEN)")
    parser.add_argument("--kv-len", type=int, default=_VLEN, help="Key/Value sequence length (== VLEN)")
    parser.add_argument("--head-dim", type=int, default=_MLEN, help="Head dimension (== MLEN)")
    parser.add_argument("--build-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"],
                        help="MX format (default: use precision.svh)")
    parser.add_argument("--no-causal", dest="causal", action="store_false",
                        help="Disable the causal mask (default: causal, Llama-style)")
    parser.add_argument("--skip-asm-gen", action="store_true",
                        help="Reuse existing generated_asm_code.asm, only re-assemble")
    args = parser.parse_args()

    workload = AttentionWorkload(
        q_len=args.q_len,
        kv_len=args.kv_len,
        head_dim=args.head_dim,
        build_dir=args.build_dir,
        seed=args.seed,
        mx_format=args.mx_format,
        causal=args.causal,
        skip_asm_gen=args.skip_asm_gen,
    )

    print("Generating single-head attention workload:")
    print(f"  Q@K^T: ({args.q_len}, {args.head_dim}) @ ({args.kv_len}, {args.head_dim})^T")
    print(f"  P@V:   ({args.q_len}, {args.kv_len}) @ ({args.kv_len}, {args.head_dim})")
    print(f"  Build dir: {workload.build_dir}")

    paths = workload.generate()
    print("\nGenerated files:")
    for key, value in paths.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {key}/{subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
