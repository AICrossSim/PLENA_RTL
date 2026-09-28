"""Multi-head attention workload for PLENA (bare MHA, no output projection).

At MLEN=16 the ISA's head-packed M_BTMM degenerates to one head, so multi-head
attention is just the proven single-head pipeline (Q@Kᵀ → softmax → P@V, see
attention.py) looped over `num_heads` head-slices and concatenated.

Per head h (head_dim == MLEN, kv_len == VLEN):
    Q_h,K_h,V_h are (seq, head_dim) slices; O_h = softmax(Q_h@K_hᵀ·scale [+mask]) @ V_h
Output O is head-major (num_heads, seq, head_dim) in VRAM.

VRAM: Q_h @0 and S/P @s_base are REUSED each head; the causal mask is staged once;
each head's O_h persists at o_base + h*(seq*head_dim). HBM holds [Q, Kᵀ, V, mask]
head-major so each head's tensors are contiguous.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
for _p in ("tools", "PLENA_Tools", "PLENA_Compiler"):
    _pp = str(_PROJECT_PATH / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)

from .base import WorkloadGenerator
from .attention import softmax_rows_asm

from asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm, projection_asm
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")
# Defaults follow the tile sizes (head_dim == MLEN, kv_len == VLEN are hard constraints).
_MLEN, _VLEN = _HW_TILE_SIZES["MLEN"], _HW_TILE_SIZES["VLEN"]
DEFAULT_MX_FORMAT = None


class MHAWorkload(WorkloadGenerator):
    """Multi-head attention (bare, no W_o), single-tile per head at MLEN=16."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    MASK_NEG = -12.0

    def __init__(self, num_heads=4, q_len=_VLEN, kv_len=_VLEN, head_dim=_MLEN,
                 mx_format=None, causal=True, skip_asm_gen=False, **kwargs):
        super().__init__(**kwargs)
        assert q_len % self.BLEN == 0, f"q_len must be divisible by BLEN={self.BLEN}"
        assert head_dim == self.MLEN, f"head_dim must == MLEN={self.MLEN}"
        assert kv_len == self.VLEN, f"kv_len must == VLEN={self.VLEN}"
        self.num_heads = num_heads
        self.q_len = q_len
        self.kv_len = kv_len
        self.head_dim = head_dim
        self.hidden = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.causal = causal
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.skip_asm_gen = skip_asm_gen

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()
        H, S, D, KV = self.num_heads, self.q_len, self.head_dim, self.kv_len

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # 1-2. Random per-head Q/K/V (head-major), quantized to the HW MX format.
        q = torch.randn(H, S, D, dtype=torch.bfloat16)
        k = torch.randn(H, KV, D, dtype=torch.bfloat16)
        v = torch.randn(H, KV, D, dtype=torch.bfloat16)
        # Quantizer is 2D-only; block-8 along the feature dim is identical whether
        # rows come from one head or many, so flatten (H,·,D)->(H·,D) and reshape back.
        q_q = self._quantize(q.reshape(H * S, D)).reshape(H, S, D)
        q_k = self._quantize(k.reshape(H * KV, D)).reshape(H, KV, D)
        q_v = self._quantize(v.reshape(H * KV, D)).reshape(H, KV, D)

        # 2b. Causal mask (shared across heads).
        if self.causal:
            qi = torch.arange(S).unsqueeze(1); ki = torch.arange(KV).unsqueeze(0)
            mask = torch.where(ki <= qi, 0.0, self.MASK_NEG).to(torch.float32)
        else:
            mask = torch.zeros(S, KV, dtype=torch.float32)

        # 3. Golden: per-head single-head attention (datapath-accurate P re-quant), stacked.
        O_heads = []
        for h in range(H):
            s = torch.matmul(q_q[h].float(), q_k[h].float().transpose(-1, -2)) * self.scale + mask
            p_q = self._quantize(torch.softmax(s, dim=-1).to(torch.bfloat16))
            O_heads.append(torch.matmul(p_q.float(), q_v[h].float()))   # (S, D)
        golden_result = torch.stack(O_heads, dim=0)                     # (H, S, D)

        # 4. HBM tensors. Each head's Kᵀ/V is packed as its OWN tensor (a self-
        #    contained single-head (D,KV)/(KV,D) weight), NOT one (H*D,KV)/(H*KV,D)
        #    tensor. The MXINT weight layout keeps elements and scales in separate
        #    per-tensor regions, and the HBM controller finds a weight's scale via
        #    C_SET_SCALE_REG = in_features*out_features (one head's size). A single
        #    H-packed weight tensor has an H*-sized element region, so per-head scale
        #    lookups miss the scale region entirely (only the last head's lands on it).
        #    Per-head tensors give each the single-head layout the controller expects.
        #    Order: q, k_0..k_{H-1}, v_0..v_{H-1}, [mask].
        q_k_hbm = q_k.transpose(-1, -2).contiguous()                   # (H, D, KV)
        input_tensors = {"q_tensor": q_q.to(torch.bfloat16)}
        specified_data_order = ["q_tensor"]
        tensor_shapes = [(H * S, D)]
        for h in range(H):
            input_tensors[f"k{h}_tensor"] = q_k_hbm[h].contiguous().to(torch.bfloat16)
            specified_data_order.append(f"k{h}_tensor")
            tensor_shapes.append((D, KV))
        for h in range(H):
            input_tensors[f"v{h}_tensor"] = q_v[h].contiguous().to(torch.bfloat16)
            specified_data_order.append(f"v{h}_tensor")
            tensor_shapes.append((KV, D))
        if self.causal:
            input_tensors["mask_tensor"] = mask.to(torch.bfloat16)
            specified_data_order.append("mask_tensor")
            tensor_shapes.append((S, KV))

        # 5. Instruction offset.
        auto_offset = calculate_instr_storage_offset_from_shapes(tensor_shapes, precision_settings, hbm_row_width)
        forced = getattr(self, "force_instr_offset", None)
        instr_offset = forced if forced is not None else auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        paths["tensors"] = self._save_tensors(input_tensors)

        # 6-7. Assembly.
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(f"--skip-asm-gen set but no asm at {asm_path}")
        else:
            asm_path.write_text(self._generate_assembly())
        paths["asm"] = asm_path

        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        paths["fp_sram"] = write_fp_sram_hex([0.0, self.scale], self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        create_mem_for_sim(
            precision_settings=precision_settings, data_size=256, mode="behave_sim",
            asm="mha", data=None, specified_data_order=specified_data_order,
            build_path=self.build_dir, hbm_row_width=hbm_row_width,
            mx_format=self.mx_format, instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 8. Golden VRAM: O head-major (H*S rows of VLEN=D). All Q staged first
        #    (H*S*D), then per-head S/P scratch (H*S*KV, one region per head so a
        #    later head's QK^T can't clobber an earlier head's P mid-P@V), then O.
        o_base = H * S * D + H * S * KV
        total_vram_rows = H * S * (D // self.VLEN)
        golden_vram = golden_result.reshape(total_vram_rows, self.VLEN).to(torch.float32)
        paths["golden_vram"] = save_golden_vram(golden_vram, output_dir=self.build_dir,
                                                filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")

        # 9. Verification params: check the whole O region.
        fmt = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False, "check_vram": True,
            "vram_start_row_idx": o_base // self.VLEN, "vram_num_rows": total_vram_rows,
            "row_dim": self.VLEN, "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_vram_rows, "vram_total_rows": total_vram_rows,
            "golden_vram_file": "golden_vram_result.pt", "workload_type": "mha",
            "num_heads": H, "q_len": S, "kv_len": KV, "head_dim": D,
            "output_shape": list(golden_result.shape), "mx_format": fmt,
            "exp_width": self.quant_config["exp_width"], "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"
        self._set_env_vars(paths)
        return paths

    def _generate_assembly(self) -> str:
        vlen, mlen, blen = self.VLEN, self.MLEN, self.BLEN
        H, S, D, KV = self.num_heads, self.q_len, self.head_dim, self.kv_len
        preload_len = self.HBM_V_Prefetch_Amount

        # HBM element offsets. K/V are packed one tensor PER HEAD (see generate),
        # so each head's Kᵀ_h / V_h is a self-contained single-head weight. The HBM
        # controller derives a weight's MXINT scale region from its BASE address reg
        # (base + in*out), so each head's base must go in the a-reg itself — not a
        # gp offset — else every head reads head 0's scale. a1/a2 are set per head.
        # Each tensor's base is the hbm_row_width-aligned cumulative size of the
        # tensors PRECEDING it (element and scale regions land on separate aligned
        # rows), matching the stager. The old packed int(size * ratio) estimate
        # under-counted the scale-row padding at MLEN=8 so the reader fetched the
        # wrong rows (same fix as attention.py). Order matches specified_data_order:
        # q, k_0..k_{H-1}, v_0..v_{H-1}, [mask].
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        q_shape, k_shape, v_shape = (H * S, D), (D, KV), (KV, D)
        _hbm = lambda shapes: (calculate_instr_storage_offset_from_shapes(
            shapes, precision_settings, hbm_row_width) if shapes else 0)
        k_hbm = lambda h: _hbm([q_shape] + [k_shape] * h)                  # head h's Kᵀ tensor base
        v_hbm = lambda h: _hbm([q_shape] + [k_shape] * H + [v_shape] * h)  # head h's V tensor base
        m_hbm = _hbm([q_shape] + [k_shape] * H + [v_shape] * H)

        # VRAM element bases. All heads' Q staged once (head-major, H*S rows); each
        # head gets its OWN S/P scratch (so head h+1's QK^T write can't clobber head
        # h's P while head h's P@V is still reading it); O head-major. a0 stays 0.
        q_base = 0
        s_base = H * S * D
        o_base = s_base + H * S * KV
        m_base = o_base + H * S * D
        q_slice = lambda h: q_base + h * S * D    # head h's Q rows in VRAM
        s_slice = lambda h: s_base + h * S * KV   # head h's S/P scratch

        code = f"; Multi-Head Attention: {H} heads, seq={S}, d={D}, kv={KV}, causal={self.causal}\n"
        code += f"; MLEN={mlen} BLEN={blen} VLEN={vlen}  VRAM: Q@{q_base} S/P@{s_base} O@{o_base} mask@{m_base}\n\n"

        # a3 = mask HBM base (set once, for the shared mask preload). a0 stays 0 (Q).
        # a1 = Kᵀ_h, a2 = V_h are set per head inside the loop (base-reg drives the
        # MXINT scale lookup, so they must hold each head's own tensor base).
        if self.causal:
            code += preload_addr_reg_asm(addr_reg_to_set=[3], available_registers=[3], addr_reg_val=[m_hbm])
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
        code += "; --- preload all Q (head-major) ---\n"
        code += preload_act_asm(vlen=vlen, preload_len=preload_len, batch=H * S, hidden_size=D,
                                alive_registers=[1, 2, 3, 4, 5], act_vram_offset=q_base,
                                activation_offset_reg=0, stride_size=D)
        if self.causal:
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
            code += "; --- preload causal mask (shared) ---\n"
            code += preload_act_asm(vlen=vlen, preload_len=preload_len, batch=S, hidden_size=KV,
                                    alive_registers=[1, 2, 3, 4, 5], act_vram_offset=m_base,
                                    activation_offset_reg=3, stride_size=KV)

        for h in range(H):
            code += f"\n; ===== head {h} =====\n"
            # a1 = Kᵀ_h base, a2 = V_h base (each its own tensor -> correct MXINT scale).
            code += preload_addr_reg_asm(addr_reg_to_set=[1, 2], available_registers=[1, 2],
                                         addr_reg_val=[k_hbm(h), v_hbm(h)])
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])
            # S = Q_h @ K_hᵀ  (S -> head h scratch).
            code += f"; head {h}: Q@K^T -> S\n"
            code += projection_asm(mlen=mlen, blen=blen, batch=S, hidden_size=D,
                                   alive_registers=[1, 2, 3, 4, 5, 6], w_base_hbm_offset_reg=1,
                                   activation_base_address=q_slice(h), result_base_address=s_slice(h),
                                   out_features=KV)
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])
            # softmax(S [+mask]).
            code += f"; head {h}: softmax\n"
            code += softmax_rows_asm(s_base_address=s_slice(h), q_len=S, kv_len=KV, vlen=vlen,
                                     scale_fp_address=1, alive_registers=[1, 2, 3],
                                     mask_base_address=(m_base if self.causal else None))
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])
            # O_h = P @ V_h -> o_base + h*(S*D).
            code += f"; head {h}: P@V -> O[{h}]\n"
            code += projection_asm(mlen=mlen, blen=blen, batch=S, hidden_size=KV,
                                   alive_registers=[1, 2, 3, 4, 5, 6], w_base_hbm_offset_reg=2,
                                   activation_base_address=s_slice(h), result_base_address=o_base + h * S * D,
                                   out_features=D)

        code += "\n; End of MHA\nC_BREAK\n"
        return code

    def get_config(self) -> dict:
        return {"workload_type": "mha", "num_heads": self.num_heads, "q_len": self.q_len,
                "kv_len": self.kv_len, "head_dim": self.head_dim, "hidden": self.hidden,
                "causal": self.causal, "mlen": self.MLEN, "blen": self.BLEN, "vlen": self.VLEN,
                "seed": self.seed, "mx_format": self.mx_format}


def main():
    p = argparse.ArgumentParser(description="Generate multi-head attention workload")
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--q-len", type=int, default=_VLEN)
    p.add_argument("--kv-len", type=int, default=_VLEN)
    p.add_argument("--head-dim", type=int, default=_MLEN)
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"])
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--skip-asm-gen", action="store_true")
    a = p.parse_args()
    w = MHAWorkload(num_heads=a.num_heads, q_len=a.q_len, kv_len=a.kv_len, head_dim=a.head_dim,
                    build_dir=a.build_dir, seed=a.seed, mx_format=a.mx_format,
                    causal=a.causal, skip_asm_gen=a.skip_asm_gen)
    print(f"Generating MHA: {a.num_heads} heads, seq={a.q_len}, d={a.head_dim}, hidden={w.hidden}")
    paths = w.generate()
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
