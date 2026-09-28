"""Single-head attention with 1D RoPE (rotary position embedding).

Demonstrates RoPE on the PLENA RTL:
  * Q (a preloaded activation) gets RoPE applied ON-CHIP via rope_asm:
        Q = Q*cos + rotate_half(Q)*sin
    with rotate_half(Q), cos, sin pre-staged in VRAM (rotate_half is a fixed
    sign-flip+swap of the given Q, and cos/sin are position tables — all
    data-independent, so precomputed and preloaded).
  * K is consumed as an HBM matmul weight, so its RoPE is baked in OFFLINE
    (K_roped stored transposed in HBM).
  * V is not rotated.
Then the proven single-head pipeline: S=Q_roped@K_roped^T -> softmax -> O=P@V.

HF/NeoX RoPE convention: rotate_half(x)=[-x[d/2:], x[:d/2]]; theta_j=base^(-2j/d)
for j in 0..d/2-1; cos/sin repeat the d/2 angles across the two halves.
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

from asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm, projection_asm, rope_asm
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW = load_hardware_tile_sizes(SRC_PATH / "definitions")
# Defaults follow the tile sizes; the golden assumes q_len == kv_len (self-attention).
_MLEN, _VLEN = _HW["MLEN"], _HW["VLEN"]


def _rope_tables(S, d, base=10000.0):
    """cos/sin tables (S, d) in HF layout (d/2 angles repeated across halves)."""
    j = torch.arange(d // 2, dtype=torch.float32)
    theta = base ** (-2.0 * j / d)                       # (d/2,)
    pos = torch.arange(S, dtype=torch.float32)
    ang = pos[:, None] * theta[None, :]                  # (S, d/2)
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)      # (S, d)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos, sin


def _rotate_half(x):
    d = x.shape[-1]
    return torch.cat([-x[..., d // 2:], x[..., : d // 2]], dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class RoPEWorkload(WorkloadGenerator):
    MLEN = _HW["MLEN"]; BLEN = _HW["BLEN"]; VLEN = _HW["VLEN"]
    HBM_V_Prefetch_Amount = _HW["HBM_V_Prefetch_Amount"]
    MASK_NEG = -12.0

    def __init__(self, q_len=_VLEN, kv_len=_VLEN, head_dim=_MLEN, rope_base=10000.0,
                 mx_format=None, causal=True, skip_asm_gen=False, stage="full",
                 use_aten_compiler=False, **kwargs):
        super().__init__(**kwargs)
        assert q_len % self.BLEN == 0 and head_dim == self.MLEN and kv_len == self.VLEN
        self.stage = stage
        self.S = q_len; self.KV = kv_len; self.Dh = head_dim
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(head_dim)
        self.causal = causal; self.mx_format = mx_format; self.skip_asm_gen = skip_asm_gen
        # When True, emit ISA via the ATen PlenaCompiler. The ATen backend only
        # exposes RoPE as the in-place RoPE(Q) primitive (ops.rope), NOT the full
        # single-head attention pipeline the template builds. So the ATen path is a
        # self-contained RoPE(Q) test: load Q/Qrot/cos/sin, apply ops.rope in place,
        # and verify the roped-Q output. The ATen-emitted RoPE writes Q back at Q's
        # VRAM address (row 0), so verification targets Q's region.
        self.use_aten_compiler = use_aten_compiler
        # Stashed by _generate_assembly_with_aten_compiler(): the VRAM element
        # address where ops.rope leaves the roped Q (== Q's load address).
        self._aten_output_vram_addr = None

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()
        S, KV, Dh = self.S, self.KV, self.Dh
        ps, cs = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, ps)
        hbm_row_width = cs.get("HBM_WIDTH", 256)
        q = self._quantize
        g = torch.Generator().manual_seed(self.seed)

        Q = torch.randn(S, Dh, generator=g, dtype=torch.bfloat16)
        K = torch.randn(KV, Dh, generator=g, dtype=torch.bfloat16)
        V = torch.randn(KV, Dh, generator=g, dtype=torch.bfloat16)
        cos, sin = _rope_tables(S, Dh, self.rope_base)           # (S, Dh) — S==KV here

        qQ = q(Q); qcos = q(cos); qsin = q(sin)
        qQrot = _rotate_half(qQ)                                  # sign-flip+swap of quantized Q
        # On-chip RoPE applies to qQ; model it: Q_roped = qQ*qcos + qQrot*qsin
        Q_roped = qQ.float() * qcos.float() + qQrot.float() * qsin.float()
        # K RoPE baked offline (K is an HBM weight). Use the same quantized tables.
        qK = q(K)
        K_roped = qK.float() * qcos.float() + _rotate_half(qK).float() * qsin.float()
        qK_roped = q(K_roped.to(torch.bfloat16))
        qV = q(V)

        # Golden: attention(Q_roped, K_roped, V)
        if self.causal:
            qi = torch.arange(S).unsqueeze(1); ki = torch.arange(KV).unsqueeze(0)
            mask = torch.where(ki <= qi, 0.0, self.MASK_NEG).to(torch.float32)
        else:
            mask = torch.zeros(S, KV, dtype=torch.float32)
        raw_scores = q(Q_roped.to(torch.bfloat16)).float() @ qK_roped.float().t()   # (S, KV) unscaled
        s = raw_scores * self.scale + mask
        p = q(torch.softmax(s, dim=-1).to(torch.bfloat16))
        golden = (p.float() @ qV.float())                        # (S, Dh)
        if self.stage == "scores":
            golden_result = raw_scores.to(torch.float32)         # debug: verify raw QK^T (roped Q)
        elif self.stage == "scores_raw":
            golden_result = (qQ.float() @ qK_roped.float().t())  # debug: raw preloaded-Q @ K^T (no on-chip rope)
        else:
            golden_result = golden.to(torch.float32)

        # HBM: Q, Qrot, cos, sin (activations, preloaded), K_roped^T, V (weights), mask
        qK_hbm = qK_roped.transpose(-1, -2).contiguous()         # (Dh, KV) transposed
        input_tensors = {
            "q_tensor": qQ.to(torch.bfloat16), "qrot_tensor": qQrot.to(torch.bfloat16),
            "cos_tensor": qcos.to(torch.bfloat16), "sin_tensor": qsin.to(torch.bfloat16),
            "k_tensor": qK_hbm.to(torch.bfloat16), "v_tensor": qV.to(torch.bfloat16),
        }
        order = ["q_tensor", "qrot_tensor", "cos_tensor", "sin_tensor", "k_tensor", "v_tensor"]
        shapes = [(S, Dh), (S, Dh), (S, Dh), (S, Dh), (Dh, KV), (KV, Dh)]
        if self.causal:
            input_tensors["mask_tensor"] = mask.to(torch.bfloat16); order.append("mask_tensor"); shapes.append((S, KV))

        instr_offset = getattr(self, "force_instr_offset", None) or \
            calculate_instr_storage_offset_from_shapes(shapes, ps, hbm_row_width)
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")
        paths["tensors"] = self._save_tensors(input_tensors)

        # The ATen path only exercises RoPE(Q) (ops.rope), so its golden is the
        # roped Q, not the full-attention output the template computes.
        if self.use_aten_compiler:
            golden_result = Q_roped.to(torch.float32)

        asm_code = self._generate_assembly_with_aten_compiler() if self.use_aten_compiler else self._asm()
        (self.build_dir / "generated_asm_code.asm").write_text(asm_code)
        paths["asm"] = self.build_dir / "generated_asm_code.asm"
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        paths["fp_sram"] = write_fp_sram_hex([0.0, self.scale], self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)
        create_mem_for_sim(precision_settings=ps, data_size=256, mode="behave_sim", asm="rope",
                           data=None, specified_data_order=order, build_path=self.build_dir,
                           hbm_row_width=hbm_row_width, mx_format=self.mx_format, instr_storage_offset=instr_offset)
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # Template leaves the attention output in the 'o' VRAM region; the ATen
        # RoPE(Q) path leaves the roped Q in place at Q's load address (captured
        # while emitting the ISA, defaulting to VRAM row 0).
        o_base = self._aten_output_vram_addr if self.use_aten_compiler else self._vram()["o"]
        total_rows = S * (Dh // self.VLEN)
        golden_vram = golden_result.reshape(S, Dh // self.VLEN, self.VLEN).permute(1, 0, 2).reshape(total_rows, self.VLEN)
        paths["golden_vram"] = save_golden_vram(golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")
        fmt = self.quant_config.get("format", "mxfp")
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump({"check_hbm": False, "check_vram": True, "vram_start_row_idx": o_base // self.VLEN,
                       "vram_num_rows": total_rows, "row_dim": self.VLEN, "vram_compare_start_row": 0,
                       "vram_compare_num_rows": total_rows, "vram_total_rows": total_rows,
                       "golden_vram_file": "golden_vram_result.pt", "workload_type": "rope",
                       "q_len": S, "kv_len": KV, "head_dim": Dh, "output_shape": list(golden_result.shape),
                       "mx_format": fmt, "exp_width": self.quant_config["exp_width"],
                       "man_width": self.quant_config["man_width"], "scale_width": self.quant_config["exp_bias_width"]}, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"
        self._set_env_vars(paths)
        return paths

    def _vram(self):
        S, Dh, KV = self.S, self.Dh, self.KV
        m = {"q": 0}
        m["qrot"] = m["q"] + S * Dh
        m["cos"] = m["qrot"] + S * Dh
        m["sin"] = m["cos"] + S * Dh
        m["sp"] = m["sin"] + S * Dh
        m["o"] = m["sp"] + S * KV
        m["mask"] = m["o"] + S * Dh
        m["scr"] = m["mask"] + S * KV
        return m

    def _asm(self) -> str:
        vlen, mlen, blen = self.VLEN, self.MLEN, self.BLEN
        S, KV, Dh = self.S, self.KV, self.Dh
        pl = self.HBM_V_Prefetch_Amount
        v = self._vram()
        # HBM offsets in order q, qrot, cos, sin, k, v, mask. Each base is the
        # hbm_row_width-aligned cumulative size of the preceding tensors (matching
        # the stager); the old int(size * ratio) estimate under-counted the
        # scale-row padding at MLEN=8 (same fix as attention.py).
        ps, cs = load_precision_from_svh(SRC_PATH / "definitions")
        hbm_row_width = cs.get("HBM_WIDTH", 256)
        _hbm = lambda shapes: (calculate_instr_storage_offset_from_shapes(shapes, ps, hbm_row_width)
                               if shapes else 0)
        qs, ks, vs = (S, Dh), (Dh, KV), (KV, Dh)
        q_hbm = 0; qrot_hbm = _hbm([qs]); cos_hbm = _hbm([qs] * 2); sin_hbm = _hbm([qs] * 3)
        k_hbm = _hbm([qs] * 4); v_hbm = _hbm([qs] * 4 + [ks]); mask_hbm = _hbm([qs] * 4 + [ks, vs])
        R6 = [1, 2, 3, 4, 5, 6]
        code = f"; Single-head attention + 1D RoPE, S={S} d={Dh} causal={self.causal}\n"
        # Preload Q/Qrot/cos/sin/mask FIRST using only a0-a3 as HBM-base regs (a0=0=Q base).
        # Then set a1=K, a2=V for the attention (mask lives in VRAM, needs no a-reg there).
        for nm, hbm, dst, areg, hid in [("Q", q_hbm, v["q"], 0, Dh), ("Qrot", qrot_hbm, v["qrot"], 1, Dh),
                                        ("cos", cos_hbm, v["cos"], 2, Dh), ("sin", sin_hbm, v["sin"], 3, Dh)]:
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
            if areg != 0:
                code += preload_addr_reg_asm(addr_reg_to_set=[areg], available_registers=[areg], addr_reg_val=[hbm])
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
            code += f"; preload {nm}\n"
            code += preload_act_asm(vlen=vlen, preload_len=pl, batch=S, hidden_size=hid,
                                    alive_registers=[1, 2, 3, 4, 5], act_vram_offset=dst,
                                    activation_offset_reg=areg, stride_size=hid)
        if self.causal:
            code += preload_addr_reg_asm(addr_reg_to_set=[1], available_registers=[1], addr_reg_val=[mask_hbm])
            code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
            code += "; preload mask\n"
            code += preload_act_asm(vlen=vlen, preload_len=pl, batch=S, hidden_size=KV,
                                    alive_registers=[1, 2, 3, 4, 5], act_vram_offset=v["mask"],
                                    activation_offset_reg=1, stride_size=KV)
        # Point a1=K_roped base, a2=V base for the attention matmuls (mask lives in VRAM).
        code += preload_addr_reg_asm(addr_reg_to_set=[1, 2], available_registers=[1, 2, 3],
                                     addr_reg_val=[k_hbm, v_hbm])
        # On-chip RoPE on Q:  Q = Q*cos + rotate_half(Q)*sin  (skipped in scores_raw to isolate the matmul)
        if self.stage != "scores_raw":
            code += reset_reg_asm(alive_registers=R6)
            code += "; ===== RoPE(Q) on-chip =====\n"
            code += rope_asm(alive_registers=[1, 2, 3, 4, 5], x_base_address=v["q"],
                             x_rot_base_address=v["qrot"], cos_base_address=v["cos"],
                             sin_base_address=v["sin"], scratchpad_base_address=v["scr"],
                             vlen=vlen, seq_len=S, head_dim=Dh, unroll=True)
        # S = Q_roped @ K_roped^T   (scores stages write raw scores straight to the verified 'o' region)
        qkt_dst = v["o"] if self.stage in ("scores", "scores_raw") else v["sp"]
        code += reset_reg_asm(alive_registers=R6)
        code += "; ===== Q@K^T =====\n"
        code += projection_asm(mlen=mlen, blen=blen, batch=S, hidden_size=Dh, alive_registers=R6,
                               w_base_hbm_offset_reg=1, activation_base_address=v["q"],
                               result_base_address=qkt_dst, out_features=KV)
        if self.stage in ("scores", "scores_raw"):
            code += "\n; End (scores stage: raw QK^T)\nC_BREAK\n"
            return code
        code += reset_reg_asm(alive_registers=R6)
        code += "; ===== softmax =====\n"
        code += softmax_rows_asm(s_base_address=v["sp"], q_len=S, kv_len=KV, vlen=vlen, scale_fp_address=1,
                                 alive_registers=[1, 2, 3], mask_base_address=(v["mask"] if self.causal else None))
        code += reset_reg_asm(alive_registers=R6)
        code += "; ===== P@V =====\n"
        code += projection_asm(mlen=mlen, blen=blen, batch=S, hidden_size=KV, alive_registers=R6,
                               w_base_hbm_offset_reg=2, activation_base_address=v["sp"],
                               result_base_address=v["o"], out_features=Dh)
        code += "\n; End RoPE attention\nC_BREAK\n"
        return code

    def _generate_assembly_with_aten_compiler(self) -> str:
        """Generate ISA via the ATen PlenaCompiler (ops.rope / prog.rope).

        The ATen backend exposes RoPE only as the in-place RoPE(Q) primitive:
            Q = Q*cos + rotate_half(Q)*sin
        with rotate_half(Q), cos, sin already resident in VRAM. It does NOT emit
        the full single-head attention pipeline (QK^T -> softmax -> P@V) that the
        template builds, so this is a self-contained RoPE(Q) test whose golden is
        the roped Q (set in generate()).

        Inputs are declared in HBM order (Q, Qrot, cos, sin) so their
        auto-allocated hbm_addr matches specified_data_order. Each is loaded to
        VRAM, then prog.rope applies RoPE in place at Q's VRAM address.
        """
        from compiler.aten.plena import PlenaCompiler
        import compiler.aten.ops as ops

        mlen = self.MLEN
        blen = self.BLEN
        S, Dh = self.S, self.Dh
        real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MXINT8 format ratio
        _, _cfg = load_precision_from_svh(SRC_PATH / "definitions")
        mram_tile_capacity = _cfg.get("MATRIX_SRAM_DEPTH", 1024) // mlen

        prog = PlenaCompiler(
            mlen=mlen,
            blen=blen,
            real_data_ratio=real_data_ratio,
            unroll_loops=True,
            mram_tile_capacity=mram_tile_capacity,
        )

        prog.emit("; RoPE(Q) Test using ATen PlenaCompiler (ops.rope)\n")
        prog.emit(f"; Q shape: ({S}, {Dh})  MLEN={mlen} BLEN={blen}\n\n")

        # Declare inputs in HBM order (matches specified_data_order = q, qrot, cos, sin).
        q_input = prog.input("Q", shape=(S, Dh))
        qrot_input = prog.input("Qrot", shape=(S, Dh))
        cos_input = prog.input("cos", shape=(S, Dh))
        sin_input = prog.input("sin", shape=(S, Dh))

        # Load all four to VRAM (Q lands at row 0, the others follow sequentially).
        q = prog.load_batch(q_input, name="Q")
        q_rot = prog.load_batch(qrot_input, name="Qrot")
        cos = prog.load_batch(cos_input, name="cos")
        sin = prog.load_batch(sin_input, name="sin")

        # Apply RoPE in place: Q = Q*cos + rotate_half(Q)*sin.
        ops.rope(prog, q, q_rot, cos, sin)

        # Capture the VRAM address where the roped Q lives (== Q's load address).
        self._aten_output_vram_addr = prog.get_vram_addr(q.name)

        prog.emit("\n; End of RoPE(Q) test\n")
        prog.emit("C_BREAK\n")
        return prog.compile()

    def get_config(self):
        return {"workload_type": "rope", "q_len": self.S, "kv_len": self.KV, "head_dim": self.Dh,
                "rope_base": self.rope_base, "causal": self.causal, "mlen": self.MLEN,
                "blen": self.BLEN, "vlen": self.VLEN, "seed": self.seed, "mx_format": self.mx_format}


def main():
    p = argparse.ArgumentParser(description="Single-head attention + 1D RoPE")
    p.add_argument("--q-len", type=int, default=_VLEN); p.add_argument("--kv-len", type=int, default=_VLEN)
    p.add_argument("--head-dim", type=int, default=_MLEN); p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--build-dir", type=str, default=None); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"])
    p.add_argument("--no-causal", dest="causal", action="store_false"); p.add_argument("--skip-asm-gen", action="store_true")
    p.add_argument("--stage", type=str, default="full", choices=["full", "scores", "scores_raw"])
    p.add_argument("--use-aten-compiler", action="store_true",
                   help="Use ATen PlenaCompiler (ops.rope) instead of the full-attention template. "
                        "The ATen path is RoPE(Q)-only (the ATen backend does not expose the full "
                        "attention pipeline as a single op); it verifies the roped-Q output.")
    a = p.parse_args()
    w = RoPEWorkload(q_len=a.q_len, kv_len=a.kv_len, head_dim=a.head_dim, rope_base=a.rope_base,
                     build_dir=a.build_dir, seed=a.seed, mx_format=a.mx_format, causal=a.causal,
                     skip_asm_gen=a.skip_asm_gen, stage=a.stage, use_aten_compiler=a.use_aten_compiler)
    print(f"Generating RoPE attention: S={a.q_len}, d={a.head_dim}, base={a.rope_base}")
    for k, val in w.generate().items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
