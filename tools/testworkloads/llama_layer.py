"""Full transformer decoder layer for PLENA (KV-cache / decode variant).

A rotary Llama-style decoder layer composed of proven blocks:

    residual = x
    xn  = rms_norm(x) * gamma_attn           # gamma-weighted RMSNorm
    q_h = RoPE(xn @ Wq_h)                     # per-head Q projection + rotary embedding
    O_h = softmax(q_h @ K_hᵀ·scale [+mask]) @ V_h   # attention; RoPE'd K/V given (cache)
    attn = concat_h(O_h) @ Wo                # output projection
    x   = residual + attn                    # residual 1
    residual = x
    xn2 = rms_norm(x) * gamma_ffn
    ffn = (silu(xn2 @ Wup) * (xn2 @ Wgate)) @ Wdown   # SwiGLU MLP
    x   = residual + ffn                     # residual 2

RoPE (HF/NeoX): Q is rotated ON-CHIP without an on-chip rotate_half — since
rotate_half(xn@Wq) = xn @ rotate_half_cols(Wq), a second projection with a
column-rotated weight (Wq_rot) yields rotate_half(q), then q = q*cos + q_rot*sin
(cos/sin pre-staged (S,Dh) tables). K's RoPE is baked into the K HBM weight offline
(like rope.py). Toggle with rope=False.

Why KV-cache variant: the matrix unit is weight(MRAM←HBM) @ activation(VRAM) — there
is NO activation@activation matmul (M_BTMM is unimplemented). So Q@Kᵀ / P@V require
one operand to be an HBM weight. Q is *computed* from the normed input (activation);
K/V are *given* in HBM as the cache (weights) — exactly the operand roles bare MHA
uses. Computing K/V (prefill) would need a VRAM→HBM→MRAM roundtrip (deferred).

Per-head weights (Wq_h, K_h, V_h) are packed as one HBM tensor PER HEAD so the MXINT
scale region is found correctly (see mha.py). Wo/Wup/Wgate/Wdown are single tensors.

Stages: `attn` verifies the attention sub-layer output (x after residual 1);
`full` verifies the whole layer. Remaining gaps toward a full layer: Wk/Wv
projections (compute K/V — needs an HBM roundtrip), GQA, and larger/multi-tile dims.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
for _p in ("tools", "PLENA_Tools", "PLENA_Compiler"):
    _pp = str(_PROJECT_PATH / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)

from .base import WorkloadGenerator
from .rope import _rope_tables, _rotate_half

# ATen PlenaCompiler: the same fused-attention path GQA/mha use, whose compute_pv
# P@V is the fixed one (V _H precision etc.).
from compiler.aten.plena import PlenaCompiler
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")
# Defaults follow the tile sizes (head_dim == MLEN, seq_len == VLEN are hard constraints).
_MLEN, _VLEN = _HW_TILE_SIZES["MLEN"], _HW_TILE_SIZES["VLEN"]


class LlamaLayerWorkload(WorkloadGenerator):
    """Full decoder layer (KV-cache variant), single tile per head at MLEN=16."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    MASK_NEG = -12.0
    EPS = 1e-6

    def __init__(self, seq_len=_VLEN, num_heads=4, head_dim=_MLEN, inter_size=64,
                 mx_format=None, causal=True, stage="full", skip_asm_gen=False,
                 rope=True, rope_base=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.rope = rope
        self.rope_base = rope_base
        assert seq_len % self.BLEN == 0, f"seq_len must be divisible by BLEN={self.BLEN}"
        assert head_dim == self.MLEN, f"head_dim must == MLEN={self.MLEN}"
        assert inter_size % self.MLEN == 0
        assert stage in ("attn", "full", "attn_qgiven")
        self.S = seq_len
        self.H = num_heads
        self.Dh = head_dim
        self.hidden = num_heads * head_dim
        self.KV = self.S            # kv_len == seq_len (self-attention, single tile)
        assert self.KV == self.VLEN, f"kv_len must == VLEN={self.VLEN}"
        self.inter = inter_size
        self.scale = 1.0 / math.sqrt(head_dim)
        self.causal = causal
        self.stage = stage
        self.mx_format = mx_format
        self.skip_asm_gen = skip_asm_gen

    # ---- golden ------------------------------------------------------------
    def _rms(self, x):
        # weightless RMS over the last dim
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.EPS)

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()
        if self.stage == "attn_qgiven":
            return self._generate_qgiven(paths)
        S, H, Dh, hid, KV, I = self.S, self.H, self.Dh, self.hidden, self.KV, self.inter

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        q = self._quantize

        # ---- random weights / inputs ----
        g = torch.Generator().manual_seed(self.seed)
        x = torch.randn(S, hid, generator=g, dtype=torch.bfloat16)
        Wq = (torch.randn(hid, hid, generator=g) / hid**0.5).to(torch.bfloat16)
        K = torch.randn(H, KV, Dh, generator=g, dtype=torch.bfloat16)   # per-head keys (cache)
        V = torch.randn(H, KV, Dh, generator=g, dtype=torch.bfloat16)   # per-head values (cache)
        Wo = (torch.randn(hid, hid, generator=g) / hid**0.5).to(torch.bfloat16)
        Wup = (torch.randn(hid, I, generator=g) / hid**0.5).to(torch.bfloat16)
        Wgate = (torch.randn(hid, I, generator=g) / hid**0.5).to(torch.bfloat16)
        Wdown = (torch.randn(I, hid, generator=g) / I**0.5).to(torch.bfloat16)
        # learned RMSNorm weights (gamma), ~1.0 like real Llama
        gamma_attn = (1.0 + 0.1 * torch.randn(hid, generator=g)).to(torch.bfloat16)
        gamma_ffn  = (1.0 + 0.1 * torch.randn(hid, generator=g)).to(torch.bfloat16)

        # ---- quantize (per stored tensor) ----
        qx = q(x)
        qWq_h = [q(Wq[:, h*Dh:(h+1)*Dh].contiguous()) for h in range(H)]     # (hid, Dh) each
        qK = torch.stack([q(K[h]) for h in range(H)])                       # (H, KV, Dh)
        qV = torch.stack([q(V[h]) for h in range(H)])                       # (H, KV, Dh)
        qWo = q(Wo)
        qWup, qWgate, qWdown = q(Wup), q(Wgate), q(Wdown)
        # gamma is loaded as a normal activation -> broadcast to (S,hid), quantized like the rest
        qGa = q(gamma_attn.unsqueeze(0).expand(S, hid).contiguous())    # (S, hid)
        qGf = q(gamma_ffn.unsqueeze(0).expand(S, hid).contiguous())
        # RoPE: cos/sin tables (S,Dh), a column-rotated Q weight so rotate_half(q)=xn@Wq_rot
        # (avoids on-chip rotate_half of a computed q), and K's RoPE baked into the K weight.
        if self.rope:
            cos, sin = _rope_tables(S, Dh, self.rope_base)             # (S, Dh); KV==S
            qcos, qsin = q(cos), q(sin)
            qWqrot_h = [q(_rotate_half(Wq[:, h*Dh:(h+1)*Dh].contiguous())) for h in range(H)]
            qK = torch.stack([q((qK[h].float()*qcos.float()
                                 + _rotate_half(qK[h].float())*qsin.float()).to(torch.bfloat16))
                              for h in range(H)])                       # K_roped (baked offline)

        # ---- golden (datapath-accurate: requant at each matmul boundary) ----
        if self.causal:
            qi = torch.arange(S).unsqueeze(1); ki = torch.arange(KV).unsqueeze(0)
            mask = torch.where(ki <= qi, 0.0, self.MASK_NEG).to(torch.float32)
        else:
            mask = torch.zeros(S, KV, dtype=torch.float32)

        if getattr(self, "skip_rms", False):
            xn = qx                                                         # feed x directly (no norm)
        else:
            xn = q((self._rms(qx.float()) * qGa.float()).to(torch.bfloat16))  # (S, hid) rms*gamma
        O_heads = []
        for h in range(H):
            qh = q((xn.float() @ qWq_h[h].float()).to(torch.bfloat16))      # (S, Dh)
            if self.rope:
                qr = q((xn.float() @ qWqrot_h[h].float()).to(torch.bfloat16))   # = rotate_half(q)
                qh = q((qh.float()*qcos.float() + qr.float()*qsin.float()).to(torch.bfloat16))  # q_roped
            s = (qh.float() @ qK[h].float().t()) * self.scale + mask        # (S, KV); qK is K_roped
            p = q(torch.softmax(s, dim=-1).to(torch.bfloat16))
            O_heads.append(q((p.float() @ qV[h].float()).to(torch.bfloat16)))  # (S, Dh)
        O = torch.cat(O_heads, dim=-1)                                      # (S, hid)
        attn = q((O.float() @ qWo.float()).to(torch.bfloat16))             # (S, hid)
        x1 = qx.float() + attn.float()                                     # residual 1

        if self.stage == "attn":
            golden = x1
        else:
            xn2 = q((self._rms(x1) * qGf.float()).to(torch.bfloat16))
            up = xn2.float() @ qWup.float()
            gate = xn2.float() @ qWgate.float()
            hidv = q((F.silu(up) * gate).to(torch.bfloat16))
            ffn = q((hidv.float() @ qWdown.float()).to(torch.bfloat16))
            x2 = x1 + ffn.float()                                          # residual 2
            golden = x2
        golden_result = golden.to(torch.float32)                           # (S, hid)

        # ---- HBM tensors: x, Wq_0..3, K_0..3, V_0..3, Wo, Wup, Wgate, Wdown, [mask] ----
        qK_hbm = qK.transpose(-1, -2).contiguous()                         # (H, Dh, KV) transposed keys
        input_tensors = {"x_tensor": qx.to(torch.bfloat16)}
        order = ["x_tensor"]; shapes = [(S, hid)]
        for h in range(H):
            input_tensors[f"wq{h}_tensor"] = qWq_h[h].to(torch.bfloat16); order.append(f"wq{h}_tensor"); shapes.append((hid, Dh))
        if self.rope:
            for h in range(H):
                input_tensors[f"wqrot{h}_tensor"] = qWqrot_h[h].to(torch.bfloat16); order.append(f"wqrot{h}_tensor"); shapes.append((hid, Dh))
        for h in range(H):
            input_tensors[f"k{h}_tensor"] = qK_hbm[h].contiguous().to(torch.bfloat16); order.append(f"k{h}_tensor"); shapes.append((Dh, KV))
        for h in range(H):
            input_tensors[f"v{h}_tensor"] = qV[h].contiguous().to(torch.bfloat16); order.append(f"v{h}_tensor"); shapes.append((KV, Dh))
        for nm, t, sh in [("wo", qWo, (hid, hid)), ("wup", qWup, (hid, I)),
                          ("wgate", qWgate, (hid, I)), ("wdown", qWdown, (I, hid)),
                          ("gamma_attn", qGa, (S, hid)), ("gamma_ffn", qGf, (S, hid))]:
            input_tensors[f"{nm}_tensor"] = t.to(torch.bfloat16); order.append(f"{nm}_tensor"); shapes.append(sh)
        if self.rope:
            for nm, t in [("cos", qcos), ("sin", qsin)]:
                input_tensors[f"{nm}_tensor"] = t.to(torch.bfloat16); order.append(f"{nm}_tensor"); shapes.append((S, Dh))
        if self.causal:
            # ATen adds the mask to the RAW QK^T then the online softmax scales the sum, so
            # pre-divide by scale to land the golden's post-scale MASK_NEG (template adds it
            # after scale). Without this, causal rows 0-3 (most masked) leak softmax weight.
            mask_hbm = mask / self.scale
            input_tensors["mask_tensor"] = mask_hbm.to(torch.bfloat16); order.append("mask_tensor"); shapes.append((S, KV))

        # ---- instruction offset ----
        auto_offset = calculate_instr_storage_offset_from_shapes(shapes, precision_settings, hbm_row_width)
        instr_offset = getattr(self, "force_instr_offset", None) or auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        paths["tensors"] = self._save_tensors(input_tensors)

        # ---- assembly ----
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(f"--skip-asm-gen set but no asm at {asm_path}")
        else:
            asm_path.write_text(self._generate_assembly_aten())
        paths["asm"] = asm_path

        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        # Template: f0=0, f1=eps, f2=1/hidden, f3=1.0 (silu one + copy), f4=scale.
        # ATen: online softmax reads f1=scale/f2=-inf; rms reads f3=eps/f4=1/hid.
        # ATen: f1=scale, f2=-inf (softmax); f3=eps, f4=1/hid (rms); f5=1.0 (ffn silu const_one).
        fp_slots = [0.0, self.scale, -60000.0, self.EPS, 1.0 / hid, 1.0]
        paths["fp_sram"] = write_fp_sram_hex(fp_slots, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        create_mem_for_sim(
            precision_settings=precision_settings, data_size=256, mode="behave_sim",
            asm="llama_layer", data=None, specified_data_order=order,
            build_path=self.build_dir, hbm_row_width=hbm_row_width,
            mx_format=self.mx_format, instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # ---- golden VRAM (result region depends on stage) ----
        # The (S,hid) output is stored TILED in VRAM (hid/VLEN tiles, each S rows of
        # VLEN), same convention as ffn/linear — NOT row-major. Permute to match.
        out_vram = self._aten_out_base
        total_rows = S * (hid // self.VLEN)
        golden_vram = golden_result.reshape(S, hid // self.VLEN, self.VLEN).permute(1, 0, 2).reshape(total_rows, self.VLEN)
        paths["golden_vram"] = save_golden_vram(golden_vram, output_dir=self.build_dir,
                                                filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")

        fmt = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False, "check_vram": True,
            "vram_start_row_idx": out_vram // self.VLEN, "vram_num_rows": total_rows,
            "row_dim": self.VLEN, "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_rows, "vram_total_rows": total_rows,
            "golden_vram_file": "golden_vram_result.pt", "workload_type": "llama_layer",
            "stage": self.stage, "num_heads": H, "seq_len": S, "hidden": hid,
            "output_shape": list(golden_result.shape), "mx_format": fmt,
            "exp_width": self.quant_config["exp_width"], "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"
        self._set_env_vars(paths)
        return paths

    # ---- isolation: attention with Q GIVEN (no rms, no Q-proj) -------------
    def _generate_qgiven(self, paths) -> dict:
        """Isolation stage: preload a given large-magnitude Q (like mha) straight
        into the layer's q VRAM region, then run the SAME attention (QKt→softmax→
        P@V) the failing stage uses. Only Q's source differs (preload vs Q-proj).
        Verifies O_h head-major. Pass => the computed-Q path is what breaks P@V."""
        S, H, Dh, KV = self.S, self.H, self.Dh, self.KV
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        q = self._quantize
        g = torch.Generator().manual_seed(self.seed)

        qscale = getattr(self, "q_scale", 1.0)   # scale Q to emulate small computed-Q (flat softmax)
        Q = (torch.randn(H, S, Dh, generator=g) * qscale).to(torch.bfloat16)  # given, ~N(0,qscale)
        K = torch.randn(H, KV, Dh, generator=g, dtype=torch.bfloat16)
        V = torch.randn(H, KV, Dh, generator=g, dtype=torch.bfloat16)
        qQ = q(Q.reshape(H * S, Dh)).reshape(H, S, Dh)
        qK = torch.stack([q(K[h]) for h in range(H)])
        qV = torch.stack([q(V[h]) for h in range(H)])

        if self.causal:
            qi = torch.arange(S).unsqueeze(1); ki = torch.arange(KV).unsqueeze(0)
            mask = torch.where(ki <= qi, 0.0, self.MASK_NEG).to(torch.float32)
        else:
            mask = torch.zeros(S, KV, dtype=torch.float32)

        O_heads = []
        for h in range(H):
            s = (qQ[h].float() @ qK[h].float().t()) * self.scale + mask
            p = q(torch.softmax(s, dim=-1).to(torch.bfloat16))
            O_heads.append(q((p.float() @ qV[h].float()).to(torch.bfloat16)))
        golden_result = torch.stack(O_heads).reshape(H * S, Dh).to(torch.float32)   # head-major (H*S, Dh)

        input_tensors = {"q_tensor": qQ.to(torch.bfloat16)}
        order = ["q_tensor"]; shapes = [(H * S, Dh)]
        # Per-head K TRANSPOSED (Dh, KV) — usable as a linear_projection weight so
        # linear_projection(Q_h, K_h) = Q_h @ K^T — and V NON-transposed (KV, Dh) for
        # compute_pv. Same layout the template path uses; the ATen path reuses it.
        qK_hbm = qK.transpose(-1, -2).contiguous()                     # (H, Dh, KV)
        for h in range(H):
            input_tensors[f"k{h}_tensor"] = qK_hbm[h].contiguous().to(torch.bfloat16); order.append(f"k{h}_tensor"); shapes.append((Dh, KV))
        for h in range(H):
            input_tensors[f"v{h}_tensor"] = qV[h].contiguous().to(torch.bfloat16); order.append(f"v{h}_tensor"); shapes.append((KV, Dh))
        if self.causal:
            # The ATen online softmax adds this mask to the RAW QK^T then scales the sum,
            # so pre-divide by scale to land the golden's post-scale MASK_NEG (the template
            # path adds it after scale, so it uses MASK_NEG directly).
            mask_hbm = mask / self.scale
            input_tensors["mask_tensor"] = mask_hbm.to(torch.bfloat16); order.append("mask_tensor"); shapes.append((S, KV))

        auto_offset = calculate_instr_storage_offset_from_shapes(shapes, precision_settings, hbm_row_width)
        instr_offset = getattr(self, "force_instr_offset", None) or auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")
        paths["tensors"] = self._save_tensors(input_tensors)

        qg_asm = self._assembly_qgiven_aten()
        (self.build_dir / "generated_asm_code.asm").write_text(qg_asm)
        paths["asm"] = self.build_dir / "generated_asm_code.asm"

        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        # ATen flash_attention's online softmax reads slot1=scale, slot2=-inf seed
        # (like gqa.py); the template path uses the [0,EPS,1/hid,1.0,scale] layout.
        fp_slots = [0.0, self.scale, -60000.0]
        paths["fp_sram"] = write_fp_sram_hex(fp_slots, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)
        create_mem_for_sim(precision_settings=precision_settings, data_size=256, mode="behave_sim",
                           asm="llama_layer", data=None, specified_data_order=order,
                           build_path=self.build_dir, hbm_row_width=hbm_row_width,
                           mx_format=self.mx_format, instr_storage_offset=instr_offset)
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        o_base = self._aten_o_base
        total_rows = H * S * Dh // self.VLEN
        golden_vram = golden_result.reshape(total_rows, self.VLEN)
        paths["golden_vram"] = save_golden_vram(golden_vram, output_dir=self.build_dir,
                                                filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")
        fmt = self.quant_config.get("format", "mxfp")
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump({"check_hbm": False, "check_vram": True,
                       "vram_start_row_idx": o_base // self.VLEN, "vram_num_rows": total_rows,
                       "row_dim": self.VLEN, "vram_compare_start_row": 0,
                       "vram_compare_num_rows": total_rows, "vram_total_rows": total_rows,
                       "golden_vram_file": "golden_vram_result.pt", "workload_type": "llama_layer",
                       "stage": "attn_qgiven", "num_heads": H, "seq_len": S,
                       "output_shape": list(golden_result.shape), "mx_format": fmt,
                       "exp_width": self.quant_config["exp_width"], "man_width": self.quant_config["man_width"],
                       "scale_width": self.quant_config["exp_bias_width"]}, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"
        self._set_env_vars(paths)
        return paths

    def _assembly_qgiven_aten(self) -> str:
        """attn_qgiven via prog.flash_attention (batch_size=H): all H heads'
        QK^T -> online-softmax -> P@V (the fixed compute_pv) in ONE call, output
        head-major (H*S, Dh). K/V are single stacked NON-transposed (H*KV, Dh) tensors;
        the FP SRAM must be [0, scale, -inf] (slot1=scale, slot2=-inf seed) as the ATen
        online softmax reads (set in _generate_qgiven). Sets self._aten_o_base."""
        S, H, Dh, KV = self.S, self.H, self.Dh, self.KV
        mlen, blen = self.MLEN, self.BLEN
        real_data_ratio = (8 * 8 + 8) / (8 * 8)
        _, cfg = load_precision_from_svh(SRC_PATH / "definitions")
        mram_tile_capacity = cfg.get("MATRIX_SRAM_DEPTH", 1024) // mlen

        prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio,
                             unroll_loops=True, mram_tile_capacity=mram_tile_capacity)
        prog.emit(f"; llama attn_qgiven via ATen flash_attention  H={H} S={S} Dh={Dh} causal={self.causal}\n\n")

        q_in = prog.input("q_tensor", shape=(H * S, Dh))
        k_ins = [prog.input(f"k{h}_tensor", shape=(Dh, KV)) for h in range(H)]   # K^T weights
        v_ins = [prog.input(f"v{h}_tensor", shape=(KV, Dh)) for h in range(H)]   # V (KV, Dh)
        mask_in = prog.input("mask_tensor", shape=(S, KV)) if self.causal else None

        prog.load_batch(q_in, name="q_tensor")
        mask_v = prog.load_batch(mask_in, name="mask_tensor") if self.causal else None

        # Head-major output O (H*S, Dh): head h at row-block h. Per head: QK^T via the
        # PROVEN linear_projection (Q_h @ K_h, K_h stored = K^T) -> S; single-block
        # inline-normalize softmax (register-direct, no racy scale_o/final_scale) -> P;
        # compute_pv (the fixed P@V) -> PV; O_h += PV.
        prog.alloc("O_all", H * S, Dh, physical_shape=(H * S, mlen))
        q_base = prog.get_vram_addr("q_tensor")
        o_base = prog.get_vram_addr("O_all")
        for h in range(H):
            Q_h = prog.alloc_at(f"_qg_Qh{h}", S, Dh, q_base + h * S * mlen, physical_shape=(S, mlen))
            O_h = prog.alloc_at(f"_qg_Oh{h}", S, Dh, o_base + h * S * mlen, physical_shape=(S, mlen))
            prog.init_online_softmax(0, O_h, rows=S)
            S_blk = prog.linear_projection(Q_h, k_ins[h], name=f"_qg_S{h}")   # Q_h @ K^T
            if self.causal:
                prog.vram_add(S_blk, mask_v)
            prog.online_softmax_block(S_blk, self.scale, rows=S, inline_normalize=True)
            # P@V via the proven linear_projection too (V_h is a (KV, Dh) weight): P @ V.
            PV = prog.linear_projection(S_blk, v_ins[h], name=f"_qg_PV{h}")
            prog.vram_add(O_h, PV, num_rows=S)

        self._aten_o_base = o_base
        prog.emit("\n; End attn_qgiven (ATen, linear_projection QK^T + P@V)\nC_BREAK\n")
        return prog.compile()

    def _generate_assembly_aten(self) -> str:
        """Full attention (+FFN) sub-layer via the ATen PlenaCompiler. Reuses the proven
        attn_qgiven attention (linear_projection QK^T + P@V, inline-normalize softmax) and
        adds rms_norm*gamma, per-head Q-proj + RoPE, O-projection, residual, and (full)
        rms2*gamma_ffn + fused FFN + residual2. FP SRAM = [0, scale, -inf, eps, 1/hid]:
        softmax reads slot1/slot2, rms reads slot3/slot4. Sets self._aten_out_base."""
        S, H, Dh, KV, hid, I = self.S, self.H, self.Dh, self.KV, self.hidden, self.inter
        mlen, blen = self.MLEN, self.BLEN
        rope, causal = self.rope, self.causal
        skip_rms = getattr(self, "skip_rms", False)
        real_data_ratio = (8 * 8 + 8) / (8 * 8)
        _, cfg = load_precision_from_svh(SRC_PATH / "definitions")
        # The FFN picks loop-vs-unrolled codegen from mram_tile_capacity: use_loop_instructions
        # = (max_k_tiles <= capacity). ffn.py forces the UNROLLED path (capacity=max_k_tiles-1)
        # because the loop path miscomputes; match it for the full stage. Attn K=hid tiles just
        # K-split accordingly (still correct).
        if self.stage == "full":
            max_k_tiles = max(hid, I) // mlen
            mram_tile_capacity = max(1, max_k_tiles - 1)
        else:
            mram_tile_capacity = cfg.get("MATRIX_SRAM_DEPTH", 1024) // mlen
        prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio,
                             unroll_loops=True, mram_tile_capacity=mram_tile_capacity)
        prog.emit(f"; llama layer stage={self.stage} via ATen  H={H} S={S} hid={hid} rope={rope} causal={causal}\n\n")

        # Inputs in the exact HBM order from generate(): x, wq*, [wqrot*], k*, v*, wo, wup,
        # wgate, wdown, gamma_attn, gamma_ffn, [cos, sin], [mask]. (k stored transposed = K^T.)
        x_in = prog.input("x_tensor", shape=(S, hid))
        wq_ins = [prog.input(f"wq{h}_tensor", shape=(hid, Dh)) for h in range(H)]
        wqrot_ins = [prog.input(f"wqrot{h}_tensor", shape=(hid, Dh)) for h in range(H)] if rope else []
        k_ins = [prog.input(f"k{h}_tensor", shape=(Dh, KV)) for h in range(H)]
        v_ins = [prog.input(f"v{h}_tensor", shape=(KV, Dh)) for h in range(H)]
        wo_in = prog.input("wo_tensor", shape=(hid, hid))
        wup_in = prog.input("wup_tensor", shape=(hid, I))
        wgate_in = prog.input("wgate_tensor", shape=(hid, I))
        wdown_in = prog.input("wdown_tensor", shape=(I, hid))
        ga_in = prog.input("gamma_attn_tensor", shape=(S, hid))
        gf_in = prog.input("gamma_ffn_tensor", shape=(S, hid))
        cos_in = prog.input("cos_tensor", shape=(S, Dh)) if rope else None
        sin_in = prog.input("sin_tensor", shape=(S, Dh)) if rope else None
        mask_in = prog.input("mask_tensor", shape=(S, KV)) if causal else None

        x_resid = prog.load_batch(x_in, name="x_tensor")             # kept for residual 1
        ga = prog.load_batch(ga_in, name="gamma_attn_tensor")
        cos = prog.load_batch(cos_in, name="cos_tensor") if rope else None
        sin = prog.load_batch(sin_in, name="sin_tensor") if rope else None
        mask_v = prog.load_batch(mask_in, name="mask_tensor") if causal else None

        # xn = rms(x) * gamma_attn. Second load of x so x_resid stays the raw input.
        xn = prog.load_batch(x_in, name="xn")
        if not skip_rms:
            prog.rms_norm(xn, eps_offset=3, reci_hid_offset=4)
            prog.vram_mul(xn, ga)

        # O (S, hid): per head Q-proj (+RoPE) then the proven attention, into col-block h.
        O_all = prog.alloc("O_all", S, hid, physical_shape=(S, hid))
        o_base = prog.get_vram_addr("O_all")
        for h in range(H):
            Q_h = prog.linear_projection(xn, wq_ins[h], name=f"_q{h}")          # xn @ Wq_h
            if rope:
                Qr_h = prog.linear_projection(xn, wqrot_ins[h], name=f"_qr{h}")  # = rotate_half(Q)
                prog.rope(Q_h, Qr_h, cos, sin)                                   # Q = Q*cos + Qr*sin
            # init_online_softmax BEFORE the softmax (sets per-row m_old=-inf that the
            # softmax reads). Putting it after leaves heads 1-3's softmax reading a stale
            # running max -> a row's max stays stuck -> uniform P. Matches attn_qgiven order.
            O_h = prog.alloc_at(f"_oh{h}", S, Dh, o_base + h * S * mlen, physical_shape=(S, mlen))
            prog.init_online_softmax(0, O_h, rows=S)
            S_blk = prog.linear_projection(Q_h, k_ins[h], name=f"_s{h}")         # Q_h @ K_h^T
            if causal:
                prog.vram_add(S_blk, mask_v)
            prog.online_softmax_block(S_blk, self.scale, rows=S, inline_normalize=True)
            PV = prog.linear_projection(S_blk, v_ins[h], name=f"_pv{h}")         # P @ V
            prog.vram_add(O_h, PV, num_rows=S)

        # attn = O @ Wo (K-split over hid); x1 = x + attn.
        attn = prog.linear_projection(O_all, wo_in, name="_attn")
        prog.vram_add(x_resid, attn, num_rows=S)

        if self.stage == "full":
            gf = prog.load_batch(gf_in, name="gamma_ffn_tensor")
            xn2 = prog.alloc("xn2", S, hid, physical_shape=(S, hid))
            prog.init_online_softmax(0, xn2, rows=S)
            prog.vram_add(xn2, x_resid, num_rows=S)                              # xn2 = x1 (copy)
            if not skip_rms:
                prog.rms_norm(xn2, eps_offset=3, reci_hid_offset=4)
                prog.vram_mul(xn2, gf)
            prog.ffn(xn2, wgate_in, wup_in, wdown_in)                            # fused SwiGLU
            prog.vram_add(x_resid, xn2, num_rows=S)                              # x2 = x1 + ffn

        self._aten_out_base = prog.get_vram_addr("x_tensor")
        prog.emit(f"\n; End stage={self.stage} (ATen)\nC_BREAK\n")
        return prog.compile()

    def get_config(self) -> dict:
        return {"workload_type": "llama_layer", "stage": self.stage, "seq_len": self.S,
                "num_heads": self.H, "head_dim": self.Dh, "hidden": self.hidden,
                "inter_size": self.inter, "causal": self.causal, "mlen": self.MLEN,
                "blen": self.BLEN, "vlen": self.VLEN, "seed": self.seed, "mx_format": self.mx_format}


def main():
    p = argparse.ArgumentParser(description="Generate a full Llama decoder layer (KV-cache variant)")
    p.add_argument("--seq-len", type=int, default=_VLEN)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=_MLEN)
    p.add_argument("--inter-size", type=int, default=64)
    p.add_argument("--stage", type=str, default="full", choices=["attn", "full", "attn_qgiven"])
    p.add_argument("--q-scale", type=float, default=1.0, help="scale given Q (attn_qgiven) to emulate small computed-Q")
    p.add_argument("--skip-rms", action="store_true", help="stage=attn/full: skip RMSNorm (feed x directly) to isolate rms as trigger")
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"])
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--skip-asm-gen", action="store_true")
    a = p.parse_args()
    w = LlamaLayerWorkload(seq_len=a.seq_len, num_heads=a.num_heads, head_dim=a.head_dim,
                           inter_size=a.inter_size, stage=a.stage, build_dir=a.build_dir,
                           seed=a.seed, mx_format=a.mx_format, causal=a.causal,
                           skip_asm_gen=a.skip_asm_gen)
    w.q_scale = a.q_scale
    w.skip_rms = a.skip_rms
    print(f"Generating Llama layer: stage={a.stage}, H={a.num_heads}, S={a.seq_len}, hidden={w.hidden}, inter={a.inter_size}")
    paths = w.generate()
    for k, val in paths.items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
