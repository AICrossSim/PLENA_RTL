"""Grouped-Query Attention (GQA) workload for PLENA via the ATen PlenaCompiler.

One KV head shared by ``hq`` query heads, single batch, non-causal, fused packed
online-softmax flash attention (prog.flash_attention GQA path). This is a pure
ATen-compiler workload — the whole point is exercising the compiler's packed-head
GQA lowering (_flash_attention_gqa_fused / _emit_packed_attention_group_internal).

Packing (MLEN=16 default):
  * Q is an ACTIVATION prestaged in VRAM at addr 0. It is MLEN-wide: the hq query
    heads occupy HLEN-sized lanes, head h at lanes [h*h_qkv:(h+1)*h_qkv].
  * K and V are the single KV head, HBM matrix weights (MXINT), physical-padded to
    MLEN wide (like mha.py's per-head K/V, so the MXINT scale region resolves).
  * Packing constraint: broadcast_amount * h_qkv == MLEN  and  hq/hkv <= broadcast.
    Default: broadcast_amount = MLEN//h_qkv = 4, h_qkv = 4, ratio = hq/hkv = 4.

Per query head h (all sharing the one K,V):
    Qh = Q_packed[:, h*h_qkv:(h+1)*h_qkv]        # (seq, h_qkv)
    S  = (Qh @ K.T) * scale                       # (seq, kv_seq), NON-causal
    P  = softmax(S, dim=-1)
    Oh = P @ V                                     # (seq, h_qkv)
Output is packed the same way: O_packed[:, h*h_qkv:(h+1)*h_qkv] = Oh.

Output lane order (verified in PLENA_Compiler/aten/plena/program_attention.py):
  _emit_packed_attention_group_internal loops head in range(group_heads); for each,
  output_head = output_head_base(0) + head, output_lane = (output_head*h_qkv)%mlen
  //h_qkv == head (for h_qkv*group_heads <= mlen). _pack_o_head_to_output shifts by
  head_slot*head_slot_dim = head*h_qkv, so head h -> output lanes [h*h_qkv:(h+1)*h_qkv].
  This is the SAME lane layout as the packed Q input, i.e. an identity head->lane map.
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

from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

# ATen PlenaCompiler (fused GQA path).
from compiler.aten.plena import PlenaCompiler

_HW = load_hardware_tile_sizes(SRC_PATH / "definitions")
# Defaults follow the tile sizes: one seq/KV tile, h_qkv lanes per head,
# hq = MLEN // h_qkv heads (packing constraint broadcast*h_qkv == MLEN, hq/hkv <= broadcast).
_MLEN = _HW["MLEN"]
_H_QKV = min(4, max(1, _MLEN // 2))
_HQ = _MLEN // _H_QKV
DEFAULT_MX_FORMAT = None
# MXINT8: one shared scale per block of 8 -> (8*8 + 8) / (8*8) = 1.125.
_REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)


class GQAWorkload(WorkloadGenerator):
    """Fused packed GQA via the ATen PlenaCompiler (one KV head, hq query heads)."""

    MLEN = _HW["MLEN"]
    BLEN = _HW["BLEN"]
    VLEN = _HW["VLEN"]
    HBM_V_Prefetch_Amount = _HW["HBM_V_Prefetch_Amount"]

    def __init__(self, seq_len=_MLEN, kv_seq_len=_MLEN, hq=_HQ, hkv=1, h_qkv=_H_QKV,
                 mx_format=None, skip_asm_gen=False, use_aten_compiler=True, **kwargs):
        super().__init__(**kwargs)
        assert hkv == 1, "GQA ATen fused path currently supports a single KV head (hkv=1)."
        assert hq % hkv == 0, f"hq={hq} must be divisible by hkv={hkv}"
        assert seq_len % self.BLEN == 0, f"seq_len must be divisible by BLEN={self.BLEN}"
        assert seq_len <= self.MLEN, f"packed GQA supports one seq tile (seq_len <= MLEN={self.MLEN})"
        assert kv_seq_len <= self.MLEN, f"packed GQA supports one KV tile (kv_seq_len <= MLEN={self.MLEN})"
        # Packing constraint: broadcast_amount * h_qkv == MLEN, and ratio <= broadcast.
        broadcast_amount = self.MLEN // h_qkv
        assert broadcast_amount * h_qkv == self.MLEN, (
            f"broadcast_amount*h_qkv must equal MLEN ({broadcast_amount}*{h_qkv} != {self.MLEN})")
        ratio = hq // hkv
        assert ratio <= broadcast_amount, (
            f"GQA ratio hq/hkv={ratio} exceeds broadcast lanes {broadcast_amount}")
        assert hq * h_qkv <= self.MLEN, (
            f"packed Q must fit one MLEN row: hq*h_qkv={hq * h_qkv} > MLEN={self.MLEN}")

        self.seq_len = seq_len
        self.kv_seq_len = kv_seq_len
        self.hq = hq
        self.hkv = hkv
        self.h_qkv = h_qkv
        self.broadcast_amount = broadcast_amount
        self.scale = 1.0 / math.sqrt(h_qkv)
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.skip_asm_gen = skip_asm_gen
        # Kept for CLI symmetry with the other workloads; GQA is always ATen.
        self.use_aten_compiler = use_aten_compiler
        # VRAM element address of the packed output O (captured while emitting ISA).
        self._aten_output_vram_addr = None

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()
        S, KV, D = self.seq_len, self.kv_seq_len, self.h_qkv
        HQ = self.hq
        mlen = self.MLEN

        ps, cs = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, ps)
        hbm_row_width = cs.get("HBM_WIDTH", 256)
        q = self._quantize
        g = torch.Generator().manual_seed(self.seed)

        # ------------------------------------------------------------------
        # 1. Tensors. Q is packed MLEN-wide: head h in lanes [h*D:(h+1)*D]. The
        #    remaining lanes (hq*D .. mlen) are zero pad. K,V are the ONE KV head,
        #    logical (seq, D) but physically MLEN-padded (physical_shape width mlen)
        #    so each is a self-contained single-head weight whose MXINT scale region
        #    resolves off its own base (same reasoning as mha.py per-head K/V).
        # ------------------------------------------------------------------
        Q_heads = torch.randn(HQ, S, D, generator=g, dtype=torch.bfloat16)     # per-head Q
        K = torch.randn(KV, D, generator=g, dtype=torch.bfloat16)             # shared KV head
        V = torch.randn(KV, D, generator=g, dtype=torch.bfloat16)

        # Quantize per-head Q (block-8 along the feature dim; flatten heads).
        qQ_heads = q(Q_heads.reshape(HQ * S, D)).reshape(HQ, S, D)
        qK = q(K)
        qV = q(V)

        # Pack quantized Q into an MLEN-wide activation: head h -> lanes [h*D:(h+1)*D].
        Q_packed = torch.zeros(S, mlen, dtype=torch.bfloat16)
        for h in range(HQ):
            Q_packed[:, h * D:(h + 1) * D] = qQ_heads[h]

        # ------------------------------------------------------------------
        # 2. Golden: per query head, non-causal SDPA, sharing the one K,V. P is
        #    re-quantized before P@V (the RTL re-quantizes the softmax output to the
        #    HW MX format when it reads P as the P@V activation, matching mha.py).
        # ------------------------------------------------------------------
        O_packed = torch.zeros(S, mlen, dtype=torch.float32)
        for h in range(HQ):
            Qh = qQ_heads[h].float()                                  # (S, D)
            s = (Qh @ qK.float().t()) * self.scale                    # (S, KV), NON-causal
            p = torch.softmax(s, dim=-1)
            p_q = q(p.to(torch.bfloat16))
            Oh = p_q.float() @ qV.float()                            # (S, D)
            O_packed[:, h * D:(h + 1) * D] = Oh
        golden_result = O_packed                                      # (S, mlen)

        # ------------------------------------------------------------------
        # 3. HBM tensors + order. Q (activation) is declared FIRST so its auto HBM
        #    slot precedes the weights, but since Q is prestaged in VRAM the ISA never
        #    prefetches it; it still occupies an HBM slot in specified_data_order so
        #    the offsets of K/V land correctly. K/V are MLEN-padded weights.
        #    Physical (KV -> mlen padded rows AND cols) so the packed matmul reads a
        #    self-contained single-head weight.
        # ------------------------------------------------------------------
        # HBM stores the MLEN-padded physical tensors (zeros outside logical region).
        q_hbm = torch.zeros(S, mlen, dtype=torch.bfloat16)           # same as Q_packed
        q_hbm[:, :] = Q_packed
        k_hbm = torch.zeros(mlen, mlen, dtype=torch.bfloat16)
        k_hbm[:KV, :D] = qK
        v_hbm = torch.zeros(mlen, mlen, dtype=torch.bfloat16)
        v_hbm[:KV, :D] = qV

        input_tensors = {
            "q_tensor": q_hbm,
            "k_tensor": k_hbm,
            "v_tensor": v_hbm,
        }
        specified_data_order = ["q_tensor", "k_tensor", "v_tensor"]
        # HBM physical shapes for the instruction-offset calculation.
        tensor_shapes = [(S, mlen), (mlen, mlen), (mlen, mlen)]

        instr_offset = getattr(self, "force_instr_offset", None) or \
            calculate_instr_storage_offset_from_shapes(tensor_shapes, ps, hbm_row_width)
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")
        paths["tensors"] = self._save_tensors(input_tensors)

        # ------------------------------------------------------------------
        # 4. Assembly via the ATen PlenaCompiler (fused GQA path).
        # ------------------------------------------------------------------
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(f"--skip-asm-gen set but no asm at {asm_path}")
            # Reuse the existing (hand-edited) asm, but still run codegen to
            # resolve the output VRAM addr used by verification.
            self._generate_assembly_with_aten_compiler()
        else:
            asm_path.write_text(self._generate_assembly_with_aten_compiler())
        paths["asm"] = asm_path

        # ------------------------------------------------------------------
        # 5. FP SRAM preload. Online softmax fp-slot conventions (memory.py):
        #    slot 0 = 0.0, slot 1 = attn scale, slot 2 = -inf seed (row-max init).
        # ------------------------------------------------------------------
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        # -60000 fits the FP12 (6e5m) scalar format and is far below any real score.
        fp_preload = [0.0, self.scale, -60000.0]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        create_mem_for_sim(
            precision_settings=ps, data_size=256, mode="behave_sim", asm="gqa",
            data=None, specified_data_order=specified_data_order,
            build_path=self.build_dir, hbm_row_width=hbm_row_width,
            mx_format=self.mx_format, instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # ------------------------------------------------------------------
        # 6. Golden VRAM. Packed O is (S, mlen). VRAM rows are VLEN wide (== mlen),
        #    so O occupies S*(mlen//VLEN) == S rows starting at out_addr//VLEN.
        # ------------------------------------------------------------------
        o_base = self._aten_output_vram_addr
        total_rows = S * (mlen // self.VLEN)
        golden_vram = golden_result.reshape(S, mlen // self.VLEN, self.VLEN).permute(1, 0, 2).reshape(total_rows, self.VLEN)
        paths["golden_vram"] = save_golden_vram(golden_vram, output_dir=self.build_dir,
                                                filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")

        # ------------------------------------------------------------------
        # 7. Verification params (VRAM check of the packed output region).
        # ------------------------------------------------------------------
        fmt = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False, "check_vram": True,
            "vram_start_row_idx": o_base // self.VLEN, "vram_num_rows": total_rows,
            "row_dim": self.VLEN, "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_rows, "vram_total_rows": total_rows,
            "golden_vram_file": "golden_vram_result.pt", "workload_type": "gqa",
            "seq_len": S, "kv_seq_len": KV, "hq": HQ, "hkv": self.hkv, "h_qkv": D,
            "broadcast_amount": self.broadcast_amount,
            "output_shape": list(golden_result.shape), "mx_format": fmt,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        with open(self.build_dir / "test_params.json", "w") as f:
            json.dump({"workload_type": "gqa", "seq_len": S, "kv_seq_len": KV,
                       "hq": HQ, "hkv": self.hkv, "h_qkv": D,
                       "broadcast_amount": self.broadcast_amount, "scale": self.scale,
                       "mlen": mlen, "blen": self.BLEN, "vlen": self.VLEN,
                       "mx_format": fmt, "seed": self.seed}, f, indent=2)
        paths["test_params"] = self.build_dir / "test_params.json"

        self._set_env_vars(paths)
        return paths

    def _generate_assembly_with_aten_compiler(self) -> str:
        """Emit the fused packed GQA ISA via prog.flash_attention (GQA path)."""
        mlen = self.MLEN
        blen = self.BLEN
        S, KV, D = self.seq_len, self.kv_seq_len, self.h_qkv
        HQ, HKV = self.hq, self.hkv

        _ps, _cfg = load_precision_from_svh(SRC_PATH / "definitions")
        mram_tile_capacity = _cfg.get("MATRIX_SRAM_DEPTH", 1024) // mlen

        prog = PlenaCompiler(
            mlen=mlen,
            blen=blen,
            real_data_ratio=_REAL_DATA_RATIO,
            unroll_loops=True,
            mram_tile_capacity=mram_tile_capacity,
        )
        # Packed-head params (mirrors the canonical test_packed_gqa_* setup).
        prog.hlen = D
        prog.broadcast_amount = self.broadcast_amount

        prog.emit("; Grouped-Query Attention (fused packed) via ATen PlenaCompiler\n")
        prog.emit(f"; hq={HQ} hkv={HKV} h_qkv={D} seq={S} kv_seq={KV} "
                  f"broadcast={self.broadcast_amount}  MLEN={mlen} BLEN={blen}\n\n")

        # Declare inputs in HBM order (matches specified_data_order = q, k, v).
        # The compiler's allocator now row-aligns each tensor base to match the stager,
        # so no explicit hbm_addr override is needed (was a MLEN=8 workaround).
        #   Q: packed activation, loaded HBM->VRAM (physical MLEN-wide).
        #   K, V: the single KV head, HBM weights, physical MLEN-padded.
        q_input = prog.input("Q", shape=(S, mlen), physical_shape=(max(S, mlen), mlen))
        k_input = prog.input("K", shape=(KV, D), physical_shape=(mlen, mlen))
        v_input = prog.input("V", shape=(KV, D), physical_shape=(mlen, mlen))

        # Load Q from HBM into VRAM (prestaged_vram_addr suppressed the prefetch, so
        # VRAM was never populated -> Q read as 0 -> uniform softmax -> identical O rows).
        qb = prog.load_batch(q_input, name="Q")

        o = prog.flash_attention(
            qb, k_input, v_input,
            scale=self.scale,
            hq=HQ, hkv=HKV, h_qkv=D,
            batch_size=1, seq_len=S, kv_seq_len=KV,
        )
        # Capture the packed output's VRAM element address for verification.
        self._aten_output_vram_addr = prog.get_vram_addr(o.name)

        prog.emit("\n; End of GQA test\n")
        prog.emit("C_BREAK\n")
        asm = prog.compile()
        # The ATen packed-output packing (_pack_o_head_to_output) emits the lane
        # shift as "V_SHIFT_V", but the RTL decoder / vector_machine and the
        # assembler's operation.svh spell that opcode "V_SHFT_V" (6'h32,
        # SHIFT_V_LANES_ELEMENT). The two are the same instruction with the same
        # `rd, rs1, rs2` operand order (spec: V_SHIFT_V rd, rs1, rs2). Rename so
        # the assembler encodes it and the RTL decodes it. (Same fix as the
        # im2col_asm_no_shift.py note about the "undocumented V_SHFT_V".)
        asm = asm.replace("V_SHIFT_V", "V_SHFT_V")
        return asm

    def get_config(self) -> dict:
        return {"workload_type": "gqa", "seq_len": self.seq_len, "kv_seq_len": self.kv_seq_len,
                "hq": self.hq, "hkv": self.hkv, "h_qkv": self.h_qkv,
                "broadcast_amount": self.broadcast_amount, "scale": self.scale,
                "mlen": self.MLEN, "blen": self.BLEN, "vlen": self.VLEN,
                "quant_config": self.quant_config, "seed": self.seed,
                "mx_format": self.mx_format, "use_aten_compiler": self.use_aten_compiler}


def main():
    p = argparse.ArgumentParser(description="Generate Grouped-Query Attention workload (ATen PlenaCompiler)")
    p.add_argument("--seq-len", type=int, default=_MLEN)
    p.add_argument("--kv-seq-len", type=int, default=_MLEN)
    p.add_argument("--hq", type=int, default=_HQ, help="number of query heads")
    p.add_argument("--hkv", type=int, default=1, help="number of KV heads (only 1 supported)")
    p.add_argument("--h-qkv", type=int, default=_H_QKV, help="per-head dim")
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"])
    p.add_argument("--skip-asm-gen", action="store_true")
    p.add_argument("--use-aten-compiler", dest="use_aten_compiler", action="store_true", default=True,
                   help="Use the ATen PlenaCompiler fused GQA path (default: on; GQA is ATen-only).")
    p.add_argument("--no-aten-compiler", dest="use_aten_compiler", action="store_false",
                   help="(unsupported) GQA has no non-ATen template path.")
    a = p.parse_args()

    if not a.use_aten_compiler:
        raise SystemExit("GQA is ATen-compiler only; --no-aten-compiler is unsupported.")

    w = GQAWorkload(seq_len=a.seq_len, kv_seq_len=a.kv_seq_len, hq=a.hq, hkv=a.hkv,
                    h_qkv=a.h_qkv, build_dir=a.build_dir, seed=a.seed,
                    mx_format=a.mx_format, skip_asm_gen=a.skip_asm_gen,
                    use_aten_compiler=a.use_aten_compiler)
    print(f"Generating GQA: hq={a.hq} hkv={a.hkv} h_qkv={a.h_qkv} "
          f"seq={a.seq_len} kv_seq={a.kv_seq_len} broadcast={w.broadcast_amount}")
    print(f"  Build dir: {w.build_dir}")
    paths = w.generate()
    print(f"  Output VRAM addr: {w._aten_output_vram_addr} "
          f"(row {w._aten_output_vram_addr // w.VLEN})")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
