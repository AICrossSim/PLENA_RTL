"""Scratchpad workload: minimal vector-op HAZARD reproducer for PLENA.

Purpose: isolate the "scalar<->vector RAW" hazard that corrupts the looped
SiLU / gate-mul / softmax vector-op chains (see memory: ffn-aten-compiler-path,
broadcast-op-reduction-deadlock). The documented failure (ffn._gate_mul_asm) is
that an in-loop incremental `S_ADDI gp += vlen` that advances a VRAM pointer used
by a vector op drifts ahead of the serialized element op, so the vector op
reads/writes the WRONG address -> corruption (or C_LOOP_END deadlock).

This harness preloads ONE known tensor A into VRAM (HBM prefetch, like silu.py),
then emits a SELECTABLE minimal sequence (`--seq`) that transforms A in place and
leaves the result in the activation region for verification. Compare the sequences
to pin down exactly what triggers the drift:

  unroll_add   : A[i] = A[i]+A[i], UNROLLED, absolute addrs        (control, should PASS)
  loop_add     : A[i] = A[i]+A[i], C_LOOP + in-loop S_ADDI += vlen (hazard candidate)
  loop_add_nop : loop_add with NOP spacing between V_ADD and S_ADDI (settle test)
  unroll_mul   : A[i] = A[i]*A[i], UNROLLED                        (control for mul)
  loop_mul     : A[i] = A[i]*A[i], C_LOOP + in-loop S_ADDI

Golden = the same op applied to the quantized A (in float); match_rate absorbs
the FP12 quantization error. A drift/hazard shows as a low match rate (or a hang).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJECT_PATH / "tools", _PROJECT_PATH / "PLENA_Tools", _PROJECT_PATH / "PLENA_Compiler"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from .base import WorkloadGenerator

from asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from asm_templates._imm import load_large_int_str as _load_large_int
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")

SEQUENCES = ("unroll_add", "loop_add", "loop_add_nop", "unroll_mul", "loop_mul",
             "chain_unroll", "chain_loop", "chain_loop_nop", "chain_inplace",
             "chain_out_mul", "chain_redsum")


class ScratchpadWorkload(WorkloadGenerator):
    """Minimal, selectable vector-op sequence for hazard isolation."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    def __init__(self, batch_size=8, hidden_size=64, seq="loop_add",
                 mx_format=None, skip_asm_gen=False, **kwargs):
        super().__init__(**kwargs)
        assert batch_size % self.BLEN == 0, f"batch_size must be divisible by {self.BLEN}"
        assert hidden_size % self.VLEN == 0, f"hidden_size must be divisible by {self.VLEN}"
        assert seq in SEQUENCES, f"seq must be one of {SEQUENCES}"
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq = seq
        self.mx_format = mx_format
        self.skip_asm_gen = skip_asm_gen

    # -------------------------------------------------------------- golden
    def _seq_golden(self, qA: Tensor) -> Tensor:
        a = qA.float()
        if self.seq == "chain_redsum":
            # per VLEN-block: scratch=A^2; s=sum(A^2); A=A*s  (out-of-place producer,
            # REDUCTION consumer — the rms_norm pattern)
            a3 = a.reshape(self.batch_size, self.hidden_size // self.VLEN, self.VLEN)
            s = (a3 * a3).sum(dim=-1, keepdim=True)
            return (a3 * s).reshape(self.batch_size, self.hidden_size)
        if self.seq == "chain_out_mul":
            return a * a * a * a  # scratch=A^2; A=scratch^2=A^4 (out-of-place, ELEMENT consumer)
        if self.seq.startswith("chain"):
            return 4.0 * a  # scratch = 2A; A = scratch+scratch = 4A
        if "mul" in self.seq:
            return a * a
        return a + a  # all *_add variants compute 2A

    # -------------------------------------------------------------- asm
    def _emit_op(self, gp_addr: int) -> str:
        """One in-place element op at VRAM[gp{gp_addr}]."""
        if "mul" in self.seq:
            return f"V_MUL_VV gp{gp_addr}, gp{gp_addr}, gp{gp_addr}, 0\n"
        return f"V_ADD_VV gp{gp_addr}, gp{gp_addr}, gp{gp_addr}, 0\n"

    def _generate_assembly(self) -> str:
        vlen = self.VLEN
        num_vectors = (self.batch_size * self.hidden_size) // vlen
        real_data_ratio = (8 * 8 + 8) / (8 * 8)
        act_hbm_size = int(self.hidden_size * self.batch_size * real_data_ratio)

        code = f"; Scratchpad hazard repro — seq={self.seq}\n"
        code += f"; Shape: ({self.batch_size}, {self.hidden_size})  VLEN={vlen}  num_vectors={num_vectors}\n\n"

        # Stage A: HBM -> VRAM[0..] (same proven path as silu.py)
        code += preload_addr_reg_asm(addr_reg_to_set=[1], available_registers=[1], addr_reg_val=[act_hbm_size])
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
        code += preload_act_asm(
            vlen=vlen, preload_len=self.HBM_V_Prefetch_Amount, batch=self.batch_size,
            hidden_size=self.hidden_size, alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=0, activation_offset_reg=0, stride_size=self.hidden_size,
        )

        code += f"\n; ===== sequence: {self.seq} =====\n"
        ADDR, LOOP, SCR = 1, 2, 3  # gp1 = A addr, gp2 = loop counter, gp3 = scratch addr
        scratch_base = self.batch_size * self.hidden_size  # VRAM region after A

        if self.seq.startswith("chain"):
            # Producer->consumer RAW between CONSECUTIVE vector ops:
            #   op1: scratch = A + A            (V_ADD_VV writes scratch)
            #   op2: A       = scratch + scratch (V_ADD_VV READS scratch op1 just wrote)
            # This is the silu/softmax-style vector->vector dependency.
            if self.seq == "chain_inplace":
                # IN-PLACE producer->consumer (rd==rs2 -> write addr = addr_2, trackable).
                # Matches the real silu/gate-mul final V_MUL_VV (in-place). op2 reads the
                # SAME row op1 just wrote.  op1: A=A+A (2A); op2: A=A+A (4A).
                for i in range(num_vectors):
                    code += _load_large_int(ADDR, i * vlen)
                    code += f"V_ADD_VV gp{ADDR}, gp{ADDR}, gp{ADDR}, 0\n"
                    code += f"V_ADD_VV gp{ADDR}, gp{ADDR}, gp{ADDR}, 0\n"
            elif self.seq == "chain_out_mul":
                # OUT-of-place producer, ELEMENT consumer (V_MUL_VV). Control for
                # chain_redsum: same out-of-place producer, but element (not reduction) consume.
                for i in range(num_vectors):
                    code += _load_large_int(ADDR, i * vlen)
                    code += _load_large_int(SCR, scratch_base + i * vlen)
                    code += f"V_MUL_VV gp{SCR}, gp{ADDR}, gp{ADDR}, 0\n"   # scratch = A^2
                    code += f"V_MUL_VV gp{ADDR}, gp{SCR}, gp{SCR}, 0\n"    # A = scratch^2 = A^4
            elif self.seq == "chain_redsum":
                # OUT-of-place producer (V_MUL_VV scratch=A^2), REDUCTION consumer
                # (V_RED_SUM reads scratch -> scalar f2), then broadcast A*=f2. The rms_norm
                # pattern. f2 written by V_RED_SUM; slots f0=0,f1=1 preloaded.
                for i in range(num_vectors):
                    code += _load_large_int(ADDR, i * vlen)
                    code += _load_large_int(SCR, scratch_base)
                    code += f"V_MUL_VV gp{SCR}, gp{ADDR}, gp{ADDR}, 0\n"   # scratch = A^2
                    code += f"V_RED_SUM f2, gp{SCR}\n"                     # f2 = sum(scratch)
                    code += f"V_MUL_VF gp{ADDR}, gp{ADDR}, f2, 0\n"        # A = A * f2
            elif self.seq == "chain_unroll":
                for i in range(num_vectors):
                    code += _load_large_int(ADDR, i * vlen)
                    code += _load_large_int(SCR, scratch_base + i * vlen)
                    code += f"V_ADD_VV gp{SCR}, gp{ADDR}, gp{ADDR}, 0\n"
                    code += f"V_ADD_VV gp{ADDR}, gp{SCR}, gp{SCR}, 0\n"
            else:  # chain_loop / chain_loop_nop
                code += _load_large_int(ADDR, 0)
                code += _load_large_int(SCR, scratch_base)
                code += f"C_LOOP_START gp{LOOP}, {num_vectors}\n"
                code += f"V_ADD_VV gp{SCR}, gp{ADDR}, gp{ADDR}, 0\n"
                if self.seq == "chain_loop_nop":
                    for _ in range(4):
                        code += "S_ADDI_INT gp0, gp0, 0\n"  # NOP between producer and consumer
                code += f"V_ADD_VV gp{ADDR}, gp{SCR}, gp{SCR}, 0\n"
                code += f"S_ADDI_INT gp{ADDR}, gp{ADDR}, {vlen}\n"
                code += f"S_ADDI_INT gp{SCR}, gp{SCR}, {vlen}\n"
                code += f"C_LOOP_END gp{LOOP}\n"

        elif self.seq in ("unroll_add", "unroll_mul"):
            # UNROLLED: absolute address each iteration, no increment. CONTROL.
            for i in range(num_vectors):
                code += _load_large_int(ADDR, i * vlen)
                code += self._emit_op(ADDR)

        else:
            # loop_add / loop_mul / loop_add_nop:
            # C_LOOP with in-loop INCREMENTAL S_ADDI (the documented hazard).
            code += _load_large_int(ADDR, 0)
            code += f"C_LOOP_START gp{LOOP}, {num_vectors}\n"
            code += self._emit_op(ADDR)
            if self.seq == "loop_add_nop":
                # spacing between the element op and the pointer advance
                for _ in range(4):
                    code += "S_ADDI_INT gp0, gp0, 0\n"  # NOP (write to gp0 = discard)
            code += f"S_ADDI_INT gp{ADDR}, gp{ADDR}, {vlen}\n"
            code += f"C_LOOP_END gp{LOOP}\n"

        code += "\n; End of scratchpad test\nC_BREAK\n"
        return code

    # -------------------------------------------------------------- generate
    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        A = torch.randn(self.batch_size, self.hidden_size, dtype=torch.bfloat16)
        qA = self._quantize(A)
        golden_result = self._seq_golden(qA)

        input_tensors = {"act_tensor": qA.to(torch.bfloat16)}
        tensor_shapes = [(self.batch_size, self.hidden_size)]

        auto_offset = calculate_instr_storage_offset_from_shapes(tensor_shapes, precision_settings, hbm_row_width)
        forced = getattr(self, "force_instr_offset", None)
        instr_offset = forced if forced is not None else auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        paths["tensors"] = self._save_tensors(input_tensors)

        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(f"--skip-asm-gen set but no asm at {asm_path}")
        else:
            asm_path.write_text(self._generate_assembly())
        paths["asm"] = asm_path

        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        paths["fp_sram"] = write_fp_sram_hex([0.0, 1.0], self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        create_mem_for_sim(
            precision_settings=precision_settings, data_size=256, mode="behave_sim",
            asm="scratchpad", data=None, specified_data_order=["act_tensor"],
            build_path=self.build_dir, hbm_row_width=hbm_row_width,
            mx_format=self.mx_format, instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # In-place result in activation region; strided VRAM layout (same as silu/rms).
        g2d = golden_result.reshape(self.batch_size, self.hidden_size).to(torch.float32)
        g3d = g2d.reshape(self.batch_size, self.hidden_size // self.VLEN, self.VLEN).permute(1, 0, 2)
        total_vram_rows = self.batch_size * (self.hidden_size // self.VLEN)
        golden_vram = g3d.reshape(total_vram_rows, self.VLEN)
        paths["golden_vram"] = save_golden_vram(golden_vram, output_dir=self.build_dir,
                                                filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")

        fmt = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False, "check_vram": True,
            "vram_start_row_idx": 0, "vram_num_rows": total_vram_rows, "row_dim": self.VLEN,
            "vram_compare_start_row": 0, "vram_compare_num_rows": total_vram_rows,
            "vram_total_rows": total_vram_rows, "golden_vram_file": "golden_vram_result.pt",
            "workload_type": "scratchpad", "batch_size": self.batch_size, "hidden_size": self.hidden_size,
            "output_shape": list(golden_result.shape), "mx_format": fmt,
            "exp_width": self.quant_config["exp_width"], "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        with open(self.build_dir / "test_params.json", "w") as f:
            json.dump({"workload_type": "scratchpad", "seq": self.seq, "batch_size": self.batch_size,
                       "hidden_size": self.hidden_size, "mlen": self.MLEN, "blen": self.BLEN,
                       "vlen": self.VLEN, "mx_format": fmt, "seed": self.seed}, f, indent=2)
        paths["test_params"] = self.build_dir / "test_params.json"

        self._set_env_vars(paths)
        return paths

    def get_config(self) -> dict:
        return {"workload_type": "scratchpad", "seq": self.seq, "batch_size": self.batch_size,
                "hidden_size": self.hidden_size, "mlen": self.MLEN, "blen": self.BLEN,
                "vlen": self.VLEN, "quant_config": self.quant_config, "seed": self.seed,
                "mx_format": self.mx_format}


def main():
    p = argparse.ArgumentParser(description="Scratchpad vector-op hazard reproducer")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--seq", type=str, default="loop_add", choices=SEQUENCES)
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"])
    p.add_argument("--skip-asm-gen", action="store_true")
    args = p.parse_args()

    w = ScratchpadWorkload(batch_size=args.batch, hidden_size=args.hidden_size, seq=args.seq,
                           build_dir=args.build_dir, seed=args.seed, mx_format=args.mx_format,
                           skip_asm_gen=args.skip_asm_gen)
    print(f"Generating scratchpad workload: seq={args.seq} shape=({args.batch},{args.hidden_size})")
    paths = w.generate()
    print(f"  Build dir: {w.build_dir}")


if __name__ == "__main__":
    main()
