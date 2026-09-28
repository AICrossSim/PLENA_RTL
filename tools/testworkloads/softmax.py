"""Row-wise softmax workload generator for PLENA.

A minimal, focused test of the softmax vector-op chain used by attention:

    softmax(x)[i] = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

over each row (one VLEN-wide vector = one batch item's `hidden` features, with
hidden == VLEN so a row is contiguous and the reductions reduce over the row).

This is the standalone repro / regression test for the "broadcast-op -> V_RED_SUM"
pipeline interlock: the natural softmax chain interleaves broadcast vector ops
(V_SUB_VF / V_EXP_V / V_MUL_VF, which set v_broadcast_en/update_v_waddr) with
reductions (V_RED_MAX / V_RED_SUM), and a reduction that consumes data produced by
a broadcast op currently deadlocks. See doc/design/attention_isa_roadmap.md.

Pipeline: preload activation HBM->VRAM, run softmax_asm in place, leave the result
in the activation VRAM region for verification. Modeled on silu.py / rms_norm.py.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
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
)
from asm_templates._imm import load_large_int_str as _load_large_int
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")
# Defaults follow the tile sizes so `just rtl-sim softmax` works at any MLEN/VLEN.
_MLEN, _VLEN = _HW_TILE_SIZES["MLEN"], _HW_TILE_SIZES["VLEN"]

DEFAULT_MX_FORMAT = None  # None means use precision.svh setting


def softmax_cpu(x: Tensor) -> Tensor:
    """CPU reference: row-wise softmax over the last dim."""
    return torch.softmax(x.float(), dim=-1)


def softmax_asm(
    activation_base_address: int,
    batch: int,
    vlen: int,
    alive_registers: list[int],
    use_max_sub: bool = True,
    unroll: bool = False,
) -> str:
    """Row-wise softmax over `batch` contiguous VLEN-wide rows, in place.

    Per row (address `base + q*vlen`):
        m = max(row)              V_RED_MAX  -> f2      (if use_max_sub)
        row = row - m             V_SUB_VF   (broadcast)
        row = exp(row)            V_EXP_V    (broadcast)
        l = sum(row)              V_RED_SUM  -> f3
        l = 1/l                   S_RECI_FP
        row = row * l             V_MUL_VF   (broadcast) = softmax row

    NOTE: this is the "natural" chain that currently deadlocks the RTL because a
    reduction (V_RED_SUM / V_RED_MAX) consumes data produced by a broadcast op.
    Kept as the faithful repro; it should pass once the interlock is fixed.
    """
    addr = alive_registers[0]
    loop = alive_registers[1]
    nop = "S_ADDI_INT gp0, gp0, 0\n"

    code = "; Row-wise Softmax\n"

    def row_body(addr_reg: str) -> str:
        c = ""
        if use_max_sub:
            c += f"V_RED_MAX f2, gp{addr_reg}, 0\n"
            c += f"V_SUB_VF gp{addr_reg}, gp{addr_reg}, f2, 0, 0\n"
        c += f"V_EXP_V gp{addr_reg}, gp{addr_reg}, 0\n"
        c += "S_ADD_FP f3, f0, f0\n"
        c += f"V_RED_SUM f3, gp{addr_reg}\n"
        c += "S_RECI_FP f3, f3\n"
        c += nop * 4
        c += f"V_MUL_VF gp{addr_reg}, gp{addr_reg}, f3, 0\n"
        return c

    if unroll:
        for q in range(batch):
            code += _load_large_int(addr, activation_base_address + q * vlen)
            code += row_body(addr)
    else:
        code += _load_large_int(addr, activation_base_address)
        code += f"C_LOOP_START gp{loop}, {batch}\n"
        code += row_body(addr)
        code += f"S_ADDI_INT gp{addr}, gp{addr}, {vlen}\n"
        code += f"C_LOOP_END gp{loop}\n"
    return code


class SoftmaxWorkload(WorkloadGenerator):
    """Row-wise softmax workload (single vector-reduction+broadcast layer)."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    def __init__(
        self,
        batch_size: int = _MLEN,
        hidden_size: int = _VLEN,
        mx_format: str = None,
        use_max_sub: bool = True,
        unroll: bool = False,
        skip_asm_gen: bool = False,
        use_aten_compiler: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert batch_size % self.BLEN == 0, f"batch_size must be divisible by BLEN={self.BLEN}"
        assert hidden_size == self.VLEN, (
            f"hidden_size must equal VLEN={self.VLEN} so each row is one contiguous "
            f"vector for in-row reduction (got {hidden_size})."
        )
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.use_max_sub = use_max_sub
        self.unroll = unroll
        self.skip_asm_gen = skip_asm_gen
        # When True, emit ISA via the ATen PlenaCompiler (ops.softmax -> the online
        # softmax backend). ops.softmax operates on a single (MLEN, MLEN) block and
        # writes the result into a freshly-allocated S region (NOT in place), so the
        # ATen path requires batch_size == MLEN and verification targets S's address.
        self.use_aten_compiler = use_aten_compiler
        if use_aten_compiler:
            assert batch_size == self.MLEN, (
                f"use_aten_compiler requires batch_size == MLEN ({self.MLEN}); ops.softmax "
                f"lowers one (MLEN, MLEN) block (got batch_size={batch_size})."
            )
        # Stashed by _generate_assembly_with_aten_compiler(): VRAM element address
        # of the ops.softmax output block S.
        self._aten_output_vram_addr = None

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        actual_quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        self.quant_config = actual_quant_config
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # 1. Random activation, modest magnitude (keep exp in FP12 range).
        activation = torch.randn(self.batch_size, self.hidden_size, dtype=torch.bfloat16)

        # 2. Quantize.
        q_activation = self._quantize(activation)

        # 3. Golden on the quantized input.
        golden_result = softmax_cpu(q_activation.float())

        # 4. Memory layout inputs.
        input_tensors = {"act_tensor": q_activation.to(torch.bfloat16)}
        specified_data_order = ["act_tensor"]
        tensor_shapes = [(self.batch_size, self.hidden_size)]

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

        # 5. Save tensors.
        tensor_paths = self._save_tensors(input_tensors)
        paths["tensors"] = tensor_paths

        # 6. Assembly.
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(
                    f"--skip-asm-gen was set but no existing assembly was found at {asm_path}."
                )
            print(f"Skipping assembly generation; reusing existing {asm_path}")
        elif self.use_aten_compiler:
            asm_path.write_text(self._generate_assembly_with_aten_compiler())
        else:
            asm_path.write_text(self._generate_assembly_with_template())
        paths["asm"] = asm_path

        # 7. FP SRAM preload.
        #   Template path:  f0=0.0, f1=1.0.
        #   ATen path (online softmax): f0=0.0, f1=scale(=1.0), f2=-inf (m_old init).
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        fp_preload = [0.0, 1.0, float("-inf")] if self.use_aten_compiler else [0.0, 1.0]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        # 8. HBM mem files.
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="softmax",
            data=None,
            specified_data_order=specified_data_order,
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=self.mx_format,
            instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 9. Golden VRAM (strided; in-place in the activation region).
        golden_2d = golden_result.reshape(self.batch_size, self.hidden_size).to(torch.float32)
        golden_3d = golden_2d.reshape(self.batch_size, self.hidden_size // self.VLEN, self.VLEN)
        golden_transposed = golden_3d.permute(1, 0, 2)
        total_vram_rows = self.batch_size * (self.hidden_size // self.VLEN)
        golden_vram = golden_transposed.reshape(total_vram_rows, self.VLEN)
        vram_golden_paths = save_golden_vram(
            golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN
        )
        paths["golden_vram"] = vram_golden_paths

        golden_paths = self._save_golden(golden_result, filename="golden_result.pt")
        paths["golden"] = golden_paths

        # 10. Verification params.
        # Template softmax runs in place at VRAM row 0; the ATen ops.softmax writes
        # its result into a freshly-allocated S block (captured while emitting ISA).
        aten_start_row = (self._aten_output_vram_addr or 0) // self.VLEN
        actual_format = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False,
            "check_vram": True,
            "vram_start_row_idx": aten_start_row if self.use_aten_compiler else 0,
            "vram_num_rows": total_vram_rows,
            "row_dim": self.VLEN,
            "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_vram_rows,
            "vram_total_rows": total_vram_rows,
            "golden_vram_file": "golden_vram_result.pt",
            "workload_type": "softmax",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "output_shape": list(golden_result.shape),
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        # 11. Test params.
        test_params = {
            "workload_type": "softmax",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "use_max_sub": self.use_max_sub,
            "unroll": self.unroll,
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

        # 12. Env vars.
        self._set_env_vars(paths)
        return paths

    def _generate_assembly_with_template(self) -> str:
        vlen = self.VLEN
        preload_len = self.HBM_V_Prefetch_Amount
        real_data_ratio = (8 * 8 + 8) / (8 * 8)

        code = "; Softmax Test using softmax_asm Template\n"
        code += f"; Shape: ({self.batch_size}, {self.hidden_size})  (row-wise softmax)\n"
        code += f"; VLEN={vlen}  use_max_sub={self.use_max_sub}  unroll={self.unroll}\n\n"

        act_hbm_size = int(self.hidden_size * self.batch_size * real_data_ratio)

        code += preload_addr_reg_asm(
            addr_reg_to_set=[1], available_registers=[1], addr_reg_val=[act_hbm_size]
        )
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
        code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=self.batch_size,
            hidden_size=self.hidden_size,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=0,
            activation_offset_reg=0,
            stride_size=self.hidden_size,
        )

        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
        code += softmax_asm(
            activation_base_address=0,
            batch=self.batch_size,
            vlen=vlen,
            alive_registers=[1, 2],
            use_max_sub=self.use_max_sub,
            unroll=self.unroll,
        )

        code += "\n; End of softmax test\n"
        code += "C_BREAK\n"
        return code

    def _generate_assembly_with_aten_compiler(self) -> str:
        """Generate ISA via the ATen PlenaCompiler (ops.softmax).

        ops.softmax lowers the numerically-stable ONLINE softmax over one
        (MLEN, MLEN) block: it allocates an S block, copies the input into it,
        runs the online-softmax kernel (using the flash-attention FP-SRAM state
        layout), and returns S. The result therefore lands in a freshly-allocated
        VRAM region, not in place. The block is row-wise softmax(input * scale)
        with scale=1.0, so the golden matches softmax_cpu (the template golden).

        fp_preload (set in generate()): [0]=0.0, [1]=scale(=1.0), [2]=-inf.
        """
        from compiler.aten.plena import PlenaCompiler
        import compiler.aten.ops as ops

        mlen = self.MLEN
        blen = self.BLEN
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

        prog.emit("; Softmax Test using ATen PlenaCompiler (ops.softmax)\n")
        prog.emit(f"; Shape: ({self.batch_size}, {self.hidden_size})  (row-wise online softmax)\n")
        prog.emit(f"; MLEN={mlen} BLEN={blen}\n\n")

        # Declare input (activation) in HBM and load it to VRAM (row 0).
        act_input = prog.input("X", shape=(self.batch_size, self.hidden_size))
        x = prog.load_batch(act_input, name="X")

        # Row-wise online softmax over the (MLEN, MLEN) block -> new S region.
        s = ops.softmax(prog, x, scale=1.0)
        self._aten_output_vram_addr = prog.get_vram_addr(s.name)

        prog.emit("\n; End of softmax test\n")
        prog.emit("C_BREAK\n")
        return prog.compile()

    def get_config(self) -> dict:
        return {
            "workload_type": "softmax",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "use_max_sub": self.use_max_sub,
            "unroll": self.unroll,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "quant_config": self.quant_config,
            "seed": self.seed,
            "mx_format": self.mx_format,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate row-wise softmax workload for PLENA RTL simulation")
    parser.add_argument("--batch", type=int, default=_MLEN, help="Batch size (divisible by BLEN)")
    parser.add_argument("--hidden-size", type=int, default=_VLEN, help="Hidden dim (== VLEN)")
    parser.add_argument("--build-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"],
                        help="MX format (default: use precision.svh)")
    parser.add_argument("--no-max-sub", dest="use_max_sub", action="store_false",
                        help="Skip the max-subtraction (softmax is shift-invariant)")
    parser.add_argument("--unroll", action="store_true", help="Unroll the per-row loop (no C_LOOP)")
    parser.add_argument("--skip-asm-gen", action="store_true",
                        help="Reuse existing generated_asm_code.asm, only re-assemble")
    parser.add_argument("--use-aten-compiler", action="store_true",
                        help="Use ATen PlenaCompiler (ops.softmax, online softmax over one "
                             "(MLEN, MLEN) block) instead of the softmax_asm template. Requires "
                             "batch == MLEN; verifies the freshly-allocated S output block.")
    args = parser.parse_args()

    workload = SoftmaxWorkload(
        batch_size=args.batch,
        hidden_size=args.hidden_size,
        build_dir=args.build_dir,
        seed=args.seed,
        mx_format=args.mx_format,
        use_max_sub=args.use_max_sub,
        unroll=args.unroll,
        skip_asm_gen=args.skip_asm_gen,
        use_aten_compiler=args.use_aten_compiler,
    )

    print("Generating softmax workload:")
    print(f"  Shape: ({args.batch}, {args.hidden_size})  row-wise softmax")
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
