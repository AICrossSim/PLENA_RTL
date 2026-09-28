"""SiLU activation workload generator for PLENA.

Generates a minimal test that exercises the vector non-linearities used by the
FFN's SiLU gate, which are otherwise UNPROVEN at the system level:
    V_EXP_V  (exp)      via fp_vector_element_alu / fp_fix_exp
    V_RECI_V (1/x)      via fp_vector_element_alu / fp_fix_reciprocal
    V_SUB_VF (reverse)  -x  = f0 - x
    V_ADD_VF            1 + exp(-x)
    V_MUL_VV            x * sigmoid(x)

SiLU(x) = x * sigmoid(x) = x * 1/(1 + exp(-x)).

Pipeline: load activation HBM->VRAM, run silu_asm in place, leave the result in
the activation VRAM region for verification. Modeled on rms_norm.py.
"""

import argparse
import json
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
    silu_asm,
)
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")

DEFAULT_MX_FORMAT = None  # None means use precision.svh setting


def silu_cpu(input: Tensor) -> Tensor:
    """CPU reference: SiLU(x) = x * sigmoid(x)."""
    return F.silu(input.float())


class SiLUWorkload(WorkloadGenerator):
    """SiLU activation workload generator (single vector non-linearity layer)."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    def __init__(
        self,
        batch_size: int = 8,
        hidden_size: int = 128,
        mx_format: str = None,
        skip_asm_gen: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert batch_size % self.BLEN == 0, f"batch_size must be divisible by {self.BLEN}"
        assert hidden_size % self.VLEN == 0, f"hidden_size must be divisible by {self.VLEN}"
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.skip_asm_gen = skip_asm_gen

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        actual_quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        self.quant_config = actual_quant_config
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # 1. Random activation. Keep magnitudes modest so exp(-x) stays in range
        #    of the FP12 special-function unit (large |x| saturates exp/reci).
        activation = torch.randn(self.batch_size, self.hidden_size, dtype=torch.bfloat16)

        # 2. Quantize
        q_activation = self._quantize(activation)

        # 3. Golden on the quantized input
        golden_result = silu_cpu(q_activation.float())

        # 4. Memory layout inputs
        input_tensors = {"act_tensor": q_activation.to(torch.bfloat16)}
        specified_data_order = ["act_tensor"]
        tensor_shapes = [(self.batch_size, self.hidden_size)]

        # Suite mode may force a fixed instruction offset shared by all cases
        # (so a single Verilator build serves every case). When forced, the
        # data must still fit below it.
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

        # 5. Save tensors + debug txt
        tensor_paths = self._save_tensors(input_tensors)
        paths["tensors"] = tensor_paths

        act_txt_path = self.build_dir / "act_tensor.txt"
        q_act_2d_debug = q_activation.reshape(self.batch_size, self.hidden_size).to(torch.float32)
        with open(act_txt_path, "w") as f:
            f.write("# Activation Tensor (quantized)\n")
            f.write(f"# Shape: ({self.batch_size}, {self.hidden_size})\n")
            f.write(f"# VLEN: {self.VLEN}\n#\n")
            for row_idx in range(self.batch_size):
                f.write(f"Row {row_idx:4d}:")
                for col_idx in range(self.hidden_size):
                    f.write(f" {q_act_2d_debug[row_idx, col_idx].item():12.6f}")
                f.write("\n")
        paths["act_txt"] = act_txt_path

        # 6. Assembly
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

        # 7. FP SRAM preload: f0=0.0, f1=1.0 (the const_one used by silu).
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        fp_preload = [0.0, 1.0]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        # 8. HBM mem files
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="silu",
            data=None,
            specified_data_order=specified_data_order,
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=self.mx_format,
            instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 9. Golden VRAM (strided layout, same as rms_norm: result is in-place in
        #    the activation region).
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

        # 10. Verification params
        actual_format = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False,
            "check_vram": True,
            "vram_start_row_idx": 0,
            "vram_num_rows": total_vram_rows,
            "row_dim": self.VLEN,
            "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_vram_rows,
            "vram_total_rows": total_vram_rows,
            "golden_vram_file": "golden_vram_result.pt",
            "workload_type": "silu",
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

        # 11. Test params
        test_params = {
            "workload_type": "silu",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
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

        # 12. Env vars
        self._set_env_vars(paths)
        return paths

    def _generate_assembly_with_template(self) -> str:
        vlen = self.VLEN
        preload_len = self.HBM_V_Prefetch_Amount
        real_data_ratio = (8 * 8 + 8) / (8 * 8)

        code = "; SiLU Test using silu_asm Template\n"
        code += f"; Shape: ({self.batch_size}, {self.hidden_size})\n"
        code += f"; VLEN={vlen}\n\n"

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

        activation_base_address = 0
        scratchpad_base_address = self.batch_size * self.hidden_size

        # FP SRAM slot 1 holds 1.0 (const_one).
        code += silu_asm(
            const_one_fp_address=1,
            alive_registers=[1, 2, 3],
            activation_base_address=activation_base_address,
            scratchpad_base_address=scratchpad_base_address,
            vlen=vlen,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_size,
        )

        code += "\n; End of SiLU test\n"
        code += "C_BREAK\n"
        return code

    def get_config(self) -> dict:
        return {
            "workload_type": "silu",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "quant_config": self.quant_config,
            "seed": self.seed,
            "mx_format": self.mx_format,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate SiLU workload for PLENA RTL simulation")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (divisible by BLEN)")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden dim (divisible by VLEN)")
    parser.add_argument("--build-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"],
                        help="MX format (default: use precision.svh)")
    parser.add_argument("--skip-asm-gen", action="store_true",
                        help="Reuse existing generated_asm_code.asm, only re-assemble")
    args = parser.parse_args()

    workload = SiLUWorkload(
        batch_size=args.batch,
        hidden_size=args.hidden_size,
        build_dir=args.build_dir,
        seed=args.seed,
        mx_format=args.mx_format,
        skip_asm_gen=args.skip_asm_gen,
    )

    print("Generating SiLU workload:")
    print(f"  Shape: ({args.batch}, {args.hidden_size})")
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
