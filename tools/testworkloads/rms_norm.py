"""RMS normalization workload generator for PLENA.

Generates test workloads for RMS normalization operations:
- Loads activation data from HBM to VRAM
- Performs RMS normalization: y = x / sqrt(mean(x^2) + eps)
- Stores result in VRAM for verification

Uses the rms_norm_asm template or PlenaCompiler for assembly generation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

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

# Import after path setup
from asm_templates import (
    preload_act_asm,
    preload_addr_reg_asm,
    reset_reg_asm,
    rms_norm_asm,
)
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram
from cfl_tools import SRC_PATH

# Load hardware tile sizes from configuration.svh
_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")

# =============================================================================
# MX Format Configuration
# =============================================================================
# Controls the data format for HBM memory generation.
# Options:
#   - None: Use the setting from precision.svh (ACT_MX_INT_ENABLE flag)
#   - "mxfp": Force MXFP (Microscaling Floating Point) format
#   - "mxint": Force MXINT (Microscaling Integer) format
DEFAULT_MX_FORMAT = None  # None means use precision.svh setting


def rms_norm_cpu(input: Tensor, eps: float = 1e-6) -> Tensor:
    """CPU reference: RMS normalization.

    Computes: y = x / sqrt(mean(x^2) + eps)

    Args:
        input: Input tensor of shape (batch, hidden_size)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor with same shape as input
    """
    x = input.float()
    # Compute RMS along the last dimension (hidden_size)
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms


class RMSNormWorkload(WorkloadGenerator):
    """RMS normalization workload generator.

    Generates test data for RMS normalization on activation tensors.

    RMS normalization computes:
        y = x / sqrt(mean(x^2) + eps)

    This generates:
    - Random activation tensor
    - Golden result computed with the CPU reference
    - Assembly code for RMS norm operations
    - Compiled machine code
    - HBM memory files in hex format
    """

    # Hardware tile sizes (loaded from configuration.svh)
    MLEN = _HW_TILE_SIZES["MLEN"]  # Matrix tile size
    BLEN = _HW_TILE_SIZES["BLEN"]  # Vector tile size
    VLEN = _HW_TILE_SIZES["VLEN"]  # Vector length
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]  # Prefetch amount

    def __init__(
        self,
        batch_size: int = 8,
        hidden_size: int = 128,
        eps: float = 1.0,
        mx_format: str = None,
        use_aten_compiler: bool = False,
        skip_asm_gen: bool = False,
        **kwargs,
    ):
        """Initialize RMS norm workload generator.

        Args:
            batch_size: Batch size (must be divisible by BLEN)
            hidden_size: Hidden dimension / feature size (must be divisible by VLEN)
            eps: Epsilon value for numerical stability
            mx_format: MX format to use ("mxfp" or "mxint"). If None, uses DEFAULT_MX_FORMAT.
            use_aten_compiler: If True, use ATen PlenaCompiler. If False, use rms_norm_asm template.
            skip_asm_gen: If True, do NOT regenerate generated_asm_code.asm. Reuse the existing
                (e.g. manually edited) file in the build dir and only assemble it to machine code.
            **kwargs: Additional arguments passed to WorkloadGenerator
        """
        super().__init__(**kwargs)

        # Validate dimensions against hardware constraints
        assert batch_size % self.BLEN == 0, f"batch_size must be divisible by {self.BLEN}"
        assert hidden_size % self.VLEN == 0, f"hidden_size must be divisible by {self.VLEN}"

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.eps = eps
        # Use provided format or fall back to module default
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.use_aten_compiler = use_aten_compiler
        self.skip_asm_gen = skip_asm_gen

    def generate(self) -> dict:
        """Generate RMS norm workload files.

        Returns:
            Dictionary with paths to all generated files
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        # Load precision settings and update quant_config based on mx_format
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        actual_quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        self.quant_config = actual_quant_config  # Update the instance quant_config
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # Scale input RMS != 1 so normalization is a clear ~4x contraction; with
        # unit-RMS input RMS-norm is ~= identity and a broken normalize goes undetected.
        ACT_RMS_SCALE = 4.0
        activation = torch.randn(self.batch_size, self.hidden_size, dtype=torch.bfloat16) * ACT_RMS_SCALE
        print(f"activation shape: {activation.shape} (RMS scale {ACT_RMS_SCALE})")

        # 2. Quantize tensor
        q_activation = self._quantize(activation)

        # 3. Compute golden result using CPU reference (on quantized input)
        golden_result = rms_norm_cpu(q_activation.float(), eps=self.eps)

        # 4. Prepare input tensors for memory generation
        input_tensors = {
            "act_tensor": q_activation.to(torch.bfloat16),
        }
        specified_data_order = ["act_tensor"]
        tensor_shapes = [(self.batch_size, self.hidden_size)]

        # Calculate and update INSTRUCTION_STORAGE_OFFSET in configuration.svh and env var
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

        # 5. Save tensors
        tensor_paths = self._save_tensors(input_tensors)
        paths["tensors"] = tensor_paths

        # Save act_tensor as txt for debugging (row-based format)
        act_txt_path = self.build_dir / "act_tensor.txt"
        q_act_2d_debug = q_activation.reshape(self.batch_size, self.hidden_size).to(torch.float32)
        with open(act_txt_path, "w") as f:
            f.write(f"# Activation Tensor (quantized)\n")
            f.write(f"# Shape: ({self.batch_size}, {self.hidden_size})\n")
            f.write(f"# VLEN: {self.VLEN}\n")
            f.write(f"#\n")
            for row_idx in range(self.batch_size):
                f.write(f"Row {row_idx:4d}:")
                for col_idx in range(self.hidden_size):
                    f.write(f" {q_act_2d_debug[row_idx, col_idx].item():12.6f}")
                f.write("\n")
        paths["act_txt"] = act_txt_path

        # 6. Generate assembly code (must be done BEFORE create_mem_for_sim)
        #    When skip_asm_gen is set, reuse the existing (possibly hand-edited) asm
        #    file and only let step 8 assemble it into machine code.
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(
                    f"--skip-asm-gen was set but no existing assembly was found at "
                    f"{asm_path}. Run once without the flag to generate it, then edit "
                    f"it manually and re-run with --skip-asm-gen."
                )
            print(f"Skipping assembly generation; reusing existing {asm_path}")
        else:
            if self.use_aten_compiler:
                asm_code = self._generate_assembly_with_aten_compiler()
            else:
                asm_code = self._generate_assembly_with_template()
            asm_path.write_text(asm_code)
        paths["asm"] = asm_path

        # 7. Initialize FP and INT SRAM preload files
        # FP SRAM slots:
        #   f0: 0.0 (always zero)
        #   f1: epsilon (for numerical stability)
        #   f2: 1/hidden_size (for mean computation)
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        fp_preload = [0.0, self.eps, 1.0 / self.hidden_size]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        # 8. Use simulator's create_mem_for_sim to generate HBM memory files
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="rms_norm",
            data=None,
            specified_data_order=specified_data_order,
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=self.mx_format,
            instr_storage_offset=instr_offset,
        )

        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 9. Generate golden results for verification
        # Output shape is same as input: (batch_size, hidden_size)
        # VRAM layout: strided - (hidden_size // VLEN, batch_size, VLEN)
        golden_2d = golden_result.reshape(self.batch_size, self.hidden_size).to(torch.float32)
        # Reshape to (batch, hidden_size // VLEN, VLEN)
        golden_3d = golden_2d.reshape(self.batch_size, self.hidden_size // self.VLEN, self.VLEN)
        # Transpose to (hidden_size // VLEN, batch, VLEN) to match strided layout
        golden_transposed = golden_3d.permute(1, 0, 2)
        # Flatten to (total_vram_rows, VLEN)
        total_vram_rows = self.batch_size * (self.hidden_size // self.VLEN)
        golden_vram = golden_transposed.reshape(total_vram_rows, self.VLEN)
        vram_golden_paths = save_golden_vram(
            golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN
        )
        paths["golden_vram"] = vram_golden_paths

        # Also save the raw golden result
        golden_paths = self._save_golden(golden_result, filename="golden_result.pt")
        paths["golden"] = golden_paths

        # 10. Save verification parameters for RTL simulation
        actual_format = self.quant_config.get("format", "mxfp")
        verification_params = {
            # Disable HBM verification (RMS norm result stays in VRAM)
            "check_hbm": False,
            # VRAM verification params
            "check_vram": True,
            "vram_start_row_idx": 0,
            # Check ALL VRAM rows (was min(8,...) = only 1/8 of hidden features).
            "vram_num_rows": total_vram_rows,
            "row_dim": self.VLEN,
            "vram_compare_start_row": 0,
            "vram_compare_num_rows": total_vram_rows,
            "vram_total_rows": total_vram_rows,
            "golden_vram_file": "golden_vram_result.pt",
            # General info
            "workload_type": "rms_norm",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "eps": self.eps,
            "output_shape": list(golden_result.shape),
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        # 11. Save test parameters
        test_params = {
            "workload_type": "rms_norm",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "eps": self.eps,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
            "seed": self.seed,
            "use_aten_compiler": self.use_aten_compiler,
        }
        params_path = self.build_dir / "test_params.json"
        with open(params_path, "w") as f:
            json.dump(test_params, f, indent=2)
        paths["test_params"] = params_path

        # 12. Set environment variables
        self._set_env_vars(paths)

        return paths

    def _generate_assembly_with_template(self) -> str:
        """Generate assembly code using rms_norm_asm template.

        Uses the rms_norm_asm template from asm_templates.

        Returns:
            Assembly code string
        """
        vlen = self.VLEN
        preload_len = self.HBM_V_Prefetch_Amount
        real_data_ratio = (8 * 8 + 8) / (8 * 8)  # Account for scale overhead

        gen_assembly_code = "; RMS Norm Test using rms_norm_asm Template\n"
        gen_assembly_code += f"; Shape: ({self.batch_size}, {self.hidden_size})\n"
        gen_assembly_code += f"; eps: {self.eps}\n"
        gen_assembly_code += f"; VLEN={vlen}\n\n"

        # Calculate HBM offset for activation
        act_hbm_size = int(self.hidden_size * self.batch_size * real_data_ratio)

        # Set address registers
        gen_assembly_code += preload_addr_reg_asm(
            addr_reg_to_set=[1],
            available_registers=[1],
            addr_reg_val=[act_hbm_size]
        )

        # Reset the registers
        gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

        # Generate Activation Preload (HBM -> VRAM)
        gen_assembly_code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=self.batch_size,
            hidden_size=self.hidden_size,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=0,
            activation_offset_reg=0,
            stride_size=self.hidden_size,
        )

        # Calculate VRAM addresses
        activation_base_address = 0
        # Scratchpad goes after the activation in VRAM
        scratchpad_base_address = self.batch_size * self.hidden_size

        # Generate RMS normalization
        # FP SRAM slots:
        #   f1 = eps (offset 1)
        #   f2 = 1/hidden_size (offset 2)
        gen_assembly_code += rms_norm_asm(
            _eps_offset=1,  # epsilon at FP SRAM slot 1
            reci_hid_offset=2,  # 1/hidden_size at FP SRAM slot 2
            alive_registers=[1, 2, 3],
            activation_base_address=activation_base_address,
            scratchpad_base_address=scratchpad_base_address,
            vlen=vlen,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_size,
        )

        # Add break instruction at end
        gen_assembly_code += "\n; End of RMS norm test\n"
        gen_assembly_code += "C_BREAK\n"

        return gen_assembly_code

    def _generate_assembly_with_aten_compiler(self) -> str:
        """Generate assembly code using PlenaCompiler from ATen compiler.

        Uses the PlenaCompiler's rms_norm method.

        Returns:
            Assembly code string
        """
        # Import PlenaCompiler
        from compiler.aten.plena import PlenaCompiler

        mlen = self.MLEN
        blen = self.BLEN
        real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MXINT8 format ratio

        # Create PlenaCompiler instance. unroll_loops=True avoids the decoder nested-loop
        # deadlock (see memory/aten-compiler-hang).
        prog = PlenaCompiler(
            mlen=mlen,
            blen=blen,
            real_data_ratio=real_data_ratio,
            unroll_loops=True,
        )

        # Add header comment
        prog.emit(f"; RMS Norm Test using ATen PlenaCompiler\n")
        prog.emit(f"; Shape: ({self.batch_size}, {self.hidden_size})\n")
        prog.emit(f"; eps: {self.eps}\n")
        prog.emit(f"; MLEN={mlen}, BLEN={blen}\n\n")

        # Declare input (activation)
        act_input = prog.input(
            "X",
            shape=(self.batch_size, self.hidden_size)
        )

        # Load activation from HBM to VRAM
        x = prog.load_batch(act_input, name="X")

        # Perform RMS normalization (in-place)
        y = prog.rms_norm(x, eps_offset=1, reci_hid_offset=2)

        # Note: Result stays in VRAM for verification

        # Add break instruction at end
        prog.emit("\n; End of RMS norm test\n")
        prog.emit("C_BREAK\n")

        # Compile and return the ISA code
        return prog.compile()

    def get_config(self) -> dict:
        """Get the workload configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            "workload_type": "rms_norm",
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "eps": self.eps,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "quant_config": self.quant_config,
            "seed": self.seed,
            "mx_format": self.mx_format,
            "use_aten_compiler": self.use_aten_compiler,
        }


def main():
    """CLI entry point for RMS norm workload generation."""
    parser = argparse.ArgumentParser(
        description="Generate RMS norm workload for PLENA RTL simulation"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (must be divisible by BLEN)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128,
        help="Hidden dimension / feature size (must be divisible by VLEN)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6,
        help="Epsilon value for numerical stability"
    )
    parser.add_argument(
        "--build-dir", type=str, default=None,
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--mx-format", type=str, default=None, choices=["mxfp", "mxint"],
        help="MX format to use (mxfp or mxint). Default: use precision.svh settings"
    )
    parser.add_argument(
        "--use-aten-compiler", action="store_true",
        help="Use ATen PlenaCompiler instead of rms_norm_asm template (default: use template)"
    )
    parser.add_argument(
        "--skip-asm-gen", action="store_true",
        help="Skip regenerating generated_asm_code.asm; reuse the existing (e.g. manually "
             "edited) file in the build dir and only assemble it to machine code. "
             "Pair with 'just rtl-sim rms_norm false ...' so the build dir is not wiped."
    )
    args = parser.parse_args()

    # Create workload generator
    workload = RMSNormWorkload(
        batch_size=args.batch,
        hidden_size=args.hidden_size,
        eps=args.eps,
        build_dir=args.build_dir,
        seed=args.seed,
        mx_format=args.mx_format,
        use_aten_compiler=args.use_aten_compiler,
        skip_asm_gen=args.skip_asm_gen,
    )

    # Generate files
    compiler_type = "ATen PlenaCompiler" if args.use_aten_compiler else "rms_norm_asm template"
    print(f"Generating RMS norm workload using {compiler_type}:")
    print(f"  Shape: ({args.batch}, {args.hidden_size})")
    print(f"  eps: {args.eps}")
    print(f"  Build dir: {workload.build_dir}")

    paths = workload.generate()

    print(f"\nGenerated files:")
    for key, value in paths.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {key}/{subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")

    print(f"\nConfiguration:")
    config = workload.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
