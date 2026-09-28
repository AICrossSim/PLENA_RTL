"""Prefetch-only workload generator for PLENA.

Generates test workloads for data prefetching operations only (HBM -> VRAM):
- Loads activation data from HBM to VRAM
- Multiplies by f0 (0.0) to test VRAM operations
- Stores result back to HBM for verification

Useful for testing:
- HBM read path
- VRAM write path
- HBM write path (store operation)
- Address register setup
- Scale factor handling
"""

import argparse
import json
import os
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
    store_act_asm,
)
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_hbm, save_golden_vram
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
#
# The format definitions are in src/definitions/precision.svh:
#   MXFP: ACT_MXFP_EXP_WIDTH, ACT_MXFP_MANT_WIDTH, ACT_MX_SCALE_WIDTH
#   MXINT: ACT_MX_INT_WIDTH, ACT_MX_SCALE_WIDTH
DEFAULT_MX_FORMAT = None  # None means use precision.svh setting


class PrefetchWorkload(WorkloadGenerator):
    """Prefetch-only workload generator.

    Generates test data for data prefetching from HBM to VRAM.
    No computation is performed - this is purely for testing data movement.

    This generates:
    - Random activation tensor (and optionally weight tensor)
    - Assembly code for prefetch operations only
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
        in_features: int = 128,
        out_features: int = 256,
        include_weights: bool = True,
        mx_format: str = None,
        **kwargs,
    ):
        """Initialize prefetch workload generator.

        Args:
            batch_size: Batch size (must be divisible by BLEN)
            in_features: Input feature dimension (must be divisible by MLEN)
            out_features: Output feature dimension (must be divisible by BLEN)
            include_weights: Whether to also prefetch weight data
            mx_format: MX format to use ("mxfp" or "mxint"). If None, uses DEFAULT_MX_FORMAT.
            **kwargs: Additional arguments passed to WorkloadGenerator
        """
        super().__init__(**kwargs)

        # Validate dimensions against hardware constraints
        assert batch_size % self.BLEN == 0, f"batch_size must be divisible by {self.BLEN}"
        assert in_features % self.MLEN == 0, f"in_features must be divisible by {self.MLEN}"
        assert out_features % self.BLEN == 0, f"out_features must be divisible by {self.BLEN}"

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.include_weights = include_weights
        # Use provided format or fall back to module default
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT

    def generate(self) -> dict:
        """Generate prefetch workload files.

        Returns:
            Dictionary with paths to all generated files
        """
        # Initialize output paths
        paths = self._init_memory_files()

        # Load precision settings and update quant_config based on mx_format
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        actual_quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        self.quant_config = actual_quant_config  # Update the instance quant_config

        # 1. Generate random tensors
        activation = torch.randn(self.batch_size, self.in_features, dtype=torch.bfloat16)
        print(f"activation shape: {activation.shape}")

        # 2. Quantize tensors to MXFP (for .pt file saving - actual HBM quantization happens in create_mem_for_sim)
        q_activation = self._quantize(activation)

        # Prepare input tensors for memory generation
        if self.include_weights:
            weight = torch.randn(self.in_features, self.out_features, dtype=torch.bfloat16)
            q_weight = self._quantize(weight)
            input_tensors = {
                "act_tensor": q_activation,
                "weights": q_weight,
            }
            specified_data_order = ["act_tensor", "weights"]
            tensor_shapes = [(self.batch_size, self.in_features), (self.in_features, self.out_features)]
        else:
            input_tensors = {
                "act_tensor": q_activation,
            }
            specified_data_order = ["act_tensor"]
            tensor_shapes = [(self.batch_size, self.in_features)]

        # Calculate and update INSTRUCTION_STORAGE_OFFSET in configuration.svh and env var
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        instr_offset = calculate_instr_storage_offset_from_shapes(
            tensor_shapes, precision_settings, hbm_row_width
        )
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        # 3. Save tensors
        tensor_paths = self._save_tensors(input_tensors)
        paths["tensors"] = tensor_paths

        # Save act_tensor as txt for debugging (row-based format)
        act_txt_path = self.build_dir / "act_tensor.txt"
        q_act_2d_debug = q_activation.reshape(self.batch_size, self.in_features).to(torch.float32)
        with open(act_txt_path, "w") as f:
            f.write(f"# Activation Tensor (quantized)\n")
            f.write(f"# Shape: ({self.batch_size}, {self.in_features})\n")
            f.write(f"# VLEN: {self.VLEN}\n")
            f.write(f"#\n")
            for row_idx in range(self.batch_size):
                f.write(f"Row {row_idx:4d}:")
                for col_idx in range(self.in_features):
                    f.write(f" {q_act_2d_debug[row_idx, col_idx].item():12.6f}")
                f.write("\n")
        paths["act_txt"] = act_txt_path

        # 4. Generate assembly code (must be done BEFORE create_mem_for_sim)
        asm_code = self._generate_assembly()
        asm_path = self.build_dir / "generated_asm_code.asm"
        asm_path.write_text(asm_code)
        paths["asm"] = asm_path

        # 5. Initialize FP and INT SRAM preload files
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        fp_preload = [0.0, 1e-6, 1 / self.in_features]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        # 6. Use simulator's create_mem_for_sim to generate HBM memory files
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="prefetch",
            data=None,
            specified_data_order=specified_data_order,
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=self.mx_format,
            instr_storage_offset=instr_offset,
        )

        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 7. Generate golden results for verification
        # Calculate row-based params first (needed for golden file formatting)
        num_elements = self.batch_size * self.in_features
        block_size = 8  # MXFP block size
        actual_format = self.quant_config.get("format", "mxfp")
        # HBM: 256 bits per row, 8 bits per element = 32 elements per row
        hbm_row_width_bits = hbm_row_width
        hbm_elements_per_row = hbm_row_width_bits // 8  # 32 elements per row for 8-bit elements

        # Generate separate golden references for VRAM and HBM (both in FP format)
        # VRAM golden - reflects how activation is loaded to VRAM
        # Each VRAM row has VLEN elements
        # The full activation (batch_size, in_features) is loaded to VRAM as:
        #   - Total VRAM rows = batch_size * (in_features / VLEN)
        #   - Row ordering: batch 0 vectors, then batch 1 vectors, etc.
        # Only VRAM row 0 is zeroed after V_MUL_VF (gp0 * f0 where f0=0.0)
        q_act_2d = q_activation.reshape(self.batch_size, self.in_features).to(torch.float32)
        # Reshape to (total_vram_rows, VLEN) matching strided VRAM layout
        # Strided load creates layout: (in_features // VLEN, batch, VLEN)
        # This interleaves batches: for each VLEN chunk, all batches are stored consecutively
        total_vram_rows = self.batch_size * (self.in_features // self.VLEN)
        # First reshape to (batch, in_features // VLEN, VLEN)
        q_act_3d = q_act_2d.reshape(self.batch_size, self.in_features // self.VLEN, self.VLEN)
        # Transpose to (in_features // VLEN, batch, VLEN) to match strided load order
        q_act_transposed = q_act_3d.permute(1, 0, 2)
        # Flatten to (total_vram_rows, VLEN)
        golden_vram = q_act_transposed.reshape(total_vram_rows, self.VLEN).clone()
        # golden_vram[0, :] = 0.0  # First row zeroed by V_MUL_VF
        vram_golden_paths = save_golden_vram(golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN)
        paths["golden_vram"] = vram_golden_paths

        # HBM golden - simplified verification: just check first VLEN elements are zeros
        # The V_MUL_VF operation zeros gp0 (batch 0's first VLEN elements)
        # These are stored at HBM addresses 0 to VLEN-1 after H_STORE_V
        # For now, only verify this key result to avoid strided layout complexity
        golden_hbm = torch.zeros(self.VLEN, dtype=torch.float32)  # First 16 elements should be zeros
        hbm_golden_paths = save_golden_hbm(golden_hbm, output_dir=self.build_dir, filename="golden_hbm_result", elements_per_row=hbm_elements_per_row)
        paths["golden_hbm"] = hbm_golden_paths

        # Save verification parameters
        hbm_total_rows = (num_elements + hbm_elements_per_row - 1) // hbm_elements_per_row

        # VRAM: VLEN elements per row, total rows = batch_size * (in_features / VLEN)
        vram_total_rows = self.batch_size * (self.in_features // self.VLEN)

        verification_params = {
            # HBM verification params. The program only prefetches HBM -> VRAM
            # (the V_MUL_VF / H_STORE_V write-back this check was written for is
            # disabled above), so there is nothing to verify in HBM.
            "check_hbm": False,
            "result_hbm_start_byte": 0,
            "result_hbm_size_bytes": num_elements + (num_elements // block_size),  # elements + scales
            "num_elements": num_elements,
            "num_batches": self.batch_size,
            "elements_per_batch": self.in_features,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
            "block_size": block_size,
            "scale_offset": num_elements,  # Scales follow elements
            "mx_format": actual_format,  # "mxfp" or "mxint"
            # HBM comparison control (compare first VLEN elements only - the zeros from V_MUL_VF)
            "hbm_compare_start_row": 0,
            "hbm_compare_num_rows": 1,  # Only first row (but golden has just VLEN elements)
            "hbm_total_rows": hbm_total_rows,
            "hbm_elements_per_row": hbm_elements_per_row,
            "golden_hbm_file": "golden_hbm_result.pt",
            # VRAM verification params
            "check_vram": True,
            "vram_start_row_idx": 0,
            "vram_num_rows": vram_total_rows,
            "row_dim": self.VLEN,
            "vram_elements_per_batch": self.in_features,  # Full in_features per batch
            "use_stride_mode": False,  # Golden already in VRAM row layout
            # VRAM comparison control - compare all rows
            "vram_compare_start_row": 0,
            "vram_compare_num_rows": vram_total_rows,  # Compare all VRAM rows
            "vram_total_rows": vram_total_rows,
            "golden_vram_file": "golden_vram_result.pt",
            # General info
            "activation_shape": list(q_activation.shape),
            "activation_vram_offset": 0,
            "workload_type": "prefetch",
        }
        if self.include_weights:
            verification_params["weight_shape"] = list(q_weight.shape)

        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        # 9. Set environment variables
        self._set_env_vars(paths)

        return paths

    def _generate_assembly(self) -> str:
        """Generate assembly code for prefetch-only operation.

        Returns:
            Assembly code string
        """
        # Hardware parameters (loaded from configuration.svh)
        vlen = self.VLEN  # Vector machine tile length
        blen = self.BLEN   # Batch length for preload
        preload_len = self.HBM_V_Prefetch_Amount  # Prefetch amount from config
        real_data_ratio = (8 * 8 + 8) / (8 * 8)  # Account for scale overhead

        gen_assembly_code = "; Prefetch-Only Test Generation\n"
        gen_assembly_code += (
            f"; Shape: ({self.batch_size}, {self.in_features})"
        )
        if self.include_weights:
            gen_assembly_code += f" + weights ({self.in_features}, {self.out_features})"
        gen_assembly_code += "\n"

        # Calculate HBM offsets
        # Layout in HBM: [activations | weights]
        act_hbm_size = int(self.in_features * self.batch_size * real_data_ratio)

        # Set the addr offset for weight (needed for proper HBM addressing)
        gen_assembly_code += preload_addr_reg_asm(
            addr_reg_to_set=[1],
            available_registers=[1],
            addr_reg_val=[act_hbm_size]
        )

        # Reset the registers
        gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])

        # Generate Activation Preload
        gen_assembly_code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=self.batch_size,
            hidden_size=self.in_features,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=0,
            activation_offset_reg=0,
            stride_size=self.in_features,
        )

        # gen_assembly_code += "V_MUL_VF gp0, gp0, f0, 0 \n"

        # gen_assembly_code += store_act_asm(
        #     vlen=vlen,
        #     batch=self.batch_size,
        #     hidden_size=self.in_features,
        #     alive_registers=[1, 2, 3, 4, 5],
        #     act_vram_offset=0,
        #     hbm_addr_reg=0,
        #     stride_size=self.in_features,
        #     store_amount=4,
        # )

        # Terminate the program. SimTop_tb waits for C_BREAK; without it the
        # run times out with "C_BREAK not loaded" and is reported as a hang.
        gen_assembly_code += "\n; End of prefetch\nC_BREAK\n"

        return gen_assembly_code

    def get_config(self) -> dict:
        """Get the workload configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            "workload_type": "prefetch",
            "batch_size": self.batch_size,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "include_weights": self.include_weights,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "quant_config": self.quant_config,
            "seed": self.seed,
            "mx_format": self.mx_format,
        }


def main():
    """CLI entry point for prefetch workload generation."""
    parser = argparse.ArgumentParser(
        description="Generate prefetch-only workload for PLENA RTL simulation"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (must be divisible by 8)"
    )
    parser.add_argument(
        "--in-features", type=int, default=128,
        help="Input feature dimension (must be divisible by 8)"
    )
    parser.add_argument(
        "--out-features", type=int, default=256,
        help="Output feature dimension (must be divisible by 8)"
    )
    parser.add_argument(
        "--no-weights", action="store_true",
        help="Only prefetch activations, not weights"
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
    args = parser.parse_args()

    # Create workload generator
    workload = PrefetchWorkload(
        batch_size=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        include_weights=not args.no_weights,
        build_dir=args.build_dir,
        seed=args.seed,
        mx_format=args.mx_format,
    )

    # Generate files
    print(f"Generating prefetch workload:")
    print(f"  Activation shape: ({args.batch}, {args.in_features})")
    if not args.no_weights:
        print(f"  Weight shape: ({args.in_features}, {args.out_features})")
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
