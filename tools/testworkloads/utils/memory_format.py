"""Memory format utilities for RTL simulation.

Provides functions for converting tensors to MXFP format and writing
hex-formatted memory files for RTL simulation.
"""

import os
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import Tensor

from plena_quant.mxfp import _mx_fp_quantize_hardware
from cfl_cocotb.torch_fp_conversion import pack_fp_to_bin
from verification.view_vector_result import float_to_fp


def _scalar_fp_format() -> Tuple[int, int]:
    """Return (exp_width, man_width) of the RTL scalar FP format.

    Reads S_FP_EXP_WIDTH / S_FP_MANT_WIDTH from precision.svh so encoded
    constants always match the hardware fp_reg_file width. Falls back to the
    FP12 (6e + 5m) defaults if the file can't be read.
    """
    try:
        from plena_utils.load_config import load_svh_settings
        from cfl_tools import SRC_PATH

        svh = Path(SRC_PATH) / "definitions" / "precision.svh"
        settings = load_svh_settings(str(svh))
        return (
            int(settings.get("S_FP_EXP_WIDTH", 6)),
            int(settings.get("S_FP_MANT_WIDTH", 5)),
        )
    except Exception:
        return 6, 5


def tensor_to_mxfp_blocks(
    tensor: Tensor,
    quant_config: dict,
) -> Tuple[List[List[int]], List[int]]:
    """Convert tensor to MXFP blocks and bias.

    Args:
        tensor: Input tensor to quantize
        quant_config: Quantization configuration dictionary with:
            - exp_width: Exponent width
            - man_width: Mantissa width
            - exp_bias_width: Block scaling factor width
            - block_size: Block dimensions [h, w]
            - skip_first_dim: Whether to skip first dimension

    Returns:
        Tuple of (blocks_list, scaling_list):
            - blocks_list: List of blocks, each block is a list of integer values
            - scaling_list: List of integer scaling factors
    """
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    bm_x, per_block_exponent, per_block_mantissa, per_block_scaling = _mx_fp_quantize_hardware(
        tensor,
        width=quant_config["exp_width"] + quant_config["man_width"] + 1,
        exponent_width=quant_config["exp_width"],
        exponent_bias_width=quant_config["exp_bias_width"],
        block_size=quant_config["block_size"],
        skip_first_dim=quant_config.get("skip_first_dim", False),
    )

    blocks_list = []
    scaling_list = []

    for i in range(per_block_mantissa.shape[0]):
        bin_block = pack_fp_to_bin(
            per_block_exponent[i],
            per_block_mantissa[i],
            quant_config["exp_width"],
            quant_config["man_width"],
        )
        blocks_list.append(bin_block.tolist())
        scaling_list.append(int(per_block_scaling[i]))

    return blocks_list, scaling_list


def write_fp_sram_hex(
    data: Union[List[float], Tensor],
    directory: Union[str, Path],
    filename: str = "fp_sram.mem",
    exp_width: int | None = None,
    man_width: int | None = None,
) -> Path:
    """Write scalar FP preload data as hex in the RTL's scalar FP format.

    Each float is encoded into the minifloat format the hardware scalar FP
    register file / FP SRAM actually use — ``1 + S_FP_EXP_WIDTH +
    S_FP_MANT_WIDTH`` bits — rather than IEEE fp16. ``exp_width``/``man_width``
    default to the values in precision.svh, so fp_sram.mem stays consistent
    with the register width for every layer (rms_norm, layer_norm, linear,
    attention, ...). Non-float entries are written as raw bit patterns.

    Args:
        data: List of float values or tensor to write
        directory: Output directory path
        filename: Output filename
        exp_width: Scalar FP exponent width (defaults to S_FP_EXP_WIDTH)
        man_width: Scalar FP mantissa width (defaults to S_FP_MANT_WIDTH)

    Returns:
        Path to the generated file
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    output_file = directory / filename

    if isinstance(data, Tensor):
        data = data.flatten().tolist()

    default_exp, default_man = _scalar_fp_format()
    if exp_width is None:
        exp_width = default_exp
    if man_width is None:
        man_width = default_man

    total_width = 1 + exp_width + man_width
    hex_digits = (total_width + 3) // 4

    with open(output_file, "w") as f:
        for value in data:
            if isinstance(value, float):
                # Encode into the RTL scalar FP format (matches fp_reg_file).
                int_val = float_to_fp(value, exp_width, man_width)
            else:
                # Already a raw bit pattern.
                int_val = int(value)
            f.write(f"0x{int_val:0{hex_digits}X}\n")

    return output_file


def write_int_sram_hex(
    data: Union[List[int], Tensor],
    directory: Union[str, Path],
    filename: str = "int_sram.mem",
    int_width: int = 32,
) -> Path:
    """Write int32 preload data as hex format.

    Args:
        data: List of integer values or tensor to write
        directory: Output directory path
        filename: Output filename
        int_width: Integer width in bits (default 32)

    Returns:
        Path to the generated file
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    output_file = directory / filename

    if isinstance(data, Tensor):
        data = data.flatten().tolist()

    hex_digits = int_width // 4

    with open(output_file, "w") as f:
        for value in data:
            int_val = int(value)
            # Handle negative values with two's complement
            if int_val < 0:
                int_val = (1 << int_width) + int_val
            f.write(f"0x{int_val:0{hex_digits}X}\n")

    return output_file
