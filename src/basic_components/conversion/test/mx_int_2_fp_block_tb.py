#!/usr/bin/env python3
"""
Testbench for mx_int_2_fp_block module.

Test flow:
1. Generate random FP values
2. Convert FP to MXINT using plena_quant.mxint
3. Pass MXINT (element + scale) to RTL
4. Get FP output from RTL
5. Compare with reference using verification.verify_mxint_encoding.mxint_to_float
"""

import sys
from pathlib import Path

# Add PLENA_Tools to path
PLENA_TOOLS_PATH = Path(__file__).resolve().parents[4] / "PLENA_Tools"
sys.path.insert(0, str(PLENA_TOOLS_PATH))

import logging
import torch
import cocotb
from cocotb.triggers import Timer, RisingEdge, ClockCycles
from cocotb.clock import Clock

from cfl_cocotb.runner import veri_runner, SRC_PATH
from cfl_cocotb.torch_fp_conversion import bin_2_fp
from plena_quant.mxint import Random_MXINT_Tensor_Generator
from verification.verify_mxint_encoding import mxint_to_float

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

torch.manual_seed(42)


@cocotb.test()
async def test_mx_int_2_fp_block(dut):
    """Test MXINT to FP conversion block."""

    # Get parameters from DUT
    BLOCK_DIM = dut.BLOCK_DIM.value
    MXINT_WIDTH = dut.MXINT_WIDTH.value
    MXINT_SCALE_WIDTH = dut.MXINT_SCALE_WIDTH.value
    FP_EXP_WIDTH = dut.FP_EXP_WIDTH.value
    FP_MANT_WIDTH = dut.FP_MANT_WIDTH.value
    FP_WIDTH = FP_EXP_WIDTH + FP_MANT_WIDTH + 1

    cocotb.log.info(f"Testing mx_int_2_fp_block with:")
    cocotb.log.info(f"  BLOCK_DIM={BLOCK_DIM}")
    cocotb.log.info(f"  MXINT_WIDTH={MXINT_WIDTH}")
    cocotb.log.info(f"  MXINT_SCALE_WIDTH={MXINT_SCALE_WIDTH}")
    cocotb.log.info(f"  FP_EXP_WIDTH={FP_EXP_WIDTH}, FP_MANT_WIDTH={FP_MANT_WIDTH}")

    # Setup MXINT generator config
    quant_config = {
        "man_width": MXINT_WIDTH,
        "exp_width": MXINT_SCALE_WIDTH,
        "block_size": BLOCK_DIM,
        "skip_first_dim": True,
    }

    generator = Random_MXINT_Tensor_Generator(
        shape=(1, BLOCK_DIM),
        quant_config=quant_config
    )

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.data_in_valid.value = 0
    dut.element_in.value = 0
    dut.scale_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

    # Test cases
    NUM_TESTS = 50
    test_ranges = [0.1, 1.0, 10.0, 100.0, 0.01]

    total_tests = 0
    passed_tests = 0
    max_rel_error = 0.0

    for test_range in test_ranges:
        for test_idx in range(NUM_TESTS // len(test_ranges)):
            # Generate random FP values
            fp_values = torch.randn(1, BLOCK_DIM) * test_range

            # Convert to MXINT using the library
            block_list, scaling_list = generator.quantize_tensor(fp_values)

            # Get the single block's elements and scale
            mxint_elements = block_list[0]  # List of packed elements
            biased_scale = scaling_list[0]  # Biased scale value

            # Pack elements for DUT input
            # element_in is [BLOCK_DIM-1:0][MXINT_WIDTH-1:0]
            packed_elements = 0
            for i, elem in enumerate(mxint_elements):
                packed_elements |= (elem & ((1 << MXINT_WIDTH) - 1)) << (i * MXINT_WIDTH)

            # Apply inputs
            dut.element_in.value = packed_elements
            dut.scale_in.value = biased_scale
            dut.data_in_valid.value = 1

            await RisingEdge(dut.clk)
            dut.data_in_valid.value = 0

            # Wait for output (1 cycle latency)
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")

            # Check output valid
            if dut.data_out_valid.value != 1:
                cocotb.log.warning(f"Test {total_tests}: data_out_valid not asserted")
                await ClockCycles(dut.clk, 1)

            # Read FP outputs
            fp_out_packed = dut.fp_out.value.integer

            # Unpack and convert FP outputs
            rtl_fp_values = []
            for i in range(BLOCK_DIM):
                fp_bits = (fp_out_packed >> (i * FP_WIDTH)) & ((1 << FP_WIDTH) - 1)
                fp_val = bin_2_fp(fp_bits, FP_EXP_WIDTH, FP_MANT_WIDTH)
                rtl_fp_values.append(fp_val)

            # Calculate expected values using mxint_to_float from verification lib
            expected_values = []
            for i in range(BLOCK_DIM):
                exp_val = mxint_to_float(
                    mxint_elements[i], biased_scale, MXINT_WIDTH, MXINT_SCALE_WIDTH
                )
                expected_values.append(exp_val)

            # Compare results
            test_passed = True
            for i in range(BLOCK_DIM):
                rtl_val = rtl_fp_values[i]
                exp_val = expected_values[i]

                # Calculate relative error
                if abs(exp_val) > 1e-10:
                    rel_error = abs(rtl_val - exp_val) / abs(exp_val)
                else:
                    rel_error = abs(rtl_val - exp_val)

                max_rel_error = max(max_rel_error, rel_error)

                # Allow some tolerance due to FP quantization
                tolerance = 0.1  # 10% relative error tolerance
                if rel_error > tolerance and abs(rtl_val - exp_val) > 1e-6:
                    test_passed = False
                    cocotb.log.error(
                        f"Test {total_tests}, Element {i}: "
                        f"RTL={rtl_val:.6f}, Expected={exp_val:.6f}, "
                        f"RelErr={rel_error:.4f}, "
                        f"MXINT elem=0x{mxint_elements[i]:02X}, scale=0x{biased_scale:02X}"
                    )

            if test_passed:
                passed_tests += 1
            else:
                # Debug output for failed test
                cocotb.log.debug(f"Input FP values: {fp_values.tolist()}")
                cocotb.log.debug(f"MXINT elements: {[hex(e) for e in mxint_elements]}")
                cocotb.log.debug(f"MXINT scale: 0x{biased_scale:02X} (unbiased: {biased_scale - 127})")
                cocotb.log.debug(f"RTL outputs: {rtl_fp_values}")
                cocotb.log.debug(f"Expected: {expected_values}")

            total_tests += 1

    # Summary
    cocotb.log.info(f"=" * 60)
    cocotb.log.info(f"Test Summary: {passed_tests}/{total_tests} passed")
    cocotb.log.info(f"Max relative error: {max_rel_error:.6f}")
    cocotb.log.info(f"=" * 60)

    assert passed_tests == total_tests, f"Failed {total_tests - passed_tests} tests"


@cocotb.test()
async def test_specific_values(dut):
    """Test with specific known values for debugging."""

    BLOCK_DIM = dut.BLOCK_DIM.value
    MXINT_WIDTH = dut.MXINT_WIDTH.value
    MXINT_SCALE_WIDTH = dut.MXINT_SCALE_WIDTH.value
    FP_EXP_WIDTH = dut.FP_EXP_WIDTH.value
    FP_MANT_WIDTH = dut.FP_MANT_WIDTH.value
    FP_WIDTH = FP_EXP_WIDTH + FP_MANT_WIDTH + 1

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.data_in_valid.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

    # Test specific values
    # Scale = 127 (bias 127) -> actual exponent = 0 -> multiplier = 1
    # Element = 0x40 -> sign=0, magnitude=64 -> value = 64/128 = 0.5
    test_cases = [
        # (elements for block, scale, description)
        ([0x40] * BLOCK_DIM, 127, "0.5 with exp=0"),  # 64/128 * 2^0 = 0.5
        ([0x00] * BLOCK_DIM, 127, "0.0"),  # 0/128 * 2^0 = 0
        ([0x7F] * BLOCK_DIM, 127, "~1.0 with exp=0"),  # 127/128 * 2^0 ≈ 0.992
        ([0xC0] * BLOCK_DIM, 127, "-0.5 with exp=0"),  # -64/128 * 2^0 = -0.5
        ([0x40] * BLOCK_DIM, 128, "1.0 with exp=1"),  # 64/128 * 2^1 = 1.0
        ([0x40] * BLOCK_DIM, 126, "0.25 with exp=-1"),  # 64/128 * 2^-1 = 0.25
    ]

    for elements, scale, desc in test_cases:
        cocotb.log.info(f"Testing: {desc}")

        # Pack elements
        packed_elements = 0
        for i, elem in enumerate(elements):
            packed_elements |= (elem & ((1 << MXINT_WIDTH) - 1)) << (i * MXINT_WIDTH)

        dut.element_in.value = packed_elements
        dut.scale_in.value = scale
        dut.data_in_valid.value = 1

        await RisingEdge(dut.clk)
        dut.data_in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")

        # Read outputs
        fp_out_packed = dut.fp_out.value.integer

        for i in range(BLOCK_DIM):
            fp_bits = (fp_out_packed >> (i * FP_WIDTH)) & ((1 << FP_WIDTH) - 1)
            rtl_val = bin_2_fp(fp_bits, FP_EXP_WIDTH, FP_MANT_WIDTH)
            exp_val = mxint_to_float(elements[i], scale, MXINT_WIDTH, MXINT_SCALE_WIDTH)

            cocotb.log.info(
                f"  Element {i}: MXINT=0x{elements[i]:02X}, scale=0x{scale:02X} -> "
                f"RTL={rtl_val:.6f}, Expected={exp_val:.6f}"
            )


if __name__ == "__main__":
    veri_runner(
        trace=True,
        module="mx_int_2_fp_block",
        group="conversion",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
        ],
        module_param_list=[
            {
                # Use settings from precision.svh
                "BLOCK_DIM": 8,
                "MXINT_WIDTH": 8,
                "MXINT_SCALE_WIDTH": 8,
                "FP_EXP_WIDTH": 6,  # V_FP_EXP_WIDTH
                "FP_MANT_WIDTH": 5,  # V_FP_MANT_WIDTH
            }
        ]
    )
