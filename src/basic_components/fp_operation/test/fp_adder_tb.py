#!/usr/bin/env python3

import logging
import pytest
import cocotb
import sys
import os

from cocotb.triggers import Timer
from cocotb.log import SimLog

from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

from plena_quant.common import _minifloat_ieee_quantize_hardware
from plena_quant.quant_operations import fp_add_hardware

import torch


def _sext(v, bits):
    """Two's-complement encode a signed int into `bits` bits for cocotb."""
    return int(v) & ((1 << bits) - 1)


@cocotb.test()
async def test(dut):
    log = SimLog("FPAddTB")
    log.setLevel(logging.DEBUG)

    config = {
        "IN_EXP_WIDTH" : dut.IN_EXP_WIDTH.value,
        "IN_FIX_WIDTH" : dut.IN_FIX_WIDTH.value,
        "IN_FIX_FRAC_WIDTH" : dut.IN_FIX_FRAC_WIDTH.value,
        "OUT_EXP_WIDTH" : dut.OUT_EXP_WIDTH.value,
        "OUT_FIX_WIDTH" : dut.OUT_FIX_WIDTH.value,
        "OUT_FIX_FRAC_WIDTH" : dut.OUT_FIX_FRAC_WIDTH.value,
        "FLOOR" : True,
    }

    in_exp_width = config["IN_EXP_WIDTH"]
    out_exp_width = config["OUT_EXP_WIDTH"]
    num = 10

    torch.manual_seed(0)
    torch_a = torch.randn(num)
    torch_b = torch.randn(num)

    width = config["IN_FIX_FRAC_WIDTH"] + config["IN_EXP_WIDTH"] + 1
    exponent_width = config["IN_EXP_WIDTH"]

    qa, a_exp, a_mant = _minifloat_ieee_quantize_hardware(torch_a, width, exponent_width)
    qb, b_exp, b_mant = _minifloat_ieee_quantize_hardware(torch_b, width, exponent_width)

    exp_sum, mant_sum = fp_add_hardware(a_exp, a_mant, b_exp, b_mant, config)

    passed = 0
    failed = 0
    for i in range(num):
        # Drive inputs — fp_adder is purely combinational
        # exp_a/exp_b are `signed [IN_EXP_WIDTH:0]`, i.e. IN_EXP_WIDTH+1 bits wide.
        # Masking to IN_EXP_WIDTH bits turned e.g. -1 into +63 and made the RTL
        # pick the wrong alignment branch.
        dut.exp_a.value = _sext(a_exp[i], in_exp_width + 1)
        dut.mant_a.value = int(a_mant[i] * 2**config["IN_FIX_FRAC_WIDTH"])
        dut.exp_b.value = _sext(b_exp[i], in_exp_width + 1)
        dut.mant_b.value = int(b_mant[i] * 2**config["IN_FIX_FRAC_WIDTH"])

        # Wait for combinational propagation
        await Timer(10, units="ns")

        # Read outputs
        got_exp = dut.exp_out.value.signed_integer
        got_mant = dut.mant_out.value.signed_integer

        exp_golden = _sext(exp_sum[i], out_exp_width)
        mant_golden = int(mant_sum[i] * 2**config["OUT_FIX_FRAC_WIDTH"])

        # Compare (sign-extend both to same width for comparison)
        got_exp_s = dut.exp_out.value.signed_integer
        exp_golden_s = int(exp_sum[i])

        if got_exp_s == exp_golden_s and got_mant == mant_golden:
            log.debug(f"PASS [{i}]: exp={got_exp_s}, mant={got_mant}")
            passed += 1
        else:
            log.error(f"FAIL [{i}]: got exp={got_exp_s} mant={got_mant}, "
                       f"expected exp={exp_golden_s} mant={mant_golden}")
            failed += 1

    log.info(f"Results: {passed} passed, {failed} failed out of {num}")
    assert failed == 0, f"{failed} tests failed!"


@pytest.mark.dev
def test_simple_fp_addition():
    # Run tests with different params
    veri_runner(
        group = "fp_operation",
        module = "fp_adder",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer")
        ],
        module_param_list=[
            # basic functionality of FP 8 addition
            {
                "IN_EXP_WIDTH" : 6,
                "IN_FIX_WIDTH" : 7,
                "IN_FIX_FRAC_WIDTH" : 5,

                "OUT_EXP_WIDTH" : 6,
                "OUT_FIX_WIDTH" : 8,
                "OUT_FIX_FRAC_WIDTH" : 5,
            },
        ],
        trace = False,
    )


if __name__ == "__main__":
    test_simple_fp_addition()
