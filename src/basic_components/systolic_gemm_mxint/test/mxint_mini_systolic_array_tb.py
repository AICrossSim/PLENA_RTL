#!/usr/bin/env python3

import logging
import pytest
import cocotb
import torch

from cocotb.triggers import RisingEdge
from cocotb.log import SimLog

from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.testbench import Testbench

from plena_quant.mxint import _mx_int_quantize_hardware
from plena_quant.common.utils import block


def quantize_line(values, element_width, scale_width):
    """Quantize one 1-D line as a single MXINT block (two's-complement elements).

    The RTL multiplies elements with $signed, so the elements are two's complement
    (NOT sign-magnitude). The element concatenation the DUT port expects is done here.

    Returns:
        signed_ints : the signed integer mantissa of each element (golden reference)
        biased_scale: the line's shared scale, biased
        element_bus : all elements concatenated (two's complement) into one bus word
    """
    block_length = values.shape[0]
    _, mantissa, scaling = _mx_int_quantize_hardware(
        values.unsqueeze(0), width=element_width, exponent_width=scale_width,
        block_size=[block_length], skip_first_dim=True,
    )
    blocked, _, _, _ = block(values.unsqueeze(0), block_shape=[block_length], skip_first_dim=True)
    signed_mantissa = torch.sign(blocked + 1e-9) * mantissa          # normalized to (-1, 1)

    mantissa_bits = element_width - 1
    scale_bias = (1 << (scale_width - 1)) - 1
    biased_scale = int((scaling + scale_bias)[0, 0].item())
    signed_ints = (signed_mantissa * (1 << mantissa_bits)).round().int().flatten().tolist()

    element_mask = (1 << element_width) - 1
    element_bus = 0
    for k, value in enumerate(signed_ints):
        element_bus |= (value & element_mask) << (k * element_width)
    return signed_ints, biased_scale, element_bus


class MXINTMiniArrayTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)
        self.param = self.read_dut_param()

    def read_dut_param(self):
        """Read the actual elaborated parameters back from the DUT, then derive the
        rest. The literal values live in module_param_list (at the veri_runner call);
        sourcing them from the built RTL keeps the reference model matching whatever
        was elaborated."""
        param = {
            "MX_T_INT_WIDTH":    int(self.dut.MX_T_INT_WIDTH.value),
            "MX_L_INT_WIDTH":    int(self.dut.MX_L_INT_WIDTH.value),
            "MXINT_SCALE_WIDTH": int(self.dut.MXINT_SCALE_WIDTH.value),
            "BLOCK_DIM":         int(self.dut.BLOCK_DIM.value),
            "ACC_DEPTH":         int(self.dut.ACC_DEPTH.value),
        }
        # derived (these are localparams in RTL — derived here, not read)
        param["SCALE_BIAS"]      = (1 << (param["MXINT_SCALE_WIDTH"] - 1)) - 1
        param["ACC_EXPAND_WIDTH"] = max(1, (param["ACC_DEPTH"] - 1).bit_length())
        param["OUT_INT_WIDTH"]   = param["MX_T_INT_WIDTH"] + param["MX_L_INT_WIDTH"] + param["ACC_EXPAND_WIDTH"]
        param["OUT_SCALE_WIDTH"] = param["MXINT_SCALE_WIDTH"] + 1
        return param

    async def init_signals(self):
        self.dut.load_a_row.value = 0
        self.dut.load_b_col.value = 0
        self.dut.load_a_scale.value = 0
        self.dut.load_b_scale.value = 0
        self.dut.load_valid.value = 0

    def generate_inputs(self, seed):
        """A[BLOCK_DIM][ACC_DEPTH] · B[ACC_DEPTH][BLOCK_DIM] = C[BLOCK_DIM][BLOCK_DIM].
        Each A row is one MXINT block (own scale); each B column is one MXINT block."""
        param = self.param
        output_dim      = param["BLOCK_DIM"]
        reduction_depth = param["ACC_DEPTH"]
        scale_width     = param["MXINT_SCALE_WIDTH"]
        left_width      = param["MX_L_INT_WIDTH"]
        top_width       = param["MX_T_INT_WIDTH"]

        torch.manual_seed(seed)
        a_float = torch.randn(output_dim, reduction_depth)
        b_float = torch.randn(reduction_depth, output_dim)

        # Each A row is one MXINT block (own scale); quantize_line also builds the
        # row's element bus word.
        a_signed = []
        self.a_scale, self.a_row_bus = [], []
        for row in range(output_dim):
            signed_ints, scale, element_bus = quantize_line(a_float[row], left_width, scale_width)
            a_signed.append(signed_ints)
            self.a_scale.append(scale)
            self.a_row_bus.append(element_bus)

        # Each B column is one MXINT block; build per-column signed ints + bus word.
        b_signed = [[0] * output_dim for _ in range(reduction_depth)]
        self.b_scale, self.b_col_bus = [0] * output_dim, [0] * output_dim
        for col in range(output_dim):
            signed_ints, scale, element_bus = quantize_line(b_float[:, col], top_width, scale_width)
            for k in range(reduction_depth):
                b_signed[k][col] = signed_ints[k]
            self.b_scale[col] = scale
            self.b_col_bus[col] = element_bus

        # Expected per-PE result: lossless integer accumulate + biased scale sum.
        scale_mask = (1 << param["OUT_SCALE_WIDTH"]) - 1
        self.expected_int   = [[0] * output_dim for _ in range(output_dim)]
        self.expected_scale = [[0] * output_dim for _ in range(output_dim)]
        for row in range(output_dim):
            for col in range(output_dim):
                self.expected_int[row][col] = sum(
                    a_signed[row][k] * b_signed[k][col] for k in range(reduction_depth))
                self.expected_scale[row][col] = (self.a_scale[row] + self.b_scale[col]
                                                 - param["SCALE_BIAS"]) & scale_mask

    async def in_driver(self):
        """Drive BLOCK_DIM load cycles: cycle t presents A row t and B column t.
        The element concatenation is already done by quantize_line, so each cycle
        just assigns the pre-built bus words and scales."""
        output_dim = self.param["BLOCK_DIM"]
        for cycle in range(output_dim):
            await RisingEdge(self.dut.clk)
            self.dut.load_a_row.value   = self.a_row_bus[cycle]
            self.dut.load_b_col.value   = self.b_col_bus[cycle]
            self.dut.load_a_scale.value = self.a_scale[cycle]
            self.dut.load_b_scale.value = self.b_scale[cycle]
            self.dut.load_valid.value   = 1

        await RisingEdge(self.dut.clk)
        self.dut.load_valid.value = 0

    async def out_monitor(self, timeout_cycles=200):
        """Valid-only monitor: wait for out_valid, then sample the packed per-PE
        outputs and compare each against the reference. Returns the mismatch count."""
        param = self.param
        output_dim      = param["BLOCK_DIM"]
        out_int_width   = param["OUT_INT_WIDTH"]
        out_scale_width = param["OUT_SCALE_WIDTH"]

        for _ in range(timeout_cycles):
            await RisingEdge(self.dut.clk)
            if int(self.dut.out_valid.value) == 1:
                break
        else:
            raise AssertionError(f"out_valid never asserted within {timeout_cycles} cycles")

        packed_int   = int(self.dut.out_int.value)
        packed_scale = int(self.dut.out_scale.value)
        mismatches = 0
        for row in range(output_dim):
            for col in range(output_dim):
                position = row * output_dim + col
                got_int = (packed_int >> (position * out_int_width)) & ((1 << out_int_width) - 1)
                if got_int & (1 << (out_int_width - 1)):
                    got_int -= (1 << out_int_width)
                got_scale = (packed_scale >> (position * out_scale_width)) & ((1 << out_scale_width) - 1)
                ok = (got_int == self.expected_int[row][col]
                      and got_scale == self.expected_scale[row][col])
                tag = "OK " if ok else "BAD"
                self.log.info(
                    f"  [{tag}] PE({row},{col}): out_int got={got_int:>6} "
                    f"exp={self.expected_int[row][col]:>6}, "
                    f"out_scale got={got_scale:>4} exp={self.expected_scale[row][col]:>4}")
                if not ok:
                    mismatches += 1
        return mismatches

    async def run_test(self, seed):
        param = self.param
        await self.reset()
        await self.init_signals()
        self.log.info(f"Reset finished, seed={seed}")

        self.generate_inputs(seed)
        await self.in_driver()
        mismatches = await self.out_monitor()
        assert mismatches == 0, f"{mismatches} PE mismatches"
        self.log.info(f"All {param['BLOCK_DIM'] * param['BLOCK_DIM']} PEs match")


@cocotb.test()
async def test(dut):
    tb = MXINTMiniArrayTB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test(seed=0)


@pytest.mark.dev
def test_mxint_mini_systolic_array():
    veri_runner(
        group="systolic_gemm_mxint",
        module="mxint_mini_systolic_array",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fixed_operation"),
        ],
        module_param_list=[{
            "MX_T_INT_WIDTH":    4,
            "MX_L_INT_WIDTH":    4,
            "MXINT_SCALE_WIDTH": 8,
            "BLOCK_DIM":         4,    # mini-array edge (= BLEN, the M/N output tile dim)
            "ACC_DEPTH":         16,   # K-segment depth per mini-array (= MXINT scale block size)
        }],
        trace=True,
    )


if __name__ == "__main__":
    test_mxint_mini_systolic_array()
