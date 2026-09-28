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
from cfl_cocotb.valid_only import ValidOnlyDriver

from plena_quant.mxint import _mx_int_quantize_hardware
from plena_quant.common.utils import block


def quantize_mxint_for_hw(values, element_width, scale_width, block_length):
    """Quantize a 1-D float tensor to MXINT hardware format (one block).

    The RTL multiplies elements with $signed, so elements are two's complement
    (NOT sign-magnitude).

    Returns:
        signed_ints     : signed integer mantissa of each element (golden reference)
        biased_scale    : the block's shared scale, biased
        quantized_float : the quantized float tensor (for reference/logging)
    """
    scale_bias = 2 ** (scale_width - 1) - 1
    mantissa_bits = element_width - 1

    quantized_float, per_block_mantissa, scaling = _mx_int_quantize_hardware(
        values.unsqueeze(0),
        width=element_width,
        exponent_width=scale_width,
        block_size=[block_length],
        skip_first_dim=True,
    )

    blocked, _, _, _ = block(values.unsqueeze(0), block_shape=[block_length], skip_first_dim=True)
    sign = torch.sign(blocked + 1e-9)
    signed_mantissa = sign * per_block_mantissa          # normalized to (-1, 1)

    signed_ints = (signed_mantissa * (1 << mantissa_bits)).round().int()
    biased_scale = (scaling + scale_bias).int()
    return signed_ints, biased_scale, quantized_float


class MXINTDefaultPETB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.param = self.read_dut_param()

        # Valid-only drivers — append (element, scale) tuples, driver streams them.
        self.top_driver = ValidOnlyDriver(
            dut.clk,
            (dut.in_top_element, dut.in_top_scale),
            dut.in_top_valid,
        )
        self.left_driver = ValidOnlyDriver(
            dut.clk,
            (dut.in_left_element, dut.in_left_scale),
            dut.in_left_valid,
        )

        # No StreamMonitor — out_valid latches high once the dot product completes,
        # so it would fire on every cycle thereafter. We sample manually in out_monitor.

    def read_dut_param(self):
        """Read the actual elaborated parameters back from the DUT, then derive the
        rest. The literal values live in module_param_list (at the veri_runner call);
        sourcing them from the built RTL keeps the reference model matching whatever
        was elaborated."""
        param = {
            "MX_L_INT_WIDTH":    int(self.dut.MX_L_INT_WIDTH.value),
            "MX_T_INT_WIDTH":    int(self.dut.MX_T_INT_WIDTH.value),
            "MXINT_SCALE_WIDTH": int(self.dut.MXINT_SCALE_WIDTH.value),
            "ACC_EXPAND_WIDTH":  int(self.dut.ACC_EXPAND_WIDTH.value),
            "ACC_DEPTH":         int(self.dut.ACC_DEPTH.value),
            "OUT_SCALE_WIDTH":   int(self.dut.OUT_SCALE_WIDTH.value),
        }
        param["SCALE_BIAS"] = (1 << (param["MXINT_SCALE_WIDTH"] - 1)) - 1
        return param

    async def init_signals(self):
        """Initialize all DUT inputs to known values."""
        self.dut.in_top_valid.value = 0
        self.dut.in_left_valid.value = 0

    def generate_inputs(self, dot_length, seed):
        """Quantize → pack inputs → software-simulate → pack expected outputs.

        The software simulation runs on the quantized values (quantized_top,
        quantized_left), and the expected hardware outputs are derived from the
        same packed integer representation the DUT sees.
        """
        param = self.param
        scale_bias = param["SCALE_BIAS"]
        left_width = param["MX_L_INT_WIDTH"]
        top_width  = param["MX_T_INT_WIDTH"]

        torch.manual_seed(seed)
        torch_top = torch.randn(dot_length)
        torch_left = torch.randn(dot_length)

        # === Quantize + pack inputs as one MXINT block of size dot_length ===
        # All elements of the top vector share ONE scale; all elements of the left
        # vector share another. This matches real MXINT usage: one shared exponent
        # per block.
        top_elements, top_scale_packed, quantized_top = quantize_mxint_for_hw(
            torch_top,
            element_width=top_width,
            scale_width=param["MXINT_SCALE_WIDTH"],
            block_length=dot_length,
        )
        left_elements, left_scale_packed, quantized_left = quantize_mxint_for_hw(
            torch_left,
            element_width=left_width,
            scale_width=param["MXINT_SCALE_WIDTH"],
            block_length=dot_length,
        )
        # Shapes: *_elements      → [1, dot_length]  (per-element packed mantissas)
        #         *_scale_packed  → [1, 1]           (one shared scale across the block)

        # Software simulation on quantized float values (per-element product) — for
        # reference/logging only.
        quantized_product = (quantized_top * quantized_left).squeeze(0)

        # === Expected hardware outputs (lossless accumulator) ===
        # out_int is the full-width signed accumulator: MX_T + MX_L + ACC_EXPAND bits.
        # No rounding or saturation — the value is just (sum of signed products).
        scale_mask = (1 << param["OUT_SCALE_WIDTH"]) - 1

        # signed_ints tensors are [dot_length, 1]; flatten to plain signed Python ints.
        top_values  = [int(v) for v in top_elements.flatten().tolist()]
        left_values = [int(v) for v in left_elements.flatten().tolist()]

        # One shared scale per block — extract scalars once.
        top_scale  = int(top_scale_packed[0, 0].item())
        left_scale = int(left_scale_packed[0, 0].item())

        # Software simulation of the PE: signed mantissa multiply, accumulate.
        # Keep the accumulator as a Python signed int — the dut value is read back as
        # a signed Python int via .signed_integer.
        accumulator = sum(top_values[i] * left_values[i] for i in range(dot_length))

        # === Store for driving and checking ===
        # Drive each element as a two's-complement bit pattern (masked to its width);
        # the shared scale is broadcast to every MAC cycle.
        top_mask  = (1 << top_width) - 1
        left_mask = (1 << left_width) - 1
        self.inputs_top  = [(top_values[i] & top_mask, top_scale) for i in range(dot_length)]
        self.inputs_left = [(left_values[i] & left_mask, left_scale) for i in range(dot_length)]
        self.expected_int   = accumulator                                      # signed Python int
        self.expected_scale = (top_scale + left_scale - scale_bias) & scale_mask

        # Keep float references for logging / cross-checking against quantized sim.
        self.quantized_top     = quantized_top.squeeze(0)
        self.quantized_left    = quantized_left.squeeze(0)
        self.quantized_product = quantized_product

    async def in_driver(self):
        """Queue the full dot product into both valid-only drivers; they stream the
        (element, scale) pairs one per cycle. The PE self-triggers on
        (in_top_valid && in_left_valid) — no separate mult pulse needed."""
        self.top_driver.load_driver(self.inputs_top)
        self.left_driver.load_driver(self.inputs_left)

    async def out_monitor(self, timeout_cycles):
        """Wait for out_valid to latch high, then sample the latched accumulator and
        scale and compare against the reference. Returns the mismatch count."""
        for _ in range(timeout_cycles):
            await RisingEdge(self.dut.clk)
            if int(self.dut.out_valid.value) == 1:
                break
        else:
            raise AssertionError(f"out_valid never asserted within {timeout_cycles} cycles")

        got_int   = self.dut.out_int.value.signed_integer
        got_scale = int(self.dut.out_scale.value)
        ok_int   = (got_int == self.expected_int)
        ok_scale = (got_scale == self.expected_scale)
        self.log.info(
            f"FINAL: out_int got={got_int} exp={self.expected_int}, "
            f"out_scale got={got_scale} exp={self.expected_scale}")
        return int(not ok_int) + int(not ok_scale)

    async def run_one_batch(self, seed):
        """Reset DUT, drive one full dot product, sample latched output, check.

        PE semantics: out_valid latches high after ACC_DEPTH MACs and HOLDS until
        rst, so we reset between batches to start fresh.
        """
        dot_length = self.param["ACC_DEPTH"]
        await self.reset()
        await self.init_signals()

        self.generate_inputs(dot_length, seed=seed)
        await self.in_driver()
        self.log.info(
            f"seed={seed}: {dot_length} MACs, expecting "
            f"(out_int={self.expected_int}, out_scale={self.expected_scale})")

        mismatches = await self.out_monitor(timeout_cycles=dot_length + 10)
        assert mismatches == 0, f"seed={seed}: {mismatches} output mismatches"

    async def run_test(self, num_batches=1):
        """Run `num_batches` independent dot products (reset between)."""
        for seed in range(num_batches):
            await self.run_one_batch(seed=seed)
        self.log.info(f"All {num_batches} batches passed")


@cocotb.test()
async def test_independent_batches(dut):
    """Reset between batches. Checks out_int and out_scale per batch."""
    tb = MXINTDefaultPETB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test(num_batches=2)


@pytest.mark.dev
def test_mxint_default_pe():
    veri_runner(
        group="systolic_gemm_mxint",
        module="mxint_default_pe",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/conversion"),
        ],
        module_param_list=[
            {
                "MX_L_INT_WIDTH":    4,
                "MX_T_INT_WIDTH":    4,
                "MXINT_SCALE_WIDTH": 8,
                "ACC_DEPTH":         1,
            },
            {
                "MX_L_INT_WIDTH":    4,
                "MX_T_INT_WIDTH":    4,
                "MXINT_SCALE_WIDTH": 8,
                "ACC_DEPTH":         8,
            },
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_mxint_default_pe()
