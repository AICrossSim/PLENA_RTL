#!/usr/bin/env python3

import logging
import pytest
import cocotb
import torch

from cocotb.triggers import RisingEdge, ReadOnly
from cocotb.log import SimLog

from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.testbench import Testbench

from plena_quant.common import _minifloat_ieee_quantize_hardware
from cfl_cocotb.torch_fp_conversion import pack_fp_to_bin


def decode_fp(bits, exp_width, mant_width):
    """Decode a packed minifloat {sign, exp, mant} integer to a Python float."""
    sign = (bits >> (exp_width + mant_width)) & 1
    exponent = (bits >> mant_width) & ((1 << exp_width) - 1)
    mantissa = bits & ((1 << mant_width) - 1)
    bias = (1 << (exp_width - 1)) - 1
    if exponent == 0:
        value = (mantissa / 2**mant_width) * 2.0 ** (1 - bias)
    else:
        value = (1.0 + mantissa / 2**mant_width) * 2.0 ** (exponent - bias)
    return -value if sign else value


class FPFixAccumulatorTB(Testbench):
    """Direct testbench for fp_fix_accumulator (NO handshake / ready signals).

    Interface : clk, rst, clear_accumulator, data_in_valid, data_in
                -> data_out, data_out_valid
    Behaviour : data_in (a minifloat) is converted to signed fixed-point
                (ACC_INT_WIDTH.ACC_FRAC_WIDTH) and added into an EXACT fixed-point
                accumulator (no per-step requantisation). The running sum is
                converted back to minifloat and registered, so data_out lags the
                accumulator by 1 cycle (2-cycle pipeline). clear_accumulator/rst
                zero the accumulator.
    """

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)
        self.param = self.read_dut_param()

    def read_dut_param(self):
        """Read the elaborated parameters back from the DUT, then derive the rest.
        The literal values live in module_param_list (at the veri_runner call)."""
        param = {
            "EXP_WIDTH":      int(self.dut.EXP_WIDTH.value),
            "MANT_WIDTH":     int(self.dut.MANT_WIDTH.value),
            "ACC_INT_WIDTH":  int(self.dut.ACC_INT_WIDTH.value),
            "ACC_FRAC_WIDTH": int(self.dut.ACC_FRAC_WIDTH.value),
        }
        param["FP_W"] = param["EXP_WIDTH"] + param["MANT_WIDTH"] + 1
        return param

    async def init_signals(self):
        self.dut.clear_accumulator.value = 0
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0

    def generate_inputs(self, num, seed):
        """Quantize `num` random minifloats and pre-compute the exact prefix sums.

        The hardware keeps full fixed-point precision, so the running sum is exact
        for our inputs; the only error is the final fixed->FP conversion, which
        *truncates* the mantissa. Golden = exact prefix sum, checked to ~1 ULP.
        """
        param = self.param
        exp_width, mant_width = param["EXP_WIDTH"], param["MANT_WIDTH"]
        width = param["FP_W"]

        torch.manual_seed(seed)
        torch_a = torch.randn(num) * 2.0   # modest magnitudes -> 16.16 acc stays exact

        qa, a_exp, a_mant = _minifloat_ieee_quantize_hardware(torch_a, width, exp_width)
        packed = pack_fp_to_bin(a_exp, a_mant, exp_width, mant_width)

        self.input_bus = [int(packed[i]) for i in range(num)]
        self.qvals = [float(qa[i]) for i in range(num)]   # exact represented values

        self.expected_real = []
        acc = 0.0
        for v in self.qvals:
            acc += v
            self.expected_real.append(acc)

    async def in_driver(self, num):
        """Stream one input per cycle with data_in_valid high."""
        for t in range(num):
            await RisingEdge(self.dut.clk)
            self.dut.data_in.value = self.input_bus[t]
            self.dut.data_in_valid.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0

    async def out_monitor(self, num, collected):
        """Record the first `num` cycles where data_out_valid is high."""
        while len(collected) < num:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if int(self.dut.data_out_valid.value) == 1:
                collected.append(int(self.dut.data_out.value))

    def check_outputs(self, collected, label):
        """Compare each drained prefix sum against the exact golden value."""
        param = self.param
        exp_width, mant_width = param["EXP_WIDTH"], param["MANT_WIDTH"]
        # truncation of the FP mantissa loses up to ~1 ULP -> allow ~2 ULP rel margin
        rel_tol = 2.0 ** -(mant_width - 1)
        abs_tol = 4.0 * (2.0 ** -param["ACC_FRAC_WIDTH"])

        mismatches = 0
        for i, bits in enumerate(collected):
            got = decode_fp(bits, exp_width, mant_width)
            want = self.expected_real[i]
            tol = abs_tol + rel_tol * abs(want)
            ok = abs(got - want) <= tol
            tag = "OK " if ok else "BAD"
            self.log.info(f"  [{tag}] {label} [{i:2d}] in={self.qvals[i]:+.5f}  "
                          f"acc(hw)={got:+.5f}  acc(ref)={want:+.5f}  err={got - want:+.5f}")
            if not ok:
                mismatches += 1
        return mismatches

    async def run_accumulate(self, num=16, seed=0):
        """Stream `num` minifloats back-to-back, check every running prefix sum."""
        label = f"accumulate(num={num})"
        self.generate_inputs(num, seed)
        collected = []
        monitor = cocotb.start_soon(self.out_monitor(num, collected))
        await self.in_driver(num)
        for _ in range(8):                      # drain the pipeline
            await RisingEdge(self.dut.clk)
            if len(collected) >= num:
                break
        await monitor

        assert len(collected) >= num, \
            f"only captured {len(collected)} valid outputs, expected {num}"
        mismatches = self.check_outputs(collected, label)
        self.log.info(f"{label}: {mismatches} mismatch(es)")
        return mismatches

    async def run_clear(self):
        """clear_accumulator must zero the running sum."""
        param = self.param
        exp_width, mant_width = param["EXP_WIDTH"], param["MANT_WIDTH"]

        val = torch.tensor([3.5])
        _, e, m = _minifloat_ieee_quantize_hardware(val, param["FP_W"], exp_width)
        bits = int(pack_fp_to_bin(e, m, exp_width, mant_width)[0])

        await RisingEdge(self.dut.clk)
        self.dut.data_in.value = bits
        self.dut.data_in_valid.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0

        self.dut.clear_accumulator.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.clear_accumulator.value = 0

        for _ in range(3):
            await RisingEdge(self.dut.clk)
        await ReadOnly()
        out = decode_fp(int(self.dut.data_out.value), exp_width, mant_width)
        self.log.info(f"  clear: data_out decodes to {out:+.5f}")
        return 0 if abs(out) < 1e-6 else 1

    async def run_test(self):
        param = self.param
        await self.reset()
        await self.init_signals()
        self.log.info(f"Reset finished, EXP_WIDTH={param['EXP_WIDTH']} "
                      f"MANT_WIDTH={param['MANT_WIDTH']} "
                      f"ACC={param['ACC_INT_WIDTH']}.{param['ACC_FRAC_WIDTH']}")

        total_mismatch = 0
        total_mismatch += await self.run_accumulate(num=16, seed=0)
        await self.init_signals()
        total_mismatch += await self.run_clear()

        assert total_mismatch == 0, f"{total_mismatch} mismatch(es)"
        self.log.info("All checks passed")


@cocotb.test()
async def test(dut):
    tb = FPFixAccumulatorTB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test()


@pytest.mark.dev
def test_fp_fix_accumulator():
    veri_runner(
        group="fp_operation",
        module="fp_fix_accumulator",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/synopsis"),
        ],
        module_param_list=[
            {"EXP_WIDTH": 6, "MANT_WIDTH": 5},
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_fp_fix_accumulator()
