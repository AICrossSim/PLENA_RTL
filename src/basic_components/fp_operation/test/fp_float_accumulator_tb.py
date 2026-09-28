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


class FPFloatAccumulatorTB(Testbench):
    """Direct testbench for fp_float_accumulator (NO handshake / ready signals).

    Interface : clk, rst, clear_accumulator, data_in_valid, data_in
                -> data_out, data_out_valid
    Behaviour : each input is FP-aligned to the running accumulator (larger
                exponent wins), mantissas added, result renormalized. Unlike the
                fixed-point accumulator this ROUNDS every step, so error grows
                with the reduction length; the check tolerates that drift and the
                log reports the per-step error so the two designs can be compared.
    """

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)
        self.param = self.read_dut_param()

    def read_dut_param(self):
        param = {
            "EXP_WIDTH":  int(self.dut.EXP_WIDTH.value),
            "MANT_WIDTH": int(self.dut.MANT_WIDTH.value),
        }
        param["FP_W"] = param["EXP_WIDTH"] + param["MANT_WIDTH"] + 1
        return param

    async def init_signals(self):
        self.dut.clear_accumulator.value = 0
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0

    def generate_inputs(self, num, seed):
        """Quantize `num` random minifloats; golden = EXACT prefix sums (ideal).
        The FP accumulator drifts from this; the tolerance below absorbs it."""
        param = self.param
        exp_width, mant_width = param["EXP_WIDTH"], param["MANT_WIDTH"]
        width = param["FP_W"]

        torch.manual_seed(seed)
        torch_a = torch.randn(num) * 2.0

        qa, a_exp, a_mant = _minifloat_ieee_quantize_hardware(torch_a, width, exp_width)
        packed = pack_fp_to_bin(a_exp, a_mant, exp_width, mant_width)

        self.input_bus = [int(packed[i]) for i in range(num)]
        self.qvals = [float(qa[i]) for i in range(num)]

        self.expected_real = []
        acc = 0.0
        for v in self.qvals:
            acc += v
            self.expected_real.append(acc)

    async def in_driver(self, num):
        for t in range(num):
            await RisingEdge(self.dut.clk)
            self.dut.data_in.value = self.input_bus[t]
            self.dut.data_in_valid.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0

    async def out_monitor(self, num, collected):
        while len(collected) < num:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if int(self.dut.data_out_valid.value) == 1:
                collected.append(int(self.dut.data_out.value))

    def check_outputs(self, collected, label):
        param = self.param
        exp_width, mant_width = param["EXP_WIDTH"], param["MANT_WIDTH"]
        # final output truncation ~1 ULP, plus accumulated per-step rounding
        # (no guard bits: each alignment shift truncates at the mantissa LSB).
        rel_tol = 2.0 ** -(mant_width - 1)
        step_eps = 2.0 ** -mant_width

        mismatches = 0
        max_abs_err = 0.0
        for i, bits in enumerate(collected):
            got = decode_fp(bits, exp_width, mant_width)
            want = self.expected_real[i]
            # drift bound: (i+1) steps, each rounding at ~step_eps of the running mag
            drift = (i + 1) * step_eps * max(abs(want), 1.0)
            tol = rel_tol * abs(want) + drift + step_eps
            err = abs(got - want)
            max_abs_err = max(max_abs_err, err)
            ok = err <= tol
            tag = "OK " if ok else "BAD"
            self.log.info(f"  [{tag}] {label} [{i:2d}] in={self.qvals[i]:+.5f}  "
                          f"acc(hw)={got:+.5f}  acc(ref)={want:+.5f}  err={got - want:+.6f}")
            if not ok:
                mismatches += 1
        self.log.info(f"  {label}: max |error| over run = {max_abs_err:.6f}")
        return mismatches

    async def run_accumulate(self, num=16, seed=0):
        label = f"accumulate(num={num})"
        self.generate_inputs(num, seed)
        collected = []
        monitor = cocotb.start_soon(self.out_monitor(num, collected))
        await self.in_driver(num)
        for _ in range(8):
            await RisingEdge(self.dut.clk)
            if len(collected) >= num:
                break
        await monitor

        assert len(collected) >= num, \
            f"only captured {len(collected)} valid outputs, expected {num}"
        mismatches = self.check_outputs(collected, label)
        self.log.info(f"{label}: {mismatches} mismatch(es)")
        return mismatches

    def make_bits(self, sign, exp_field, mant_field):
        ew, mw = self.param["EXP_WIDTH"], self.param["MANT_WIDTH"]
        return (sign << (ew + mw)) | (exp_field << mw) | mant_field

    async def clear_acc(self):
        await RisingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0
        self.dut.clear_accumulator.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.clear_accumulator.value = 0

    async def drive_settle(self, bits_list):
        for b in bits_list:
            await RisingEdge(self.dut.clk)
            self.dut.data_in.value = b
            self.dut.data_in_valid.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0
        for _ in range(4):            # drain the 2-cycle pipeline
            await RisingEdge(self.dut.clk)

    def read_out_fields(self):
        ew, mw = self.param["EXP_WIDTH"], self.param["MANT_WIDTH"]
        out = int(self.dut.data_out.value)
        return ((out >> (ew + mw)) & 1, (out >> mw) & ((1 << ew) - 1), out & ((1 << mw) - 1))

    async def run_saturation(self):
        """Directed over/underflow: overflow -> +/-max, cancellation -> 0."""
        ew, mw = self.param["EXP_WIDTH"], self.param["MANT_WIDTH"]
        max_exp, max_mant = (1 << ew) - 2, (1 << mw) - 1
        big_pos = self.make_bits(0, max_exp, max_mant)   # largest representable
        big_neg = self.make_bits(1, max_exp, max_mant)
        mism = 0

        # --- positive overflow: sum several max values -> saturate to +max ---
        await self.clear_acc()
        await self.drive_settle([big_pos] * 4)
        await ReadOnly()
        s, e, m = self.read_out_fields()
        ok = (s == 0 and e == max_exp and m == max_mant)
        self.log.info(f"  +overflow: out=({s},{e},{m}) expect=(0,{max_exp},{max_mant}) "
                      f"-> {'OK' if ok else 'BAD'}")
        mism += 0 if ok else 1

        # --- negative overflow: saturate to -max (sign preserved) ---
        await self.clear_acc()
        await self.drive_settle([big_neg] * 4)
        await ReadOnly()
        s, e, m = self.read_out_fields()
        ok = (s == 1 and e == max_exp and m == max_mant)
        self.log.info(f"  -overflow: out=({s},{e},{m}) expect=(1,{max_exp},{max_mant}) "
                      f"-> {'OK' if ok else 'BAD'}")
        mism += 0 if ok else 1

        # --- cancellation/underflow: x + (-x) -> cap to 0 ---
        val = torch.tensor([3.5])
        _, e_q, m_q = _minifloat_ieee_quantize_hardware(val, self.param["FP_W"], ew)
        x = int(pack_fp_to_bin(e_q, m_q, ew, mw)[0])
        x_neg = x ^ (1 << (ew + mw))     # flip sign bit
        await self.clear_acc()
        await self.drive_settle([x, x_neg])
        await ReadOnly()
        s, e, m = self.read_out_fields()
        ok = (e == 0 and m == 0)
        self.log.info(f"  underflow/cancel: out=({s},{e},{m}) expect exp=0,mant=0 "
                      f"-> {'OK' if ok else 'BAD'}")
        mism += 0 if ok else 1

        await RisingEdge(self.dut.clk)   # leave the read-only phase before caller writes
        self.log.info(f"saturation: {mism} mismatch(es)")
        return mism

    async def run_clear(self):
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
                      f"MANT_WIDTH={param['MANT_WIDTH']}")

        total_mismatch = 0
        total_mismatch += await self.run_accumulate(num=16, seed=0)
        await self.init_signals()
        total_mismatch += await self.run_saturation()
        await self.init_signals()
        total_mismatch += await self.run_clear()

        assert total_mismatch == 0, f"{total_mismatch} mismatch(es)"
        self.log.info("All checks passed")


@cocotb.test()
async def test(dut):
    tb = FPFloatAccumulatorTB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test()


@pytest.mark.dev
def test_fp_float_accumulator():
    veri_runner(
        group="fp_operation",
        module="fp_float_accumulator",
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
    test_fp_float_accumulator()
