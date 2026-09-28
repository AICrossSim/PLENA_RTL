#!/usr/bin/env python3
"""Deterministic characterization of fp_fix_exp at FP12 (exp=6, mant=5).

Drives a handful of known inputs one at a time (with ample settle time) and
prints HW output vs torch.exp, to localize the exp unit's numeric bug.
"""
import logging
import sys

import cocotb
from cocotb.triggers import RisingEdge
import torch

from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.testbench import Testbench
from plena_quant.common import _minifloat_ieee_quantize_hardware
from cfl_cocotb.torch_fp_conversion import pack_fp_to_bin
from cocotb.log import SimLog


def decode_fp(bits, ew, mw):
    bits = int(bits)
    s = (bits >> (ew + mw)) & 1
    e = (bits >> mw) & ((1 << ew) - 1)
    m = bits & ((1 << mw) - 1)
    bias = (1 << (ew - 1)) - 1
    v = (m / (1 << mw)) * 2.0 ** (1 - bias) if e == 0 else (1 + m / (1 << mw)) * 2.0 ** (e - bias)
    return -v if s else v


class CharTB(Testbench):
    def __init__(self, dut):
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("char")

    async def run(self):
        ew = int(self.dut.EXP_WIDTH.value)
        mw = int(self.dut.MANT_WIDTH.value)
        await self.reset()
        try:
            self.dut.data_in_ready.value = 1
        except Exception:
            pass
        self.dut.data_in_valid.value = 0
        await RisingEdge(self.dut.clk)

        # internal probe handles (fp_cp_exp_inst.fp_exp_inst.*)
        try:
            ex = self.dut.fp_cp_exp_inst.fp_exp_inst
        except Exception:
            ex = None

        def gi(name):
            try:
                return int(getattr(ex, name).value)
            except Exception:
                return None

        def gs(name, width):
            # signed read
            try:
                v = int(getattr(ex, name).value)
                if v >> (width - 1):
                    v -= 1 << width
                return v
            except Exception:
                return None

        logf = open("/tmp/exp_char.log", "w")
        test_vals = [0.5, 0.8125, 1.0, 1.5, 2.0, -0.5, -1.0, -2.0, 0.25, 3.0]
        for xv in test_vals:
            tx = torch.tensor([float(xv)])
            qx, x_exp, x_mant = _minifloat_ieee_quantize_hardware(tx, mw + ew, ew)
            inbits = int(pack_fp_to_bin(x_exp, x_mant, ew, mw)[0])
            ref = float(torch.exp(qx)[0])
            self.dut.data_in.value = inbits
            self.dut.data_in_valid.value = 1
            await RisingEdge(self.dut.clk)
            self.dut.data_in_valid.value = 0
            got = None
            for c in range(30):
                await RisingEdge(self.dut.clk)
                try:
                    if int(self.dut.data_out_valid.value) == 1:
                        got = decode_fp(self.dut.data_out.value, ew, mw)
                        break
                except Exception:
                    pass
            rel = abs(got - ref) / ref if (got is not None and ref) else None
            logf.write(f"exp({float(qx[0]):+.4f}) HW={got} ref={ref:.4f}"
                       + (f"  relerr={rel:.1%}" if rel is not None else "") + "\n")
        logf.close()


@cocotb.test()
async def test(dut):
    tb = CharTB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run()


def main():
    veri_runner(
        group="fp_operation",
        module="fp_fix_exp",
        test_module="fp_fix_exp_char_tb",
        test_dir=__import__("pathlib").Path(__file__).parent,
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        module_param_list=[{"EXP_WIDTH": 6, "MANT_WIDTH": 5}],
        trace=False,
    )


if __name__ == "__main__":
    main()
