#!/usr/bin/env python3

import logging
import pytest
import cocotb

from cocotb.triggers import Timer

import torch

from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.testbench import Testbench
from cfl_cocotb.valid_only import ValidOnlyDriver, ValidOnlyMonitor

from plena_quant.common import _minifloat_ieee_quantize_hardware
from cfl_cocotb.torch_fp_conversion import pack_fp_to_bin
from cocotb.log import SimLog


class FPFixMultTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # Valid-only pipeline: no ready on either side.
        self.in_driver = ValidOnlyDriver(
            dut.clk, (dut.data_a, dut.data_b), dut.data_in_valid
        )

        self.out_monitor = ValidOnlyMonitor(
            dut.clk,
            dut.data_out,
            dut.data_out_valid,
            check=True,
            unsigned=True,
        )
        self.out_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, num):
        q_config = {
            "EXP_WIDTH" : self.dut.EXP_WIDTH.value,
            "MANT_WIDTH" : self.dut.MANT_WIDTH.value,
        }

        torch.manual_seed(0)
        torch_a = torch.randn(num)
        torch_b = torch.randn(num)

        width = q_config["MANT_WIDTH"] + q_config["EXP_WIDTH"] + 1
        exponent_width = q_config["EXP_WIDTH"]

        qa, a_exp, a_mant = _minifloat_ieee_quantize_hardware(torch_a, width, exponent_width)
        qb, b_exp, b_mant = _minifloat_ieee_quantize_hardware(torch_b, width, exponent_width)

        inputs_a = pack_fp_to_bin(a_exp, a_mant, q_config["EXP_WIDTH"], q_config["MANT_WIDTH"])
        inputs_b = pack_fp_to_bin(b_exp, b_mant, q_config["EXP_WIDTH"], q_config["MANT_WIDTH"])

        out = qa * qb

        qout, out_exp, out_mant = _minifloat_ieee_quantize_hardware(out, width, exponent_width)
        outputs_out = pack_fp_to_bin(
            out_exp, out_mant, 
            q_config["EXP_WIDTH"], 
            q_config["MANT_WIDTH"])
        
        self.inputs = [(int(inputs_a[i]), int(inputs_b[i])) for i in range(num)]
        self.outputs = [int(outputs_out[i]) for i in range(num)]

    async def run_test(self, us, num):
        self.dut.data_in_valid.value = 0
        await self.reset()
        self.log.info(f"Reset finished")

        self.generate_inputs(num)   

        self.in_driver.load_driver(self.inputs)

        self.out_monitor.load_monitor(self.outputs)

        await Timer(100, units="ns")

        await Timer(us, units="us")
        assert self.out_monitor.exp_queue.empty()

@cocotb.test()
async def test(dut):
    tb = FPFixMultTB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test(10, 10)



@pytest.mark.dev
def test_simple_fp_addition():
    # Run tests with different params
    veri_runner(
        group = "fp_operation",
        module = "fp_fix_mult",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/synopsis")
        ],
        definitions_path = [
            str(SRC_PATH / "definitions"), 
        ],
        module_param_list=[
            {"EXP_WIDTH" : 6, "MANT_WIDTH" : 5},
        ],
        trace = False,
    )

if __name__ == "__main__":
    test_simple_fp_addition()
