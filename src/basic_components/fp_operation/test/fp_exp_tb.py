#!/usr/bin/env python3

# This script tests the floating point exponential function
import logging

import torch

from pathlib import Path

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from cfl_cocotb.testbench import Testbench
from cfl_cocotb.valid_only import ValidOnlyDriver, ValidOnlyMonitor
from cfl_cocotb.runner import veri_runner, SRC_PATH
from plena_quant.common import _minifloat_ieee_quantize_hardware
from plena_quant.quant_operations import fp_exp_hardware

logger = logging.getLogger("testbench")
logger_level = logging.DEBUG
logger.setLevel(logger_level)

src_path = Path(__file__).parent.parent.parent

torch.manual_seed(10)


class FPExpTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # Valid-only pipeline: no ready on either side.
        self.in_driver = ValidOnlyDriver(
            dut.clk, (dut.signed_exp_in, dut.signed_mant_in), dut.data_in_valid
        )

        self.out_monitor = ValidOnlyMonitor(
            dut.clk,
            (dut.signed_exp_out, dut.signed_mant_out),
            dut.data_out_valid,
            check=True,
        )
        self.out_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, num):
        torch.manual_seed(0)
        q_config = {
            "in_exp_width": self.dut.IN_EXP_WIDTH.value,
            "in_fix_width": self.dut.IN_FIX_WIDTH.value,
            "in_fix_frac_width": self.dut.IN_FIX_FRAC_WIDTH.value,
            "extend_width": self.dut.EXTEND_WIDTH.value,
            "out_exp_width": self.dut.OUT_EXP_WIDTH.value,
            "out_fix_width": self.dut.OUT_FIX_WIDTH.value,
            "out_fix_frac_width": self.dut.OUT_FIX_FRAC_WIDTH.value,
        }
        
        # Generate test inputs
        a = torch.randn(num) * 3
        qa, a_exp, a_mant = _minifloat_ieee_quantize_hardware(a, q_config["in_fix_frac_width"] + q_config["in_exp_width"] + 1, q_config["in_exp_width"])
        
        # Calculate expected outputs using hardware model
        expected_exp, expected_mant = fp_exp_hardware(a_exp, a_mant, q_config)

        self.inputs = [(int(a_exp[i]), int(a_mant[i]*2**(q_config["in_fix_frac_width"]))) for i in range(num)]
        self.outputs = [(int(expected_exp[i]), int(expected_mant[i]*2**(q_config["out_fix_frac_width"]))) for i in range(num)]

    async def run_test(self, us, num):
        self.dut.data_in_valid.value = 0
        await self.reset()
        self.log.info(f"Reset finished")

        self.generate_inputs(num)   

        self.in_driver.load_driver(self.inputs)

        self.out_monitor.load_monitor(self.outputs)

        await Timer(us, units="us")
        assert self.out_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = FPExpTB(dut)
    tb.log.setLevel(logger_level)
    await tb.run_test(20, 10)


if __name__ == "__main__":
    veri_runner(
        trace=False, 
        module="fp_exp",
        group="fp_operation",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/buffer"),
        ],
        module_param_list=[
            {
                "IN_EXP_WIDTH": 8,
                "IN_FIX_WIDTH": 7,
                "IN_FIX_FRAC_WIDTH": 5,
                "EXTEND_WIDTH": 5,
                "OUT_EXP_WIDTH": 8,
                "OUT_FIX_WIDTH": 8,
                "OUT_FIX_FRAC_WIDTH": 5
            }
        ]
    )
