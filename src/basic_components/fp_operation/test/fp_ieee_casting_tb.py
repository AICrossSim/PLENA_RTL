#!/usr/bin/env python3

import logging
import torch
from pathlib import Path

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from cfl_cocotb.testbench import Testbench
from cfl_cocotb.valid_only import ValidOnlyDriver, ValidOnlyMonitor
from cfl_cocotb.runner import veri_runner, SRC_PATH
from cfl_cocotb.torch_fp_conversion import fp_2_bin

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

src_path = Path(__file__).parent.parent.parent

torch.manual_seed(10)


class FPIEEECasting(Testbench):
    """fp_ieee_casting is a 2-stage valid-only pipeline (exp cast -> mant cast),
    so it is driven as a clocked stream rather than a combinational block."""

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.in_driver = ValidOnlyDriver(
            dut.clk, dut.data_in, dut.data_in_valid
        )

        self.out_monitor = ValidOnlyMonitor(
            dut.clk,
            dut.data_out,
            dut.data_out_valid,
            check=True,
            unsigned=True,
        )

    def generate_inputs(self, num):
        self.q_config = {
            "in_exp_width": self.dut.IN_EXP_WIDTH.value,
            "in_man_width": self.dut.IN_MANT_WIDTH.value,
            "out_exp_width": self.dut.OUT_EXP_WIDTH.value,
            "out_man_width": self.dut.OUT_MANT_WIDTH.value,
        }
        
        # Generate random inputs between -5 and 5
        x = torch.rand(num) * 10 - 5

        q_x, fp_inputs = fp_2_bin(
            x, 
            self.q_config["in_exp_width"], 
            self.q_config["in_man_width"]
        )

        self.log.debug(f"Input Packed FP: {fp_inputs}")

        q_x, fp_outputs = fp_2_bin(
            q_x, 
            self.q_config["out_exp_width"], 
            self.q_config["out_man_width"]
        )

        self.log.debug(f"Output Packed FP: {fp_outputs}")

        self.inputs = [int(v) for v in fp_inputs.int().tolist()]
        self.outputs = [int(v) for v in fp_outputs.int().tolist()]

    async def run_test(self, us, num):
        self.dut.data_in_valid.value = 0
        await self.reset()
        self.log.info("Reset finished")

        self.generate_inputs(num)

        self.in_driver.load_driver(self.inputs)
        self.out_monitor.load_monitor(self.outputs)

        await Timer(us, units="us")
        assert self.out_monitor.exp_queue.empty()


@cocotb.test()
async def test(dut):
    tb = FPIEEECasting(dut)
    tb.log.setLevel(logging.INFO)
    await tb.run_test(5, 10)

if __name__ == "__main__":
    veri_runner(
        trace=False, 
        module="fp_ieee_casting",
        group="fp_operation",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation")
        ],
        module_param_list=[
            {
                "IN_EXP_WIDTH": 5,
                "IN_MANT_WIDTH": 10,
                "OUT_EXP_WIDTH": 4,
                "OUT_MANT_WIDTH": 7
            }
        ]
    )
