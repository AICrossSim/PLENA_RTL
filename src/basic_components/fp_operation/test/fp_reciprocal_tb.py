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
from cfl_cocotb.torch_fp_conversion import pack_fp_to_bin
from plena_quant.quant_operations import fp_reciprocal

from plena_quant.common import _minifloat_ieee_quantize_hardware

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)

src_path = Path(__file__).parent.parent.parent

torch.manual_seed(10)
class FPReciprocalTB(Testbench):
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
            "out_exp_width": self.dut.OUT_EXP_WIDTH.value,
            "out_fix_width": self.dut.OUT_FIX_WIDTH.value,
            "out_fix_frac_width": self.dut.OUT_FIX_FRAC_WIDTH.value,
        }
        
        # Generate random inputs between -1 and 1, avoiding very small numbers
        torch.manual_seed(0)
        x = torch.rand(num) * self.range - self.range / 2
        
        qx, exp_x, mant_x = _minifloat_ieee_quantize_hardware(
            x, q_config["in_fix_frac_width"] + q_config["in_exp_width"] + 1, q_config["in_exp_width"])
        # Convert inputs to binary format
        
        self.log.debug(f"input : {exp_x}, {mant_x}")
        # Compute expected outputs
        expected_exp, expected_mant = fp_reciprocal(exp_x, mant_x, q_config)
        self.log.debug(f"expected_exp : {expected_exp}, expected_mant : {expected_mant}")

        expected_outputs = pack_fp_to_bin(expected_exp, expected_mant, q_config["out_exp_width"], q_config["out_fix_width"]).tolist()
        
        self.inputs = [(int(exp_x[i]), int(mant_x[i]*2**q_config["in_fix_frac_width"])) for i in range(num)]
        self.outputs = [(int(expected_exp[i]), int(expected_mant[i]*2**q_config["out_fix_frac_width"])) for i in range(num)]

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
    tb = FPReciprocalTB(dut)
    tb.log.setLevel(logging.DEBUG)
    tb.range = 1
    await tb.run_test(10,10)

if __name__ == "__main__":
    veri_runner(
        trace=False, 
        module="fp_reciprocal",
        group="fp_operation",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
        ],
        module_param_list=[
            {
                "IN_EXP_WIDTH": 4,
                "IN_FIX_WIDTH": 7,
                "IN_FIX_FRAC_WIDTH": 5,
                "OUT_EXP_WIDTH": 4,
                "OUT_FIX_WIDTH": 7,
                "OUT_FIX_FRAC_WIDTH": 5
            }
        ]
    )
