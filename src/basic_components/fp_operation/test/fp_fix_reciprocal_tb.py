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


class FPFixReciTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        # Valid-only pipeline: no ready on either side.
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
        self.out_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, num):
        q_config = {
            "exp_width" : self.dut.EXP_WIDTH.value,
            "mant_width" : self.dut.MANT_WIDTH.value,
        }

        # Generate random inputs, avoiding values too close to zero to prevent overflow
        torch_x = torch.randn(num) * 2.5
        # Replace values too close to zero with reasonable values
        torch_x[torch.abs(torch_x) < 0.1] = torch.sign(torch_x[torch.abs(torch_x) < 0.1]) * 0.5

        in_width = q_config["mant_width"] + q_config["exp_width"]
        in_exponent_width = q_config["exp_width"]

        # Quantize input
        qx, x_exp, x_mant = _minifloat_ieee_quantize_hardware(torch_x, in_width, in_exponent_width)

        # Quantize output
        out_width = q_config["mant_width"] + q_config["exp_width"] + 1
        out_exponent_width = q_config["exp_width"]

        # Pack inputs and outputs to binary format
        inputs_x = pack_fp_to_bin(x_exp, x_mant, q_config["exp_width"], q_config["mant_width"])
        q_out = torch.reciprocal(qx)
        q_out, q_out_exp, q_out_mant = _minifloat_ieee_quantize_hardware(q_out, out_width, out_exponent_width)
        outputs_out = pack_fp_to_bin(q_out_exp, q_out_mant, q_config["exp_width"], q_config["mant_width"])

        self.inputs = [(int(inputs_x[i])) for i in range(num)]

        self.outputs = [int(outputs_out[i]) for i in range(num)]

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
    tb = FPFixReciTB(dut)
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test(10, 10)



@pytest.mark.dev
def test_simple_fp_addition():
    # Run tests with different params
    veri_runner(
        group = "fp_operation",
        module = "fp_fix_reciprocal",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/cast"),
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
