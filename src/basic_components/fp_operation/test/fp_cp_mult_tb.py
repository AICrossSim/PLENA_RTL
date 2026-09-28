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
from cfl_tools.debugger import set_excepthook
from cocotb.log import SimLog

class FPCPMultTB(Testbench):
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
    def generate_inputs(self, num):
        # seed = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(0)
        # self.log.info(f"seed : {seed}")
        q_config = {
            "exp_width" : self.dut.EXP_WIDTH.value,
            "mant_width" : self.dut.MANT_WIDTH.value,
            "ext_mant_width" : self.dut.EXT_MANT_WIDTH.value,
            "ext_exp_width" : self.dut.EXT_EXP_WIDTH.value,
        }

        exp_width = q_config["exp_width"]
        mant_width = q_config["mant_width"]
        ext_mant_width = q_config["ext_mant_width"]
        ext_exp_width = q_config["ext_exp_width"]

        torch_a = torch.randn(num) * 10 - 5
        torch_b = torch.randn(num) * 10 - 5

        width = q_config["mant_width"] + q_config["exp_width"]
        exponent_width = q_config["exp_width"]

        qa, a_exp, a_mant = _minifloat_ieee_quantize_hardware(torch_a, width, exponent_width)
        qb, b_exp, b_mant = _minifloat_ieee_quantize_hardware(torch_b, width, exponent_width)

        out_width = q_config["mant_width"] + q_config["exp_width"] + q_config["ext_mant_width"] + q_config["ext_exp_width"] + 1
        out_exponent_width = q_config["exp_width"] + q_config["ext_exp_width"]

        out = qa * qb
        self.log.debug(f"out : {out}")
        debug_out, debug_exp, debug_mant = _minifloat_ieee_quantize_hardware(out, 6 + 8 + 1, 6)
        debug_out_bin = pack_fp_to_bin(debug_exp, debug_mant, 6, 8)
        self.log.debug(f"debug_out_bin : {debug_out_bin}")
        self.log.debug(f"debug_out : {debug_out}")
        self.log.debug(f"debug_exp : {debug_exp}")
        self.log.debug(f"debug_mant : {debug_mant}")
        _minifloat_ieee_quantize_hardware(torch.tensor([196]), 4,3)
        qout, out_exp, out_mant = _minifloat_ieee_quantize_hardware(out, out_width, out_exponent_width)

        inputs_a = pack_fp_to_bin(a_exp, a_mant, q_config["exp_width"], q_config["mant_width"])
        inputs_b = pack_fp_to_bin(b_exp, b_mant, q_config["exp_width"], q_config["mant_width"])

        outputs_out = pack_fp_to_bin(out_exp, out_mant, q_config["exp_width"] + q_config["ext_exp_width"], q_config["mant_width"] + q_config["ext_mant_width"])

        self.inputs = [(int(inputs_a[i]), int(inputs_b[i])) for i in range(num)]

        self.log.debug(f"input_0 : {qa}, {a_exp}, {a_mant}")
        self.log.debug(f"input_1 : {qb}, {b_exp}, {b_mant}")
        self.log.debug(f"output : {qout}, {out_exp}, {out_mant}")
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
async def test_fp_cp_mult(dut):
    set_excepthook()
    tb = FPCPMultTB(dut)
    tb.log.setLevel(logging.INFO)
    await tb.run_test(100, 10)



@pytest.mark.dev
def test_simple_fp_addition():
    # Run tests with different params
    veri_runner(
        group = "fp_operation",
        module = "fp_cp_mult",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation")
        ],
        module_param_list=[
            {"EXP_WIDTH" : 6, "MANT_WIDTH" : 5, "EXT_MANT_WIDTH" : 0, "EXT_EXP_WIDTH" : 0},
            # {"EXP_WIDTH" : 1, "MANT_WIDTH" : 6, "EXT_MANT_WIDTH" : 0, "EXT_EXP_WIDTH" : 0},
        ],
        trace = False,
    )

if __name__ == "__main__":
    test_simple_fp_addition()
