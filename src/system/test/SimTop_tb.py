#!/usr/bin/env python3
"""PLENA System Top-Level RTL Testbench (production / verification).

This is the canonical testbench used by `just rtl-sim <workload>` for ALL
workloads. It is intentionally probe-free: it resets the DUT, runs until
C_BREAK, drains the pipeline, and lets the $finish memory dump produce
vector_result.mem / hbm_result.mem, which the verifier checks.

Workload-specific cocotb signal probes used during bring-up/debug live in
SimTop_tb_debug.py — keep them OUT of this file so every workload runs the
same lean, fast, side-effect-free testbench.

Usage:
    # Generate workload first
    python -m tools.testworkloads.linear --batch 8 --in-features 128 --out-features 256

    # Run simulation
    just rtl-sim linear
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pytest
import cocotb
from cocotb.log import SimLog
from cocotb.triggers import RisingEdge

# Setup paths
_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"
_TEST_PATH = Path(__file__).resolve().parent

for p in [str(_TOOLS_PATH), str(_TEST_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cfl_cocotb.runner import veri_runner, SRC_PATH
from cfl_cocotb.testbench import Testbench

from cfl_tools.logger import get_logger
from cfl_tools.debugger import set_excepthook

from test_platform import PLENATestPlatform

logger = get_logger("testbench")
logger.setLevel(logging.DEBUG)

INSTRUCTION_LENGTH = 32
set_excepthook()


class SimTOP(Testbench):
    """PLENA System Top-Level Testbench.

    Drives the PLENA system with instructions from the HBM file, runs until the
    program's C_BREAK, then drains the compute pipelines so the final VRAM/HBM
    dump is complete. No signal probing — verification reads the dumped files.
    """

    def __init__(
        self,
        dut,
        hbm_file: str,
    ) -> None:
        """Initialize the testbench.

        Args:
            dut: cocotb DUT handle
            hbm_file: Path to combined HBM file (contains data and instructions)
        """
        super().__init__(dut, dut.clk, dut.rst)
        self.hbm_file = hbm_file

        # Execution clock counter state
        self.execution_clocks = 0
        self.c_break_detected = False

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

    async def run_test(self):
        """Run the main test sequence.

        Instructions are loaded from hbm_file (HBM_INSTRUCTIONS section) at simulation start.
        """
        await self.reset()
        logger.info("Reset finished")
        logger.info(f"HBM data and instructions loaded from: {self.hbm_file}")

        # Count execution clocks until C_BREAK is detected
        exec_clocks = await self.count_execution_clocks(timeout_us=10000)

        if exec_clocks < 0:
            raise RuntimeError("BUG: C_BREAK instruction was not loaded/decoded")

        # C_BREAK is detected at DECODE (the first pipeline stage), but vector /
        # matrix ops issued in the cycles just before C_BREAK are still draining
        # through the multi-stage compute pipelines and their write-backs to the
        # vector SRAM have not committed yet. Ending the test (and triggering the
        # $finish memory dump) immediately would capture a half-written VRAM
        # (e.g. the last normalization chunk of an RMS loop left un-normalized).
        # The PC keeps fetching past C_BREAK but post-C_BREAK words decode to
        # STALL/NOP, so these extra cycles cannot corrupt state — they only let
        # in-flight write-backs land before the VRAM is dumped.
        # Cover the full matrix-writeback drain: a final M_MM_WO before C_BREAK
        # commits its write-back well past a 32-cycle window (dropped last tile).
        drain_cycles = 128
        logger.info(f"Draining pipeline for {drain_cycles} cycles after C_BREAK")
        for _ in range(drain_cycles):
            await RisingEdge(self.dut.clk)

    async def count_execution_clocks(self, timeout_us=50):
        """Count execution clocks from reset release until C_BREAK is decoded.

        Args:
            timeout_us: Maximum time to wait for C_BREAK in microseconds

        Returns:
            Number of clock cycles if C_BREAK detected, -1 if timeout (bug)
        """
        self.execution_clocks = 0
        self.c_break_detected = False

        # Calculate timeout in clock cycles (assuming 20ns clock period from testbench.py)
        clock_period_ns = 20
        timeout_cycles = int(timeout_us * 1000 / clock_period_ns)

        logger.info(f"Starting execution clock counter (timeout: {timeout_us}us = {timeout_cycles} cycles)")

        while self.execution_clocks < timeout_cycles:
            await RisingEdge(self.dut.clk)
            self.execution_clocks += 1

            try:
                c_break = self.dut.dut.c_break_detected.value
                if int(c_break) == 1:
                    self.c_break_detected = True
                    logger.info(f"C_BREAK detected at clock cycle {self.execution_clocks}")
                    logger.info(f"=== EXECUTION CLOCKS: {self.execution_clocks} ===")
                    return self.execution_clocks
            except (AttributeError, ValueError):
                pass

        logger.error(f"BUG: C_BREAK not loaded within {timeout_us}us ({timeout_cycles} cycles)")
        logger.error("=== BUG: C_BREAK NOT DETECTED ===")
        return -1


@cocotb.test()
async def test(dut):
    """Main cocotb test entry point."""
    tb = SimTOP(
        dut,
        os.environ["FAKE_HBM_INIT_FILE"],
    )
    tb.log.setLevel(logging.DEBUG)
    await tb.run_test()


def get_module_params() -> dict:
    """Get module parameters from environment variables."""
    return {
        "INSTRUCTION_LENGTH": INSTRUCTION_LENGTH,
        "FAKE_HBM_INIT_FILE": f"\"{os.environ['FAKE_HBM_INIT_FILE']}\"",  # Combined HBM init file (data + instructions)
        "FP_MEM_INIT_FILE": f"\"{os.environ['FP_MEM_INIT_FILE']}\"",
        "INT_MEM_INIT_FILE": f"\"{os.environ['INT_MEM_INIT_FILE']}\"",
        "VECTOR_MEM_RESULT_FILE": f"\"{os.environ['VECTOR_MEM_RESULT_FILE']}\"",
        "FP_REG_RESULT_FILE": f"\"{os.environ.get('FP_REG_RESULT_FILE', '')}\"",  # FP register-file dump for debug
        "HBM_RESULT_FILE": f"\"{os.environ.get('HBM_RESULT_FILE', '')}\"",  # HBM dump for verification
    }


@pytest.mark.dev
def test_SimTop():
    """Run SimTop RTL test."""
    # Check if SKIP_BUILD environment variable is set
    skip_build = os.environ.get("SKIP_BUILD", "0") == "1"

    # Waveform dumping dominates run time on long programs. SIMTOP_TRACE=0
    # builds without --trace and skips the VCD dump; SIMTOP_TRACE=1 (default)
    # enables it. Cycle counts are unaffected either way.
    trace = os.environ.get("SIMTOP_TRACE", "1") == "1"

    # Get workload directory for simulation artifacts (VCD, etc)
    workload_dir = os.environ.get("WORKLOAD_DIR")
    sim_build_dir = Path(workload_dir) if workload_dir else None

    return veri_runner(
        group="system",
        module="SimTop",
        test_dir=Path(__file__).parent,  # Tell cocotb where to find SimTop_tb.py
        extra_build_args=["-DSIMULATION"],
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/systolic_gemm_mx"),
            str(SRC_PATH / "basic_components/systolic_gemm_mxint"),
            str(SRC_PATH / "basic_components/gemv"),
            str(SRC_PATH / "basic_components/synopsis/rtl"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/hadamard_transform"),
            str(SRC_PATH / "frontend"),
            str(SRC_PATH / "control"),
            str(SRC_PATH / "matrix_machine"),
            str(SRC_PATH / "vector_machine"),
            str(SRC_PATH / "scalar_machine"),
            str(SRC_PATH / "memory/matrix_sram"),
            str(SRC_PATH / "memory/vector_sram"),
            str(SRC_PATH / "memory/scratch_sram"),
            str(SRC_PATH / "memory/scalar_sram"),
            str(SRC_PATH / "memory/HBM"),
            str(SRC_PATH / "core"),
        ],
        definitions_path=[
            str(SRC_PATH / "definitions"),
            str(SRC_PATH / "memory/HBM/TileLink_Lib"),
        ],
        module_param_list=[get_module_params()],
        trace=trace,
        skip_build=skip_build,
        sim_build_dir=sim_build_dir,
    )


def run_with_workload(workload_dir: str):
    """Run simulation with a generated workload directory.

    Args:
        workload_dir: Path to workload build directory
    """
    os.environ["WORKLOAD_DIR"] = workload_dir

    # Load platform to set all env vars
    platform = PLENATestPlatform.from_workload(workload_dir)
    platform.prepare()

    # Propagate the cocotb fail count as the exit code so a hung run reports FAILED.
    # Use sys.exit, not raise (the module's pdb.post_mortem excepthook would hang).
    num_failed = test_SimTop()
    if num_failed:
        logger.error(f"RTL simulation FAILED: {num_failed} cocotb test(s) did not pass "
                     f"(e.g. C_BREAK not reached — program hung).")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLENA SimTop RTL Testbench")
    parser.add_argument(
        "--workload-dir",
        type=str,
        required=True,
        help="Path to workload build directory (from testworkloads)"
    )
    args = parser.parse_args()

    run_with_workload(args.workload_dir)
