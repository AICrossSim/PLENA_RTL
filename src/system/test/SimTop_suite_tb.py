#!/usr/bin/env python3
"""PLENA looped suite testbench — one Verilator build, many layer cases.

Unlike SimTop_tb.py (one workload per build), this tb builds the DUT ONCE and
then iterates a suite of pre-generated cases, backdoor-loading each case's HBM
and scalar SRAM contents into the memory arrays between runs, resetting, running
to C_BREAK, and dumping the vector-SRAM result. Verification runs afterward in
Python (reusing verify_rtl_sim per case).

Flow:
    1. tools/testworkloads/test_suite.py generates all cases at a SHARED
       instruction-storage offset (so one build serves them all) and writes
       <root>/suite_manifest.json.
    2. run_with_suite(root) sets env (build params from case 0) and calls
       veri_runner -> builds once, runs the cocotb test() which loops cases.
    3. After the sim, each case is verified and a summary table is printed.

Backdoor memory paths (flat arrays, simulation-visible):
    HBM bytes : dut.fake_hbm_inst.mem[byte]                       (256-bit rows)
    fp sram   : dut.dut.scalar_machine_init.fp_scalar_sram.sram.mem[word]
    int sram  : dut.dut.scalar_machine_init.int_scalar_sram.sram.mem[word]
    vec result: dut.dut.vector_sram.vect_storage.mem_flat[row]    (VLEN*12 bits)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import RisingEdge

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

logger = get_logger("suite")
logger.setLevel(logging.INFO)

INSTRUCTION_LENGTH = 32
VEC_RESULT_ROWS = 1024   # SRAM_DEPTH for the vector SRAM result mirror
VEC_RESULT_HEXLEN = 48   # VLEN*12 bits = 192 bits = 48 hex chars
set_excepthook()


def _parse_hbm_rows(hbm_path):
    """Parse an hbm.mem file into a list of (row_index, 256-bit int) entries.

    Each non-comment line is a 32-byte (256-bit) hex row at byte address
    row_index*32 (matching fake_hbm_5port's initial-block loader).
    """
    rows = []
    idx = 0
    with open(hbm_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if s[:2].lower() == "0x":
                s = s[2:]
            try:
                val = int(s, 16)
            except ValueError:
                continue
            rows.append((idx, val))
            idx += 1
    return rows


def _parse_hex_words(mem_path):
    """Parse a scalar-sram .mem file (one hex word per line)."""
    words = []
    with open(mem_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if s[:2].lower() == "0x":
                s = s[2:]
            try:
                words.append(int(s, 16))
            except ValueError:
                continue
    return words


class SimTopSuite(Testbench):
    """Looped suite driver."""

    def __init__(self, dut):
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("SimTopSuite")

    def _diag(self, msg):
        """File-based diagnostics (cocotb logger isn't captured by the runner)."""
        root = Path(os.environ["SUITE_MANIFEST"]).parent
        with open(root / "suite_diag.log", "a") as f:
            f.write(msg + "\n")

    # ---- backdoor loaders -------------------------------------------------
    def _load_hbm(self, hbm_path, natural_offset, fixed_offset):
        """Load a case's hbm.mem into the HBM byte array, RELOCATING the
        instruction region from the case's natural offset (where generate_hbm
        packs it, right after data) to the fixed build offset
        (INSTRUCTION_STORAGE_OFFSET, which is baked into this one build).

        rows [0, natural_offset/32)  -> data, written in place
        rows [natural_offset/32, end) -> instructions, written at fixed_offset
        """
        mem = self.dut.fake_hbm_inst.mem
        rows = _parse_hbm_rows(hbm_path)
        nat_row = natural_offset // 32
        n_data = n_instr = 0
        for row_idx, val in rows:
            if row_idx < nat_row:
                base = row_idx * 32                       # data: in place
                n_data += 1
            else:
                base = fixed_offset + (row_idx - nat_row) * 32   # instr: relocated
                n_instr += 1
            for i in range(32):
                mem[base + i].value = (val >> (i * 8)) & 0xFF
        self._diag(f"  HBM: {len(rows)} rows (data={n_data}, instr={n_instr}), "
                   f"nat_off={natural_offset} -> fixed_off={fixed_offset}")

    def _load_sram(self, handle, mem_path, label):
        words = _parse_hex_words(mem_path)
        for addr, w in enumerate(words):
            handle.mem[addr].value = w
        logger.info(f"  {label}: loaded {len(words)} words")

    def _clear_vec_result(self):
        mf = self.dut.dut.vector_sram.vect_storage.mem_flat
        for i in range(VEC_RESULT_ROWS):
            mf[i].value = 0

    def _dump_vec_result(self, out_path):
        mf = self.dut.dut.vector_sram.vect_storage.mem_flat
        with open(out_path, "w") as f:
            for i in range(VEC_RESULT_ROWS):
                try:
                    v = int(mf[i].value)
                except ValueError:
                    v = 0
                f.write(f"{v:0{VEC_RESULT_HEXLEN}x}\n")

    # ---- per-case run -----------------------------------------------------
    async def run_case(self, name, case_dir):
        case_dir = Path(case_dir)
        logger.info(f"=== CASE {name} ===")

        # Backdoor-load this case's memory image, relocating instructions to
        # the shared build offset.
        self._clear_vec_result()
        self._load_hbm(case_dir / "hbm.mem", self._case_natural[name], self._fixed_offset)
        self._load_sram(self.dut.dut.scalar_machine_init.fp_scalar_sram.sram,
                        case_dir / "fp_sram.mem", "fp_sram")
        self._load_sram(self.dut.dut.scalar_machine_init.int_scalar_sram.sram,
                        case_dir / "int_sram.mem", "int_sram")

        # Reset so the PC/instr buffer restart against the new HBM image.
        await self.reset()

        exec_clocks = await self.count_execution_clocks(timeout_us=10000)

        drain_cycles = 32
        for _ in range(drain_cycles):
            await RisingEdge(self.dut.clk)

        # Dump the vector-SRAM result for offline verification. Use a dedicated
        # filename (NOT vector_result.mem): the build's $writememh fires once at
        # $finish and writes the LAST case's mem_flat to the baked
        # VECTOR_MEM_RESULT_FILE path, which would clobber a per-case
        # vector_result.mem. run_with_suite copies this to vector_result.mem
        # after the sim, before verifying.
        self._dump_vec_result(case_dir / "suite_vec_result.mem")

        # Count nonzero result rows as a liveness signal.
        nz = 0
        try:
            mf = self.dut.dut.vector_sram.vect_storage.mem_flat
            for i in range(VEC_RESULT_ROWS):
                if int(mf[i].value) != 0:
                    nz += 1
        except Exception as e:
            nz = f"err:{e}"
        self._diag(f"  {name}: exec_clocks={exec_clocks} mem_flat_nonzero={nz}")
        return exec_clocks > 0

    async def run_suite(self):
        manifest_path = Path(os.environ["SUITE_MANIFEST"])
        manifest = json.loads(manifest_path.read_text())
        cases = manifest["cases"]
        self._fixed_offset = manifest["shared_instr_offset"]
        self._case_natural = {c["name"]: c["natural_offset"] for c in cases}
        # Reset the diag log at suite start.
        (manifest_path.parent / "suite_diag.log").write_text(
            f"Suite: {len(cases)} cases, fixed build offset "
            f"0x{self._fixed_offset:X}\n")

        for c in cases:
            await self.run_case(c["name"], c["dir"])

    async def count_execution_clocks(self, timeout_us=50):
        """Run until C_BREAK is decoded; return cycle count, or -1 on timeout."""
        clocks = 0
        clock_period_ns = 20
        timeout_cycles = int(timeout_us * 1000 / clock_period_ns)
        while clocks < timeout_cycles:
            await RisingEdge(self.dut.clk)
            clocks += 1
            try:
                if int(self.dut.dut.c_break_detected.value) == 1:
                    return clocks
            except (AttributeError, ValueError):
                pass
        return -1


@cocotb.test()
async def test(dut):
    tb = SimTopSuite(dut)
    tb.log.setLevel(logging.INFO)
    await tb.run_suite()


def get_module_params() -> dict:
    return {
        "INSTRUCTION_LENGTH": INSTRUCTION_LENGTH,
        "FAKE_HBM_INIT_FILE": f"\"{os.environ['FAKE_HBM_INIT_FILE']}\"",
        "FP_MEM_INIT_FILE": f"\"{os.environ['FP_MEM_INIT_FILE']}\"",
        "INT_MEM_INIT_FILE": f"\"{os.environ['INT_MEM_INIT_FILE']}\"",
        "VECTOR_MEM_RESULT_FILE": f"\"{os.environ['VECTOR_MEM_RESULT_FILE']}\"",
        "FP_REG_RESULT_FILE": f"\"{os.environ.get('FP_REG_RESULT_FILE', '')}\"",
        "HBM_RESULT_FILE": f"\"{os.environ.get('HBM_RESULT_FILE', '')}\"",
    }


def _run_sim():
    """Build once + run the looped cocotb test."""
    skip_build = os.environ.get("SKIP_BUILD", "0") == "1"
    workload_dir = os.environ.get("WORKLOAD_DIR")
    sim_build_dir = Path(workload_dir) if workload_dir else None
    veri_runner(
        group="system",
        module="SimTop",
        test_dir=Path(__file__).parent,
        test_module="SimTop_suite_tb",
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
        trace=os.environ.get("SIMTOP_TRACE", "0") == "1",
        skip_build=skip_build,
        sim_build_dir=sim_build_dir,
    )


def _verify_case(case_dir):
    """Run verify_rtl_sim on one case dir; return (passed, summary_line)."""
    proc = subprocess.run(
        [sys.executable, "-m", "verification.verify_rtl_sim",
         "--workload-dir", str(case_dir)],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    match = None
    for line in out.splitlines():
        if "match rate" in line.lower() or "Match Rate" in line:
            match = line.strip()
    passed = proc.returncode == 0
    return passed, (match or out.strip().splitlines()[-1] if out.strip() else "")


def run_with_suite(root: str):
    """Build once, run all cases, verify, print summary."""
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    cases = manifest["cases"]

    # Build params come from case 0 (time-0 $readmemh is harmless; the loop
    # backdoor-loads every case including the first).
    case0 = Path(cases[0]["dir"])
    os.environ["WORKLOAD_DIR"] = str(root)
    os.environ["SUITE_MANIFEST"] = str(root / "suite_manifest.json")
    platform = PLENATestPlatform.from_workload(str(case0))
    platform.prepare()
    # The result-dump env var is unused by the suite tb (it dumps per case in
    # Python), but veri_runner needs the params populated.
    os.environ.setdefault("VECTOR_MEM_RESULT_FILE", str(root / "_unused_result.mem"))

    _run_sim()

    # Verify each case offline and summarize.
    print("\n" + "=" * 60)
    print("SUITE RESULTS")
    print("=" * 60)
    n_pass = 0
    import shutil
    for c in cases:
        # Restore the per-case dump over the $finish-clobbered vector_result.mem.
        src = Path(c["dir"]) / "suite_vec_result.mem"
        if src.exists():
            shutil.copy(src, Path(c["dir"]) / "vector_result.mem")
        passed, summary = _verify_case(c["dir"])
        n_pass += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {c['name']:24s} {summary}")
    print("=" * 60)
    print(f"  {n_pass}/{len(cases)} cases passed")
    print("=" * 60)
    return n_pass == len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLENA looped suite testbench")
    parser.add_argument("--suite-root", type=str, default="build/test/suite",
                        help="Suite root dir (must contain suite_manifest.json)")
    args = parser.parse_args()
    ok = run_with_suite(args.suite_root)
    sys.exit(0 if ok else 1)
