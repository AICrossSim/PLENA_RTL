#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools"))

import logging
import pytest
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.fp_generation import FpGenerator

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "PLENA_Tools"))
from plena_utils.load_config import load_svh_settings, load_hardware_tile_sizes

_DEFS = SRC_PATH / "definitions"
_TILES = load_hardware_tile_sizes(_DEFS)
_PREC = load_svh_settings(str(_DEFS / "precision.svh"))
VLEN = _TILES["VLEN"]
EXP = _PREC["V_FP_EXP_WIDTH"]
MANT = _PREC["V_FP_MANT_WIDTH"]

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

def pack_v(bus_elems, elem_width):
    """Pack list of element words [0..VLEN-1] into single int bus (lane 0 at LSB)."""
    v = 0
    mask = (1 << elem_width) - 1
    for i, w in enumerate(bus_elems):
        v |= (int(w) & mask) << (i * elem_width)
    return v

def unpack_v(bus, lanes, elem_w):
    mask = (1 << elem_w) - 1
    return [ (bus >> (i * elem_w)) & mask for i in range(lanes) ]

def pretty_list(xs):
    return "[" + ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in xs) + "]"

@cocotb.test()
async def shift_vm_test(dut):
    cocotb.log.info("========== Vector Machine Shift Test ==========")

    ELEM_W = 1 + EXP + MANT
    gen = FpGenerator(EXP, MANT)

    # Clock and reset
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)

    # Defaults
    dut.v_out_ready.value = 1
    dut.broadcast_fp2.value = 0
    dut.reduct_v_control.value = 0  # STALL_V_REDUCT

    ELEM_W = 1 + EXP + MANT

    # Program writeback addr (optional)
    dut.result_waddr.value = 0
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    # Prepare vector A [1..VLEN]
    vals = list(range(1, VLEN + 1))
    v_a_bus = pack_v(vals, ELEM_W)

    # Shift immediate (lanes)
    shift_amount = 2
    dut.s_wtarget.value = shift_amount

    # Issue SHIFT op: SHIFT_V_LANES_ELEMENT = 4'h8 = 8
    SHIFT_OP = 8
    dut.element_v_control.value = SHIFT_OP

    # Drive A for one cycle
    dut.v_a_in.value = v_a_bus
    dut.v_a_valid.value = 1
    await RisingEdge(dut.clk)
    dut.v_a_valid.value = 0

    # Deassert op back to STALL
    dut.element_v_control.value = 0

    # Wait up to N cycles for non-zero v_out
    got_packed = 0
    seen_nonzero = False
    when = -1
    MAX_WAIT = 32
    for t in range(MAX_WAIT):
        await RisingEdge(dut.clk)
        got = int(dut.v_out.value)
        if got != 0 and not seen_nonzero:
            got_packed = got
            seen_nonzero = True
            when = t  # cycles after deasserting v_a_valid
    if not seen_nonzero:
        got_packed = int(dut.v_out.value)

    # Expected shifted lanes
    expected = ([0] * shift_amount) + vals[:VLEN - shift_amount]
    actual = unpack_v(got_packed, VLEN, ELEM_W)

    cocotb.log.info(f"Observed v_out after {when if seen_nonzero else MAX_WAIT} cycles")
    cocotb.log.info(f"v_out (packed)=0x{got_packed:0{(VLEN*ELEM_W+3)//4}X}")
    cocotb.log.info(f"expected lanes: {expected}")
    cocotb.log.info(f"actual   lanes: {actual}")

    assert actual == expected, "Shift output mismatch"


@cocotb.test()
async def add_vv_latency_test(dut):
    """Measure RTL latency for ADD_V_ELEMENT (V_ADD_VV)."""
    import cocotb.utils
    CLOCK_PERIOD_NS = 10  # 10ns clock
    ELEM_W = 1 + EXP + MANT
    gen = FpGenerator(EXP, MANT)
    # VECTOR_ADD_CYCLES = 7 in RTL (configuration.svh, non-DC_LIB_EN mode)
    SIM_PREDICTED_CYCLES = 7

    cocotb.log.info("========== V_ADD_VV Latency Test (RTL vs Simulator) ==========")

    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    dut.rst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)

    # Default signals
    if hasattr(dut, "broadcast_fp2"):    dut.broadcast_fp2.value = 0
    if hasattr(dut, "reduct_v_control"): dut.reduct_v_control.value = 0
    if hasattr(dut, "s_in"):             dut.s_in.value = 0
    if hasattr(dut, "s_in_valid"):       dut.s_in_valid.value = 0
    if hasattr(dut, "result_waddr"):        dut.result_waddr.value = 0
    if hasattr(dut, "result_waddr_update"): dut.result_waddr_update.value = 0
    if hasattr(dut, "v_out_ready"):      dut.v_out_ready.value = 1

    # Set ADD_V_ELEMENT = 4'h1 = 1
    dut.element_v_control.value = 1

    # Build input vectors A=[1..8], B=[1..8]
    vals_a = [float(i + 1) for i in range(VLEN)]
    vals_b = [float(i + 1) for i in range(VLEN)]
    _, enc_a = gen.generate_specified_value_fp_input(vals_a)
    _, enc_b = gen.generate_specified_value_fp_input(vals_b)
    v_a_bus = pack_v(enc_a[:VLEN], ELEM_W)
    v_b_bus = pack_v(enc_b[:VLEN], ELEM_W)

    dut.v_a_in.value = v_a_bus
    if hasattr(dut, "v_b_in"):
        dut.v_b_in.value = v_b_bus

    # Record start time and pulse valid
    start_ns = cocotb.utils.get_sim_time(units='ns')
    if hasattr(dut, "v_a_valid"):
        dut.v_a_valid.value = 1
    if hasattr(dut, "v_b_valid"):
        dut.v_b_valid.value = 1
    await RisingEdge(dut.clk)
    if hasattr(dut, "v_a_valid"):
        dut.v_a_valid.value = 0
    if hasattr(dut, "v_b_valid"):
        dut.v_b_valid.value = 0

    # Wait for element_v_out_valid
    rtl_cycles = None
    for cycle in range(1, 100):
        await RisingEdge(dut.clk)
        if hasattr(dut, "element_v_out_valid") and int(dut.element_v_out_valid.value) == 1:
            end_ns = cocotb.utils.get_sim_time(units='ns')
            rtl_cycles = (end_ns - start_ns) / CLOCK_PERIOD_NS
            break

    if rtl_cycles is not None:
        cocotb.log.info("=" * 60)
        cocotb.log.info(f"[LATENCY] V_ADD_VV (VLEN={VLEN}, EXP={EXP}, MANT={MANT})")
        cocotb.log.info(f"[LATENCY]   RTL measured cycles:      {rtl_cycles:.0f}")
        cocotb.log.info(f"[LATENCY]   Simulator prediction:     {SIM_PREDICTED_CYCLES}")
        cocotb.log.info(f"[LATENCY]   Match: {'YES' if abs(rtl_cycles - SIM_PREDICTED_CYCLES) <= 1 else 'NO (within 1 cycle tolerance)'}")
        cocotb.log.info("=" * 60)
    else:
        cocotb.log.error("[LATENCY] TIMEOUT: element_v_out_valid never asserted")


@cocotb.test()
async def reduct_sum_test(dut):
    """Drive a SUM_V_REDUCT and verify the vector machine emits s_out_valid."""
    cocotb.log.info("========== V_RED_SUM s_out_valid Test ==========")
    cocotb.log.info(f"config from svh: VLEN={VLEN} EXP={EXP} MANT={MANT} (elem {1+EXP+MANT}b)")
    ELEM_W = 1 + EXP + MANT
    gen = FpGenerator(EXP, MANT)

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)

    # Defaults: everything idle. (Outputs have no backpressure; s_out_valid is fire-and-forget.)
    dut.broadcast_fp2.value = 0
    dut.element_v_control.value = 0
    dut.reduct_v_control.value = 0
    dut.v_a_valid.value = 0
    if hasattr(dut, "v_b_valid"):
        dut.v_b_valid.value = 0
    dut.s_in_valid.value = 0
    dut.result_waddr_update.value = 0

    # Program writeback addr.
    dut.result_waddr.value = 0
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    # Inputs: vector A = [1..VLEN], scalar accumulator seed = 0.
    vals_a = [float(i + 1) for i in range(VLEN)]
    _, enc_a = gen.generate_specified_value_fp_input(vals_a)
    v_a_bus = pack_v(enc_a[:VLEN], ELEM_W)
    _, enc_s = gen.generate_specified_value_fp_input([0.0])
    s_seed = int(enc_s[0])

    TARGET_RD = 2  # mimic `V_RED_SUM f2, gp2`
    dut.s_wtarget.value = TARGET_RD

    # Hold valids several cycles so v_port_a_valid / s_acc_in_valid overlap and prepare can fire.
    SUM_V_REDUCT = 1
    dut.reduct_v_control.value = SUM_V_REDUCT
    dut.v_a_in.value = v_a_bus
    dut.v_a_valid.value = 1
    dut.s_in.value = s_seed
    dut.s_in_valid.value = 1
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.v_a_valid.value = 0
    dut.s_in_valid.value = 0
    dut.reduct_v_control.value = 0

    # Watch for the result strobe, tracing the reduction-prepare internals.
    def rd(name):
        try:
            return int(getattr(dut, name).value)
        except Exception:
            return "?"

    INTERNALS = ["recorded_reduct_v_control", "v_port_a_valid", "s_acc_in_valid",
                 "complete_reduct_prepare", "red_push", "red_head_valid", "red_pop",
                 "s_out_valid"]
    fired = False
    when = -1
    s_out_bits = None
    s_out_rd = None
    MAX_WAIT = 64
    for t in range(MAX_WAIT):
        await RisingEdge(dut.clk)
        if t < 24:
            cocotb.log.info(f"  t={t:2d} " + " ".join(f"{n}={rd(n)}" for n in INTERNALS))
        if int(dut.s_out_valid.value) == 1:
            fired = True
            when = t
            s_out_bits = int(dut.s_out.value)
            s_out_rd = int(dut.s_out_rd.value)
            break

    if not fired:
        cocotb.log.error(f"s_out_valid NEVER pulsed within {MAX_WAIT} cycles for SUM reduction")
    else:
        got = gen.custom_fp_to_float(s_out_bits)
        expected = sum(vals_a)
        cocotb.log.info(f"s_out_valid fired after {when} cycles")
        cocotb.log.info(f"  s_out_rd = {s_out_rd} (expected {TARGET_RD})")
        cocotb.log.info(f"  s_out    = {got} (expected ~{expected})")

    assert fired, f"s_out_valid never asserted for SUM_V_REDUCT within {MAX_WAIT} cycles"
    assert s_out_rd == TARGET_RD, f"s_out_rd={s_out_rd}, expected {TARGET_RD}"
    got = gen.custom_fp_to_float(s_out_bits)
    expected = sum(vals_a)
    assert abs(got - expected) <= 0.1 * expected, f"sum mismatch: got {got}, expected {expected}"


@cocotb.test()
async def reduct_stress_test(dut):
    """Stress the reduction path: hammer SUM reductions and verify red_push == s_out_valid (1:1)."""
    cocotb.log.info("========== V_RED_SUM stress test ==========")
    ELEM_W = 1 + EXP + MANT
    gen = FpGenerator(EXP, MANT)
    N = 8

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)

    dut.broadcast_fp2.value = 0
    dut.element_v_control.value = 0
    dut.reduct_v_control.value = 0
    dut.v_a_valid.value = 0
    if hasattr(dut, "v_b_valid"):
        dut.v_b_valid.value = 0
    dut.s_in_valid.value = 0
    dut.result_waddr.value = 0
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    # Count red_push (accepted) and s_out_valid (produced) to tell drop from duplicate.
    sends = [0]
    pushes = [0]

    def rd(name):
        try:
            return int(getattr(dut, name).value)
        except Exception:
            return 0

    async def tick():
        await RisingEdge(dut.clk)
        if int(dut.s_out_valid.value) == 1:
            sends[0] += 1
        if rd("red_push") == 1:
            pushes[0] += 1

    # HAMMER: hold reduction inputs high continuously so red_push fires every cycle prepare is met.
    SUM_V_REDUCT = 1
    dut.reduct_v_control.value = SUM_V_REDUCT
    dut.s_wtarget.value = 2
    HAMMER = 80
    for i in range(HAMMER):
        vals = [float((i + j) % 7 + 1) for j in range(VLEN)]
        _, enc = gen.generate_specified_value_fp_input(vals)
        _, encs = gen.generate_specified_value_fp_input([0.0])
        dut.v_a_in.value = pack_v(enc[:VLEN], ELEM_W)
        dut.s_in.value = int(encs[0])
        dut.v_a_valid.value = 1
        dut.s_in_valid.value = 1
        await tick()
    dut.v_a_valid.value = 0
    dut.s_in_valid.value = 0
    dut.reduct_v_control.value = 0

    # Drain — let every accepted reduction produce its result.
    for _ in range(HAMMER * 50):
        await tick()

    cocotb.log.error(f"STRESS: intended {N} | red_push={pushes[0]} | s_out_valid={sends[0]}")
    assert pushes[0] == sends[0], (
        f"vector machine reduction NOT 1:1: red_push={pushes[0]} but s_out_valid={sends[0]} "
        f"-> {'drops' if pushes[0] > sends[0] else 'duplicates'} reductions under load"
    )


@pytest.mark.dev
def test_vector_machine():
    veri_runner(
        group="vector_machine",
        module="vector_machine",
        additional_include_paths=[
            str(SRC_PATH / "basic_components" / "buffer"),
            str(SRC_PATH / "basic_components" / "common"),
            str(SRC_PATH / "basic_components" / "fp_operation"),
            str(SRC_PATH / "basic_components" / "hadamard_transform"),
            str(SRC_PATH / "basic_components" / "synopsis_ip_inst"),
            str(SRC_PATH / "basic_components" / "conversion"),
            str(SRC_PATH / "basic_components" / "fixed_operation"),
            str(SRC_PATH / "basic_components" / "int_operation"),
            str(SRC_PATH / "basic_components" / "synopsis"),
            str(SRC_PATH / "basic_components" / "cast"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=True,
    )

if __name__ == "__main__":
    test_vector_machine()