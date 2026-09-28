#!/usr/bin/env python3
"""Testbench for fp_reduction_compute_unit (the FP reduction tree / accumulate unit)."""

from pathlib import Path
import sys

import logging
import math
import pytest
import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
from cfl_cocotb import veri_runner, FpGenerator, SRC_PATH

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "PLENA_Tools"))
from plena_utils.load_config import load_svh_settings, load_hardware_tile_sizes

_DEFS = SRC_PATH / "definitions"
_TILES = load_hardware_tile_sizes(_DEFS)
_PREC = load_svh_settings(str(_DEFS / "precision.svh"))

VLEN = _TILES["VLEN"]
EXP = _PREC["V_FP_EXP_WIDTH"]
MANT = _PREC["V_FP_MANT_WIDTH"]

VEC_DIM = VLEN + 1                      # +1 scalar-machine accumulator seed lane
ELEM_W = 1 + EXP + MANT                 # IN_WIDTH; OUT_WIDTH == IN_WIDTH (ACC_EXT=0)
LEVELS = math.ceil(math.log2(VEC_DIM))

# MAX is latency-matched to SUM inside fp_vector_reduce_layer (uniform per-level latency).
SUM_LAYER_LAT = 6
MAX_LAYER_LAT = 6
SUM_LATENCY = 2 + LEVELS * (SUM_LAYER_LAT + 1)
MAX_LATENCY = 2 + LEVELS * (MAX_LAYER_LAT + 1)
LATENCY = max(SUM_LATENCY, MAX_LATENCY)

OP_STALL = 0
OP_SUM = 1
OP_MAX = 2

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

generator = FpGenerator(EXP, MANT)


def pack_vec(values):
    """Pack a list of VEC_DIM python floats into the packed v_in bus word (lane 0 at LSB)."""
    assert len(values) == VEC_DIM, f"need {VEC_DIM} lanes, got {len(values)}"
    _, enc = generator.generate_specified_value_fp_input(values)
    word = 0
    for n in range(VEC_DIM):
        word |= int(enc[n]) << (ELEM_W * n)
    return word


def decode_s_out(dut):
    """Decode the s_out scalar (same EXP/MANT as inputs, ACC_EXT=0)."""
    return generator.custom_fp_to_float(int(dut.s_out.value))


def fp_round(val):
    """Round a python float through the DUT's FP format (encode->decode)."""
    return generator.custom_fp_to_float(generator.float_to_custom_fp(val))


def tree_sum_reference(values):
    """Reference SUM mimicking the hardware pairwise tree reduction order (FP non-associative)."""
    cur = [fp_round(v) for v in values]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(fp_round(cur[i] + cur[i + 1]))
        if len(cur) % 2 == 1:        # odd remainder lane passes through
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def tree_max_reference(values):
    cur = [fp_round(v) for v in values]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(max(cur[i], cur[i + 1]))
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


async def reset_dut(dut):
    """Active-high reset (register_slice clears on rst==1)."""
    dut.rst.value = 1
    dut.v_in_valid.value = 0
    dut.s_out_ready.value = 1
    dut.operation.value = OP_STALL
    dut.v_in.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)


async def drive_single_reduction(dut, values, op):
    """Pulse v_in_valid one cycle, capture the single s_out_valid pulse, return decoded scalar."""
    dut.operation.value = op
    dut.s_out_ready.value = 1
    dut.v_in.value = pack_vec(values)
    dut.v_in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.v_in_valid.value = 0
    dut.v_in.value = 0
    # operation MUST be held for the whole drain (layer valid-mux is gated by live op).

    result = None
    pulses = 0
    for _ in range(LATENCY + 8):
        await RisingEdge(dut.clk)
        if int(dut.s_out_valid.value) == 1:
            pulses += 1
            if result is None:
                result = decode_s_out(dut)
    assert pulses == 1, (
        f"expected exactly 1 s_out_valid pulse for one-cycle input, got {pulses}"
    )
    return result


def tol_for(expected):
    """A few ULP tolerance at the magnitude of `expected` (and >= one ULP near 0)."""
    mag = max(abs(expected), 1.0)
    exp = math.floor(math.log2(mag)) if mag > 0 else 0
    ulp = 2.0 ** (exp - MANT)
    return 4.0 * ulp


@cocotb.test()
async def reduction_latency_test(dut):
    """Measure and ASSERT the pipeline latency (LEVELS+1) for SUM and MAX."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    for op, name, exp_lat in (
        (OP_SUM, "SUM", SUM_LATENCY),
        (OP_MAX, "MAX", MAX_LATENCY),
    ):
        await reset_dut(dut)
        vals = [float(i + 1) for i in range(VEC_DIM)]
        dut.v_in.value = pack_vec(vals)
        dut.operation.value = op
        dut.s_out_ready.value = 1
        dut.v_in_valid.value = 1

        cycles = 0
        fired = False
        for _ in range(LATENCY + 50):
            await RisingEdge(dut.clk)
            cycles += 1
            if int(dut.s_out_valid.value) == 1:
                fired = True
                break
        dut.v_in_valid.value = 0
        dut.operation.value = OP_STALL

        assert fired, f"{name}: s_out_valid never asserted"
        assert cycles == exp_lat, (
            f"{name}: latency {cycles} != expected {exp_lat} "
            f"(1 + LEVELS*(layer_lat+1), LEVELS={LEVELS})"
        )


@cocotb.test()
async def sum_canonical_exact_test(dut):
    """SUM of [1..VLEN] with scalar seed 0; catches the odd-lane-drop bug (136 vs 120)."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    data = [float(i + 1) for i in range(VLEN)]
    values = data + [0.0]                       # lane VLEN = seed
    expected = tree_sum_reference(values)       # FP-tree-ordered reference
    naive = sum(data)

    got = await drive_single_reduction(dut, values, OP_SUM)
    cocotb.log.info(f"[SUM 1..{VLEN}] got={got} ref(tree)={expected} naive={naive}")
    assert got is not None, "no s_out_valid"
    assert abs(got - expected) <= tol_for(expected), (
        f"SUM [1..{VLEN}] mismatch: got {got}, expected ~{expected} "
        f"(naive {naive}); a value far below {naive} indicates a dropped lane "
        f"(the 136->120 bug)"
    )


@cocotb.test()
async def sum_all_equal_test(dut):
    """All VLEN lanes = x, seed 0 -> N*x; a single dropped lane shows as (N-1)*x."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    for x in (1.0, 2.0):
        values = [x] * VLEN + [0.0]
        expected = tree_sum_reference(values)
        got = await drive_single_reduction(dut, values, OP_SUM)
        cocotb.log.info(f"[SUM all={x}] got={got} expected={expected} (N={VLEN})")
        assert abs(got - expected) <= tol_for(expected), (
            f"all-equal SUM x={x}: got {got}, expected {expected}; "
            f"(N-1)*x={ (VLEN - 1) * x} would indicate one lane dropped"
        )


@cocotb.test()
async def sum_single_lane_sweep_test(dut):
    """For each lane: drive it = 2.0, others 0, SUM; localizes which lane is dropped."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    dropped = []
    for lane in range(VEC_DIM):
        await reset_dut(dut)
        values = [0.0] * VEC_DIM
        values[lane] = 2.0
        got = await drive_single_reduction(dut, values, OP_SUM)
        cocotb.log.info(f"[single-lane SUM] lane {lane}: got={got} (expect 2.0)")
        if abs(got - 2.0) > tol_for(2.0):
            dropped.append((lane, got))
    assert not dropped, (
        f"these lanes did not propagate through the SUM tree (dropped): {dropped}; "
        f"a non-empty list localizes the off-by-one lane-drop bug"
    )


@cocotb.test()
async def sum_nonzero_seed_test(dut):
    """The scalar seed lane (index VLEN) must contribute to the sum (accumulator chaining)."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    await reset_dut(dut)
    values = [0.0] * VLEN + [7.0]
    got = await drive_single_reduction(dut, values, OP_SUM)
    cocotb.log.info(f"[seed only] got={got} (expect 7.0)")
    assert abs(got - 7.0) <= tol_for(7.0), (
        f"scalar seed lane (index {VLEN}) dropped: got {got}, expected 7.0; "
        f"breaks accumulator chaining"
    )

    await reset_dut(dut)
    values = [1.0] * VLEN + [8.0]
    expected = tree_sum_reference(values)
    got = await drive_single_reduction(dut, values, OP_SUM)
    cocotb.log.info(f"[data+seed] got={got} expected={expected}")
    assert abs(got - expected) <= tol_for(expected), (
        f"data+seed SUM: got {got}, expected {expected}"
    )


@cocotb.test()
async def sum_endpoint_asymmetry_test(dut):
    """Large value in the FIRST vs LAST data lane; both must sum equal (catches last-lane drop)."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    big = 16.0
    a = [big] + [1.0] * (VLEN - 1) + [0.0]
    b = [1.0] * (VLEN - 1) + [big] + [0.0]
    exp_a = tree_sum_reference(a)
    exp_b = tree_sum_reference(b)

    await reset_dut(dut)
    got_a = await drive_single_reduction(dut, a, OP_SUM)
    await reset_dut(dut)
    got_b = await drive_single_reduction(dut, b, OP_SUM)

    cocotb.log.info(f"[endpoint] firstA={got_a}/{exp_a} lastB={got_b}/{exp_b}")
    assert abs(got_a - exp_a) <= tol_for(exp_a), (
        f"first-lane vector: got {got_a}, expected {exp_a}"
    )
    assert abs(got_b - exp_b) <= tol_for(exp_b), (
        f"last-lane vector: got {got_b}, expected {exp_b}; the big value in the "
        f"LAST data lane was dropped -> last-lane off-by-one"
    )


@cocotb.test()
async def sum_signed_test(dut):
    """Sign-bit and subtraction path through the tree, plus a sign-sensitive last-lane drop check."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    # all-negative [-1..-VLEN]
    await reset_dut(dut)
    neg = [float(-(i + 1)) for i in range(VLEN)] + [0.0]
    exp_neg = tree_sum_reference(neg)
    got = await drive_single_reduction(dut, neg, OP_SUM)
    cocotb.log.info(f"[neg sum] got={got} expected={exp_neg}")
    assert abs(got - exp_neg) <= tol_for(exp_neg), (
        f"all-negative SUM: got {got}, expected {exp_neg}"
    )

    # +4 x (VLEN-1), -4 in the LAST data lane
    await reset_dut(dut)
    vals = [4.0] * (VLEN - 1) + [-4.0] + [0.0]
    exp = tree_sum_reference(vals)              # (VLEN-1)*4 - 4
    got = await drive_single_reduction(dut, vals, OP_SUM)
    cocotb.log.info(f"[signed last] got={got} expected={exp}")
    assert abs(got - exp) <= tol_for(exp), (
        f"signed last-lane SUM: got {got}, expected {exp}; if the -4 last lane is "
        f"dropped the result is {(VLEN - 1) * 4.0}"
    )


# All-zero SUM (+0 not -0)
@cocotb.test()
async def sum_all_zero_test(dut):
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)
    values = [0.0] * VEC_DIM
    dut.operation.value = OP_SUM
    dut.s_out_ready.value = 1
    dut.v_in.value = pack_vec(values)
    dut.v_in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.v_in_valid.value = 0
    got = None
    raw = None
    for _ in range(LATENCY + 8):
        await RisingEdge(dut.clk)
        if int(dut.s_out_valid.value) == 1:
            got = decode_s_out(dut)
            raw = int(dut.s_out.value)
            break
    assert got is not None, "no s_out_valid for all-zero SUM"
    assert got == 0.0, f"all-zero SUM != 0: got {got}"
    sign_bit = (raw >> (EXP + MANT)) & 0x1
    assert sign_bit == 0, f"all-zero SUM produced -0 (sign bit set), raw=0x{raw:x}"


# MAX: positive, negative, tie, max-in-last-lane (odd-lane drop for MAX)
@cocotb.test()
async def max_value_test(dut):
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    # (a) all-positive distinct, max somewhere in the middle
    await reset_dut(dut)
    a = [1.0, 9.0, 3.0, 7.0] + [2.0] * (VLEN - 4) + [0.0]
    got = await drive_single_reduction(dut, a, OP_MAX)
    assert abs(got - 9.0) <= tol_for(9.0), f"MAX(a): got {got}, expected 9.0"

    # (b) max sits in the LAST data lane (index VLEN-1) -> catches MAX odd-drop
    await reset_dut(dut)
    b = [1.0] * (VLEN - 1) + [13.0] + [0.0]
    got = await drive_single_reduction(dut, b, OP_MAX)
    assert abs(got - 13.0) <= tol_for(13.0), (
        f"MAX with max in LAST lane: got {got}, expected 13.0; last-lane dropped"
    )

    # (c) all-negative -> the least-negative wins
    await reset_dut(dut)
    c = [-5.0, -1.0, -9.0, -2.0] + [-8.0] * (VLEN - 4) + [-100.0]
    exp_c = max(c)
    got = await drive_single_reduction(dut, c, OP_MAX)
    assert abs(got - exp_c) <= tol_for(exp_c), f"MAX(neg): got {got}, expected {exp_c}"

    # (d) all equal
    await reset_dut(dut)
    d = [7.0] * VEC_DIM
    got = await drive_single_reduction(dut, d, OP_MAX)
    assert abs(got - 7.0) <= tol_for(7.0), f"MAX(all 7): got {got}, expected 7.0"


@cocotb.test()
async def stall_op_no_output_test(dut):
    """STALL op with v_in_valid high must NOT emit s_out_valid; a following SUM must still work."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    dut.operation.value = OP_STALL
    dut.s_out_ready.value = 1
    dut.v_in.value = pack_vec([float(i + 1) for i in range(VEC_DIM)])
    dut.v_in_valid.value = 1
    for _ in range(LATENCY + 6):
        await RisingEdge(dut.clk)
        assert int(dut.s_out_valid.value) == 0, "STALL op produced an output"
    dut.v_in_valid.value = 0

    # Now a real SUM must work.
    await reset_dut(dut)
    values = [1.0] * VLEN + [0.0]
    expected = tree_sum_reference(values)
    got = await drive_single_reduction(dut, values, OP_SUM)
    assert abs(got - expected) <= tol_for(expected), (
        f"SUM after STALL: got {got}, expected {expected}"
    )


@cocotb.test()
async def single_pulse_one_output_test(dut):
    """A one-cycle v_in_valid must yield exactly one s_out_valid (no drop, no duplicate)."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)
    values = [float(i + 1) for i in range(VLEN)] + [0.0]
    expected = tree_sum_reference(values)
    got = await drive_single_reduction(dut, values, OP_SUM)
    assert abs(got - expected) <= tol_for(expected), (
        f"single-pulse SUM: got {got}, expected {expected}"
    )


@cocotb.test()
async def streaming_throughput_test(dut):
    """Stream N distinct vectors on consecutive cycles; exactly N outputs must appear in order."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    N = 32
    dut.operation.value = OP_SUM
    dut.s_out_ready.value = 1

    inputs = []
    expected = []
    for k in range(1, N + 1):
        x = float(k)
        vec = [x] * VLEN + [0.0]
        inputs.append(vec)
        expected.append(tree_sum_reference(vec))

    captured = []

    async def driver():
        for vec in inputs:
            dut.v_in.value = pack_vec(vec)
            dut.v_in_valid.value = 1
            await RisingEdge(dut.clk)
        dut.v_in_valid.value = 0
        dut.v_in.value = 0
        # Keep operation = SUM held through the drain (layer valid-mux is gated by live op).

    drv = cocotb.start_soon(driver())
    for _ in range(N + LATENCY + 16):
        await RisingEdge(dut.clk)
        if int(dut.s_out_valid.value) == 1:
            captured.append(decode_s_out(dut))
    await drv

    assert len(captured) == N, (
        f"streaming: count(s_out_valid)={len(captured)} != count(v_in_valid)={N}; "
        f"{'dropped' if len(captured) < N else 'duplicated'} reductions under load"
    )
    for k, (got, exp) in enumerate(zip(captured, expected)):
        assert abs(got - exp) <= tol_for(exp), (
            f"streaming vector #{k}: got {got}, expected {exp} (in-order mismatch)"
        )


@cocotb.test()
async def sparse_valid_pattern_test(dut):
    """A bubbled input-valid pattern must emerge as the same pattern delayed by LATENCY."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    pattern = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    dut.operation.value = OP_SUM
    dut.s_out_ready.value = 1

    in_valids = []
    out_valids = []
    k = 1
    drive_cycles = len(pattern)
    total = drive_cycles + LATENCY + 8

    for c in range(total):
        v = pattern[c] if c < drive_cycles else 0
        if v:
            dut.v_in.value = pack_vec([float(k)] * VLEN + [0.0])
            k += 1
        dut.v_in_valid.value = v
        in_valids.append(v)
        await RisingEdge(dut.clk)
        out_valids.append(int(dut.s_out_valid.value))
    dut.v_in_valid.value = 0

    n_in = sum(pattern)
    n_out = sum(out_valids)
    assert n_out == n_in, (
        f"sparse valid: {n_out} outputs for {n_in} inputs (bubbles mishandled)"
    )
    # Output pattern must equal input pattern shifted by LATENCY (allow +/-1 sampling skew).
    for i, iv in enumerate(in_valids):
        if iv:
            oi = i + LATENCY
            window = [oj for oj in (oi - 1, oi, oi + 1) if 0 <= oj < len(out_valids)]
            assert any(out_valids[oj] == 1 for oj in window), (
                f"input valid at cycle {i} did not appear near output cycle {oi} "
                f"(pipeline depth {LATENCY})"
            )


@cocotb.test()
async def reset_midstream_test(dut):
    """rst mid-flight must flush the pipeline (no phantom output), and a fresh op still works."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    await reset_dut(dut)

    # Launch a SUM, let it advance partway into the tree.
    dut.operation.value = OP_SUM
    dut.s_out_ready.value = 1
    dut.v_in.value = pack_vec([float(i + 1) for i in range(VLEN)] + [0.0])
    dut.v_in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.v_in_valid.value = 0
    # advance ~2 cycles (data mid-tree) then reset
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    dut.rst.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst.value = 0

    # After reset, drive nothing; assert no phantom output for the aborted op.
    for _ in range(LATENCY + 8):
        await RisingEdge(dut.clk)
        assert int(dut.s_out_valid.value) == 0, (
            "phantom s_out_valid after mid-stream reset (pipeline not flushed)"
        )

    # Fresh op must work and produce exactly one pulse.
    values = [2.0] * VLEN + [0.0]
    expected = tree_sum_reference(values)
    got = await drive_single_reduction(dut, values, OP_SUM)
    assert abs(got - expected) <= tol_for(expected), (
        f"SUM after mid-stream reset: got {got}, expected {expected}"
    )


async def _run_feed(dut, feed, op, sout_ready_pattern=None):
    """Drive v_in_valid per `feed` (0/1) holding operation=op; returns (n_in, n_out)."""
    await reset_dut(dut)
    dut.operation.value = op
    dut.s_out_ready.value = 1

    n_in = 0
    n_out = 0
    k = 1
    total = len(feed) + LATENCY + 12
    for c in range(total):
        v = feed[c] if c < len(feed) else 0
        if sout_ready_pattern is not None:
            dut.s_out_ready.value = (
                sout_ready_pattern[c] if c < len(sout_ready_pattern) else 1
            )
        if v:
            dut.v_in.value = pack_vec([float(k % 7 + 1)] * VLEN + [0.0])
            k += 1
            n_in += 1
        dut.v_in_valid.value = v
        await RisingEdge(dut.clk)
        if int(dut.s_out_valid.value) == 1:
            n_out += 1
    dut.v_in_valid.value = 0
    return n_in, n_out


@cocotb.test()
async def adversarial_feed_sweep_test(dut):
    """Sweep feed timings with operation held constant (SUM/MAX); each must be exactly 1:1."""
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    import random
    random.seed(0xC0FFEE)

    patterns = {}
    patterns["continuous_40"] = [1] * 40
    patterns["single_pulses"] = sum(([1, 0, 0, 0, 0] for _ in range(12)), [])
    patterns["burst3_gap2"] = sum(([1, 1, 1, 0, 0] for _ in range(12)), [])
    patterns["burst5_gap1"] = sum(([1, 1, 1, 1, 1, 0] for _ in range(10)), [])
    patterns["gap1"] = sum(([1, 0] for _ in range(24)), [])
    # Valids exactly SUM_LATENCY apart (result emerges as next is fed)
    p = []
    for _ in range(8):
        p.append(1)
        p += [0] * (SUM_LATENCY - 1)
    patterns["exactly_latency_apart"] = p
    patterns["random_gaps"] = [random.randint(0, 1) for _ in range(80)]
    patterns["bursty_random"] = [
        1 if random.random() < 0.6 else 0 for _ in range(80)
    ]

    # s_out_ready stress: prove the compute ignores it (no backpressure path).
    ready_low = {}
    ready_low["ready_low_all"] = ([0] * 200)
    ready_low["ready_low_window"] = ([1] * 8 + [0] * 20 + [1] * 200)

    failures = []
    for op, opname in ((OP_SUM, "SUM"), (OP_MAX, "MAX")):
        for name, feed in patterns.items():
            n_in, n_out = await _run_feed(dut, feed, op)
            cocotb.log.info(f"[feed {opname} {name}] in={n_in} out={n_out}")
            if n_in != n_out:
                failures.append((opname, name, n_in, n_out, "ready=1"))
        # s_out_ready stress with a continuous feed
        for rname, rp in ready_low.items():
            n_in, n_out = await _run_feed(
                dut, [1] * 40, op, sout_ready_pattern=rp
            )
            cocotb.log.info(
                f"[feed {opname} continuous + {rname}] in={n_in} out={n_out}"
            )
            if n_in != n_out:
                failures.append((opname, rname, n_in, n_out, "ready_stress"))

    assert not failures, (
        "constant-op feed sweep dropped reductions (should be 1:1): " + repr(failures)
    )


# Op-change-under-flight: interleave SUM/MAX so op switches mid-tree (full-system drop reproducer).
@cocotb.test()
async def op_change_under_flight_test(dut):
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    # Each issue is (op, gap_after); switches land while prior reductions are mid-tree.
    schedules = [
        [(OP_SUM, 1), (OP_MAX, 1), (OP_SUM, 1), (OP_MAX, 1),
         (OP_SUM, 1), (OP_MAX, 1)],
        [(OP_SUM, 3), (OP_MAX, 3)] * 6,
        [(OP_SUM, 1)] * 6 + [(OP_MAX, 1)] * 6,
        [(OP_MAX, 2), (OP_SUM, 2)] * 6,
    ]

    failures = []
    for idx, sched in enumerate(schedules):
        await reset_dut(dut)
        dut.s_out_ready.value = 1
        n_in = 0
        n_out = 0
        k = 1

        async def driver(sched_local):
            nonlocal n_in, k
            for op, gap in sched_local:
                dut.operation.value = op
                dut.v_in.value = pack_vec([float(k % 5 + 1)] * VLEN + [0.0])
                k += 1
                dut.v_in_valid.value = 1
                n_in += 1
                await RisingEdge(dut.clk)
                dut.v_in_valid.value = 0
                # Hold the last issued op during the gap (recorded_reduct_v_control is sticky).
                for _ in range(gap):
                    await RisingEdge(dut.clk)

        drv = cocotb.start_soon(driver(sched))
        for _ in range(len(sched) * 4 + SUM_LATENCY + 20):
            await RisingEdge(dut.clk)
            if int(dut.s_out_valid.value) == 1:
                n_out += 1
        await drv
        dut.v_in_valid.value = 0
        cocotb.log.info(f"[op-change sched {idx}] in={n_in} out={n_out}")
        if n_in != n_out:
            failures.append((idx, n_in, n_out))

    assert not failures, (
        "op-change-under-flight DROPPED reductions (count out != in): "
        + repr(failures)
        + " -- the layer valid-mux is gated by the live `operation`; in-flight "
          "reductions are lost when op switches mid-tree."
    )


def _runner(param_list):
    veri_runner(
        group="vector_machine",
        module="fp_reduction_compute_unit",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        module_param_list=param_list,
        trace=True,
        test_module="reduction_compute_unit_tb",
    )


@pytest.mark.dev
def test_reduction_production():
    """Elaborate at the production geometry loaded from the .svh files."""
    _runner([{"EXP_WIDTH": EXP, "MANT_WIDTH": MANT, "VLEN": VLEN}])


if __name__ == "__main__":
    test_reduction_production()
