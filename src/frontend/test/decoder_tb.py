#!/usr/bin/env python3
"""
Decoder Testbench -- Instruction Loss During Stall Transitions

Tests the bug where the decoder's 2-cycle internal pipeline (FIFO read ->
decode_instr_info -> decode_stage_op) causes instructions to be lost when
pipeline_stall fires. The FIFO dequeues 1-2 instructions ahead, and during
stall transitions decode_instr_info is cleared to invalid before decode_stage_op
can consume it.

Test protocol:
  1. Feed sequence: [S_ADDI_INT * N, M_MM, S_ADDI_INT, S_ADDI_INT, M_MM_WO,
                      S_ADDI_INT, S_ADDI_INT, M_MM, S_ADDI_INT, S_ADDI_INT, M_MM_WO]
  2. Assert pipeline_stall when first M_MM appears in decode (dec_m_op=5)
     to simulate the Condition 3 stall
  3. Release stall after N cycles
  4. Monitor dec_m_op -- count how many times MM_IC (5) appears
  5. ASSERT: MM_IC must appear exactly 2 times (both M_MM instructions decoded)
  6. ASSERT: MM_WO must appear exactly 2 times (both M_MM_WO instructions decoded)
"""
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

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

# -- Config ------------------------------------------------------------------
CLOCK_PERIOD_NS = 2
TIMEOUT_CYCLES  = 300

# Instruction encoding parameters (from instruction_pkg)
INSTRUCTION_LENGTH = 32
OPCODE_WIDTH       = 6
OPERAND_WIDTH      = 4

# Opcodes (from operation.svh CUSTOM_ISA_OPCODE)
M_MM       = 0x01
M_MM_WO    = 0x06
S_ADDI_INT = 0x22

# M_OP enum values (from operation.svh)
STALL_M = 0x0
MM_IC   = 0x5
MM_WO   = 0x7


# -- Helpers -----------------------------------------------------------------

def _safe_int(sig):
    """Read a cocotb signal as int; return 0 on error."""
    try:
        return int(sig.value)
    except Exception:
        return 0


def encode_instruction(opcode, rd=0, rs1=0, rs2=0, rstride=0, imm_top=0):
    """
    Pack a 32-bit instruction word.
    Format: [31:22]=IMM_TOP [21:18]=RSTRIDE [17:14]=RS2 [13:10]=RS1 [9:6]=RD [5:0]=OPCODE
    """
    word = (
        ((imm_top & 0x3FF) << 22) |
        ((rstride & 0xF) << 18) |
        ((rs2 & 0xF) << 14) |
        ((rs1 & 0xF) << 10) |
        ((rd & 0xF) << 6) |
        (opcode & 0x3F)
    )
    return word & 0xFFFFFFFF


async def reset_dut(dut):
    """Assert rst for 4 cycles then release; set all inputs to safe defaults."""
    dut.rst.value = 1
    dut.system_stall_flag.value = 0
    dut.pipeline_stall.value = 0
    dut.instruction.value = 0
    dut.instruction_valid.value = 0

    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


async def feed_instructions(dut, instr_list):
    """
    Feed a list of instruction words into the decoder's instruction port
    using valid/ready handshake. Returns the number of cycles spent feeding.
    """
    idx = 0
    cycles = 0
    while idx < len(instr_list):
        dut.instruction.value = instr_list[idx]
        dut.instruction_valid.value = 1
        await RisingEdge(dut.clk)
        cycles += 1
        if _safe_int(dut.instruction_ready):
            idx += 1
    dut.instruction_valid.value = 0
    dut.instruction.value = 0
    return cycles


# -- Test 1: Instruction preservation during stall --------------------------

@cocotb.test()
async def instruction_preservation_during_stall(dut):
    """
    Feed a sequence with M_MM and M_MM_WO instructions interleaved with
    S_ADDI_INT. Assert pipeline_stall at strategic moments and verify that
    ALL matrix instructions eventually appear in dec_m_op output.

    Expected: MM_IC appears exactly 2 times, MM_WO appears exactly 2 times.
    BUG: stall transitions cause decode_instr_info to be cleared, losing
    instructions that were already dequeued from the FIFO but not yet
    latched into decode_stage_op.
    """
    cocotb.log.info("=" * 60)
    cocotb.log.info("Test 1: instruction_preservation_during_stall")
    cocotb.log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    # Build instruction sequence
    instrs = []
    # 10 x S_ADDI_INT (fill pipeline, let decoder warm up)
    for i in range(10):
        instrs.append(encode_instruction(S_ADDI_INT, rd=1, rs1=2, imm_top=i))
    # M_MM #1
    instrs.append(encode_instruction(M_MM, rd=0, rs1=1, rs2=2))
    # 2 x S_ADDI_INT
    instrs.append(encode_instruction(S_ADDI_INT, rd=3, rs1=4, imm_top=100))
    instrs.append(encode_instruction(S_ADDI_INT, rd=3, rs1=4, imm_top=101))
    # M_MM_WO #1
    instrs.append(encode_instruction(M_MM_WO, rd=0, rs1=1, rs2=2))
    # 4 x S_ADDI_INT
    for i in range(4):
        instrs.append(encode_instruction(S_ADDI_INT, rd=5, rs1=6, imm_top=200 + i))
    # M_MM #2
    instrs.append(encode_instruction(M_MM, rd=0, rs1=3, rs2=4))
    # 2 x S_ADDI_INT
    instrs.append(encode_instruction(S_ADDI_INT, rd=7, rs1=8, imm_top=300))
    instrs.append(encode_instruction(S_ADDI_INT, rd=7, rs1=8, imm_top=301))
    # M_MM_WO #2
    instrs.append(encode_instruction(M_MM_WO, rd=0, rs1=3, rs2=4))

    cocotb.log.info(f"Instruction sequence length: {len(instrs)}")

    # Start feeding instructions in background
    feed_task = cocotb.start_soon(feed_instructions(dut, instrs))

    mm_ic_count = 0
    mm_wo_count = 0
    stall_phase = 0  # 0=feeding, 1=first_stall, 2=released, 3=second_stall, 4=done
    stall_counter = 0
    feed_cycles = 0

    for cyc in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)

        m_op = _safe_int(dut.dec_m_op)
        int_op = _safe_int(dut.out_assigned_int_op)

        # Count matrix ops appearing at output
        if m_op == MM_IC:
            mm_ic_count += 1
            cocotb.log.info(f"  cyc={cyc}: MM_IC detected (count={mm_ic_count})")
        if m_op == MM_WO:
            mm_wo_count += 1
            cocotb.log.info(f"  cyc={cyc}: MM_WO detected (count={mm_wo_count})")

        # Stall state machine
        if stall_phase == 0 and cyc >= 5:
            # After 5 cycles of feeding, assert first stall
            dut.pipeline_stall.value = 1
            stall_phase = 1
            stall_counter = 0
            cocotb.log.info(f"  cyc={cyc}: STALL PHASE 1 asserted")

        elif stall_phase == 1:
            stall_counter += 1
            if stall_counter >= 20:
                dut.pipeline_stall.value = 0
                stall_phase = 2
                stall_counter = 0
                cocotb.log.info(f"  cyc={cyc}: STALL PHASE 1 released")

        elif stall_phase == 2:
            stall_counter += 1
            if stall_counter >= 5:
                dut.pipeline_stall.value = 1
                stall_phase = 3
                stall_counter = 0
                cocotb.log.info(f"  cyc={cyc}: STALL PHASE 2 asserted")

        elif stall_phase == 3:
            stall_counter += 1
            if stall_counter >= 20:
                dut.pipeline_stall.value = 0
                stall_phase = 4
                stall_counter = 0
                cocotb.log.info(f"  cyc={cyc}: STALL PHASE 2 released")

        elif stall_phase == 4:
            stall_counter += 1

        # Log every 10 cycles
        if cyc % 10 == 0:
            cocotb.log.info(
                f"cyc={cyc:3d} m_op={m_op} int_op={int_op}"
                f" stall={_safe_int(dut.pipeline_stall)}"
                f" valid={_safe_int(dut.instruction_valid)}"
                f" ready={_safe_int(dut.instruction_ready)}"
                f" mm_ic={mm_ic_count} mm_wo={mm_wo_count}"
            )

        # Early exit once everything drained
        if stall_phase == 4 and stall_counter > 40:
            break

    # Drain any remaining pipeline cycles
    dut.pipeline_stall.value = 0
    for cyc_extra in range(30):
        await RisingEdge(dut.clk)
        m_op = _safe_int(dut.dec_m_op)
        if m_op == MM_IC:
            mm_ic_count += 1
            cocotb.log.info(f"  drain cyc={cyc_extra}: MM_IC detected (count={mm_ic_count})")
        if m_op == MM_WO:
            mm_wo_count += 1
            cocotb.log.info(f"  drain cyc={cyc_extra}: MM_WO detected (count={mm_wo_count})")

    # -- Assertions ----------------------------------------------------------
    cocotb.log.info("=" * 60)
    cocotb.log.info(f"[RESULT] MM_IC count = {mm_ic_count} (expected 2)")
    cocotb.log.info(f"[RESULT] MM_WO count = {mm_wo_count} (expected 2)")
    cocotb.log.info("=" * 60)

    assert mm_ic_count == 2, (
        f"INSTRUCTION LOSS BUG: MM_IC appeared {mm_ic_count} times, expected 2. "
        f"Stall transitions caused instruction loss in decoder pipeline."
    )
    assert mm_wo_count == 2, (
        f"INSTRUCTION LOSS BUG: MM_WO appeared {mm_wo_count} times, expected 2. "
        f"Stall transitions caused instruction loss in decoder pipeline."
    )

    cocotb.log.info("PASSED: All matrix instructions preserved through stall transitions")


# -- Test 2: Stall at exact transition --------------------------------------

@cocotb.test()
async def stall_at_exact_transition(dut):
    """
    Targeted stall-timing test. Feed a short sequence with exactly one M_MM
    and assert pipeline_stall at the exact moment M_MM enters the decoder
    pipeline (1-2 cycles before it would appear in dec_m_op).

    Expected: MM_IC appears exactly 1 time.
    BUG: The instruction is lost during the stall transition because
    decode_instr_info is cleared while pipeline_stall holds.
    """
    cocotb.log.info("=" * 60)
    cocotb.log.info("Test 2: stall_at_exact_transition")
    cocotb.log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    # Build instruction sequence: 3 x S_ADDI_INT, M_MM, 3 x S_ADDI_INT
    instrs = []
    for i in range(3):
        instrs.append(encode_instruction(S_ADDI_INT, rd=1, rs1=2, imm_top=i))
    instrs.append(encode_instruction(M_MM, rd=0, rs1=1, rs2=2))
    for i in range(3):
        instrs.append(encode_instruction(S_ADDI_INT, rd=3, rs1=4, imm_top=50 + i))

    cocotb.log.info(f"Instruction sequence length: {len(instrs)}")

    # Feed all instructions
    feed_task = cocotb.start_soon(feed_instructions(dut, instrs))

    mm_ic_count = 0
    stall_asserted = False
    stall_hold_cycles = 0

    for cyc in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)

        m_op = _safe_int(dut.dec_m_op)

        if m_op == MM_IC:
            mm_ic_count += 1
            cocotb.log.info(f"  cyc={cyc}: MM_IC detected (count={mm_ic_count})")

        # Assert stall after 3 cycles (when M_MM is being read from FIFO
        # but before it reaches decode_stage_op -- the critical window)
        if not stall_asserted and cyc == 3:
            dut.pipeline_stall.value = 1
            stall_asserted = True
            stall_hold_cycles = 0
            cocotb.log.info(f"  cyc={cyc}: pipeline_stall ASSERTED (targeting M_MM transit)")

        # Hold stall for 10 cycles then release
        if stall_asserted and stall_hold_cycles < 10:
            stall_hold_cycles += 1
        elif stall_asserted and stall_hold_cycles == 10:
            dut.pipeline_stall.value = 0
            stall_hold_cycles += 1
            cocotb.log.info(f"  cyc={cyc}: pipeline_stall RELEASED")

        # Log periodically
        if cyc % 5 == 0:
            cocotb.log.info(
                f"cyc={cyc:3d} m_op={m_op}"
                f" stall={_safe_int(dut.pipeline_stall)}"
                f" ready={_safe_int(dut.instruction_ready)}"
                f" mm_ic={mm_ic_count}"
            )

        # Early exit once we've drained enough
        if stall_hold_cycles > 10 and cyc > 40:
            break

    # Let pipeline drain fully
    dut.pipeline_stall.value = 0
    for cyc_extra in range(30):
        await RisingEdge(dut.clk)
        m_op = _safe_int(dut.dec_m_op)
        if m_op == MM_IC:
            mm_ic_count += 1
            cocotb.log.info(f"  drain cyc={cyc_extra}: MM_IC detected (count={mm_ic_count})")

    # -- Assertions ----------------------------------------------------------
    cocotb.log.info("=" * 60)
    cocotb.log.info(f"[RESULT] MM_IC count = {mm_ic_count} (expected 1)")
    cocotb.log.info("=" * 60)

    assert mm_ic_count == 1, (
        f"INSTRUCTION LOSS BUG: MM_IC appeared {mm_ic_count} times, expected 1. "
        f"Stall at exact transition caused M_MM to be lost in decoder pipeline."
    )

    cocotb.log.info("PASSED: M_MM instruction preserved through exact-transition stall")


# -- Runner ------------------------------------------------------------------

@pytest.mark.dev
def test_decoder_stall():
    veri_runner(
        group="frontend",
        module="decoder_tb_wrapper",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=True,
        test_module="decoder_tb",
    )


if __name__ == "__main__":
    test_decoder_stall()
