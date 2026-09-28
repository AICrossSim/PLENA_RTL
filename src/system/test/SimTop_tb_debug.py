#!/usr/bin/env python3
"""PLENA System Top-Level RTL Testbench.

This script provides the cocotb testbench for PLENA system simulation.
It uses pre-generated workload files from tools/testworkloads.

Usage:
    # Generate workload first
    python -m tools.testworkloads.linear --batch 8 --in-features 128 --out-features 256

    # Run simulation
    just test-linear 8 128 256
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pytest
import cocotb
import torch
from cocotb.log import SimLog
from cocotb.triggers import Timer, RisingEdge, First

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

    Drives the PLENA system with instructions and monitors output.
    Uses pre-generated workload files from PLENATestPlatform.
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

        cocotb.start_soon(self.trace_silu())

        # Execution clock counter state
        self.execution_clocks = 0
        self.c_break_detected = False

        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))

    def _diag(self, msg):
        with open(Path(os.environ["WORKLOAD_DIR"]) / "silu_diag.log", "a") as f:
            f.write(msg + "\n")

    async def trace_pc_hang(self):
        """Sample the decoder PC every N cycles. When the PC stops making forward
        progress (max PC over a window stops rising = hang), log the stuck PC
        range and the loop-control state so we can locate the hang in the asm."""
        try:
            dec = self.dut.dut.decoder_init
        except AttributeError:
            logger.warning("[PCH] decoder_init not found")
            return

        def g(obj, name, default=-1):
            try:
                return int(getattr(obj, name).value)
            except Exception:
                return default

        WIN = 5000          # cycles per progress window
        cyc = 0
        win_min = win_max = None
        prev_win_max = -1
        stuck_windows = 0
        while True:
            await RisingEdge(self.dut.clk)
            cyc += 1
            pc = g(dec, 'pc_reg')
            if win_min is None or pc < win_min:
                win_min = pc
            if win_max is None or pc > win_max:
                win_max = pc
            if cyc % WIN == 0:
                logger.info(
                    f"[PCH] c{cyc} pc_win=[{win_min},{win_max}] "
                    f"exp={g(dec,'expected_decode_pc')} "
                    f"loop_exit_pc={g(dec,'loop_exit_pc')} "
                    f"cont_pc={g(dec,'loop_continue_pc')} body_pc={g(dec,'loop_body_pc')}"
                )
                if win_max <= prev_win_max:      # no forward progress this window
                    stuck_windows += 1
                    if stuck_windows >= 2:
                        logger.error(
                            f"[PCH] HANG: pc stuck in [{win_min},{win_max}] "
                            f"(instr idx {win_min//4}..{win_max//4}) at c{cyc} "
                            f"exp={g(dec,'expected_decode_pc')}"
                        )
                else:
                    stuck_windows = 0
                prev_win_max = win_max
                win_min = win_max = pc

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
        drain_cycles = 32
        logger.info(f"Draining pipeline for {drain_cycles} cycles after C_BREAK")
        for _ in range(drain_cycles):
            await RisingEdge(self.dut.clk)

    async def check_vector_sram(self):
        """Monitor vector SRAM output."""
        while True:
            await RisingEdge(self.dut.clk)
            try:
                v_out_valid = self.dut.dut.vector_machine_init.element_v_out_valid.value
                v_out_ready = self.dut.dut.vector_machine_init.element_v_out_ready.value

                if v_out_valid == 1 and v_out_ready == 1:
                    data = self.dut.dut.vector_machine_init.element_v_out.value
                    list_ = []
                    for i in range(0, len(data), 16):
                        list_.append(data[i:i+15].integer)
                    self.log.debug(f"Vector Core fp_out: {list_}")
            except AttributeError:
                pass

    async def trace_dfc_vload(self):
        """Per-cycle trace of the v-for-matrix load FSM in data_flow_control
        around the M_MM iterations where one activation group goes missing."""
        try:
            dfc = self.dut.dut.data_flow_init
        except AttributeError:
            logger.warning("[DFCV] data_flow_init not found")
            return
        cyc = 0
        logged = 0
        while logged < 400:
            await RisingEdge(self.dut.clk)
            cyc += 1
            def g(name, default=-1):
                try:
                    return int(getattr(dfc, name).value)
                except Exception:
                    return default
            if not (g('continuous_load_v_for_matrix_en') == 1 or g('m_v_load') == 1 or g('m_v_valid') == 1
                    or g('continuous_load_m_en') == 1 or g('m_m_load') == 1 or g('m_m_valid') == 1):
                continue
            logged += 1
            logger.info(
                f"[DFCV] c{cyc}"
                f" en={g('continuous_load_v_for_matrix_en')}"
                f" cnt={g('v_sram_load_for_matrix_counter')}"
                f" eol={g('end_of_load_v_for_matrix')}"
                f" mvl={g('m_v_load')} mvlc={g('m_v_load_cond')}"
                f" p1={g('p1_vport_a_load_valid')} p2={g('p2_vport_a_load_valid')}"
                f" mvv={g('m_v_valid')}"
                f" perm={g('permit_load_for_m')} mrdy={g('matrix_related_data_ready')}"
                f" rec={g('recorded_v_load_for_matrix_addr')}"
                f" addrA={g('v_sram_addr_a')} reqA={g('v_sram_req_a')}"
                f" menl={g('continuous_load_m_en')} eolm={g('end_of_load_m')}"
                f" mcnt={g('m_sram_load_counter')}"
                f" mmv={g('m_m_valid')}"
            )

    async def trace_dispatch(self):
        """Per-cycle trace of decoder dispatch + scalar regfile writes around
        the stall-release window where MM#7's PASS_ADDR loses to the trailing
        S_ADDI write (cycles ~1570-1660, v-load #7 starts ~c1640)."""
        try:
            dec = self.dut.dut.decoder_init
            sm = self.dut.dut.scalar_machine_init
            pc = self.dut.dut.pipeline_control_init
        except AttributeError:
            logger.warning("[DISP] instances not found")
            return
        cyc = 0
        while cyc < 2450:
            await RisingEdge(self.dut.clk)
            cyc += 1
            if cyc < 2100:
                continue

            def g(inst, name, default=-1):
                try:
                    return int(getattr(inst, name).value)
                except Exception:
                    return default

            wen = g(sm, 'gp_reg_wen')
            wline = f" W:gp{g(sm,'gp_reg_waddr')}={g(sm,'gp_reg_wdata')}" if wen == 1 else ""
            logger.info(
                f"[DISP] c{cyc}"
                f" stall={g(dec,'pipeline_stall')}"
                f" pc={g(dec,'pc_reg')//4} pcd={g(dec,'pc_reg_d1')//4}"
                f" ri={g(dec,'read_instr')}"
                f" opc={g(dec,'loaded_opcode'):2d}"
                f" dvalid={g(dec,'decode_instr_valid')}"
                f" aint={g(dec,'assigned_int_op')}"
                f" eint={g(dec,'exe_int_op')}"
                f" pend={g(dec,'int_op_pending')}"
                f" dadv={g(dec,'decode_advanced_d')}"
                f" rs1={g(dec,'rs1')} rs2={g(dec,'rs2')}"
                f" egp={g(sm,'exe_gp_op')}"
                f" go1={g(sm,'gp_out_1')} go2={g(sm,'gp_out_2')}"
                f" rec1={g(pc,'recorded_gp_addr_1')}"
                f" p2rec={g(pc,'p2_recover_from_stall')}"
                f"{wline}"
            )

    async def trace_fetch(self):
        """Per-cycle trace of the imem->decoder fetch handshake around the
        startup buffer refills (cycles 2150-2300): tag/data/ready vs
        expected_decode_pc and the accept (fresh_fetch) decision."""
        try:
            dec = self.dut.dut.decoder_init
            imem = self.dut.dut.instr_mem_inst
        except AttributeError:
            logger.warning("[FET] instances not found")
            return
        cyc = 0
        while cyc < 2300:
            await RisingEdge(self.dut.clk)
            cyc += 1
            if cyc < 2150:
                continue

            def g(inst, name, default=-1):
                try:
                    return int(getattr(inst, name).value)
                except Exception:
                    return default

            logger.info(
                f"[FET] c{cyc}"
                f" pc={g(dec,'pc_reg')//4}"
                f" exp={g(dec,'expected_decode_pc')//4}"
                f" tag={g(dec,'instruction_addr_from_imem')//4}"
                f" rdy={g(dec,'load_instr_valid')}"
                f" ff={g(dec,'fresh_fetch')}"
                f" ri={g(dec,'read_instr')}"
                f" opc={g(dec,'loaded_opcode'):2d}"
                f" skip={g(dec,'fetch_skipped_ahead')}"
                f" | base={g(imem,'buffer_base_addr')//4}"
                f" bv={g(imem,'buffer_valid')}"
                f" idx={g(imem,'buffer_index')}"
                f" inrange={g(imem,'pc_in_range')}"
                f" st={g(imem,'state')}"
            )

    async def trace_v_writes(self):
        """Event log of every vector-path port-A write into VRAM plus the
        v_write_request/v_write_addr handshake and f2 (fp reg) updates -
        to catch the rms_norm row-0 corruption (a V op writing batch-7 data
        with a stale f2 to address 0)."""
        try:
            dfc = self.dut.dut.data_flow_init
            sm = self.dut.dut.scalar_machine_init
        except AttributeError:
            logger.warning("[VWR] instances not found")
            return

        def g(inst, name, default=-1):
            try:
                return int(getattr(inst, name).value)
            except Exception:
                return default

        def f12(bits):
            s = (bits >> 11) & 1
            e = (bits >> 5) & 63
            m = bits & 31
            v = (m / 32) * 2.0 ** (1 - 31) if e == 0 else (1 + m / 32) * 2.0 ** (e - 31)
            return -v if s else v

        cyc = 0
        logged = 0
        prev_f2 = None
        while logged < 300:
            await RisingEdge(self.dut.clk)
            cyc += 1
            try:
                f2raw = int(sm.fp_reg_file.mem[2].value)
            except Exception:
                f2raw = -1
            if f2raw != prev_f2 and f2raw >= 0:
                logger.info(f"[VWR] c{cyc} f2 -> {f12(f2raw):.4f} (raw={f2raw:#x})")
                prev_f2 = f2raw
                logged += 1
            vreq = g(dfc, 'v_write_request')
            wen = g(dfc, 'v_sram_wen_a')
            sel = g(dfc, 'select_write_data_a')
            if vreq == 1:
                logger.info(f"[VWR] c{cyc} v_write_request addr={g(dfc,'v_write_addr')}")
                logged += 1
            if wen == 1 and sel == 0:
                logger.info(
                    f"[VWR] c{cyc} VWRITE addrA={g(dfc,'v_sram_addr_a')}"
                    f" rec={g(dfc,'recorded_v_write_addr')}"
                    f" mask={g(dfc,'v_sram_mask_a'):#x}")
                logged += 1

    async def trace_silu(self):
        """Event log of the vector-element op path (SiLU chain) + the loop
        counter, to see whether the C_LOOP iterates and which V op produces 0."""
        try:
            vm = self.dut.dut.vector_machine_init
            sm = self.dut.dut.scalar_machine_init
        except AttributeError:
            logger.warning("[SILU] instances not found")
            return

        VE = 6  # V_FP_EXP_WIDTH, V_FP_MANT_WIDTH for V FP12 (1+6+5)
        VM = 5

        def vfp(bits):
            s = (bits >> (VE + VM)) & 1
            e = (bits >> VM) & ((1 << VE) - 1)
            m = bits & ((1 << VM) - 1)
            bias = (1 << (VE - 1)) - 1
            v = (m / 2**VM) * 2.0 ** (1 - bias) if e == 0 else (1 + m / 2**VM) * 2.0 ** (e - bias)
            return -v if s else v

        # V_ELEMENT_OP encoding (operation.svh): decode names lazily.
        ELEM = {0: "STALL", 1: "ADD", 2: "SUB", 3: "MUL", 4: "EXP", 5: "RECI"}

        def g(inst, name, default=-1):
            try:
                return int(getattr(inst, name).value)
            except Exception:
                return default

        def lane0(inst, name):
            try:
                bv = getattr(inst, name).value
                raw = int(bv)
                w = len(bv) // 16
                return vfp(raw & ((1 << w) - 1))
            except Exception:
                return None

        def alllanes(inst, name):
            try:
                bv = getattr(inst, name).value
                raw = int(bv)
                w = len(bv) // 16
                return [round(vfp((raw >> (k * w)) & ((1 << w) - 1)), 3) for k in range(16)]
            except Exception:
                return None

        self._lane_logged = 0

        # Discover the hierarchy once (pipeline_control_init wasn't traversable
        # via the obvious path; find its real handle name).
        try:
            kids = [getattr(c, "_name", str(c)) for c in self.dut.dut]
            self._diag("dut.dut children: " + ", ".join(sorted(kids)))
        except Exception as e:
            self._diag(f"child-iter err: {e}")
        pc = None
        am = None
        for cand in ("pipeline_control_init", "pipeline_control_inst", "pipeline_control"):
            try:
                pc = getattr(self.dut.dut, cand)
                self._diag(f"found pc handle: {cand}")
                break
            except AttributeError:
                continue
        if pc is not None:
            for cand in ("addr_monitor_inst", "addr_monitor_init", "addr_monitor"):
                try:
                    am = getattr(pc, cand)
                    self._diag(f"found am handle: {cand}")
                    break
                except AttributeError:
                    continue
        cyc = 0
        logged = 0
        prev_loop = None
        # Log only cycles where the element chain is ACTIVE (op recorded, input
        # valid, output valid, or a write fires) so we capture the chain wherever
        # it lands in the run — no hardcoded window, since exp REG_N shifts it.
        while logged < 9000:
            await RisingEdge(self.dut.clk)
            cyc += 1
            ctrl = g(vm, "recorded_element_v_control")
            outv = g(vm, "element_v_out_valid")
            wreq = g(vm, "v_wreq")
            ina = g(vm, "element_v_in_a_valid")
            try:
                loop3 = int(sm.gp_reg_file.mem[3].value)
            except Exception:
                loop3 = -1
            try:
                gp1v = int(sm.gp_reg_file.mem[1].value)
                gp2v = int(sm.gp_reg_file.mem[2].value)
            except Exception:
                gp1v = gp2v = -1
            if loop3 != prev_loop and loop3 >= 0:
                self._diag(f"c{cyc} ---- loop gp3 -> {loop3}  gp1={gp1v} gp2={gp2v}")
                prev_loop = loop3
            # Loop-control event probe: log the decoder's loop PCs whenever a
            # loop exit or jump-back fires. Reveals whether the silu loop's exit
            # PC (loop_exit_pc/loop_continue_pc) is correct (= C_BREAK) at high
            # instruction addresses, or wrongly points back into the matmul.
            dec = self.dut.dut.decoder_init
            lex = g(dec, "loop_exit")
            ljb = g(dec, "loop_jump_back")
            if lex == 1 or (ljb == 1 and (cyc % 200 < 2)):
                self._diag(
                    f"LP c{cyc} exit={lex} jb={ljb}"
                    f" exit_pc={g(dec,'loop_exit_pc')} cont_pc={g(dec,'loop_continue_pc')}"
                    f" body_pc={g(dec,'loop_body_pc')} target={g(dec,'loop_target_pc')}"
                    f" pc={g(self.dut.dut,'decoder_pc')} pc_d1={g(dec,'pc_reg_d1')}"
                    f" srs={g(dec,'scalar_raw_stall')} gp3={loop3}")
            active = (ctrl not in (-1, 0)) or outv == 1 or wreq == 1 or ina == 1
            if not active:
                continue
            # pipeline_control serialization state (fix36)
            ebusy = g(pc, "v_elem_busy") if pc is not None else -2
            einexe = g(pc, "v_elem_in_exe") if pc is not None else -2
            pstall = g(pc, "pipeline_stall") if pc is not None else -2
            wena = g(self.dut.dut, "v_sram_wen_a")
            # Write-address capture path: result_waddr = exe_stage_op.addr_2,
            # latched into vm.recorded_result_waddr when exe_stage_op.update_v_waddr.
            rrw = g(vm, "recorded_result_waddr")
            # Scalar address outputs (gp_out_1/2 = PASS_ADDR reads) + the stall
            # snapshot/recover machinery in pipeline_control. These show whether
            # gp_out_2 leads the stalled element op by one instruction.
            dfc = self.dut.dut.data_flow_init
            rla1 = g(dfc, "recorded_v_load_addr_1")
            rla2 = g(dfc, "recorded_v_load_addr_2")
            rvw = g(dfc, "recorded_v_write_addr")
            p2r = g(pc, "p2_recover_from_stall") if pc is not None else -2
            a0 = lane0(vm, "element_in_v_a")
            o0 = lane0(vm, "element_v_out")
            opn = ELEM.get(ctrl, ctrl) if ctrl not in (-1, 0) else "."
            # All-16-lane dump on valid input/output, first ~24 events, to detect
            # a lane shift in the scratch read/write at this (failing) config.
            if self._lane_logged < 24 and (ina == 1 or outv == 1):
                self._diag(
                    f"LANES c{cyc} op={opn} ina={ina} outv={outv}\n"
                    f"   in_a ={alllanes(vm, 'element_in_v_a')}\n"
                    f"   out  ={alllanes(vm, 'element_v_out')}")
                self._lane_logged += 1
            self._diag(
                f"c{cyc} | op={opn} ina={ina} a0={a0} outv={outv} out0={o0}"
                f" | busy={ebusy} inexe={einexe} stall={pstall} wena={wena}"
                f" | rla1={rla1} rla2={rla2} rvw={rvw} p2r={p2r} rec_waddr={rrw} gp1={gp1v} gp2={gp2v}"
                + (f" WRITE@{g(vm,'v_waddr')}=>{lane0(vm,'v_out')}" if wreq == 1 else ""))
            logged += 1

    async def check_mm_operands(self):
        """Decode the MCU operands v1 (TOP=b/weight) and v2 (LEFT=a/act) on the
        first few MM_IC cycles, so we can check whether M_MM reads a row-major
        (K-vector per row) and b column-major (K-vector per column)."""
        try:
            mcu = self.dut.dut.matrix_machine_init.gen_mxint_systolic_mcu.matrix_compute_unit
        except AttributeError:
            logger.warning("[MM] matrix_compute_unit not found")
            return
        MLEN = 16

        def unpack_signed(handle, n):
            bv = handle.value
            try:
                raw = int(bv)
            except (ValueError, TypeError):
                return None
            w = len(bv) // n
            out = []
            for k in range(n):
                v = (raw >> (k * w)) & ((1 << w) - 1)
                if v >> (w - 1):
                    v -= 1 << w
                out.append(v)
            return out

        n1 = 0
        n2 = 0
        while n1 < 40 or n2 < 40:
            await RisingEdge(self.dut.clk)
            try:
                v1v = int(mcu.v1_in_valid.value)
                v2v = int(mcu.v2_in_valid.value)
            except (AttributeError, ValueError):
                continue
            if v1v and n1 < 40:
                v1 = unpack_signed(mcu.v1_element, MLEN)
                if v1 and not any(v1):
                    logger.info(f"[MM] v1#{n1} ZERO beat")
                    n1 += 1
                elif v1:
                    logger.info(f"[MM] v1#{n1} (TOP/b weight)  int={v1}")
                    try:
                        sraw = int(mcu.v1_scale.value)
                        scales = [(sraw >> (k * 8)) & 0xFF for k in range(MLEN)]
                        logger.info(f"[MM] v1#{n1} scales(biased)={scales}")
                    except (AttributeError, ValueError):
                        pass
                    n1 += 1
            if v2v and n2 < 40:
                v2 = unpack_signed(mcu.v2_element, MLEN)
                if v2 and not any(v2):
                    logger.info(f"[MM] v2#{n2} ZERO beat")
                    n2 += 1
                elif v2:
                    logger.info(f"[MM] v2#{n2} (LEFT/a act)    int={v2}")
                    try:
                        sraw = int(mcu.v2_scale.value)
                        scales = [(sraw >> (k * 8)) & 0xFF for k in range(4)]
                        logger.info(f"[MM] v2#{n2} scales(biased)={scales}")
                    except (AttributeError, ValueError):
                        pass
                    n2 += 1

    async def watch_acc_pulses(self):
        """Log gebm_result[0][0] (tile contribution to C[0][0]) at each
        accumulate pulse, to verify per-tile compute against theory."""
        try:
            mcu = self.dut.dut.matrix_machine_init.gen_mxint_systolic_mcu.matrix_compute_unit
        except AttributeError:
            return
        FP_EXP, FP_MAN = 6, 5

        def dfp(bits):
            sgn = (bits >> (FP_EXP + FP_MAN)) & 1
            e = (bits >> FP_MAN) & ((1 << FP_EXP) - 1)
            m = bits & ((1 << FP_MAN) - 1)
            bias = (1 << (FP_EXP - 1)) - 1
            v = (m / 2**FP_MAN) * 2.0 ** (1 - bias) if e == 0 else (1 + m / 2**FP_MAN) * 2.0 ** (e - bias)
            return -v if sgn else v

        n = 0
        prev = 0
        while n < 20:
            await RisingEdge(self.dut.clk)
            try:
                ap = int(mcu.accumulate_pulse.value)
            except (AttributeError, ValueError):
                continue
            if ap == 1 and prev == 0:
                try:
                    raw = int(mcu.gebm_result.value)
                    lane00 = raw & 0xFFF
                    logger.info(f"[ACC] pulse#{n} gebm[0][0]={dfp(lane00):.3f}")
                except (AttributeError, ValueError):
                    logger.info(f"[ACC] pulse#{n} gebm unreadable")
                n += 1
            prev = ap

    async def watch_gp6(self):
        """Log every change of gp6/gp4 (result base / MM_WO address regs)."""
        try:
            rf = self.dut.dut.scalar_machine_init.gp_reg_file
        except AttributeError:
            logger.warning("[GP] regfile not found")
            return
        prev6 = None
        prev4 = None
        n = 0
        cyc = 0
        while n < 120:
            await RisingEdge(self.dut.clk)
            cyc += 1
            try:
                v6 = int(rf.mem[6].value)
                v4 = int(rf.mem[4].value)
                v3 = int(rf.mem[3].value)
                v2 = int(rf.mem[2].value)
            except (AttributeError, ValueError, IndexError):
                continue
            if v6 != prev6:
                logger.info(f"[GP] @cyc{cyc} gp6 -> {v6}")
                prev6 = v6
                n += 1
            if v4 != prev4:
                logger.info(f"[GP] @cyc{cyc} gp4 -> {v4}")
                prev4 = v4
                n += 1
            if v3 != getattr(self, "_prev3", None):
                logger.info(f"[GP] @cyc{cyc} gp3 -> {v3}")
                self._prev3 = v3
                n += 1
            if v2 != getattr(self, "_prev2", None):
                logger.info(f"[GP] @cyc{cyc} gp2 -> {v2}")
                self._prev2 = v2
                n += 1

    async def check_mcu_drain(self):
        """Log each MCU result-drain row with its write address, to map the
        HW result layout against golden."""
        try:
            mcu = self.dut.dut.matrix_machine_init.gen_mxint_systolic_mcu.matrix_compute_unit
            mm = self.dut.dut.matrix_machine_init
        except AttributeError:
            logger.warning("[DRAIN] matrix_compute_unit not found")
            return

        FP_EXP, FP_MAN, BLEN = 6, 5, 4

        def dfp(bits):
            s = (bits >> (FP_EXP + FP_MAN)) & 1
            e = (bits >> FP_MAN) & ((1 << FP_EXP) - 1)
            m = bits & ((1 << FP_MAN) - 1)
            bias = (1 << (FP_EXP - 1)) - 1
            v = (m / 2**FP_MAN) * 2.0 ** (1 - bias) if e == 0 else (1 + m / 2**FP_MAN) * 2.0 ** (e - bias)
            return -v if s else v

        def unpack(handle, n):
            bv = handle.value
            try:
                raw = int(bv)
            except (ValueError, TypeError):
                return None
            w = len(bv) // n
            return [(raw >> (k * w)) & ((1 << w) - 1) for k in range(n)]

        logger.info("[DRAIN] probe attached")
        logged = 0
        while logged < 160:
            await RisingEdge(self.dut.clk)
            try:
                wr = int(mcu.v_result_write_req.value)
            except (AttributeError, ValueError):
                continue
            if wr != 1:
                continue
            vr = unpack(mcu.v_result, BLEN)
            if vr is None:
                continue
            row = [round(dfp(b), 2) for b in vr]
            try:
                waddr = int(mm.m_waddr.value)
            except (AttributeError, ValueError):
                waddr = -1
            logger.info(f"[DRAIN] waddr={waddr} row={row}")
            logged += 1

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
                # Access c_break_detected signal from plena (exposed in SIMULATION mode)
                c_break = self.dut.dut.c_break_detected.value

                if int(c_break) == 1:
                    self.c_break_detected = True
                    logger.info(f"C_BREAK detected at clock cycle {self.execution_clocks}")
                    logger.info(f"=== EXECUTION CLOCKS: {self.execution_clocks} ===")
                    return self.execution_clocks
            except (AttributeError, ValueError) as e:
                # Signal might not be accessible yet, continue counting
                pass

        # Timeout reached without C_BREAK
        logger.error(f"BUG: C_BREAK not loaded within {timeout_us}us ({timeout_cycles} cycles)")
        logger.error(f"=== BUG: C_BREAK NOT DETECTED ===")
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

    # Get workload directory for simulation artifacts (VCD, etc)
    workload_dir = os.environ.get("WORKLOAD_DIR")
    sim_build_dir = Path(workload_dir) if workload_dir else None

    veri_runner(
        group="system",
        module="SimTop",
        test_module="SimTop_tb_debug",  # use THIS file's cocotb test (with probes)
        test_dir=Path(__file__).parent,
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
        trace=True,
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

    test_SimTop()


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
