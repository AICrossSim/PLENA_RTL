# PLENA FPGA bring-up — Nexys Video, Vivado-free JTAG

This document captures the FPGA deployment work for the PLENA accelerator on the Digilent **Nexys
Video** board (AMD/Xilinx **Artix-7 XC7A200T-1SBG484C**), and the full simulation infrastructure that
validates it. The goal: drive the whole host↔core loop over the on-board **FTDI JTAG cable with no
Vivado hardware server** — load DDR3 and the program, launch, run, and read results back, all over JTAG.

Everything below the Vivado toolchain boundary is implemented, simulated, and Verilator-lint-clean.

---

## 1. What was built (one paragraph)

The PLENA core talks to HBM over five TileLink ports (instructions + matrix/vector × element/scale). On
the board those funnel through one `tl_to_axi4` bridge into the MIG DDR3 controller, and the host reaches
the core through an AXI-lite control block. We (a) fixed a fundamental deadlock in the board's HBM mux,
(b) built a behavioural-DDR3 simulation of the whole FPGA datapath and proved it data-correct, (c) built
a **custom BSCANE2→AXI-lite master** so the host drives the core over JTAG without Vivado's `hw_server`,
(d) added a **JTAG→DDR3 loader** and **VSRAM readback over JTAG**, and (e) proved the entire loop
end-to-end in cocotb. All of it is mirrored into the board top `plena_a7_top.sv`.

---

## 2. Findings / bugs fixed (in order of discovery)

| # | Problem | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Board HBM mux **deadlocks** | The hand-rolled 4-way mux (a) only grants `a_ready` to an already-arbitrated channel (plena gates `a_valid` on `a_ready`); (b) the single-outstanding bridge stalls in `TL_RESP` holding one channel's response while plena won't consume it until its `join2` partner (element↔scale) is read — which is stuck behind the stall; (c) strict priority drains a whole element burst (`LOAD_AMOUNT=MLEN`) before any scale. | Per-channel `tl_regslice` skid buffers (`a_ready=!full`), per-channel `prim_fifo_sync` response FIFOs (bridge never back-pressured), **round-robin** arbitration. |
| 2 | Matmul output wrong via DDR3 (~9–35%) | Datapath is **byte-exact** (activation path bit-identical to fake_hbm); the matmul prefetch is not robust to real memory latency — weights haven't fully landed in `matrix_sram` before the matmul reads them (35% @ latency 20, 77% @ latency 2). | Not a datapath bug — a prefetch→compute synchronization gap in the core, flagged for a future core fix. The MIG's higher/variable latency will hit this on hardware. |
| 3 | `system_break` missed over JTAG | It is a **combinational 1-cycle pulse** (`exe_stage_op.c_op==BREAK`); the core keeps fetching NOPs past it. Polling STATUS over slow JTAG always misses the pulse. | Latch it sticky (cleared per launch); STATUS reports the latch. |
| 4 | Core runs garbage before load | The board releases the core when `ui_clk_sync_rst` deasserts, so it executes empty/garbage IMEM (and issues stray HBM traffic) before the host loads the program. | `core_launched` gate: `plena_rst = ui_clk_sync_rst \| soft_rst \| ~core_launched` — hold the core until the first SOFT_RST. |
| 5 | `jtag_axi` IP needs Vivado hw_server | The Xilinx JTAG-to-AXI Master IP is only drivable from `run_hw_axi`. | Custom BSCANE2→AXI-lite master (`jtag_axi_bscan`), driven from a host with pyftdi/OpenOCD. |
| 6 | IMEM too small / wrong load addr | `IMEM_DEPTH=512` can't hold real programs once `INSTRUCTION_STORAGE_OFFSET` is added (`fetch_addr = pc + offset`, decoded to IMEM word `addr>>2`). Program word `k` lands at IMEM word `(offset>>2)+k`. | Size `IMEM_DEPTH ≥ (offset>>2)+nwords` in the tb; parse the SV literal properly (`32'h6C0`→1728, not 32). For the board, either bump `IMEM_DEPTH` or build with `INSTRUCTION_STORAGE_OFFSET=0`. |
| 7 | Loader hangs on HW (not sim) | The board's HBM datapath ran on `plena_rst`, which is **held during load** (`core_launched=0`), so the loader couldn't issue. Sim's datapath uses the always-live `rst`, so it passed. | Datapath resets on `dp_rst = ui_clk_sync_rst` (MIG reset) while the core stays on `plena_rst`. |

---

## 3. Architecture (board datapath)

```
 host (pyftdi / OpenOCD over FT2232H JTAG)
   │  40-bit DR scan: {rw, byte_addr, wdata} in / {done, addr_echo, rdata} out (LSB-first)
   ▼
 BSCANE2 (JTAG_CHAIN=4 / USER4=0x23)  ──►  jtag_axi_bscan  ──►  AXI-lite register block
   (tck domain)          toggle-req/ack CDC          │  (aclk = core clock)
                                                     ├─ IMEM_DATA/WPTR → IMEM BRAM → tl_adapter_bram → core instr port
                                                     ├─ SOFT_RST → core_launched gate → plena_rst
                                                     ├─ STATUS {init_calib, system_break_latched}
                                                     ├─ DEBUG_CTRL/ADDR/Dk → core debug_vsram readback
                                                     └─ DDR3_ADDR/W0..3/GO → loader FSM ─┐
                                                                                         ▼   2:1 mux (LOAD_MODE)
 core 4 HBM ports → skid buffers → round-robin → response FIFOs ──────────────────────►  tl_to_axi4 ─► MIG DDR3
```

- **LOAD_MODE=1** (core in reset): the loader owns the bridge, writes DDR3; the plena mux is gated off.
- **LOAD_MODE=0** (running): the plena 4-way mux owns the bridge; loader idle.
- Datapath resets on `dp_rst=ui_clk_sync_rst`; core resets on `plena_rst` (held until first SOFT_RST).

### AXI-lite register map (word address = `byte_addr[7:2]`)

| Byte | R/W | Name | Function |
|------|-----|------|----------|
| `0x00` | W | IMEM_DATA | push one 32-bit instruction word; `wptr++` |
| `0x04` | R | STATUS | `{…, init_calib_complete[1], system_break_latched[0]}` |
| `0x08` | W | DEBUG_CTRL | bit0 → `debug_vsram_en` |
| `0x0C` | W | DEBUG_ADDR | `debug_row` |
| `0x10` | RW | IMEM_WPTR | set/read IMEM write pointer |
| `0x14` | W | SOFT_RST | reset `wptr=0` + pulse a 16-cycle core reset → run from PC0 |
| `0x18` | R | RST_CNT | cycles the core was held in reset |
| `0x1C` | RW | DDR3_ADDR | DDR3 byte address (auto `+=16` after each GO) |
| `0x20..0x2C` | W/R | DDR3_W0..W3 / DDR3_R0..R3 | 128-bit beat words (write) / read-back words |
| `0x30` | W | DDR3_GO | issue one 128-bit PutFullData at DDR3_ADDR |
| `0x34` | R | DDR3_STATUS | `{…, done[1], busy[0]}` |
| `0x38` | RW | LOAD_MODE | bit0: 1 = loader owns bridge, 0 = plena mux |
| `0x3C` | W | DDR3_RD | issue one 128-bit Get at DDR3_ADDR (readback) |
| `0x40+` | R | DEBUG_Dk | VSRAM row readback words (`DBG_WORDS=6`) |

### JTAG DR command format (`jtag_axi_bscan`, 40 bits, LSB-first)

```
shift-IN : [31:0]=wdata, [38:32]=byte addr (NOT >>2; slave slices [7:2]), [39]=rw (1=wr, 0=rd)
shift-OUT: [31:0]=rdata (prev txn), [38:32]=addr echo, [39]=done
```
Pipelined: a scan launches txn N and shifts out txn N-1's result; a read = issue-scan + flush-scan.
The tap shifter is **TCK-clocked** (DRCK is dead during Update-DR) and joined to the AXI core clock by a
toggle req/ack CDC with a quasi-static payload.

---

## 4. The host↔core loop (validated end-to-end in sim)

```
DDR3 empty
  → (JTAG) LOAD_MODE=1; stream DDR3 data region as 16-byte PutFullData beats; LOAD_MODE=0
  → (JTAG) set IMEM_WPTR; stream IMEM_DATA words
  → (JTAG) SOFT_RST                      # launch: core resets to PC0 and runs
  → core reads DDR3 weights/acts + computes into VSRAM, hits C_BREAK → system_break latched
  → (JTAG) poll STATUS until system_break
  → (JTAG) DEBUG_CTRL/ADDR + read DEBUG_Dk → VSRAM rows back to host
```

The host driver that implemented this sequence over an abstract transport (cocotb BSCAN in
simulation, pyftdi on the board) is not included in this release; see §5.

---

## 5. Simulation testbenches

The cocotb testbenches that validated this path (`SimTopDDR_tb.py`, `SimTopA7_tb.py`,
`SimTopA7_jtag_tb.py`, `ddr3_loader_tb.py`, `SimTopA7_jtag_dbg_tb.py`,
`src/fpga/common/test/jtag_axi_bscan_tb.py`) and the host-side driver stack
(`plena_driver.py`, `jtag_bscan_transport.py`, `jtag_axi_transport.py`, `cocotb_axil_bfm.py`)
are **not part of this release**; they were removed from the tree in the initial open-source
commit because they are not maintained by any `just` recipe. The RTL they exercised (§6) is
still shipped and lint-clean. Note that every workload generator rewrites
`INSTRUCTION_STORAGE_OFFSET` in `src/definitions/configuration.svh` to match the generated
program; that change is a build artefact and should not be committed.

---

## 6. Files

### RTL (datapath + JTAG)
- `src/fpga/common/rtl/tl_to_axi4.sv` — TL→AXI4 burst bridge (single-outstanding).
- `src/fpga/common/rtl/jtag_axi_bscan.sv` — custom BSCANE2→AXI-lite master (Vivado-free JTAG).
- `src/memory/HBM/rtl/fake_ddr3_axi.sv` — behavioural DDR3 AXI4 slave.
- `src/system/rtl/SimTopDDR.sv` — datapath sim top.
- `src/system/rtl/SimTopA7.sv` — activation sim top (AXI-lite control + IMEM + datapath + **loader**).
- `src/system/rtl/SimTopA7_jtag.sv` — SimTopA7 driven through `jtag_axi_bscan`.
- `src/system/rtl/SimTopA7_dbg.sv` / `SimTopA7_jtag_dbg.sv` — as above but with real VSRAM readback.
- `src/core_dbg/rtl/plena.sv` — core variant with the `debug_vsram` port-A read hijack, used by the
  `*_dbg` sim tops for VSRAM readback.

### Host driver + tests

Removed from this release (see §5).

### Board top
- `src/fpga/nexys_video/plena_a7_top.sv` — the board top: mux fix + BSCANE2/`jtag_axi_bscan` +
  activation fixes + DDR3 loader + reset split. Uses the core's `debug_vsram` readback port.
- `src/fpga/nexys_video/build_nexys.tcl` — non-project Vivado build script.
- `src/fpga/nexys_video/mig_nexys_video.prj` — MIG DDR3 config.
- `src/fpga/nexys_video/nexys_a7.xdc` — pin/timing constraints.
- `src/fpga/nexys_video/xil_sim_stubs.sv` — BUFG / mig_ddr3 / BSCANE2 lint stubs.

None of these testbenches are run by a `just` recipe; invoke them directly as shown in §5.

---

## 7. Remaining: hardware bring-up (needs Vivado + the board)

1. **Vivado synth** of `plena_a7_top` via `src/fpga/nexys_video/build_nexys.tcl` → bitstream. BSCANE2 is
   a 7-series unisim (no IP to generate); the MIG is generated from `mig_nexys_video.prj`.
2. Program the FPGA (any JTAG programmer; the bitstream, not the driver).
3. Run the host: `PlenaDriver(JtagAxiTransport())` — load DDR3, load IMEM, launch, poll, read back.
4. **Bench confirms** (not sim-verifiable): the `USER4=0x23` IR opcode / IR length against the XC7A200T
   BSDL; the FT2232H URL (`ftdi://ftdi:2232h/2`, JTAG on interface B); and that direct pyftdi/libusb
   access doesn't collide with Digilent Adept / a running `hw_server` (close them first).
5. Optional: for large programs, bump `IMEM_DEPTH` in `plena_a7_top` (512 → ≥ next_pow2(offset/4 + nwords)),
   or build with `INSTRUCTION_STORAGE_OFFSET=0`.

Known caveat carried from sim: matmul numeric accuracy degrades with real DDR3 latency (finding #2) — a
core prefetch-synchronization fix, separate from this bring-up.
