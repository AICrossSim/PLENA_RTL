# AGENTS.md — working in PLENA_RTL

Guidance for coding agents (and new contributors) working in this repository.
It complements `README.md` (user-facing setup) and the `justfile` (the single
source of truth for every command). Keep this file short and factual; if a
statement here disagrees with the code, fix the file.

## 1. What this repo is

SystemVerilog RTL for **PLENA**, a programmable accelerator for long-context
transformer inference, plus the cocotb/Verilator harness that runs whole
programs (linear, RMSNorm, SiLU, softmax, attention, FFN, a full Llama decoder
layer) on the simulated core and checks the results against a PyTorch golden
model.

Two git submodules supply the software side and are imported by the harness:

| Submodule | Role | Used from here |
|---|---|---|
| `PLENA_Compiler` | assembler, ISA definition copy, hand-written asm templates, ATen compiler | `assembler.assembly_to_binary`, `asm_templates.*`, `PLENA_Compiler/doc/{operation,configuration}.svh` |
| `PLENA_Tools` | MX quantization (`plena_quant`), config loaders (`plena_utils`), result verifier (`verification.verify_rtl_sim`) | every workload generator, `just rtl-sim` step 3 |

Always run `git submodule update --init --recursive` after cloning. Do not
`git pull` inside a submodule unless you intend to move the pinned commit.

## 2. Layout

```
src/
  definitions/      configuration.svh (tile sizes, SRAM depths, INSTRUCTION_STORAGE_OFFSET),
                    precision.svh (MX / FP widths), operation.svh (ISA enums, OP_BUNDLE), global_define.vh
  core/rtl/plena.sv accelerator top: instr_mem -> decoder -> pipeline_control / data_flow_control
                    -> matrix_machine | vector_machine | scalar_machine, matrix SRAM, vector SRAM, hbm_sys
  frontend/         decoder, instruction memory, loop controller
  control/          pipeline_control (hazards/stalls), data_flow_control (HBM<->SRAM movement), addr_monitor
  matrix_machine/   systolic GEMM wrapper (mxint path is elaborated when precision.svh WT_MX_INT_ENABLE=1)
  vector_machine/   element-wise + reduction FP vector units
  scalar_machine/   scalar INT/FP ALU and register files
  memory/           matrix_sram, vector_sram, scalar_sram, HBM (fake HBM models + TileLink controllers)
  basic_components/ reusable blocks: fifo, skid buffer, register slices, FP/fixed arithmetic, MX conversion,
                    systolic PEs. Each block dir has rtl/ and (usually) test/<module>_tb.py
  system/rtl/       SimTop.sv (plena + fake_hbm_5port) — the top every rtl-sim/rtl-suite run builds
  system/test/      SimTop_tb.py (canonical), SimTop_suite_tb.py (one build, many cases), test_platform.py
  fpga/, system/rtl/SimTopA7*.sv, SimTopDDR.sv   FPGA bring-up RTL (lint-only; no testbenches shipped)
tools/
  cfl_cocotb/       veri_runner(): builds with Verilator and runs a cocotb module; _verilator_args() holds the flags
  cfl_tools/        PROJECT_PATH / SRC_PATH constants, logger, pdb excepthook
  testworkloads/    one generator per workload (<name>.py, class <Name>Workload, CLI via python -m)
  rtl_check.py / rtl_lint.py   Verilator elaboration check / lint (lint is what CI runs)
  synopsys/         DC synthesis flow used by `just synth` (synth.tcl + read_rtl.tcl; needs a licensed DC + ASAP7 lib)
doc/                design notes; doc/design/attention_isa_roadmap.md explains the ISA gaps
build/test/<workload>/   all generated artefacts for a run (git-ignored)
```

## 3. Environment

* Native: `direnv allow` (Nix flake + uv venv) or `nix develop`. Docker:
  `just docker-dev`. Check with `verilator --version` (5.034 expected) and
  `python -c "import cocotb, torch"` (cocotb 1.9.2).
* Everything is launched through `just`. Recipes set `PYTHONPATH` to
  `src/system/test:tools:PLENA_Tools:PLENA_Compiler`; if you call a Python
  file directly, export the same `PYTHONPATH` from the repo root.
* Simulations are CPU-only. Torch is only used for golden models and
  quantization.

## 4. Running tests

| Command | What it does | Typical time (64-core box, MLEN=8) |
|---|---|---|
| `just rtl-sim <workload> [rebuild] [--args]` | generate workload -> Verilator build -> cocotb run -> verify | ~6 min with rebuild, ~1–2 min with `false` |
| `just rtl-suite [rebuild]` | cases in `tools/testworkloads/test_suite.py`, one build, backdoor reload per case | ~6 min |
| `just test-hw` / `just test-fp` | FP arithmetic unit cocotb tests; a recipe fails if any tb reports `Failed: N>0` | minutes |
| `just rtl-lint` / `just rtl-check` | Verilator lint (CI) / full elaboration of SimTop | ~1 min |

Workloads (`tools/testworkloads/<name>.py`): `linear`, `rms_norm`, `silu`,
`linear_silu`, `silu_down`, `softmax`, `rope`, `mha`, `gqa`, `attention`,
`ffn`, `llama_layer` (full decoder layer), plus infrastructure tests
`prefetch`, `loop`, `scratchpad`.
Every generator's defaults are derived from the tile sizes, so
`just rtl-sim <name>` works with no arguments.

Rules that bite:

* **`rebuild` is positional and comes before `--args`.**
  `just rtl-sim linear false --batch 16`.
* **Set `SIMTOP_TRACE=0`** unless you need a VCD. Tracing multiplies run time
  and produces multi-GB `dump.vcd` files under `build/test/<workload>/`.
* **Workload shapes must be multiples of the tile sizes** in
  `src/definitions/configuration.svh` (`MLEN`, `VLEN`, `BLEN`). Generators
  read these at runtime and assert. Changing them is a hardware change: run
  with `rebuild=true` afterwards.
* **Pass bar** (`PLENA_Tools/verification/verify_rtl_sim.py`): an element
  matches if `|err| <= 0.1 + 0.1*|golden|`; a run passes at >= 90 % match
  rate. Do not "fix" a failing run by loosening this.
* A hung program shows up as `BUG: C_BREAK not loaded within ...us`. The
  testbench then exits non-zero; look at `build/test/<workload>/sim.log`.
* `tools/cfl_tools/debugger.py:set_excepthook` drops into `pdb` on any
  uncaught Python exception in `SimTop_tb.py`. In a non-interactive shell it
  exits; in a terminal it waits for input. Pipe stdin from `/dev/null` in
  automation.
* Concurrent `rtl-sim` runs are **not safe**: they share
  `src/system/test/build/SimTop/` and rewrite `configuration.svh` (below).
  Run them sequentially.

## 5. The `INSTRUCTION_STORAGE_OFFSET` contract

Programs live in HBM after the data. Each generator computes where its data
ends, writes the instructions there, and **patches the tracked file
`src/definitions/configuration.svh`** (`INSTRUCTION_STORAGE_OFFSET`) via
`plena_utils.config.update_instruction_storage_offset`. Consequences:

* After generating a new workload the RTL must be rebuilt, because the offset
  is a `localparam`. `rebuild=false` is only valid when the offset did not
  change (same workload and shapes).
* `git status` will show `configuration.svh` modified after any run. Never
  commit that hunk by accident; `git checkout -- src/definitions/configuration.svh`
  restores it.
* `rtl-suite` avoids the rebuild-per-case cost by choosing one offset large
  enough for every case and relocating each case's instructions at load time.

## 6. Adding or changing things

**A new workload generator**

1. Copy the closest generator in `tools/testworkloads/`, subclass
   `WorkloadGenerator` (`base.py`), keep the file/class naming pattern.
2. Read tile sizes with `plena_utils.load_config.load_hardware_tile_sizes`
   and the quant config with `get_quant_config_for_format`; do not hard-code
   `MLEN`/`VLEN`/`BLEN` or the MX format.
3. Emit `hbm.mem`, `fp_sram.mem`, `int_sram.mem`, `generated_machine_code.mem`,
   a golden tensor, and `verification_params.json` (the verifier's contract).
4. Prefer an asm template in `PLENA_Compiler/asm_templates` over inline
   strings, so the emulator and the RTL run the same program.
5. Add it to `default_suite()` in `test_suite.py` if it should run in
   `rtl-suite`, and to the README workload list.

**A new ISA instruction** touches three places that must stay in sync:
`src/definitions/operation.svh` (RTL), `PLENA_Compiler/doc/operation.svh`
(what the assembler reads), and `src/frontend/rtl/decoder.sv` (opcode ->
machine op). The assembler only reads the instruction-encoding widths from
`PLENA_Compiler/doc/configuration.svh`; tile sizes come from
`src/definitions/configuration.svh`.

**RTL changes**

* Reuse `src/basic_components` (`fifo.sv`, `skid_buffer.sv`,
  `register_slice*.sv`, `fp_fix_*` arithmetic) instead of re-implementing
  buffers or FP ops inline.
* Style: `` `timescale 1ns / 1ps ``, `` `include "configuration.svh" ``,
  `module name import configuration_pkg::*; #(params) (ports)`, `logic`
  everywhere, a header comment with Module / Description. Follow the
  surrounding file.
* Simulation-only code goes under `` `ifdef SIMULATION `` (the runner passes
  `-DSIMULATION`). `DC_LIB_EN` selects DesignWare FP IP for synthesis and is
  off by default.
* Unit tests live next to the RTL as `<block>/test/<module>_tb.py` and use
  `cfl_cocotb.runner.veri_runner`. Run one with
  `python src/<...>/test/<module>_tb.py`.
* Files under `src/memory/HBM/TileLink_Lib/` and the `prim_*` files in
  `src/memory/{vector_sram,scalar_sram}/rtl/` are vendored third-party code
  (TileLink library, lowRISC primitives). Do not restyle them; keep their
  license headers.
* Before opening a PR run `just rtl-lint` (this is the CI gate) and at least
  `just rtl-sim linear` plus the workload you touched.

## 7. Verification artefacts

`build/test/<workload>/` after a run:

| File | Meaning |
|---|---|
| `generated_asm_code.asm` / `generated_machine_code.mem` | program (edit the `.asm` and re-run with `--skip-asm-gen` to test hand changes) |
| `hbm.mem`, `fp_sram.mem`, `int_sram.mem` | initial memory images |
| `golden_result.pt/.txt`, `golden_vram_result.pt/.txt` | PyTorch reference |
| `vector_result.mem`, `hbm_result.mem`, `fp_reg_result.mem` | RTL dumps at `$finish` |
| `verification_params.json` | which rows/regions the verifier compares |
| `sim.log`, `dump.vcd` (if traced) | log and waveform |

## 8. Things not to do

* Do not edit `INSTRUCTION_STORAGE_OFFSET` by hand, and do not commit it.
* Do not change Verilator flags in `tools/cfl_cocotb/runner.py` without
  re-running `rtl-suite`; every test shares them.
* Do not add absolute paths, home directories, or site-specific mounts to
  code or docs. The Synopsys setup script is the one site-specific default and
  is overridden with the `SYNOPSYS_SETUP` environment variable.
* Do not shrink `simulation_pkg::FAKE_HBM_ADDR_WIDTH`: it is the per-port word
  address width of the simulation HBM model and bounds the largest `hbm.mem`
  a workload can use (2^N * 2 bytes through the scale port).
* Do not silence a failing verification by editing thresholds or by
  narrowing the compared rows in `verification_params.json`.
