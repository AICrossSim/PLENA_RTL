# Synthesis with Synopsys Design Compiler

Any RTL module in `src/` can be synthesized to the ASAP7 7 nm library with:

```
just synth <module> [clk_period_ps] [mode]

just synth fp_adder              # 1000 ps clock, standard compile
just synth fp_adder 500          # 500 ps clock
just synth fp_adder 500 ultra    # compile_ultra -retime (with optimize_registers)
just synth plena                 # the accelerator top
```

`<module>` is the SystemVerilog module name (the design `read_rtl.tcl`
elaborates as top), `clk_period_ps` defaults to `1000`, and `mode` is `normal`
(`compile`) or `ultra` (`compile_ultra -retime`).

## Requirements

- **Synopsys Design Compiler** (`dc_shell`), with a license. The recipe sources
  a site setup script before running; the default is
  `/mnt/applications/synopsys/2024-25/scripts/SYN_2024.09-SP2_RHELx86.sh`.
  Point `SYNOPSYS_SETUP` at your own script to override it:
  ```
  SYNOPSYS_SETUP=/path/to/dc_setup.sh just synth fp_adder
  ```
  The recipe also strips the Nix/venv Python variables (`PYTHONPATH`,
  `VIRTUAL_ENV`, ...) and removes Nix Python from `PATH` so they do not clash
  with DC's bundled Python.
- **ASAP7 standard-cell library**, not shipped with this repository. Place the
  `asap7sc7p5t_28` release under `tools/synopsys/lib/asap7/` so that
  `tools/synopsys/lib/asap7/asap7sc7p5t_28/` contains the compiled `.db` files
  named in `.synopsys_dc.setup` (RVT, TT corner, CCS):
  `asap7sc7p5t_SIMPLE_RVT_TT_ccs_211120.db`, `asap7sc7p5t_AO_RVT_TT_ccs_211120.db`,
  `asap7sc7p5t_SEQ_RVT_TT_ccs_220123.db`, `asap7sc7p5t_INVBUF_RVT_TT_ccs_211120.db`.
  DesignWare (`dw_foundation.sldb`) is taken from the DC installation.
- Synthesis is host-only; it is not available in the Docker images.

## Files

```
tools/synopsys/
├── .synopsys_dc.setup   # DC startup: ASAP7 target/link libraries, search_path,
│                        #   WORK library, naming rules, message suppression
├── synth.tcl            # the synthesis script `just synth` runs (dc_shell -f synth.tcl)
├── read_rtl.tcl         # sourced by synth.tcl: analyze packages + all RTL dirs, elaborate top
└── constraints/         # reference .sdc files (not sourced by synth.tcl; the clock and
                         #   I/O constraints are set inline in synth.tcl)
```

`just synth` exports `SYNTH_MODULE`, `SYNTH_CLK_PERIOD`, `SYNTH_COMPILE_MODE`
and `SYNTH_BUILD_DIR`, `cd`s into `tools/synopsys/` (so `.synopsys_dc.setup` is
picked up) and runs `dc_shell -f synth.tcl`, which:

1. sources `read_rtl.tcl`: analyzes the `src/definitions/` packages
   (`global_define.vh`, `precision.svh`, `configuration.svh`, `operation.svh`),
   then every `.sv`/`.v` under the listed `src/**/rtl` directories, skipping the
   simulation-only files in its `skip_list` (`fp_rounding.sv`, `bram.sv`,
   `fake_hbm.sv`, `peripheral_system.sv`), and elaborates `<module>`;
2. runs `check_design`, links and uniquifies;
3. constrains the design: `create_clock -period <clk_period_ps> clk`, 10% setup
   uncertainty, 8%/5% input/output delays, `set_max_delay` to all outputs and
   register data pins;
4. writes the unmapped netlist, compiles (`compile` or `compile_ultra -retime`),
   runs `check_timing`/`check_design`;
5. writes the mapped netlist, SDF, SDC and reports.

To synthesize a sub-module, just name it; the whole tree is analyzed each time
and only `<module>` is elaborated. Add a file to `skip_list` in `read_rtl.tcl`
if a non-synthesizable file breaks the analyze step. The top-level clock/reset
port names are assumed to be `clk`/`rst`.

## Output layout

Each run gets its own timestamped directory, and `latest` is a symlink to the
most recent run of that module:

```
build/synth/<module>/<YYYYMMDD_HHMMSS>/
├── logs/      synth.log (full dc_shell transcript), summary.log, area.log,
│              power.log, timing.log, <module>_{pre_check,post_check,timing_check}.log
├── reports/   <module>_{area,timing,timing_slack,qor,power,constraints,
│              reference,clock,units,port,lib}.rpt
├── netlist/   <module>_unmapped.{v,ddc}, <module>.tcl (write_script)
└── out/       <module>_mapped.{v,ddc}, <module>.sdf, <module>.sdc
build/synth/<module>/latest -> <YYYYMMDD_HHMMSS>
```

Helper recipes:

| Recipe | Purpose |
|---|---|
| `just synth-report <module>` | print `latest/logs/summary.log` and list the reports |
| `just synth-builds` | list the last five builds of every module |
| `just synth-clean <module>` / `just synth-clean '' true` | delete one module's builds / all builds |
| `just synth-clean-old [keep=3]` | keep only the newest N builds per module |

`build/` and DC's working library under `tools/synopsys/` are git-ignored.
