# Testing

All commands are `just` recipes run from the repository root. The RTL is built
at the tile configuration in `src/definitions/configuration.svh`
(`MLEN=8, VLEN=8, HLEN=8, BLEN=4`), with MXINT8 activations and an FP12
(e6m5) vector SRAM. For which workloads and shapes are verified, see
`doc/design/workload_matrix.md`.

## Full-system simulation: `just rtl-sim`

```
just rtl-sim <workload> [rebuild=true|false] [generator args...]
```

`<workload>` is a generator module under `tools/testworkloads/` (`linear`,
`rms_norm`, `silu`, `linear_silu`, `silu_down`, `ffn`, `softmax`, `attention`,
`mha`, `rope`, `llama_layer`, `gqa`, `loop`, `prefetch`). Everything for one run
lands in `build/test/<workload>/`. The recipe has three steps:

1. **Generate** — `python -m tools.testworkloads.<workload> --build-dir ... [args]`
   builds the PyTorch golden model, quantizes inputs/weights, emits
   `generated_asm_code.asm`, assembles it to `generated_machine_code.mem`, writes
   the HBM/SRAM images (`hbm.mem`, `fp_sram.mem`, `int_sram.mem`), the golden
   (`golden_result.pt`) and `verification_params.json` / `comparison_params.json`.
   It also rewrites `INSTRUCTION_STORAGE_OFFSET` in
   `src/definitions/configuration.svh` to the program's location in `hbm.mem`;
   this edit is a build artefact and should not be committed.
2. **Simulate** — `src/system/test/SimTop_tb.py --workload-dir ...` builds
   `SimTop` (plena core + fake HBM) with Verilator through
   `cfl_cocotb.runner.veri_runner`, releases reset, counts cycles until a
   `C_BREAK` is decoded (10 ms simulated-time timeout, reported as a hang),
   then keeps the clock running for 128 cycles so in-flight matrix/vector
   writebacks drain before `$finish` dumps `vector_result.mem` and
   `hbm_result.mem`. Log: `sim.log`.
3. **Verify** — `python -m verification.verify_rtl_sim --workload-dir ... --verbose`
   (from `PLENA_Tools`) decodes the dumps and compares them to the golden.

**Pass bar.** An element matches when `|rtl - gold| <= 0.1 + 0.1 * |gold|`;
the run passes when the match rate is >= 90%.

### Knobs

- `rebuild` (default `true`): `true` wipes `build/test/<workload>/` and
  rebuilds the Verilator model; `false` keeps the directory and reuses the
  cached binary (`SKIP_BUILD=1`). The workload is regenerated either way.
  `INSTRUCTION_STORAGE_OFFSET` is compiled into the model, so reuse a cached
  build only for the same workload and shape; a different shape changes the
  offset and the cached binary will load instructions from the wrong address.
- `SIMTOP_TRACE` (default `1` for `rtl-sim`): `1` builds with `--trace` and
  writes `dump.vcd` next to the results; `SIMTOP_TRACE=0 just rtl-sim ...`
  skips the waveform and is much faster on long programs.
- `--skip-asm-gen` (passed through to the generator): reuse the existing,
  possibly hand-edited, `generated_asm_code.asm` instead of regenerating it,
  only re-assemble and simulate. The recipe refuses to wipe the build directory
  when this flag is present, so it is safe together with `rebuild=true`.
  Hand-edit workflow:
  ```
  just rtl-sim rms_norm                       # produce build/test/rms_norm/generated_asm_code.asm
  $EDITOR build/test/rms_norm/generated_asm_code.asm
  just rtl-sim rms_norm false --skip-asm-gen  # assemble + simulate the edited program
  ```
- Other generator flags (`--batch`, `--in-features`, `--seed`, `--mx-format`,
  workload-specific shape flags) are listed by `python -m tools.testworkloads.<workload> --help`.

`just rtl-sim-clean <workload>` (or `just rtl-sim-clean '' true` for all)
removes build directories.

## Looped suite: `just rtl-suite [rebuild]`

`tools/testworkloads/test_suite.py` generates every case (currently
`linear_8x128x256`, `rms_norm_8x128`, `silu_8x128`) under `build/test/suite/` at
one shared instruction offset; `src/system/test/SimTop_suite_tb.py` then builds
the DUT once, and for each case backdoor-loads its HBM/SRAM images, runs to
`C_BREAK`, and verifies. `SIMTOP_TRACE` defaults to `0` here.

## Unit tests

- `just test-hw` — FP arithmetic cocotb tests (IEEE partition/normalize,
  compact-precision adder/multiplier, fixed-point reciprocal/exp/adder/mult).
- `just test-fp` — FP unit tests on the custom RTL path (adder, multiplier,
  asymmetric multiplier, casting, exp).
  Both recipes run each `*_tb.py`, parse the cocotb summary and exit non-zero if
  any test fails.
- **Per-block testbenches.** Most blocks have `src/<group>/test/<module>_tb.py`
  next to `src/<group>/rtl/<module>.sv`. Run one directly:
  ```
  PYTHONPATH=tools python3 src/basic_components/buffer/test/fifo_tb.py
  ```
  Each calls `cfl_cocotb.runner.veri_runner`, which infers `<group>`/`<module>`
  from the file path, builds the module with Verilator (parameter sets from
  `module_param_list`), and runs the cocotb tests in the same file. Useful
  arguments: `module=` to build a different top (e.g. a `*_tb_wrapper.sv` that
  flattens struct ports for cocotb, as `decoder_tb.py` does with
  `decoder_tb_wrapper`), `test_module=` to name the cocotb test module when it
  differs from the built module, `test_dir=` to choose where `results.xml` and
  the VCD land, and `trace=`/`skip_build=`. The runner passes
  `--converge-limit 10000` to Verilator to silence UNOPTFLAT residuals from the
  `register_slice`/`split_n` combinational paths.

## Static checks

- `just rtl-lint [--strict]` — `tools/rtl_lint.py`: Verilator `--lint-only` on
  `SimTop` with the bug-catching warnings enabled (PINCONNECTEMPTY, UNDRIVEN,
  MULTIDRIVEN, CASEINCOMPLETE, ...); `--strict` adds the style warnings.
- `just rtl-check [module] [--trace] [-j N]` — `tools/rtl_check.py`: full
  Verilator elaboration/compilation of `SimTop` (default) with warnings
  suppressed, no simulation. `just rtl-check-clean` removes its build directory.
