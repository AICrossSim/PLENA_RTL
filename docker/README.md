# Docker Setup for PLENA RTL

A containerized development environment so you can run RTL simulation, lint, and
the cocotb test suites without installing the toolchain (or Nix) on the host.

## Files

- `Dockerfile` — default **Ubuntu 24.04** image. Installs the Verilog/cocotb
  toolchain (Python 3.12, `just`, … from apt; Verible from its GitHub release;
  **Verilator built from source**, pinned to the version `flake.nix` resolves to
  — see below) plus the project's Python dependencies. No Nix.
- `Dockerfile.nix` — optional image that wraps the project's `flake.nix` for
  exact `nix develop` parity. Heavier; use only if you need it.
- `docker-compose.yml` — service definitions (`dev`, `dev-cuda`, `dev-nix`) and
  cache volumes.

## Quick start

From the repository root:

```bash
just docker-build         # build the dev image
just docker-dev           # build (if needed) and open a shell

# or directly:
docker compose -f docker/docker-compose.yml build dev
docker compose -f docker/docker-compose.yml run --rm dev bash
```

## `just` wrappers

| Command | Description |
|---------|-------------|
| `just docker-build` | Build the dev image |
| `just docker-dev` | Build (if needed) and enter an interactive shell |
| `just docker-run <cmd>` | Run an arbitrary command in the dev image |
| `just docker-test <recipe> [args...]` | Run a `just` recipe in the dev image |
| `just docker-clean` | Remove containers and the cache volumes |

Examples:

```bash
just docker-test test-hw                 # FP hardware cocotb tests
just docker-test rtl-sim linear          # generate workload + run a sim
just docker-test rtl-lint                # Verilator lint
just docker-run python3 -c "import cocotb, torch; print('ok')"
```

The repo is bind-mounted at `/workspace`, so edits on the host are picked up
immediately and simulation artifacts land back in your working tree. The Python
venv and ccache live in named volumes so they persist across runs.

## Why no Nix by default?

For a container, the image itself is the reproducibility boundary, so Nix's
pinning is redundant — and the whole RTL toolchain is available directly:
Python 3.12 from apt, Verible from its GitHub release, and Verilator built from
source. Verilator is pinned (via the `VERILATOR_VERSION` build arg, default
`5.034`) to match the version the locked `flake.nix` resolves to, since apt's
5.020 is too old for this repo's RTL. The result is a familiar image that still
tracks the native toolchain. The `flake.nix` remains the native (non-Docker)
path via `direnv` / `nix develop`, and `Dockerfile.nix` is kept for anyone who
wants full flake parity inside a container.

## CUDA / GPU

The default image installs **CPU-only PyTorch** because cocotb simulation runs
on CPU. **No CUDA-enabled torch is installed by any image as shipped.**

The `cuda` profile's `dev-cuda` service only exposes the host's NVIDIA devices
to the container (requires `nvidia-container-toolkit` on the host); it builds
the *same* `Dockerfile` as `dev`, so it still contains the CPU torch wheel:

```bash
docker compose -f docker/docker-compose.yml --profile cuda build dev-cuda
docker compose -f docker/docker-compose.yml --profile cuda run --rm dev-cuda bash
```

To actually get GPU torch, change the install line in `docker/Dockerfile`
(`uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`)
to a CUDA wheel index such as `https://download.pytorch.org/whl/cu126`, or
reinstall inside the running container:

```bash
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

(The venv lives in the persistent `plena-rtl-venv` volume, so a reinstall in the
container survives across `docker compose run` invocations.)

## Nix parity image

```bash
docker compose -f docker/docker-compose.yml --profile nix build dev-nix
docker compose -f docker/docker-compose.yml --profile nix run --rm dev-nix bash
```

## Not supported in Docker

`just synth …` (Synopsys Design Compiler) depends on a licensed toolchain
mounted from `/mnt/applications/...` on the host and is intentionally out of
scope. Run synthesis natively.
