# PLENA RTL development image (no Nix)
#
# A plain Ubuntu 24.04 image carrying the Verilog/cocotb toolchain and Python
# dependencies needed to run RTL simulation, lint, and the cocotb test suites.
# This is the default, lightweight path. For exact `flake.nix` parity use
# docker/Dockerfile.nix instead.
#
# What this image canNOT do: Synopsys synthesis (`just synth ...`) requires a
# licensed Design Compiler mounted from the host and is intentionally out of
# scope for Docker.

FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# --- System toolchain ---------------------------------------------------
# Ubuntu 24.04 ships Python 3.12 and just in the archive. Verilator is built
# from source below (apt's 5.020 is too old for this RTL); the deps it needs to
# compile are included here alongside the rest of the toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Compilers & build tools (Verilator emits C++ that g++/clang compile)
        build-essential \
        gcc g++ \
        clang \
        llvm \
        make cmake ninja-build \
        pkg-config \
        ccache \
        # Verilator build-from-source dependencies
        autoconf flex bison libfl-dev help2man perl zlib1g-dev \
        # Python 3.12
        python3 python3-dev python3-venv python3-pip \
        # Task runner used by the justfile recipes
        just \
        # General utilities
        git wget curl unzip ca-certificates \
        graphviz \
        vim less \
    && rm -rf /var/lib/apt/lists/*

# --- Verilator (built from source) ---------------------------------------
# Pinned to match the version the project's flake.nix resolves to (the locked
# nixpkgs provides 5.034), so the Docker toolchain matches `nix develop`.
# apt's 5.020 rejects valid RTL in this repo (e.g. non-blocking assignment to
# an array element inside a for-loop -> BLKLOOPINIT).
ARG VERILATOR_VERSION=5.034
RUN set -eux; \
    git clone --depth 1 --branch "v${VERILATOR_VERSION}" \
        https://github.com/verilator/verilator.git /tmp/verilator; \
    cd /tmp/verilator; \
    autoconf; \
    ./configure; \
    make -j "$(nproc)"; \
    make install; \
    cd /; \
    rm -rf /tmp/verilator; \
    verilator --version

# --- Verible (SystemVerilog formatter/linter; not packaged in apt) -------
# Release tags and asset names embed a git hash, so resolve the download URL
# from the GitHub release API rather than hardcoding it. Defaults to the latest
# release; override VERIBLE_RELEASE_API with a .../releases/tags/<tag> URL to pin.
ARG VERIBLE_RELEASE_API=https://api.github.com/repos/chipsalliance/verible/releases/latest
RUN set -eux; \
    url="$(curl -fsSL "${VERIBLE_RELEASE_API}" \
        | grep -o 'https://[^"]*linux-static-x86_64.tar.gz' | head -1)"; \
    test -n "$url"; \
    curl -fsSL "$url" -o /tmp/verible.tar.gz; \
    mkdir -p /opt/verible; \
    tar -xzf /tmp/verible.tar.gz -C /opt/verible --strip-components=1; \
    rm /tmp/verible.tar.gz; \
    ln -s /opt/verible/bin/* /usr/local/bin/
ENV PATH="/opt/verible/bin:${PATH}"

# --- uv (fast Python package manager, same as .envrc) --------------------
RUN pip install --no-cache-dir --break-system-packages uv

# ========================================================================
# Development image
# ========================================================================
FROM base AS dev

WORKDIR /workspace

# The repo is bind-mounted from the host (owned by the host user) while the
# container runs as root. Mark it safe so git submodule/editable installs work.
RUN git config --global --add safe.directory '*'

# Copy the repository (submodules included if already checked out on host).
COPY . .

# Initialize submodules if a .git directory was copied in.
RUN if [ -d .git ] && [ -f .gitmodules ]; then \
        git submodule update --init --recursive || true; \
    fi

# Create the venv and install Python dependencies. Mirrors .envrc, but pulls
# CPU-only PyTorch (cocotb simulation runs on CPU) to keep the image small.
# For GPU work use the dev-cuda compose service.
RUN uv venv .venv --python python3.12 \
    && . .venv/bin/activate \
    && uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install \
        numpy \
        "cocotb[bus]==1.9.2" \
        bitstring \
        colorlog \
        toml \
        tqdm \
        pytest \
        transformers \
        matplotlib \
    && uv pip install -e . \
    && (uv pip install -e PLENA_Tools    || echo "Warning: could not install PLENA_Tools") \
    && (uv pip install -e PLENA_Compiler || echo "Warning: could not install PLENA_Compiler")

# Entry point: activate the venv and expose tools + submodules on PYTHONPATH
# so both `just` recipes and ad-hoc python commands work out of the box.
COPY <<'EOF' /entrypoint.sh
#!/usr/bin/env bash
set -e
cd /workspace
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="/workspace/tools:/workspace/PLENA_Tools:/workspace/PLENA_Compiler:${PYTHONPATH:-}"
exec "$@"
EOF
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
