# PLENA RTL development image (Nix parity)
#
# Optional alternative to docker/Dockerfile. Instead of installing the toolchain
# from apt, this image wraps the project's flake.nix so the in-container
# toolchain matches `direnv`/`nix develop` on a host exactly. It is heavier and
# slower to build than the apt-based default — use it only when you specifically
# need flake parity.
#
# Usage:
#   docker compose -f docker/docker-compose.yml --profile nix build dev-nix
#   docker compose -f docker/docker-compose.yml --profile nix run --rm dev-nix bash

FROM docker.io/nixos/nix:latest AS base

# Enable flakes / nix-command.
RUN mkdir -p /etc/nix && \
    printf 'experimental-features = nix-command flakes\nsandbox = false\n' >> /etc/nix/nix.conf

WORKDIR /workspace

# Copy the flake first for better layer caching, then pre-build the dev shell.
COPY flake.nix flake.lock* ./
RUN nix develop --profile /nix-profile -c true

FROM base AS dev

# Bind-mounted repo is host-owned; container runs as root.
RUN printf '[safe]\n\tdirectory = /workspace\n\tdirectory = *\n' > /root/.gitconfig

COPY . .

RUN if [ -d .git ] && [ -f .gitmodules ]; then \
        nix develop -c git submodule update --init --recursive || true; \
    fi

# Create the venv + install Python deps inside the Nix shell (uv comes from the
# flake). Mirrors .envrc, with CPU-only torch to keep the image manageable.
RUN nix develop -c bash -c '\
    uv venv .venv --python python3.12 && \
    . .venv/bin/activate && \
    uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install numpy "cocotb[bus]==1.9.2" bitstring colorlog toml tqdm pytest transformers matplotlib && \
    uv pip install -e . && \
    (uv pip install -e PLENA_Tools    || true) && \
    (uv pip install -e PLENA_Compiler || true)'

COPY <<'EOF' /entrypoint.sh
#!/usr/bin/env bash
set -e
cd /workspace
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="/workspace/tools:/workspace/PLENA_Tools:/workspace/PLENA_Compiler:${PYTHONPATH:-}"
# Run the command inside the Nix develop shell so verilator/verible/etc resolve.
exec nix develop -c "$@"
EOF
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
