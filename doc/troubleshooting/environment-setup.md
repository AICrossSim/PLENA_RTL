# Environment Setup Troubleshooting

## Issue 1: `GLIBC_2.38 not found` when importing PyTorch

**Symptom:**
```
ImportError: /lib64/libc.so.6: version `GLIBC_2.38' not found
  (required by /nix/store/.../gcc-15.2.0-lib/lib/libstdc++.so.6)
```

**Root Cause:**
A Nix-provided `libstdc++.so.6` (built against a newer glibc than the system
one, e.g. glibc 2.38 vs. RHEL 9's 2.34) is being picked up ahead of the system
library. glibc versions cannot be mixed, and prepending Nix's glibc to
`LD_LIBRARY_PATH` fails with symbol errors as well.

**Fix:**
Make sure `LD_LIBRARY_PATH` does not put a Nix `libstdc++` ahead of the system
one. PyTorch ships its own compatible `libstdc++`, so no extra C++ runtime path
is needed. Check with:
```bash
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -n nix
```
If a `/nix/store/...-gcc-*-lib/lib` entry appears before the system paths, drop
it from whatever exported it (a shell profile or a local `.envrc` override) and
run `direnv reload` or open a new terminal.

---

## Issue 2: `ModuleNotFoundError: No module named 'plena_quant'`

**Symptom:**
```
ModuleNotFoundError: No module named 'plena_quant'
```

**Root Cause:**
The `PLENA_Tools` git submodule was not initialized. The directory exists but is empty.

**Fix:**
```bash
git submodule update --init --recursive
```

Then reinstall the editable packages:
```bash
rm .venv/.deps-installed
direnv reload
# Or manually:
uv pip install -e PLENA_Tools
uv pip install -e PLENA_Compiler
```

---

## Quick Checklist for New Environment Setup

1. Install Nix and enable flakes
2. Clone the repo with submodules: `git clone --recurse-submodules <repo-url>`
3. Enter the directory (direnv will auto-setup if `direnv allow` has been run)
4. Verify: `python3 -c "import torch; print(torch.__version__)"`
5. Verify: `python3 -c "import plena_quant; print('ok')"`
