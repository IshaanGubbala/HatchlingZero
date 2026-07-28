# Triton-MSL Experiment

Date: July 26, 2026

Branch: `exp/triton-msl-mac`

## Goal

Test whether the `HZ-0A` upstream `GatedDeltaNet-2` path can be moved forward
on Apple Silicon using `triton-msl`.

## Environment used

- macOS arm64
- Python 3.12
- PyTorch 2.13.0
- `triton-msl==0.1.0a2`
- `flash-linear-attention==0.5.2`
- Triton built from source from `triton-lang/triton`

## What worked

1. Created an isolated environment:

   ```bash
   python3 -m venv .venv-msl
   source .venv-msl/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install torch pyyaml numpy pytest
   pip install git+https://github.com/bledden/triton-msl.git
   pip install ninja cmake lit
   pip install git+https://github.com/triton-lang/triton.git
   pip install -e . einops
   pip install --no-build-isolation --no-deps git+https://github.com/fla-org/flash-linear-attention
   ```

2. Verified successful imports:

   - `import triton`
   - `import triton.language`
   - `import fla`

3. With a small compatibility shim in `src/hz0/model/backends.py`, the vendored
   `GatedDeltaNet-2` module becomes importable and:

   ```bash
   python -m hz0.backend_check
   ```

   reports:

   - `gdn2_available=True`

## Compatibility shim used

The installable FLA build no longer exports `USE_CUDA_GRAPH` from `fla.utils`,
but the vendored GDN-2 code imports it. This branch injects:

```python
fla.utils.USE_CUDA_GRAPH = False
```

when absent, which is sufficient to restore import-time compatibility.

## Current hard blocker

A real forward pass still fails.

### MPS path

The first attempted forward pass on `mps` fails with:

```text
RuntimeError: PyTorch was compiled without CUDA support
```

because the FLA short-convolution path enters a CUDA device context.

### CPU path

The CPU retry goes further but eventually fails inside Triton runtime driver
selection with:

```text
RuntimeError: 0 active drivers ([]). There should only be one.
```

This indicates that `triton-msl` plus upstream Triton is enough to satisfy the
Python import surface, but not yet enough to make the FLA/GDN-2 runtime execute
cleanly on this machine.

## Current conclusion

This Mac branch is a meaningful step forward:

- we moved from “missing Triton entirely” to
- “Triton, Triton language, FLA, and GDN-2 imports all work”

But it is not yet a runnable upstream GDN-2 execution path.

The next likely experiment would be one of:

1. patch FLA device-context logic to avoid unconditional CUDA paths on MPS
2. investigate why Triton reports zero active drivers under `triton-msl`
3. pin older or different FLA / Triton revisions closer to the vendored GDN-2 expectations
