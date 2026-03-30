# gwnum ARM64 backend (Apple Silicon)

This directory contains a C/NEON ARM64 backend for gwnum that replaces the x86 assembly dispatch path with ARM64 entry points.

## Contents

- `arm64_cpuid.c` - ARM64 CPU detection (macOS sysctl + Linux fallback)
- `arm64_gwinfo.c` - `gwinfo1()` replacement and ARM64 jmptab tables
- `arm64_fft.c` - GWPROCPTRS[0] FFT dispatcher + one-pass FFT/mul/square path
- `arm64_aux.c` - GWPROCPTRS[1..8] add/sub/addsub/copy/muls routines
- `arm64_norm.c` - GWPROCPTRS[9..12] normalization variants
- `arm64_asm_data.h` - ARM64 asm_data union/view definitions and helpers
- `gwnum_arm64.patch` - minimal `gwnum.c` integration patch outline
- `makemacarm64` - static library build file for `gwnum.a` on arm64 macOS

## Build gwnum.a for macOS ARM64

From `armcrunch/gwnum_arm64/`:

```bash
make -f makemacarm64
```

This produces `gwnum.a` in the same directory.

Environment overrides:

- `GWNUM_SRC=/path/to/gwnum/sources`
- `BUILD_DIR=/custom/build/path`

## PRST integration

A companion ARM64 makefile is provided at:

- `prst/src/macarm64/Makefile`

It links against:

- `../../framework/gwnum/macarm64/gwnum.a`

## Architecture choices

- Backend uses C + NEON intrinsics (`arm_neon.h`) for portability and maintainability.
- FFT path supports one-pass contiguous layouts for 1K..32K complex points.
- IBDWT weighting is applied in forward FFT (`arm64_forward_weight_at`), and inverse weighting is applied during normalization.
- Carry propagation and roundoff tracking are implemented in the normalization path.
- All 13 required GWPROCPTRS slots are implemented and installed by `arm64_install_gwprocptrs()`.

## Current status / limitations

- Focused on one-pass kernels (no dedicated two-pass PASS2 microkernels yet).
- Twiddle usage prefers gwnum sin/cos tables when provided, with trig fallback for completeness.
- Inner loops are NEON-vectorized where practical; hand-tuned assembly can be added later for hot spots.
- The provided patch file intentionally keeps integration changes minimal and isolated with `#ifdef ARM64`.