# gwnum ARM64 backend (Apple Silicon)

This directory contains a C/NEON ARM64 backend for gwnum that replaces the x86 assembly dispatch path with ARM64 entry points, enabling PRST and other gwnum-based programs to run natively on Apple Silicon.

## Contents

| File | Purpose |
|------|---------|
| `arm64_asm_data.h` | ARM64 asm_data union/view definitions, helper accessors, routine prototypes |
| `arm64_cpuid.c` | ARM64 CPU detection (macOS sysctl + Linux /proc/cpuinfo fallback) |
| `arm64_gwinfo.c` | `gwinfo1()` replacement with ARM64 jmptab tables (1K–32K one-pass FFTs) |
| `arm64_fft.c` | GWPROCPTRS[0]: FFT dispatcher with radix-4 forward/inverse, IBDWT weighting, pointwise mul/square, ffttype 1–5 |
| `arm64_aux.c` | GWPROCPTRS[1–8]: add/sub/addsub (quick + normalized), 4KB copy, small-constant multiply |
| `arm64_norm.c` | GWPROCPTRS[9–12]: normalization with inverse weighting, rounding, carry propagation, error check, mulbyconst |
| `gwnum_arm64_integration.c` | Hook functions for gwnum.c (`arm64_gwinfo_hook`, `arm64_gwsetup_hook`) |
| `patch_gwnum_for_arm64.sh` | Build-time script that patches gwnum.c with `#ifdef ARM64` blocks |
| `makemacarm64` | Makefile for building `gwnum.a` on arm64 macOS |

## Build gwnum.a for macOS ARM64

The build process patches gwnum.c at build time (the original is not modified), then compiles the patched core sources alongside the ARM64 backend modules.

```bash
cd armcrunch/gwnum_arm64
make -f makemacarm64
```

This produces `gwnum.a` in the current directory.

The patch script (`patch_gwnum_for_arm64.sh`) applies these changes to gwnum.c:
1. Adds `#include "arm64_asm_data.h"` and ARM64 hook declarations
2. Guards all x86 assembly extern declarations with `#if !defined(ARM64)`
3. Replaces `gwinfo1()` call with `arm64_gwinfo_hook()` on ARM64
4. Replaces x86 GWPROCPTRS assignment with `arm64_gwsetup_hook()` on ARM64
5. Guards `fpu_init()` call (no-op on ARM64)

## PRST integration

A companion ARM64 makefile is provided at `prst/src/macarm64/Makefile`.

```bash
# 1. Build gwnum.a
cd armcrunch/gwnum_arm64
make -f makemacarm64

# 2. Install gwnum.a into PRST framework
mkdir -p /path/to/prst/framework/gwnum/macarm64
cp gwnum.a /path/to/prst/framework/gwnum/macarm64/

# 3. Build PRST
cd prst/src/macarm64
make
```

## Architecture

- All 13 GWPROCPTRS entry points are implemented in C with NEON intrinsics (`arm_neon.h`).
- FFT uses radix-4 DIT (forward) and radix-4 DIF (inverse) with bit-reversal permutation.
- IBDWT pre-weighting is applied during forward FFT; inverse weighting during normalization.
- Carry propagation and roundoff error tracking are in the normalization path.
- NEON vectorization is used for pointwise multiply, add/sub, scaling, and copy operations.
- Hot inner loops can be replaced with hand-tuned assembly later for further optimization.

## Current limitations

- One-pass FFT kernels only (1K–32K complex points). Two-pass support for larger sizes is not yet implemented.
- Twiddle factors use computed trig when gwnum sin/cos tables are not populated.
- Single-threaded (no pass1/pass2 auxiliary thread dispatch yet).
- Not yet tested end-to-end with PRST (requires an ARM64 macOS build environment).