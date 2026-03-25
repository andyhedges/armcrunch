# NEON FFT — Closing the gap to gwnum

## Goal

Beat gwnum-under-Rosetta (~0.6 ms/iter on Apple Silicon) with a native ARM64
Rust implementation. Current best: 4.3 ms/iter (7.4× gap).

## Background

### Where the time goes (profiled on Apple Silicon, 1003·2^2499999−1)

| Phase | Estimated % | Notes |
|-------|-----------|-------|
| FFT (r2c) | ~35% | realfft forward transform |
| IFFT (c2r) | ~35% | realfft inverse transform |
| Pointwise square | ~5% | N/2+1 complex multiplies |
| Carry + kbn reduce | ~15% | Carry propagation + BigUint kbn |
| Signal fill + weight | ~10% | Loading limbs, applying weights |

The FFT/IFFT pair dominates at ~70% of total time. This is the target.

### Why gwnum is faster

gwnum achieves 0.6 ms/iter under Rosetta because:
1. **Hand-tuned x86 SIMD assembly** — AVX2 radix-4/8 butterflies with 4-way or 8-way parallelism
2. **Mixed-radix transform lengths** — e.g., 196608 = 3×2^16, which is smaller than our power-of-2 transform
3. **True in-domain IBDWT** — no BigUint round-trip at all (even for k>1)
4. **Careful error bound management** — balanced-digit representation reduces FFT dynamic range

Under Rosetta, the x86 SIMD instructions are translated to ARM64 equivalents with ~2-3× overhead, making gwnum run at ~0.6 ms instead of ~0.36 ms natively.

### What native ARM64 NEON provides

ARM64 NEON has 128-bit SIMD registers (32 of them: v0-v31), each holding 2×f64.
Key instructions for FFT butterflies:
- `fmla` / `fmls` — fused multiply-add/subtract (f64×2)
- `fadd` / `fsub` — vector add/subtract (f64×2)
- `fmul` — vector multiply (f64×2)
- `ld2` / `st2` — interleaved load/store of complex pairs
- `zip1` / `zip2` — interleave vector elements

A NEON radix-4 butterfly processes 2 complex values per instruction, giving 2×
throughput over scalar. Apple M-series chips have very wide execution units and
can often sustain 2 NEON operations per cycle, effectively giving 4× throughput.

### Target: how fast could we theoretically go?

gwnum on native x86 (this sandbox): 0.36 ms/iter with AVX2 (4×f64 = 256-bit).
NEON is 2×f64 = 128-bit, so theoretical peak is ~2× slower than AVX2.
Expected native ARM64 NEON: ~0.7-1.0 ms/iter (competitive with Rosetta gwnum).

Apple Silicon's high clock speed and wide execution pipeline could compensate
for the narrower SIMD width, potentially reaching 0.5-0.8 ms/iter.

## Key files

| File | Role |
|------|------|
| `src/dwt.rs` | DwtSquarer — where the FFT is called |
| `src/neon_fft.rs` | New: NEON-accelerated FFT module |
| `Cargo.toml` | May need `std::arch` feature flags |

## Shared constants

```
Target number: 1003·2^2499999 − 1  (~752K digits)
Current FFT length (IBDWT, k=1): depends on exp, chosen for f64 precision
Current FFT length (zero-padded, k>1): ~8192 16-bit limbs → fft_size ~16384
gwnum FFT length for same number: 196608 (3×2^16, mixed-radix)
```

## Task order

| Status | File | What it builds | Done when |
|--------|------|---------------|-----------|
| [ ] | [task-01-scaffold.md](task-01-scaffold.md) | Trait + module structure | `cargo test` passes with trait-based FFT dispatch |
| [ ] | [task-02-radix2-neon.md](task-02-radix2-neon.md) | NEON radix-2 butterfly | Radix-2 FFT matches realfft output |
| [ ] | [task-03-radix4-neon.md](task-03-radix4-neon.md) | NEON radix-4 butterfly | Radix-4 FFT faster than realfft |
| [ ] | [task-04-twiddle-table.md](task-04-twiddle-table.md) | Precomputed twiddle factors | No per-FFT trig computation |
| [ ] | [task-05-integration.md](task-05-integration.md) | Wire into DwtSquarer | `fermat_bench` shows improvement |
| [ ] | [task-06-mixed-radix.md](task-06-mixed-radix.md) | Radix-3/5 support | Smaller FFT lengths possible |

## Prerequisites

- **ARM64 development environment** — either macOS Apple Silicon or Linux ARM64 (Graviton)
- **Rust nightly or stable with `std::arch::aarch64`** — NEON intrinsics are stable since Rust 1.59
- **Benchmark baseline** — run `fermat_bench` before and after each task to measure improvement

## Marking a task complete

Change `[ ]` to `[x]` in the table above when the task's "Done when" criterion is met.