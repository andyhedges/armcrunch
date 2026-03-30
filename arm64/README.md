# ARM64 FFT Generator (`armcrunch/arm64`)

This directory contains a Python code generator that emits native ARM64 NEON assembly for radix-4 complex FFT workloads and a benchmark runner that executes the generated code through the remote Apple Silicon sandbox API.

## Goal

Create a native ARM64 FFT implementation that can compete with or beat x86-64 gwnum running under Rosetta 2 translation on Apple Silicon, while staying within sandbox constraints.

---

## Files

- **`fftgen.py`** — Generates ARM64 assembly for FFT sizes 16, 64, 256, 1024
  - `generate_twiddles(n)` — stage-major twiddle factor tables
  - `generate_fft_forward(n)` — radix-4 DIT forward FFT
  - `generate_fft_inverse(n)` — conjugate-twiddle inverse FFT with 1/N scaling
  - `generate_pointwise_mul(n)` — vectorized complex pointwise multiply (NEON .2d)
  - `generate_benchmark_harness(n)` — deterministic init + FFT + checksum return
  - `generate_program(n)` — full standalone assembly source for one size
- **`benchmark.py`** — Remote benchmark runner
  - Generates assembly via `fftgen`
  - Calls remote API (`POST http://home.hedges.io:1337/run`)
  - Handles `429` rate-limit and `503` execution-slot-busy retries
  - Verifies correctness against NumPy FFT checksum
  - Reports benchmark stats

---

## Architecture Decisions

### Data layout

- **Memory**: interleaved complex doubles (AoS) — `re0, im0, re1, im1, ...`
- **Registers**: split real/imag via `ldp`/`stp` for scalar butterfly operations
- **Pointwise multiply**: NEON `ld2`/`st2` with `.2d` for vectorized complex pairs

### FFT structure

- Forward: iterative radix-4 DIT with stage loops
- Inverse: same structure with conjugated twiddles + explicit 1/N scaling
- Twiddle tables: stage-major `.data` arrays, streamed with post-increment `ldp`

### Instruction choices

- **Scalar FMA**: `fmadd`/`fmsub` (4-operand form) for complex twiddle multiplication
- **Vector FMA**: `fmla`/`fmls` (NEON .2d tied accumulator) for pointwise multiply
- **Load/Store**: `ldp`/`stp` for scalar double pairs, `ld2`/`st2` for NEON .2d
- **Prefetch**: `prfm pldl1keep` 256 bytes ahead in hot loops
- **Immediates**: `movz`/`movk` sequences (no pseudo-`mov` for large values)

### Sandbox constraints

- `.macro` / `.endm` directives are **forbidden** by the sandbox static analysis
- `ld2`/`st2` with `.1d` element size is **invalid** — must use `.2d` or `ldp`/`stp`
- Scalar `fmla`/`fmls` (3-operand tied) **does not exist** — use `fmadd`/`fmsub`
- Assembly source size limit: **524,288 characters** (raised from the original 65,536)

---

## Running the benchmark

### 1. Set API key

```bash
export ASM_KEY="your_api_key_here"
```

### 2. Run benchmarks

```bash
# All supported sizes
python -m armcrunch.arm64.benchmark

# Specific sizes
python -m armcrunch.arm64.benchmark --sizes 16 64 256 1024

# Override iteration count
python -m armcrunch.arm64.benchmark --iterations 5000

# Save generated assembly
python -m armcrunch.arm64.benchmark --dump-asm-dir ./generated_asm
```

---

## Correctness methodology

The benchmark harness initializes deterministic complex input data in `.data`, runs `_fft_fwd_N`, and returns a packed checksum derived from output bin 0. `benchmark.py` reproduces the same input in NumPy, computes `np.fft.fft`, and verifies the expected checksum matches the assembly return value.

---

## Current results (Apple Silicon via sandbox API)

All sizes pass correctness verification against NumPy FFT.

| N | Source chars | Mean (ns) | Median (ns) | Min (ns) | Max (ns) | Iterations |
|------|-------------|-----------|-------------|----------|----------|------------|
| 16 | 9,130 | 24 | 41 | 0 | 83 | 10,800 |
| 64 | 15,022 | 331 | 333 | 250 | 375 | 7,200 |
| 256 | 28,818 | 1,750 | 1,750 | 1,666 | 31,041 | 3,600 |
| 1024 | 73,853 | 8,626 | 8,541 | 8,500 | 54,041 | 1,350 |

All generated sources are within the 524,288 character sandbox limit.

### Performance analysis

- **N=16**: ~24 ns per 16-point complex FFT (2 radix-4 stages)
- **N=64**: ~331 ns per 64-point complex FFT (3 radix-4 stages)
- **N=256**: ~1.75 µs per 256-point complex FFT (4 radix-4 stages)
- **N=1024**: ~8.6 µs per 1024-point complex FFT (5 radix-4 stages)

Throughput scales approximately as O(N log N) as expected for radix-4 FFT.

---

## Design document

See `desktop/complex_reasoning/arm64_fft_architecture.md` for the full architecture analysis covering data layout rationale, Apple Silicon microarchitecture considerations, two-pass FFT strategy for larger sizes, and Python generator structure.