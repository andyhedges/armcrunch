# Task 4 of 6 — Precomputed twiddle factor table

## Goal

Precompute and store all twiddle factors (cos/sin values) used in the FFT
butterflies, eliminating per-FFT trigonometric computation.

## Background

The current FFT computes `cos(theta)` and `sin(theta)` for each twiddle
factor during every FFT call. For a length-N FFT, this is O(N) trig calls
per transform — expensive on any architecture.

gwnum precomputes all twiddle factors in a table during setup, then the
FFT inner loop is pure multiply-add with no trig.

## Design

```rust
struct TwiddleTable {
    /// Twiddle factors for each FFT pass, stored as interleaved [re, im] pairs.
    /// table[pass][k] = e^{-2πi·k/2^(pass+1)} for forward FFT.
    factors: Vec<Vec<(f64, f64)>>,
    n: usize,
}

impl TwiddleTable {
    fn new(n: usize) -> Self {
        // Precompute all factors for log2(n) passes
        let mut factors = Vec::new();
        let mut half = 1;
        while half < n {
            let step = half * 2;
            let mut pass_factors = Vec::with_capacity(half);
            for k in 0..half {
                let theta = -std::f64::consts::TAU * k as f64 / step as f64;
                pass_factors.push((theta.cos(), theta.sin()));
            }
            factors.push(pass_factors);
            half *= 2;
        }
        TwiddleTable { factors, n }
    }
}
```

For NEON: store twiddle factors as contiguous f64 arrays aligned to 16 bytes
for efficient `vld1q_f64` loading.

## Done when

FFT inner loop has zero trig calls. Twiddle table is computed once in
`DwtSquarer::new()`. Measurable speedup on the FFT portion.