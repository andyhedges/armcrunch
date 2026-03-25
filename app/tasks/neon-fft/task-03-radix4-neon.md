# Task 3 of 6 — NEON radix-4 butterfly

## Goal

Replace the radix-2 FFT with a radix-4 FFT for better instruction-level
parallelism. A radix-4 butterfly does 4 points at once, reducing the number
of passes by half and improving data locality.

## Background

A radix-4 DIT butterfly computes:
```
a' = (a + c) + (b + d)
b' = (a - c) - i(b - d)    [or +i for inverse]
c' = (a + c) - (b + d)
d' = (a - c) + i(b - d)    [or -i for inverse]
```

With NEON, we can process the real and imaginary parts of two complex
values simultaneously, giving 4 complex multiplies per iteration.

## Key optimization: split-radix

For even better performance, use a split-radix approach:
- Radix-4 for the bulk of the transform
- Radix-2 for the final pass when needed

This reduces the total operation count by ~17% compared to pure radix-2.

## Implementation notes

- Process 4 complex values per butterfly iteration
- Use `vld1q_f64` pairs to load 4 complex values (8 f64 values)
- Twiddle factor multiplication uses FMA instructions for accuracy
- Unroll the innermost loop for better pipeline utilization

## Done when

Radix-4 NEON FFT is measurably faster than radix-2 NEON FFT on the target
Apple Silicon hardware. Correctness verified against realfft.