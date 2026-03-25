# Task 5 of 6 — Integration with DwtSquarer

## Goal

Replace realfft with the custom NEON FFT in DwtSquarer and verify correctness
and performance improvement.

## Steps

1. Implement `NeonFftEngine` that wraps the NEON radix-4 FFT with twiddle tables
2. Add the real FFT trick (pack/unpack) so it operates as a real-to-complex transform
3. Wire into DwtSquarer via the `FftEngine` trait from Task 1
4. Run all 32 tests to verify correctness
5. Run `fermat_bench` to measure improvement

## Expected results

| Metric | Before (realfft) | After (NEON) | Expected |
|--------|-----------------|--------------|----------|
| dwt ms/iter | 4.3 ms | ~2-3 ms | 1.5-2× faster |
| Gap to gwnum | 7.4× | ~3-5× | Significant progress |

## Correctness verification

The NEON FFT must produce results within f64 rounding tolerance of realfft.
Run the verification mode of `fermat_bench` to confirm all methods agree.

## Fallback

On non-ARM64 platforms (Linux x86_64, CI), the `RealFftEngine` fallback
ensures everything still compiles and works. The NEON code is gated behind
`#[cfg(target_arch = "aarch64")]`.

## Done when

```
cargo run --release --bin fermat_bench -- --duration 60
```

Shows dwt faster than 3.0 ms/iter on Apple Silicon.
All tests pass on both ARM64 and x86_64.