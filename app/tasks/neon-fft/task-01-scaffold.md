# Task 1 of 6 — FFT trait scaffold and module structure

## Goal

Create an abstraction layer that allows DwtSquarer to use either realfft
(current, portable) or a custom NEON FFT (new, ARM64-only), selected at
compile time or runtime.

## Design

### Trait definition

```rust
// src/fft_engine.rs (new file)

pub trait FftEngine {
    /// Forward real-to-complex FFT.  `input` is length N (modified in place),
    /// `output` is length N/2+1.
    fn forward(&mut self, input: &mut [f64], output: &mut [Complex<f64>]);

    /// Inverse complex-to-real FFT.  `input` is length N/2+1 (modified),
    /// `output` is length N.
    fn inverse(&mut self, input: &mut [Complex<f64>], output: &mut [f64]);

    /// FFT length N.
    fn len(&self) -> usize;
}
```

### Implementations

1. **`RealFftEngine`** — wraps the current `realfft` crate (portable).
   This is the default on all platforms.

2. **`NeonFftEngine`** — custom NEON implementation (ARM64 only).
   Compiled only with `#[cfg(target_arch = "aarch64")]`.
   Falls back to `RealFftEngine` on other architectures.

### DwtSquarer changes

Replace:
```rust
r2c: Arc<dyn RealToComplex<f64>>,
c2r: Arc<dyn ComplexToReal<f64>>,
scratch_r2c: Vec<Complex<f64>>,
scratch_c2r: Vec<Complex<f64>>,
```

With:
```rust
fft_engine: Box<dyn FftEngine>,
```

The `real_buf` and `spectrum` fields remain the same — the trait operates
on these buffers.

### Feature flag

```toml
# Cargo.toml
[features]
default = []
neon-fft = []  # Enable custom NEON FFT (ARM64 only)
```

When `neon-fft` is enabled AND the target is `aarch64`, use `NeonFftEngine`.
Otherwise, use `RealFftEngine`.

## Implementation steps

1. Create `src/fft_engine.rs` with the trait and `RealFftEngine` wrapper
2. Modify `DwtSquarer` to use `Box<dyn FftEngine>` instead of raw `r2c`/`c2r`
3. Verify all tests still pass (no behavioral change)
4. Add `NeonFftEngine` stub that delegates to `RealFftEngine` (placeholder for Task 2)

## Done when

```
cargo test
```

All 32 tests green. No performance regression (verify with `cargo bench`).
The trait abstraction is in place for Task 2 to plug in NEON butterflies.