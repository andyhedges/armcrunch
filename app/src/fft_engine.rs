//! FFT engine abstraction for pluggable FFT implementations.
//!
//! The `FftEngine` trait provides a uniform interface for real-to-complex
//! and complex-to-real FFT operations.  The default implementation wraps
//! the `realfft` crate (portable, works on all architectures).  A future
//! NEON-accelerated implementation can be swapped in on ARM64.

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Trait for real FFT engines used by DwtSquarer.
///
/// Implementations must provide forward (real → complex) and inverse
/// (complex → real) transforms.  The forward transform takes N real values
/// and produces N/2+1 complex values.  The inverse does the opposite.
///
/// Neither transform applies normalization — the caller must divide by N
/// after the inverse transform.
pub trait FftEngine: Send {
    /// Forward real-to-complex FFT.
    ///
    /// `input` is length N (contents are modified by the transform).
    /// `output` is length N/2+1 (overwritten with the complex spectrum).
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]);

    /// Inverse complex-to-real FFT.
    ///
    /// `input` is length N/2+1 (contents are modified by the transform).
    /// `output` is length N (overwritten with real-valued result).
    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], scratch: &mut [Complex<f64>]);

    /// FFT length N.
    fn len(&self) -> usize;

    /// Required scratch buffer length for forward transform.
    fn forward_scratch_len(&self) -> usize;

    /// Required scratch buffer length for inverse transform.
    fn inverse_scratch_len(&self) -> usize;
}

/// Default FFT engine wrapping the `realfft` crate.
///
/// This is portable and works on all architectures.  It uses RustFFT
/// internally with automatic SIMD detection where available.
pub struct RealFftEngine {
    len: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

impl RealFftEngine {
    /// Create a new RealFftEngine for transforms of length `len`.
    /// `len` must be a power of 2 and >= 4.
    pub fn new(len: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(len);
        let c2r = planner.plan_fft_inverse(len);
        RealFftEngine { len, r2c, c2r }
    }
}

impl FftEngine for RealFftEngine {
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
        self.r2c
            .process_with_scratch(input, output, scratch)
            .expect("real-to-complex FFT failed");
    }

    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], scratch: &mut [Complex<f64>]) {
        self.c2r
            .process_with_scratch(input, output, scratch)
            .expect("complex-to-real IFFT failed");
    }

    fn len(&self) -> usize {
        self.len
    }

    fn forward_scratch_len(&self) -> usize {
        self.r2c.get_scratch_len()
    }

    fn inverse_scratch_len(&self) -> usize {
        self.c2r.get_scratch_len()
    }
}