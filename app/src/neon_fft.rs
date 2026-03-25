//! Custom radix-2 DIT FFT with optional NEON acceleration.
//!
//! On `aarch64`, the butterfly inner loop uses NEON SIMD intrinsics for
//! 2×f64 parallel complex multiply-add.  On other architectures, a scalar
//! fallback is used for the butterfly (used in tests only; production code
//! on non-aarch64 uses `RealFftEngine` instead).
//!
//! The real FFT is implemented via the standard packing trick:
//! - Forward: pack N reals into N/2 complex, run N/2-point complex FFT,
//!   unpack to N/2+1 complex spectrum values.
//! - Inverse: reverse the unpack, run N/2-point complex IFFT, unpack reals.

use rustfft::num_complex::Complex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Bit-reversal permutation
// ---------------------------------------------------------------------------

fn bit_reverse_permute(data: &mut [Complex<f64>], n: usize) {
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Butterfly implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn butterfly(a_ptr: *mut f64, b_ptr: *mut f64, w_re: f64, w_im: f64) {
    use std::arch::aarch64::*;

    unsafe {
        let va = vld1q_f64(a_ptr);
        let vb = vld1q_f64(b_ptr);

        let vw_re = vdupq_n_f64(w_re);
        let vw_im = vdupq_n_f64(w_im);
        let vb_swap = vextq_f64(vb, vb, 1);

        let prod1 = vmulq_f64(vw_re, vb);
        let prod2 = vmulq_f64(vw_im, vb_swap);

        let negate = vsetq_lane_f64(-1.0, vdupq_n_f64(1.0), 0);
        let wb = vfmaq_f64(prod1, negate, prod2);

        let a_out = vaddq_f64(va, wb);
        let b_out = vsubq_f64(va, wb);

        vst1q_f64(a_ptr, a_out);
        vst1q_f64(b_ptr, b_out);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn butterfly(a_ptr: *mut f64, b_ptr: *mut f64, w_re: f64, w_im: f64) {
    unsafe {
        let a_re = *a_ptr;
        let a_im = *a_ptr.add(1);
        let b_re = *b_ptr;
        let b_im = *b_ptr.add(1);

        let wb_re = w_re * b_re - w_im * b_im;
        let wb_im = w_re * b_im + w_im * b_re;

        *a_ptr = a_re + wb_re;
        *a_ptr.add(1) = a_im + wb_im;
        *b_ptr = a_re - wb_re;
        *b_ptr.add(1) = a_im - wb_im;
    }
}

// ---------------------------------------------------------------------------
// Radix-2 DIT complex FFT (in-place)
// ---------------------------------------------------------------------------

/// In-place radix-2 decimation-in-time complex FFT.
/// No normalization is applied (caller divides by N for inverse).
fn complex_fft(data: &mut [Complex<f64>], n: usize, inverse: bool) {
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of 2");

    bit_reverse_permute(data, n);

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut half = 1;
    while half < n {
        let step = half * 2;
        let angle = sign * PI / half as f64;

        for k in 0..half {
            let theta = angle * k as f64;
            let w_re = theta.cos();
            let w_im = theta.sin();

            let mut j = k;
            while j < n {
                let i2 = j + half;
                unsafe {
                    let a_ptr = &mut data[j] as *mut Complex<f64> as *mut f64;
                    let b_ptr = &mut data[i2] as *mut Complex<f64> as *mut f64;
                    butterfly(a_ptr, b_ptr, w_re, w_im);
                }
                j += step;
            }
        }
        half = step;
    }
}

// ---------------------------------------------------------------------------
// Real FFT via packing trick
// ---------------------------------------------------------------------------

/// Forward real FFT: N reals → N/2+1 complex spectrum.
///
/// Standard algorithm:
/// 1. Pack N reals into N/2 complex: z[k] = x[2k] + i·x[2k+1]
/// 2. Compute N/2-point complex FFT of z
/// 3. Unpack using:
///    X[k] = (Z[k] + conj(Z[M-k])) / 2
///          - i · (Z[k] - conj(Z[M-k])) / 2 · W_N^k
///    where M = N/2, W_N = e^{-2πi/N}, and Z[M] wraps to Z[0].
fn real_fft_forward(input: &[f64], output: &mut [Complex<f64>]) {
    let n = input.len();
    let m = n / 2;
    debug_assert_eq!(output.len(), m + 1);

    // Step 1: Pack reals into complex
    let mut z: Vec<Complex<f64>> = (0..m)
        .map(|k| Complex::new(input[2 * k], input[2 * k + 1]))
        .collect();

    // Step 2: N/2-point complex FFT
    complex_fft(&mut z, m, false);

    // Step 3: Unpack
    for k in 0..=m {
        let zk = z[k % m];
        let zmk = z[if k == 0 { 0 } else { m - k }].conj();

        let a = zk + zmk;       // = 2 * even_part
        let b = zk - zmk;       // = 2 * odd_part (before twiddle)

        // Twiddle: W_N^k = e^{-2πik/N}
        let theta = -2.0 * PI * k as f64 / n as f64;
        let tw = Complex::new(theta.cos(), theta.sin());

        // X[k] = a/2 + b/2 * (-i) * tw
        //      = a/2 + (-i*b/2) * tw
        let neg_i_b = Complex::new(b.im, -b.re); // -i * b
        output[k] = (a + neg_i_b * tw) * 0.5;
    }
}

/// Inverse real FFT: N/2+1 complex spectrum → N reals.
/// No normalization applied (caller divides by N).
///
/// Reverses the forward unpack to recover Z[k], then IFFT, then deinterleave.
fn real_fft_inverse(input: &[Complex<f64>], output: &mut [f64]) {
    let m = input.len() - 1;
    let n = m * 2;
    debug_assert_eq!(output.len(), n);

    // Step 1: Reverse the unpack to recover Z[k]
    // From forward: X[k] = (Z[k] + Z*[M-k])/2 + (-i)(Z[k] - Z*[M-k])/2 · W^k
    // So: Z[k] = X[k] + conj(X[M-k]) + i*(X[k] - conj(X[M-k])) * conj(W^k)
    //   (divided by appropriate factor)
    //
    // More precisely, the inverse relation is:
    // Z[k] = (X[k] + conj(X[M-k])) / 2
    //       + i · (X[k] - conj(X[M-k])) / 2 · conj(W_N^k)
    // where this uses the inverse twiddle (positive angle).
    let mut z: Vec<Complex<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let xk = input[k];
        let xmk = input[if k == 0 { m } else { m - k }].conj();

        let a = xk + xmk;
        let b = xk - xmk;

        let theta = 2.0 * PI * k as f64 / n as f64;
        let tw = Complex::new(theta.cos(), theta.sin());

        // i * b = (-b.im, b.re)
        let i_b = Complex::new(-b.im, b.re);
        z.push(a + i_b * tw);
    }

    // Step 2: N/2-point complex IFFT (no normalization)
    complex_fft(&mut z, m, true);

    // Step 3: Deinterleave real and imaginary into output
    for k in 0..m {
        output[2 * k] = z[k].re;
        output[2 * k + 1] = z[k].im;
    }
}

// ---------------------------------------------------------------------------
// NeonFftEngine — only available on aarch64
// ---------------------------------------------------------------------------

/// FFT engine using a custom radix-2 DIT FFT with NEON-accelerated butterflies.
/// Only available on `aarch64` targets.
#[cfg(target_arch = "aarch64")]
pub struct NeonFftEngine {
    len: usize,
}

#[cfg(target_arch = "aarch64")]
impl NeonFftEngine {
    pub fn new(len: usize) -> Self {
        assert!(len >= 4 && len.is_power_of_two(), "FFT length must be a power of 2 >= 4");
        NeonFftEngine { len }
    }
}

#[cfg(target_arch = "aarch64")]
impl crate::fft_engine::FftEngine for NeonFftEngine {
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
        debug_assert_eq!(input.len(), self.len);
        debug_assert_eq!(output.len(), self.len / 2 + 1);
        real_fft_forward(input, output);
    }

    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], _scratch: &mut [Complex<f64>]) {
        debug_assert_eq!(input.len(), self.len / 2 + 1);
        debug_assert_eq!(output.len(), self.len);
        real_fft_inverse(input, output);
    }

    fn len(&self) -> usize { self.len }
    fn forward_scratch_len(&self) -> usize { 0 }
    fn inverse_scratch_len(&self) -> usize { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft_engine::{FftEngine, RealFftEngine};

    /// Verify that our custom FFT matches RealFftEngine for forward transforms.
    #[test]
    fn test_custom_fft_matches_realfft() {
        for &n in &[4, 8, 16, 32, 64, 128, 256, 1024] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).cos()).collect();

            let real_engine = RealFftEngine::new(n);
            let mut real_input = input.clone();
            let mut real_output = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            let mut real_scratch = vec![Complex::new(0.0, 0.0); real_engine.forward_scratch_len()];
            real_engine.forward(&mut real_input, &mut real_output, &mut real_scratch);

            let mut custom_output = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            real_fft_forward(&input, &mut custom_output);

            for k in 0..=n / 2 {
                let diff_re = (real_output[k].re - custom_output[k].re).abs();
                let diff_im = (real_output[k].im - custom_output[k].im).abs();
                assert!(
                    diff_re < 1e-6 && diff_im < 1e-6,
                    "FFT size {n}, bin {k}: realfft=({:.6}, {:.6}), custom=({:.6}, {:.6}), diff=({:.2e}, {:.2e})",
                    real_output[k].re, real_output[k].im,
                    custom_output[k].re, custom_output[k].im,
                    diff_re, diff_im
                );
            }
        }
    }

    /// Verify forward → inverse round-trip produces the original signal.
    #[test]
    fn test_custom_fft_round_trip() {
        for &n in &[4, 8, 16, 64, 256] {
            let original: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();

            let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            let mut output = vec![0.0f64; n];

            real_fft_forward(&original, &mut spectrum);
            real_fft_inverse(&spectrum, &mut output);

            let inv_n = 1.0 / n as f64;
            for v in &mut output { *v *= inv_n; }

            for i in 0..n {
                let diff = (original[i] - output[i]).abs();
                assert!(
                    diff < 1e-10,
                    "Round-trip failed at index {i} for size {n}: expected {:.10}, got {:.10}, diff={:.2e}",
                    original[i], output[i], diff
                );
            }
        }
    }
}