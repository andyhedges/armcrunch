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
use std::cell::UnsafeCell;
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
// Precomputed twiddle tables
// ---------------------------------------------------------------------------

/// Precomputed twiddle factors for all FFT passes and real FFT unpack.
/// Built once at engine construction; eliminates all per-transform trig.
struct TwiddleTable {
    /// Twiddle factors for each radix-2 pass (forward direction).
    /// forward[pass][k] = (cos(-π·k/half), sin(-π·k/half))
    forward: Vec<Vec<(f64, f64)>>,
    /// Twiddle factors for each radix-2 pass (inverse direction).
    inverse: Vec<Vec<(f64, f64)>>,
    /// Real FFT unpack twiddles (forward): e^{-2πik/N} for k=0..M
    real_fwd: Vec<Complex<f64>>,
    /// Real FFT unpack twiddles (inverse): e^{+2πik/N} for k=0..M
    real_inv: Vec<Complex<f64>>,
}

impl TwiddleTable {
    /// Build twiddle tables for a complex FFT of length `complex_n`.
    /// The corresponding real FFT length is `2 * complex_n`.
    fn new(complex_n: usize) -> Self {
        let mut forward = Vec::new();
        let mut inverse = Vec::new();

        let mut half = 1;
        while half < complex_n {
            let mut fwd_pass = Vec::with_capacity(half);
            let mut inv_pass = Vec::with_capacity(half);
            for k in 0..half {
                let fwd_theta = -PI * k as f64 / half as f64;
                let inv_theta = PI * k as f64 / half as f64;
                fwd_pass.push((fwd_theta.cos(), fwd_theta.sin()));
                inv_pass.push((inv_theta.cos(), inv_theta.sin()));
            }
            forward.push(fwd_pass);
            inverse.push(inv_pass);
            half *= 2;
        }

        let real_n = complex_n * 2;
        let mut real_fwd = Vec::with_capacity(complex_n + 1);
        let mut real_inv = Vec::with_capacity(complex_n + 1);
        for k in 0..=complex_n {
            let fwd_theta = -2.0 * PI * k as f64 / real_n as f64;
            let inv_theta = 2.0 * PI * k as f64 / real_n as f64;
            real_fwd.push(Complex::new(fwd_theta.cos(), fwd_theta.sin()));
            real_inv.push(Complex::new(inv_theta.cos(), inv_theta.sin()));
        }

        TwiddleTable { forward, inverse, real_fwd, real_inv }
    }
}

// ---------------------------------------------------------------------------
// Radix-2 DIT complex FFT with precomputed twiddle tables
// ---------------------------------------------------------------------------

/// In-place radix-2 DIT FFT using precomputed twiddle factors.
/// No normalization applied.
fn complex_fft_precomp(data: &mut [Complex<f64>], n: usize, twiddles: &[Vec<(f64, f64)>]) {
    if n <= 1 {
        return;
    }

    bit_reverse_permute(data, n);

    let stages = twiddles.len();
    let mut pass = 0usize;

    // If log2(n) is odd, do one radix-2 pass first.
    if stages & 1 == 1 {
        let half = 1usize;
        let step = 2usize;
        let tw = &twiddles[0];

        for k in 0..half {
            let (w_re, w_im) = tw[k];
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

        pass = 1;
    }

    // Fuse passes p and p+1 together over groups of 4*h.
    while pass + 1 < stages {
        let h = 1usize << pass;
        let h2 = h * 2;
        let step = h2 * 2;

        let tw1 = &twiddles[pass];
        let tw2 = &twiddles[pass + 1];

        for k in 0..h {
            let (w1_re, w1_im) = tw1[k];
            let (w2a_re, w2a_im) = tw2[k];
            let (w2b_re, w2b_im) = tw2[k + h];

            let mut j = k;
            while j < n {
                let i1 = j + h;
                let i2 = j + h2;
                let i3 = i2 + h;

                let a0 = data[j];
                let a1_raw = data[i1];
                let a2 = data[i2];
                let a3_raw = data[i3];

                // Inner radix-2 level (twiddle w1).
                let a1 = Complex::new(
                    w1_re * a1_raw.re - w1_im * a1_raw.im,
                    w1_re * a1_raw.im + w1_im * a1_raw.re,
                );
                let a3 = Complex::new(
                    w1_re * a3_raw.re - w1_im * a3_raw.im,
                    w1_re * a3_raw.im + w1_im * a3_raw.re,
                );

                let b0 = a0 + a1;
                let b1 = a0 - a1;
                let b2 = a2 + a3;
                let b3 = a2 - a3;

                // Outer radix-2 level (twiddles w2a/w2b).
                let c2 = Complex::new(
                    w2a_re * b2.re - w2a_im * b2.im,
                    w2a_re * b2.im + w2a_im * b2.re,
                );
                let c3 = Complex::new(
                    w2b_re * b3.re - w2b_im * b3.im,
                    w2b_re * b3.im + w2b_im * b3.re,
                );

                data[j] = b0 + c2;
                data[i2] = b0 - c2;
                data[i1] = b1 + c3;
                data[i3] = b1 - c3;

                j += step;
            }
        }

        pass += 2;
    }
}

// ---------------------------------------------------------------------------
// Real FFT with precomputed twiddles and external buffer
// ---------------------------------------------------------------------------

/// Forward real FFT using precomputed twiddles and caller-provided buffer.
fn real_fft_forward_precomp(
    input: &[f64], output: &mut [Complex<f64>], z: &mut [Complex<f64>],
    fft_tw: &[Vec<(f64, f64)>], real_tw: &[Complex<f64>],
) {
    let n = input.len();
    let m = n / 2;
    debug_assert_eq!(output.len(), m + 1);
    debug_assert!(z.len() >= m);

    for k in 0..m { z[k] = Complex::new(input[2 * k], input[2 * k + 1]); }
    complex_fft_precomp(&mut z[..m], m, fft_tw);

    for k in 0..=m {
        let zk = z[k % m];
        let zmk = z[if k == 0 { 0 } else { m - k }].conj();
        let a = zk + zmk;
        let b = zk - zmk;
        let tw = real_tw[k];
        let neg_i_b = Complex::new(b.im, -b.re);
        output[k] = (a + neg_i_b * tw) * 0.5;
    }
}

/// Inverse real FFT using precomputed twiddles and caller-provided buffer.
/// No normalization applied.
fn real_fft_inverse_precomp(
    input: &[Complex<f64>], output: &mut [f64], z: &mut [Complex<f64>],
    fft_tw: &[Vec<(f64, f64)>], real_tw: &[Complex<f64>],
) {
    let m = input.len() - 1;
    debug_assert_eq!(output.len(), m * 2);
    debug_assert!(z.len() >= m);

    for k in 0..m {
        let xk = input[k];
        let xmk = input[if k == 0 { m } else { m - k }].conj();
        let a = xk + xmk;
        let b = xk - xmk;
        let tw = real_tw[k];
        let i_b = Complex::new(-b.im, b.re);
        z[k] = a + i_b * tw;
    }

    complex_fft_precomp(&mut z[..m], m, fft_tw);

    for k in 0..m {
        output[2 * k] = z[k].re;
        output[2 * k + 1] = z[k].im;
    }
}

// ---------------------------------------------------------------------------
// NeonFftEngine — only available on aarch64
// ---------------------------------------------------------------------------

/// FFT engine using a custom radix-2 DIT FFT with NEON-accelerated butterflies
/// and precomputed twiddle tables.
///
/// # Interior Mutability
///
/// Uses `UnsafeCell` for the internal working buffer `z_buf` to allow mutation
/// through the `&self` interface required by the `FftEngine` trait.
///
/// ## Safety invariant
///
/// This is safe because `DwtSquarer` is single-threaded: `forward` and `inverse`
/// are never called concurrently, and no reference to `z_buf` escapes these
/// methods. The `Send` implementation is safe because the engine is never shared
/// between threads (it is owned by a single `DwtSquarer`).
#[cfg(target_arch = "aarch64")]
pub struct NeonFftEngine {
    len: usize,
    twiddles: TwiddleTable,
    /// Working buffer for the N/2-point complex FFT.
    /// Wrapped in `UnsafeCell` for interior mutability through `&self`.
    z_buf: UnsafeCell<Vec<Complex<f64>>>,
}

/// SAFETY: `NeonFftEngine` is only used within a single-threaded `DwtSquarer`.
/// The `UnsafeCell<Vec<Complex<f64>>>` is never accessed from multiple threads.
#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonFftEngine {}

#[cfg(target_arch = "aarch64")]
impl NeonFftEngine {
    pub fn new(len: usize) -> Self {
        assert!(len >= 4 && len.is_power_of_two(), "FFT length must be a power of 2 >= 4");
        let half = len / 2;
        NeonFftEngine {
            len,
            twiddles: TwiddleTable::new(half),
            z_buf: UnsafeCell::new(vec![Complex::new(0.0, 0.0); half]),
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl crate::fft_engine::FftEngine for NeonFftEngine {
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
        debug_assert_eq!(input.len(), self.len);
        debug_assert_eq!(output.len(), self.len / 2 + 1);
        // SAFETY: see struct-level safety invariant documentation.
        let z = unsafe { &mut *self.z_buf.get() };
        real_fft_forward_precomp(input, output, z, &self.twiddles.forward, &self.twiddles.real_fwd);
    }

    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], _scratch: &mut [Complex<f64>]) {
        debug_assert_eq!(input.len(), self.len / 2 + 1);
        debug_assert_eq!(output.len(), self.len);
        let z = unsafe { &mut *self.z_buf.get() };
        real_fft_inverse_precomp(input, output, z, &self.twiddles.inverse, &self.twiddles.real_inv);
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

            let half = n / 2;
            let tw = TwiddleTable::new(half);
            let mut z_buf = vec![Complex::new(0.0, 0.0); half];
            let mut custom_output = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            real_fft_forward_precomp(&input, &mut custom_output, &mut z_buf, &tw.forward, &tw.real_fwd);

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

            let half = n / 2;
            let tw = TwiddleTable::new(half);
            let mut z_buf = vec![Complex::new(0.0, 0.0); half];
            let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            let mut output = vec![0.0f64; n];

            real_fft_forward_precomp(&original, &mut spectrum, &mut z_buf, &tw.forward, &tw.real_fwd);
            real_fft_inverse_precomp(&spectrum, &mut output, &mut z_buf, &tw.inverse, &tw.real_inv);

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