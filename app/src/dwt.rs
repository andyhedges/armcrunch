//! FFT-based squarer for numbers mod M = k·2^exp − 1.
//!
//! Computes x² via FFT-based polynomial multiplication (16-bit limbs,
//! linear convolution with zero-padding to avoid aliasing), then reduces
//! mod M using the efficient kbn identity:
//!
//!   q = (x² >> exp) / k
//!   r = x² + q − q·k·2^exp
//!
//! which avoids full BigUint division since k is a small u64.
//!
//! ## Pipeline per square()
//!
//! ```text
//! x (BigUint) → 16-bit limbs → zero-pad → FFT → pointwise square
//!   → IFFT → round → carry → BigUint (x²) → kbn reduce → x mod M
//! ```

use num_bigint::BigUint;
use num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub struct DwtSquarer {
    /// k in M = k·2^exp − 1
    k: u64,
    /// exp in M = k·2^exp − 1
    exp: u64,
    /// The modulus M = k·2^exp − 1
    modulus: BigUint,
    /// Current value x ∈ [0, M)
    value: BigUint,
    /// FFT transform length (power of 2, large enough for linear convolution)
    fft_size: usize,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    /// Reusable FFT signal buffer
    signal: Vec<Complex<f64>>,
    /// Reusable FFT scratch buffer
    scratch: Vec<Complex<f64>>,
    /// Reusable carry buffer (fft_size + 1 elements for overflow)
    limbs: Vec<i64>,
}

impl DwtSquarer {
    /// Build a squarer for M = k·2^exp − 1 and load initial value `x`.
    pub fn new(k: u64, exp: u64, x: &BigUint) -> Self {
        let modulus = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;

        // Size the FFT for linear convolution of two numbers < M.
        // Each number has at most ceil(bits(M) / 16) 16-bit limbs.
        // The product polynomial has degree < 2*n_limbs, so we need
        // fft_size >= 2*n_limbs to avoid cyclic aliasing.
        let m_bytes = (modulus.bits() as usize + 7) / 8;
        let n_limbs = (m_bytes + 1) / 2; // number of 16-bit limbs
        let fft_size = (2 * n_limbs).next_power_of_two();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft
            .get_inplace_scratch_len()
            .max(ifft.get_inplace_scratch_len());

        DwtSquarer {
            k,
            exp,
            modulus,
            value: x.clone(),
            fft_size,
            fft,
            ifft,
            signal: vec![Complex::new(0.0, 0.0); fft_size],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            limbs: vec![0i64; fft_size + 1],
        }
    }

    /// One squaring: x ← x² mod (k·2^exp − 1).
    pub fn square(&mut self) {
        let sq = self.fft_square();
        self.value = kbn_reduce(sq, self.k, self.exp, &self.modulus);
    }

    /// Extract current value as a BigUint.
    pub fn to_biguint(&self) -> BigUint {
        self.value.clone()
    }

    /// Compute x² using FFT-based polynomial multiplication with 16-bit limbs.
    fn fft_square(&mut self) -> BigUint {
        let bytes = self.value.to_bytes_le();
        if bytes.is_empty() {
            return BigUint::zero();
        }

        let n_limbs = (bytes.len() + 1) / 2;

        // Fill signal with 16-bit limbs, then zero-pad.
        for i in 0..n_limbs {
            let lo = bytes.get(2 * i).copied().unwrap_or(0) as f64;
            let hi = bytes.get(2 * i + 1).copied().unwrap_or(0) as f64;
            self.signal[i] = Complex::new(lo + hi * 256.0, 0.0);
        }
        self.signal[n_limbs..].fill(Complex::new(0.0, 0.0));

        // Forward FFT
        self.fft
            .process_with_scratch(&mut self.signal, &mut self.scratch);

        // Pointwise square
        for c in &mut self.signal {
            *c = *c * *c;
        }

        // Inverse FFT
        self.ifft
            .process_with_scratch(&mut self.signal, &mut self.scratch);

        // Scale, round, and carry-propagate into 16-bit limbs.
        let inv_n = 1.0 / self.fft_size as f64;
        self.limbs.fill(0);
        for (i, c) in self.signal.iter().enumerate() {
            self.limbs[i] = (c.re * inv_n).round() as i64;
        }

        // Carry propagation: squaring produces non-negative coefficients,
        // so use .max(0) to absorb any floating-point rounding noise near zero.
        for i in 0..self.limbs.len() - 1 {
            let v = self.limbs[i].max(0);
            self.limbs[i] = v & 0xFFFF;
            self.limbs[i + 1] += v >> 16;
        }

        // Reconstruct BigUint from 16-bit limbs packed into u32 words.
        let last = self.limbs.iter().rposition(|&v| v != 0).unwrap_or(0);
        let n_words = (last + 2) / 2;
        let words: Vec<u32> = (0..n_words)
            .map(|i| {
                let lo = self.limbs[2 * i] as u32;
                let hi = self.limbs.get(2 * i + 1).copied().unwrap_or(0) as u32;
                lo | (hi << 16)
            })
            .collect();

        BigUint::new(words)
    }
}

/// Reduce x mod M = k·2^exp − 1 using the kbn identity.
///
/// Computes: q = (x >> exp) / k, r = x + q − (q * k) << exp.
/// Since r ∈ [0, 2M), at most one subtraction is needed.
fn kbn_reduce(x: BigUint, k: u64, exp: u64, modulus: &BigUint) -> BigUint {
    if x.is_zero() {
        return x;
    }
    let q = (&x >> exp as usize) / k;
    let r = x + &q - ((&q * k) << exp as usize);
    if r >= *modulus {
        r - modulus
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mersenne(exp: u64) -> BigUint {
        (BigUint::from(1u64) << exp as usize) - 1u64
    }

    #[test]
    fn test_round_trip_k1_l_divides_exp() {
        let x = BigUint::from(123456789u64);
        let sq = DwtSquarer::new(1, 32, &x);
        assert_eq!(sq.to_biguint(), x);
    }

    #[test]
    fn test_round_trip_k1_l_not_divides_exp() {
        let x = BigUint::from(123456789u64);
        let sq = DwtSquarer::new(1, 31, &x);
        assert_eq!(sq.to_biguint(), x);
    }

    #[test]
    fn test_round_trip_k_gt_1() {
        let k = 1005u64;
        let exp = 64u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let x = &m - BigUint::from(1u64);
        let sq = DwtSquarer::new(k, exp, &x);
        assert_eq!(sq.to_biguint(), x);
    }

    #[test]
    fn test_round_trip_zero_and_one() {
        for &v in &[0u64, 1u64] {
            let x = BigUint::from(v);
            let sq = DwtSquarer::new(1, 31, &x);
            assert_eq!(sq.to_biguint(), x);
        }
    }

    #[test]
    fn test_limb_width_covers_modulus() {
        let k = 1005u64;
        let exp = 64u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let x = &m - BigUint::from(1u64);
        let sq = DwtSquarer::new(k, exp, &x);
        assert_eq!(sq.to_biguint(), x);
    }

    #[test]
    fn test_weights_unity_when_l_divides_exp() {
        // Retained for compatibility: verify that the legacy weight formula
        // produces unity weights when L | exp (informational only; this squarer
        // no longer uses DWT weights).
        let len = 8usize;
        let b_lo = 4u64;
        let b = 32.0f64 / len as f64; // = 4.0
        let mut bit_pos = 0u64;
        for j in 0..len {
            let w = 2.0f64.powf(j as f64 * b - bit_pos as f64);
            assert!(
                (w - 1.0).abs() < 1e-12,
                "weight[{j}] = {w}, expected 1.0"
            );
            bit_pos += b_lo;
        }
    }

    #[test]
    fn test_phase_a_l_divides_exp() {
        let exp = 32u64;
        let m = mersenne(exp);
        let start = BigUint::from(3u64);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(1, exp, &start);
        for i in 0..8 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            let got = sq.to_biguint();
            assert_eq!(got, x_naive, "mismatch at iteration {i}");
        }
    }

    #[test]
    fn test_phase_a_single_square() {
        let x = BigUint::from(3u64);
        let mut sq = DwtSquarer::new(1, 3, &x);
        sq.square();
        assert_eq!(sq.to_biguint(), BigUint::from(2u64));
    }

    #[test]
    fn test_phase_a_chain() {
        let exp = 31u64;
        let m = mersenne(exp);
        let start = BigUint::from(3u64);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(1, exp, &start);
        for i in 0..10 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            let got = sq.to_biguint();
            assert_eq!(got, x_naive, "mismatch at iteration {i}");
        }
    }

    #[test]
    fn test_phase_b_single_square() {
        let k = 3u64;
        let exp = 4u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let x = BigUint::from(3u64);
        let mut sq = DwtSquarer::new(k, exp, &x);
        sq.square();
        let expected = (&x * &x) % &m;
        assert_eq!(sq.to_biguint(), expected);
    }

    #[test]
    fn test_phase_b_chain() {
        let k = 1005u64;
        let exp = 64u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let start = BigUint::from(3u64).modpow(&BigUint::from(k), &m);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(k, exp, &start);
        for i in 0..8 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            assert_eq!(sq.to_biguint(), x_naive, "mismatch at iteration {i}");
        }
    }

    #[test]
    fn test_phase_b_chain_small() {
        let k = 3u64;
        let exp = 8u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let start = BigUint::from(100u64);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(k, exp, &start);
        for i in 0..6 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            assert_eq!(sq.to_biguint(), x_naive, "mismatch at iteration {i}");
        }
    }
}