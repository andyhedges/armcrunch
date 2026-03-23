use num_bigint::BigUint;
use num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Square `x` in place mod `modulus`.
#[inline]
pub fn square_mod(x: &mut BigUint, modulus: &BigUint) {
    *x = (&*x * &*x) % modulus;
}

/// Original naive implementation, kept for benchmarking.
#[inline]
pub fn square_mod_naive(x: &mut BigUint, modulus: &BigUint) {
    *x = (&*x * &*x) % modulus;
}

/// Squaring mod N = k·2ⁿ-1 using structure of the modulus to avoid full bigint division.
///
/// Instead of computing x² % N directly, we use:
///   q = floor(x² / (k·2ⁿ)) = (x² >> n) / k   (cheap: shift + divide by small k)
///   r = x² - q·N = x² + q - q·k·2ⁿ           (cheap: multiply by u64 + shift)
///
/// r ∈ [0, 2N) so at most one subtraction corrects it.
#[inline]
pub fn square_mod_kbn(x: &mut BigUint, k: u64, n: u64, modulus: &BigUint) {
    let sq = &*x * &*x;
    let q = (&sq >> n as usize) / k;
    let r = sq + &q - ((&q * k) << n as usize);
    *x = if r >= *modulus { r - modulus } else { r };
}

/// FFT-based squarer for repeated squarings mod N = k·2ⁿ-1.
///
/// Pre-plans the FFT and reuses scratch buffers across calls.
pub struct FftSquarer {
    fft_size: usize,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    signal: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    limbs: Vec<i64>,
}

impl FftSquarer {
    /// Create a squarer sized for the given modulus.
    pub fn new(modulus: &BigUint) -> Self {
        let n_bytes = (modulus.bits() as usize + 7) / 8;
        let n_limbs = (n_bytes + 1) / 2; // 16-bit limbs
        let fft_size = (2 * n_limbs).next_power_of_two();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft.get_inplace_scratch_len().max(ifft.get_inplace_scratch_len());

        FftSquarer {
            fft_size,
            fft,
            ifft,
            signal: vec![Complex::new(0.0, 0.0); fft_size],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            limbs: vec![0i64; fft_size + 1],
        }
    }

    /// Square `x` in place mod N = k·2ⁿ-1.
    pub fn square_kbn(&mut self, x: &mut BigUint, k: u64, n: u64, modulus: &BigUint) {
        let sq = self.fft_square(x);
        let q = (&sq >> n as usize) / k;
        let r = sq + &q - ((&q * k) << n as usize);
        *x = if r >= *modulus { r - modulus } else { r };
    }

    fn fft_square(&mut self, x: &BigUint) -> BigUint {
        let bytes = x.to_bytes_le();
        if bytes.is_empty() {
            return BigUint::zero();
        }

        let n_limbs = (bytes.len() + 1) / 2;

        // Fill: data limbs then zero-pad — two loops avoids branch per element
        for i in 0..n_limbs {
            let lo = bytes.get(2 * i).copied().unwrap_or(0) as f64;
            let hi = bytes.get(2 * i + 1).copied().unwrap_or(0) as f64;
            self.signal[i] = Complex::new(lo + hi * 256.0, 0.0);
        }
        self.signal[n_limbs..].fill(Complex::new(0.0, 0.0));

        self.fft.process_with_scratch(&mut self.signal, &mut self.scratch);
        for c in &mut self.signal {
            *c = *c * *c;
        }
        self.ifft.process_with_scratch(&mut self.signal, &mut self.scratch);

        // Scale and round into limb array
        let inv_n = 1.0 / self.fft_size as f64;
        self.limbs.fill(0);
        for (i, c) in self.signal.iter().enumerate() {
            self.limbs[i] = (c.re * inv_n).round() as i64;
        }

        // Carry propagate: squaring gives non-negative coefficients so bitwise
        // ops are safe (>> 16 == / 65536, & 0xFFFF == % 65536 for values >= 0).
        // Use .max(0) to absorb any floating-point rounding noise near zero.
        for i in 0..self.limbs.len() - 1 {
            let v = self.limbs[i].max(0);
            self.limbs[i] = v & 0xFFFF;
            self.limbs[i + 1] += v >> 16;
        }

        // Build BigUint directly from u32 words (two 16-bit limbs per word),
        // avoiding an intermediate byte Vec.
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

    /// Returns per-phase timings (µs) for profiling: (to_bytes, fill, fft, pointwise, ifft, carry, from_bytes).
    pub fn profile_phases(&mut self, x: &BigUint, k: u64, n: u64, _modulus: &BigUint) -> [u64; 7] {
        use std::time::Instant;
        let mut t = [0u64; 7];

        let t0 = Instant::now();
        let bytes = x.to_bytes_le();
        t[0] = t0.elapsed().as_micros() as u64;

        let n_limbs = (bytes.len() + 1) / 2;

        let t1 = Instant::now();
        for i in 0..self.fft_size {
            self.signal[i] = if i < n_limbs {
                let lo = bytes.get(2 * i).copied().unwrap_or(0) as f64;
                let hi = bytes.get(2 * i + 1).copied().unwrap_or(0) as f64;
                Complex::new(lo + hi * 256.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        t[1] = t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        self.fft.process_with_scratch(&mut self.signal, &mut self.scratch);
        t[2] = t2.elapsed().as_micros() as u64;

        let t3 = Instant::now();
        for c in &mut self.signal {
            *c = *c * *c;
        }
        t[3] = t3.elapsed().as_micros() as u64;

        let t4 = Instant::now();
        self.ifft.process_with_scratch(&mut self.signal, &mut self.scratch);
        t[4] = t4.elapsed().as_micros() as u64;

        let t5 = Instant::now();
        let inv_n = 1.0 / self.fft_size as f64;
        let base = 65536i64;
        self.limbs.fill(0);
        for (i, c) in self.signal.iter().enumerate() {
            self.limbs[i] = (c.re * inv_n).round() as i64;
        }
        for i in 0..self.limbs.len() - 1 {
            let carry = self.limbs[i].div_euclid(base);
            self.limbs[i] = self.limbs[i].rem_euclid(base);
            self.limbs[i + 1] += carry;
        }
        t[5] = t5.elapsed().as_micros() as u64;

        let t6 = Instant::now();
        let mut result_bytes: Vec<u8> = self.limbs
            .iter()
            .flat_map(|&v| [v as u8, (v >> 8) as u8])
            .collect();
        while result_bytes.last() == Some(&0) { result_bytes.pop(); }
        let sq = if result_bytes.is_empty() { BigUint::zero() } else { BigUint::from_bytes_le(&result_bytes) };
        t[6] = t6.elapsed().as_micros() as u64;

        // Apply kbn reduction (not profiled, it's fast)
        let q = (&sq >> n as usize) / k;
        let _ = sq + &q - ((&q * k) << n as usize);

        t
    }
}
