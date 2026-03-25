//! DWT/FFT-based squarer for numbers mod M = k·2^exp − 1.
//!
//! For k=1 (Mersenne numbers): uses an IBDWT (Irrational Base Discrete
//! Weighted Transform) with a half-size FFT.  The weighted cyclic convolution
//! automatically reduces mod 2^exp − 1, eliminating the need for zero-padding.
//!
//! For k>1 (Riesel numbers): uses a standard zero-padded FFT with 16-bit limbs
//! and kbn reduction, since the IBDWT ring structure does not directly extend
//! to k>1 without balanced-representation carry arithmetic.
//!
//! ## IBDWT (k=1) — Crandall & Fagin, 1994
//!
//! Represent x in variable-base form: x = Σ x_j · 2^{⌈exp·j/N⌉}.
//! Weight signal: a_j = 2^{⌈exp·j/N⌉ − exp·j/N}.
//! The weighted cyclic convolution computes x² mod (2^exp − 1) directly.
//! FFT length = N (number of limbs), not 2N.
//!
//! ## Zero-padded FFT (k>1)
//!
//! Standard FFT-based squaring with 16-bit limbs and 2× zero-padding.
//! Reduction via kbn identity: q = (x² >> exp) / k, r = x² + q − q·k·2^exp.

use num_bigint::BigUint;
use num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Shift-safe helpers
// ---------------------------------------------------------------------------

/// Bitmask of `width` ones as u64.  Safe for width >= 64.
#[inline]
fn limb_mask_u64(width: u64) -> u64 {
    if width >= 64 { u64::MAX } else if width == 0 { 0 } else { (1u64 << width) - 1 }
}

/// Bitmask of `width` ones as i64.  Safe for width >= 63.
#[inline]
fn limb_mask_i64(width: u64) -> i64 {
    if width >= 63 { i64::MAX } else if width == 0 { 0 } else { (1i64 << width) - 1 }
}

/// Arithmetic right-shift, safe for width >= 64.
#[inline]
fn arith_shr(val: i64, width: u64) -> i64 {
    if width >= 64 { if val >= 0 { 0 } else { -1 } } else { val >> width }
}

/// Internal strategy tag: which squaring algorithm to use.
enum Strategy {
    /// k=1: IBDWT with half-size FFT and weighted cyclic convolution.
    Ibdwt {
        /// Bit positions: bit_pos[j] = ⌈exp·j/len⌉, length len+1
        bit_pos: Vec<u64>,
        /// Forward weights
        fwd: Vec<f64>,
        /// Inverse weights
        inv_w: Vec<f64>,
    },
    /// k>1: zero-padded FFT with 16-bit limbs and kbn reduction.
    ZeroPadded {
        k: u64,
        exp: u64,
        /// Carry buffer (fft_size + 1 elements)
        limbs: Vec<i64>,
    },
}

pub struct DwtSquarer {
    /// The modulus M = k·2^exp − 1
    modulus: BigUint,
    /// FFT transform length (power of 2)
    fft_size: usize,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    signal: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    /// Current value for ZeroPadded path (None for IBDWT)
    value: Option<BigUint>,
    strategy: Strategy,
}

impl DwtSquarer {
    /// Build a squarer for M = k·2^exp − 1 and load initial value `x`.
    pub fn new(k: u64, exp: u64, x: &BigUint) -> Self {
        let modulus = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;

        if k == 1 {
            Self::new_ibdwt(exp, &modulus, x)
        } else {
            Self::new_zero_padded(k, exp, &modulus, x)
        }
    }

    /// IBDWT constructor for k=1 (Mersenne numbers).
    fn new_ibdwt(exp: u64, modulus: &BigUint, x: &BigUint) -> Self {
        // Choose FFT length so max limb width satisfies f64 precision:
        // N · (2^b_max)² < 2^53  ⟹  b_max < (53 − log₂(N)) / 2
        let fft_size = {
            let mut n = 4usize;
            loop {
                let b_max = ((exp as usize + n - 1) / n) as f64;
                let headroom = (53.0 - (n as f64).log2()) / 2.0;
                if b_max <= headroom && n.is_power_of_two() {
                    break n;
                }
                n *= 2;
                assert!(n <= (1 << 28), "FFT length overflow");
            }
        };

        // bit_pos[j] = ceil(exp * j / N)
        let mut bit_pos = Vec::with_capacity(fft_size + 1);
        for j in 0..=fft_size {
            let bp = if j == 0 {
                0u64
            } else {
                ((exp as u128 * j as u128 + fft_size as u128 - 1) / fft_size as u128) as u64
            };
            bit_pos.push(bp);
        }

        // Weights: a[j] = 2^(bit_pos[j] - exp*j/N)
        let len_f = fft_size as f64;
        let exp_f = exp as f64;
        let mut fwd = Vec::with_capacity(fft_size);
        let mut inv_w = Vec::with_capacity(fft_size);
        for j in 0..fft_size {
            let w = 2.0f64.powf(bit_pos[j] as f64 - exp_f * (j as f64) / len_f);
            fwd.push(w);
            inv_w.push(1.0 / w);
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft.get_inplace_scratch_len().max(ifft.get_inplace_scratch_len());

        let mut sq = DwtSquarer {
            modulus: modulus.clone(),
            fft_size,
            fft,
            ifft,
            signal: vec![Complex::new(0.0, 0.0); fft_size],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            value: None,
            strategy: Strategy::Ibdwt { bit_pos, fwd, inv_w },
        };
        sq.ibdwt_load(x);
        sq
    }

    /// Zero-padded FFT constructor for k>1.
    fn new_zero_padded(k: u64, exp: u64, modulus: &BigUint, x: &BigUint) -> Self {
        let m_bytes = (modulus.bits() as usize + 7) / 8;
        let n_limbs = (m_bytes + 1) / 2;
        let fft_size = (2 * n_limbs).next_power_of_two();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft.get_inplace_scratch_len().max(ifft.get_inplace_scratch_len());

        DwtSquarer {
            modulus: modulus.clone(),
            fft_size,
            fft,
            ifft,
            signal: vec![Complex::new(0.0, 0.0); fft_size],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            value: Some(x.clone()),
            strategy: Strategy::ZeroPadded {
                k,
                exp,
                limbs: vec![0i64; fft_size + 1],
            },
        }
    }

    /// One squaring: x ← x² mod M.
    pub fn square(&mut self) {
        match &self.strategy {
            Strategy::Ibdwt { .. } => self.ibdwt_square(),
            Strategy::ZeroPadded { .. } => self.zeropad_square(),
        }
    }

    /// Extract current value as a BigUint.
    pub fn to_biguint(&self) -> BigUint {
        match &self.strategy {
            Strategy::Ibdwt { .. } => self.ibdwt_to_biguint(),
            Strategy::ZeroPadded { .. } => self.value.as_ref().unwrap().clone(),
        }
    }

    // -----------------------------------------------------------------------
    // IBDWT path (k=1)
    // -----------------------------------------------------------------------

    fn ibdwt_load(&mut self, x: &BigUint) {
        let (bit_pos, fwd) = match &self.strategy {
            Strategy::Ibdwt { bit_pos, fwd, .. } => (bit_pos, fwd),
            _ => unreachable!(),
        };
        let bytes = x.to_bytes_le();
        for j in 0..self.fft_size {
            let start = bit_pos[j];
            let width = bit_pos[j + 1] - start;
            let v = extract_bits(&bytes, start, width) as f64;
            self.signal[j] = Complex::new(v * fwd[j], 0.0);
        }
        self.fft.process_with_scratch(&mut self.signal, &mut self.scratch);
    }

    fn ibdwt_square(&mut self) {
        // 1. Pointwise square
        for c in &mut self.signal {
            *c = *c * *c;
        }

        // 2. IFFT + unweight
        self.ifft.process_with_scratch(&mut self.signal, &mut self.scratch);
        let inv_n = 1.0 / self.fft_size as f64;
        let inv_w = match &self.strategy {
            Strategy::Ibdwt { inv_w, .. } => inv_w.clone(),
            _ => unreachable!(),
        };
        for j in 0..self.fft_size {
            self.signal[j].re *= inv_n * inv_w[j];
            self.signal[j].im = 0.0;
        }

        // 3. Round + carry
        let bit_pos = match &self.strategy {
            Strategy::Ibdwt { bit_pos, .. } => bit_pos.clone(),
            _ => unreachable!(),
        };
        let mut a: Vec<i64> = self.signal.iter().map(|c| c.re.round() as i64).collect();
        ibdwt_carry(&mut a, &bit_pos, self.fft_size);

        // 4. Reconstruct + reduce + reload
        let raw = biguint_from_var_limbs(&a, &bit_pos, self.fft_size);
        let reduced = raw % &self.modulus;
        self.ibdwt_load(&reduced);
    }

    fn ibdwt_to_biguint(&self) -> BigUint {
        let (bit_pos, inv_w) = match &self.strategy {
            Strategy::Ibdwt { bit_pos, inv_w, .. } => (bit_pos.clone(), inv_w.clone()),
            _ => unreachable!(),
        };
        let mut tmp = self.signal.clone();
        let mut scratch = self.scratch.clone();
        self.ifft.process_with_scratch(&mut tmp, &mut scratch);

        let inv_n = 1.0 / self.fft_size as f64;
        for j in 0..self.fft_size {
            tmp[j].re *= inv_n * inv_w[j];
            tmp[j].im = 0.0;
        }

        let mut a: Vec<i64> = tmp.iter().map(|c| c.re.round() as i64).collect();
        ibdwt_carry(&mut a, &bit_pos, self.fft_size);
        let raw = biguint_from_var_limbs(&a, &bit_pos, self.fft_size);
        raw % &self.modulus
    }

    // -----------------------------------------------------------------------
    // Zero-padded FFT path (k>1)
    // -----------------------------------------------------------------------

    fn zeropad_square(&mut self) {
        let (k, exp, limbs) = match &mut self.strategy {
            Strategy::ZeroPadded { k, exp, limbs } => (*k, *exp, limbs),
            _ => unreachable!(),
        };
        let value = self.value.as_ref().unwrap();
        let bytes = value.to_bytes_le();

        let n_data_limbs = if bytes.is_empty() { 0 } else { (bytes.len() + 1) / 2 };

        // Fill signal with 16-bit limbs + zero-pad
        for i in 0..n_data_limbs {
            let lo = bytes.get(2 * i).copied().unwrap_or(0) as f64;
            let hi = bytes.get(2 * i + 1).copied().unwrap_or(0) as f64;
            self.signal[i] = Complex::new(lo + hi * 256.0, 0.0);
        }
        self.signal[n_data_limbs..].fill(Complex::new(0.0, 0.0));

        // FFT → square → IFFT
        self.fft.process_with_scratch(&mut self.signal, &mut self.scratch);
        for c in &mut self.signal {
            *c = *c * *c;
        }
        self.ifft.process_with_scratch(&mut self.signal, &mut self.scratch);

        // Round + signed carry propagation into 16-bit limbs.
        // Uses div_euclid/rem_euclid to correctly handle negative rounding
        // residues without clamping.
        let inv_n = 1.0 / self.fft_size as f64;
        let base: i64 = 65536; // 2^16
        limbs.fill(0);
        for (i, c) in self.signal.iter().enumerate() {
            limbs[i] = (c.re * inv_n).round() as i64;
        }
        for i in 0..limbs.len() - 1 {
            let carry = limbs[i].div_euclid(base);
            limbs[i] = limbs[i].rem_euclid(base);
            limbs[i + 1] += carry;
        }

        // Reconstruct BigUint from non-negative 16-bit limbs packed into u32 words.
        let last = limbs.iter().rposition(|&v| v != 0).unwrap_or(0);
        let n_words = (last + 2) / 2;
        let words: Vec<u32> = (0..n_words)
            .map(|i| {
                let lo = limbs[2 * i] as u32;
                let hi = limbs.get(2 * i + 1).copied().unwrap_or(0) as u32;
                lo | (hi << 16)
            })
            .collect();
        let sq = BigUint::new(words);

        // kbn reduce
        self.value = Some(kbn_reduce(sq, k, exp, &self.modulus));
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn extract_bits(bytes: &[u8], start: u64, width: u64) -> u64 {
    if width == 0 { return 0; }
    let byte_start = (start / 8) as usize;
    let bit_off = start % 8;
    let mut val: u64 = 0;
    for i in 0..9usize {
        let idx = byte_start + i;
        let b = if idx < bytes.len() { bytes[idx] as u64 } else { 0 };
        val |= b << (8 * i);
        if 8 * (i as u64 + 1) >= bit_off + width { break; }
    }
    (val >> bit_off) & limb_mask_u64(width)
}

fn write_bits(bytes: &mut [u8], start: u64, width: u64, val: u64) {
    if width == 0 { return; }
    let byte_start = (start / 8) as usize;
    let bit_off = start % 8;
    let mut v = val << bit_off;
    let n = ((bit_off + width + 7) / 8) as usize;
    for i in 0..n {
        if byte_start + i < bytes.len() {
            bytes[byte_start + i] |= (v & 0xFF) as u8;
        }
        v >>= 8;
    }
}

/// IBDWT carry normalization for k=1.
///
/// Carry overflow from the last limb represents carry × 2^exp.
/// Since M = 2^exp − 1, we have 2^exp ≡ 1 (mod M), so carry wraps
/// to a[0] with factor 1.  The caller's `% modulus` canonicalizes.
fn ibdwt_carry(a: &mut [i64], bit_pos: &[u64], len: usize) {
    let mut carry: i64 = 0;
    for j in 0..len {
        let width = bit_pos[j + 1] - bit_pos[j];
        let mask = limb_mask_i64(width);
        let v = a[j] + carry;
        carry = arith_shr(v, width);
        a[j] = v & mask;
    }

    // Wrap: carry × 2^exp ≡ carry (mod 2^exp − 1)
    let mut wrap_iters = 0;
    while carry != 0 {
        wrap_iters += 1;
        debug_assert!(wrap_iters <= 16, "carry wrap not converging: carry={carry}");
        a[0] += carry;
        carry = 0;
        for j in 0..len {
            let width = bit_pos[j + 1] - bit_pos[j];
            let mask = limb_mask_i64(width);
            let v = a[j] + carry;
            carry = arith_shr(v, width);
            a[j] = v & mask;
        }
    }

    // Handle negative limbs from FP rounding: add 2^exp ≡ 1 (mod M)
    if a.iter().any(|&v| v < 0) {
        a[0] += 1;
        carry = 0;
        for j in 0..len {
            let width = bit_pos[j + 1] - bit_pos[j];
            let mask = limb_mask_i64(width);
            let v = a[j] + carry;
            carry = arith_shr(v, width);
            a[j] = v & mask;
        }
        while carry != 0 {
            a[0] += carry;
            carry = 0;
            for j in 0..len {
                let width = bit_pos[j + 1] - bit_pos[j];
                let mask = limb_mask_i64(width);
                let v = a[j] + carry;
                carry = arith_shr(v, width);
                a[j] = v & mask;
            }
        }
    }
}

fn biguint_from_var_limbs(a: &[i64], bit_pos: &[u64], len: usize) -> BigUint {
    let total_bits = bit_pos[len];
    let total_bytes = (total_bits as usize + 7) / 8;
    let mut bytes = vec![0u8; total_bytes];
    for j in 0..len {
        let start = bit_pos[j];
        let width = bit_pos[j + 1] - start;
        write_bits(&mut bytes, start, width, a[j] as u64);
    }
    while bytes.last() == Some(&0) { bytes.pop(); }
    if bytes.is_empty() { BigUint::zero() } else { BigUint::from_bytes_le(&bytes) }
}

/// kbn reduction: x mod M = k·2^exp − 1.
fn kbn_reduce(x: BigUint, k: u64, exp: u64, modulus: &BigUint) -> BigUint {
    if x.is_zero() { return x; }
    let q = (&x >> exp as usize) / k;
    let r = x + &q - ((&q * k) << exp as usize);
    if r >= *modulus { r - modulus } else { r }
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
        let sq = DwtSquarer::new(1, 32, &BigUint::from(1u64));
        if let Strategy::Ibdwt { fwd, .. } = &sq.strategy {
            for (j, &w) in fwd.iter().enumerate() {
                assert!((w - 1.0).abs() < 1e-12, "weight[{j}] = {w}, expected 1.0");
            }
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
            assert_eq!(sq.to_biguint(), x_naive, "mismatch at iteration {i}");
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
            assert_eq!(sq.to_biguint(), x_naive, "mismatch at iteration {i}");
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

    #[test]
    fn test_carry_wrap_k1() {
        let exp = 31u64;
        let m = mersenne(exp);
        let start = &m - BigUint::from(1u64);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(1, exp, &start);
        for i in 0..5 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            assert_eq!(sq.to_biguint(), x_naive, "k=1 wrap failed at iter {i}");
        }
    }

    #[test]
    fn test_carry_wrap_k_large() {
        let k = 1005u64;
        let exp = 64u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let start = &m - BigUint::from(2u64);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(k, exp, &start);
        for i in 0..5 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            assert_eq!(sq.to_biguint(), x_naive, "k=1005 wrap failed at iter {i}");
        }
    }

    #[test]
    fn test_phase_b_chain_medium() {
        let k = 7u64;
        let exp = 100u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let start = BigUint::from(12345u64);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(k, exp, &start);
        for i in 0..20 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
            assert_eq!(sq.to_biguint(), x_naive, "k=7 chain failed at iter {i}");
        }
    }
}