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
//! ## Hot-loop allocation avoidance
//!
//! The square() method avoids per-iteration heap allocation:
//! - IBDWT: reloads spectrum from a reusable limb buffer (no BigUint round-trip).
//! - Zero-padded: fills real buffer from BigUint::iter_u32_digits() (no byte Vec).
//! - All FFT scratch buffers, limb arrays, and transform vectors are pre-allocated.

use num_bigint::BigUint;
use num_traits::Zero;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Shift-safe helpers
// ---------------------------------------------------------------------------

#[inline]
fn limb_mask_u64(width: u64) -> u64 {
    if width >= 64 {
        u64::MAX
    } else if width == 0 {
        0
    } else {
        (1u64 << width) - 1
    }
}

#[inline]
fn limb_mask_i64(width: u64) -> i64 {
    if width >= 63 {
        i64::MAX
    } else if width == 0 {
        0
    } else {
        (1i64 << width) - 1
    }
}

#[inline]
fn arith_shr(val: i64, width: u64) -> i64 {
    if width >= 64 {
        if val >= 0 {
            0
        } else {
            -1
        }
    } else {
        val >> width
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Ibdwt,
    ZeroPadded,
}

pub struct DwtSquarer {
    mode: Mode,
    modulus: BigUint,
    k: u64,
    exp: u64,
    fft_size: usize,

    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    real_buf: Vec<f64>,          // length N (real input/output)
    spectrum: Vec<Complex<f64>>, // length N/2 + 1 (frequency domain)
    scratch_r2c: Vec<Complex<f64>>,
    scratch_c2r: Vec<Complex<f64>>,

    // IBDWT fields (empty Vecs for ZeroPadded)
    bit_pos: Vec<u64>,
    fwd: Vec<f64>,
    inv_w: Vec<f64>,
    m_limbs: Vec<i64>,
    ibdwt_limbs: Vec<i64>,

    // Zero-padded fields
    value: Option<BigUint>,
    zp_limbs: Vec<i64>,
}

impl DwtSquarer {
    pub fn new(k: u64, exp: u64, x: &BigUint) -> Self {
        let modulus = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        if k == 1 {
            Self::new_ibdwt(k, exp, &modulus, x)
        } else {
            Self::new_zero_padded(k, exp, &modulus, x)
        }
    }

    fn new_ibdwt(k: u64, exp: u64, modulus: &BigUint, x: &BigUint) -> Self {
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

        let mut bit_pos = Vec::with_capacity(fft_size + 1);
        for j in 0..=fft_size {
            let bp = if j == 0 {
                0u64
            } else {
                ((exp as u128 * j as u128 + fft_size as u128 - 1) / fft_size as u128) as u64
            };
            bit_pos.push(bp);
        }

        let len_f = fft_size as f64;
        let exp_f = exp as f64;
        let mut fwd = Vec::with_capacity(fft_size);
        let mut inv_w = Vec::with_capacity(fft_size);
        for j in 0..fft_size {
            let w = 2.0f64.powf(bit_pos[j] as f64 - exp_f * (j as f64) / len_f);
            fwd.push(w);
            inv_w.push(1.0 / w);
        }

        let m_bytes = modulus.to_bytes_le();
        let mut m_limbs = vec![0i64; fft_size];
        for j in 0..fft_size {
            m_limbs[j] = extract_bits(&m_bytes, bit_pos[j], bit_pos[j + 1] - bit_pos[j]) as i64;
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);
        let scratch_r2c_len = r2c.get_scratch_len();
        let scratch_c2r_len = c2r.get_scratch_len();

        let mut sq = DwtSquarer {
            mode: Mode::Ibdwt,
            modulus: modulus.clone(),
            k,
            exp,
            fft_size,

            r2c,
            c2r,
            real_buf: vec![0.0; fft_size],
            spectrum: vec![Complex::new(0.0, 0.0); fft_size / 2 + 1],
            scratch_r2c: vec![Complex::new(0.0, 0.0); scratch_r2c_len],
            scratch_c2r: vec![Complex::new(0.0, 0.0); scratch_c2r_len],

            bit_pos,
            fwd,
            inv_w,
            m_limbs,
            ibdwt_limbs: vec![0i64; fft_size],

            value: None,
            zp_limbs: Vec::new(),
        };
        sq.ibdwt_load_biguint(x);
        sq
    }

    fn new_zero_padded(k: u64, exp: u64, modulus: &BigUint, x: &BigUint) -> Self {
        let m_bytes = (modulus.bits() as usize + 7) / 8;
        let n_limbs = (m_bytes + 1) / 2;
        let fft_size = (2 * n_limbs).next_power_of_two();

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);
        let scratch_r2c_len = r2c.get_scratch_len();
        let scratch_c2r_len = c2r.get_scratch_len();

        DwtSquarer {
            mode: Mode::ZeroPadded,
            modulus: modulus.clone(),
            k,
            exp,
            fft_size,

            r2c,
            c2r,
            real_buf: vec![0.0; fft_size],
            spectrum: vec![Complex::new(0.0, 0.0); fft_size / 2 + 1],
            scratch_r2c: vec![Complex::new(0.0, 0.0); scratch_r2c_len],
            scratch_c2r: vec![Complex::new(0.0, 0.0); scratch_c2r_len],

            bit_pos: Vec::new(),
            fwd: Vec::new(),
            inv_w: Vec::new(),
            m_limbs: Vec::new(),
            ibdwt_limbs: Vec::new(),

            value: Some(x.clone()),
            zp_limbs: vec![0i64; fft_size + 1],
        }
    }

    pub fn square(&mut self) {
        match self.mode {
            Mode::Ibdwt => self.ibdwt_square(),
            Mode::ZeroPadded => self.zeropad_square(),
        }
    }

    pub fn to_biguint(&self) -> BigUint {
        match self.mode {
            Mode::Ibdwt => self.ibdwt_to_biguint(),
            Mode::ZeroPadded => self.value.as_ref().unwrap().clone(),
        }
    }

    // -----------------------------------------------------------------------
    // IBDWT path (k=1)
    // -----------------------------------------------------------------------

    fn ibdwt_load_biguint(&mut self, x: &BigUint) {
        let bytes = x.to_bytes_le();
        for j in 0..self.fft_size {
            let start = self.bit_pos[j];
            let width = self.bit_pos[j + 1] - start;
            self.real_buf[j] = extract_bits(&bytes, start, width) as f64 * self.fwd[j];
        }

        self.r2c
            .process_with_scratch(
                &mut self.real_buf,
                &mut self.spectrum,
                &mut self.scratch_r2c,
            )
            .expect("IBDWT real-to-complex FFT failed");
    }

    fn ibdwt_square(&mut self) {
        for c in &mut self.spectrum {
            *c = *c * *c;
        }

        self.c2r
            .process_with_scratch(
                &mut self.spectrum,
                &mut self.real_buf,
                &mut self.scratch_c2r,
            )
            .expect("IBDWT complex-to-real IFFT failed");

        let inv_n = 1.0 / self.fft_size as f64;
        for j in 0..self.fft_size {
            self.ibdwt_limbs[j] = (self.real_buf[j] * inv_n * self.inv_w[j]).round() as i64;
        }

        ibdwt_carry(&mut self.ibdwt_limbs, &self.bit_pos, self.fft_size);
        limb_reduce(&mut self.ibdwt_limbs, &self.m_limbs, &self.bit_pos, self.fft_size);

        for j in 0..self.fft_size {
            self.real_buf[j] = self.ibdwt_limbs[j] as f64 * self.fwd[j];
        }

        self.r2c
            .process_with_scratch(
                &mut self.real_buf,
                &mut self.spectrum,
                &mut self.scratch_r2c,
            )
            .expect("IBDWT reload real-to-complex FFT failed");
    }

    fn ibdwt_to_biguint(&self) -> BigUint {
        let mut tmp_spectrum = self.spectrum.clone();
        let mut tmp_real = vec![0.0f64; self.fft_size];
        let mut scratch = self.scratch_c2r.clone();

        self.c2r
            .process_with_scratch(&mut tmp_spectrum, &mut tmp_real, &mut scratch)
            .expect("IBDWT to_biguint complex-to-real IFFT failed");

        let inv_n = 1.0 / self.fft_size as f64;
        for j in 0..self.fft_size {
            tmp_real[j] *= inv_n * self.inv_w[j];
        }

        let mut a: Vec<i64> = tmp_real.iter().map(|v| v.round() as i64).collect();
        ibdwt_carry(&mut a, &self.bit_pos, self.fft_size);
        biguint_from_var_limbs(&a, &self.bit_pos, self.fft_size) % &self.modulus
    }

    // -----------------------------------------------------------------------
    // Zero-padded FFT path (k>1)
    // -----------------------------------------------------------------------

    fn zeropad_square(&mut self) {
        // Fill real_buf with 16-bit limbs directly from BigUint's internal u32
        // words via iter_u32_digits().  Each u32 word is split into two
        // entries (lo 16 bits, hi 16 bits). No byte Vec is allocated.
        self.real_buf.fill(0.0);
        let mut sig_idx = 0usize;
        for word in self.value.as_ref().unwrap().iter_u32_digits() {
            if sig_idx < self.fft_size {
                self.real_buf[sig_idx] = (word & 0xFFFF) as f64;
            }
            if sig_idx + 1 < self.fft_size {
                self.real_buf[sig_idx + 1] = (word >> 16) as f64;
            }
            sig_idx += 2;
        }

        self.r2c
            .process_with_scratch(
                &mut self.real_buf,
                &mut self.spectrum,
                &mut self.scratch_r2c,
            )
            .expect("Zero-padded real-to-complex FFT failed");

        for c in &mut self.spectrum {
            *c = *c * *c;
        }

        self.c2r
            .process_with_scratch(
                &mut self.spectrum,
                &mut self.real_buf,
                &mut self.scratch_c2r,
            )
            .expect("Zero-padded complex-to-real IFFT failed");

        let inv_n = 1.0 / self.fft_size as f64;
        let base: i64 = 65536;
        self.zp_limbs.fill(0);
        for (i, &v) in self.real_buf.iter().enumerate() {
            self.zp_limbs[i] = (v * inv_n).round() as i64;
        }
        for i in 0..self.zp_limbs.len() - 1 {
            let carry = self.zp_limbs[i].div_euclid(base);
            self.zp_limbs[i] = self.zp_limbs[i].rem_euclid(base);
            self.zp_limbs[i + 1] += carry;
        }

        let last = self.zp_limbs.iter().rposition(|&v| v != 0).unwrap_or(0);
        let n_words = (last + 2) / 2;
        let words: Vec<u32> = (0..n_words)
            .map(|i| {
                let lo = self.zp_limbs[2 * i] as u32;
                let hi = self.zp_limbs.get(2 * i + 1).copied().unwrap_or(0) as u32;
                lo | (hi << 16)
            })
            .collect();

        self.value = Some(kbn_reduce(
            BigUint::new(words),
            self.k,
            self.exp,
            &self.modulus,
        ));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_bits(bytes: &[u8], start: u64, width: u64) -> u64 {
    if width == 0 {
        return 0;
    }
    let byte_start = (start / 8) as usize;
    let bit_off = start % 8;
    let mut val: u64 = 0;
    for i in 0..9usize {
        let idx = byte_start + i;
        let b = if idx < bytes.len() {
            bytes[idx] as u64
        } else {
            0
        };
        val |= b << (8 * i);
        if 8 * (i as u64 + 1) >= bit_off + width {
            break;
        }
    }
    (val >> bit_off) & limb_mask_u64(width)
}

fn write_bits(bytes: &mut [u8], start: u64, width: u64, val: u64) {
    if width == 0 {
        return;
    }
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

fn ibdwt_carry(a: &mut [i64], bit_pos: &[u64], len: usize) {
    let mut carry: i64 = 0;
    for j in 0..len {
        let width = bit_pos[j + 1] - bit_pos[j];
        let mask = limb_mask_i64(width);
        let v = a[j] + carry;
        carry = arith_shr(v, width);
        a[j] = v & mask;
    }
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

fn limb_reduce(a: &mut [i64], m: &[i64], bit_pos: &[u64], len: usize) {
    let mut ge = true;
    for j in (0..len).rev() {
        if a[j] > m[j] {
            break;
        }
        if a[j] < m[j] {
            ge = false;
            break;
        }
    }
    if ge {
        let mut borrow: i64 = 0;
        for j in 0..len {
            let width = bit_pos[j + 1] - bit_pos[j];
            let mask = limb_mask_i64(width);
            let radix = 1i128 << width;
            let v = a[j] as i128 - m[j] as i128 + borrow as i128;
            if v < 0 {
                a[j] = ((v + radix) as i64) & mask;
                borrow = -1;
            } else {
                a[j] = (v as i64) & mask;
                borrow = 0;
            }
        }
    }
}

fn biguint_from_var_limbs(a: &[i64], bit_pos: &[u64], len: usize) -> BigUint {
    let total_bits = bit_pos[len];
    let total_bytes = (total_bits as usize + 7) / 8;
    let mut bytes = vec![0u8; total_bytes];
    for j in 0..len {
        write_bits(
            &mut bytes,
            bit_pos[j],
            bit_pos[j + 1] - bit_pos[j],
            a[j] as u64,
        );
    }
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    if bytes.is_empty() {
        BigUint::zero()
    } else {
        BigUint::from_bytes_le(&bytes)
    }
}

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
        let sq = DwtSquarer::new(1, 32, &BigUint::from(1u64));
        for (j, &w) in sq.fwd.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-12, "weight[{j}] = {w}, expected 1.0");
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
        assert_eq!(sq.to_biguint(), (&x * &x) % &m);
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