//! GWnum-style weighted DWT squarer.
//!
//! Stores `x` in FFT domain between calls so each squaring costs only:
//!   pointwise square → IFFT → unweight → round → carry → weight → FFT
//! instead of two full FFT pairs (one per CRT prime in the NTT approach).
//!
//! ## Mathematical basis
//!
//! For M = k·2^exp − 1 with transform length L (power of 2), represent x as L
//! limbs of b = floor(exp/L) or b+1 bits (mixed-radix, n_hi = exp % L limbs are
//! one bit wider).
//!
//! Weight limb j by w_j = k^(j/L) before the forward FFT.  Then polynomial
//! multiplication in the weighted ring equals multiplication mod M, and carry
//! wrap-around automatically incorporates the factor of k.
//!
//! For k=1 (Mersenne): weights are all 1.0 — Phase A, exact cyclic ring.
//! For k>1: weights are irrational floats — Phase B.

use num_bigint::BigUint;
use num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub struct DwtSquarer {
    /// FFT transform length (power of 2, ≥ 2*ceil(exp/8))
    len: usize,
    /// k in k·2^exp − 1
    k: u64,
    /// exp in k·2^exp − 1
    exp: u64,
    /// ⌈log₂(k)⌉ extra bits needed to represent values up to k·2^exp − 1
    k_extra_bits: u64,
    /// floor(padded_bits / len) — bits per limb (uniform; n_hi is always 0)
    b_lo: u64,
    /// always 0: padded_bits is a multiple of len so all limbs have equal width
    n_hi: usize,
    /// padded_bits − (exp + k_extra_bits): the extra bits added so b_lo is an
    /// integer.  Carry wrap multiplies by 2^padding before dividing by k.
    padding: u64,
    /// forward weights: fwd[j] = k^(j/L) * 2^(j*exp/L - sum_{i<j} b_i)
    fwd: Vec<f64>,
    /// inverse weights: 1/fwd[j]
    inv_w: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    /// Working buffer — holds x in FFT domain between square() calls.
    pub(crate) signal: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl DwtSquarer {
    /// Build a squarer for M = k·2^exp − 1 and load initial value `x`.
    pub fn new(k: u64, exp: u64, x: &BigUint) -> Self {
        // Transform length: enough room to hold the product without aliasing.
        // Product of two (exp/8)-byte numbers has up to 2*(exp/8) bytes.
        let n_bytes = (exp as usize + 7) / 8;
        let min_len = (2 * n_bytes).next_power_of_two();

        // Extra bits needed to represent values up to M = k·2^exp − 1.
        // M has exp + ⌈log₂(k)⌉ bits; k=1 needs no extra bits.
        let k_extra_bits = if k <= 1 { 0u64 } else { u64::BITS as u64 - k.leading_zeros() as u64 };
        let total_bits = exp + k_extra_bits;

        // Pad total_bits up to the next multiple of min_len so that
        // b = padded_bits / len is an exact integer.  This eliminates the
        // non-integer-b rounding error in the cyclic wrap terms.
        let padded_bits =
            min_len as u64 * ((total_bits + min_len as u64 - 1) / min_len as u64);
        let len = min_len;
        let b_lo = padded_bits / len as u64;
        let n_hi = 0usize; // padded_bits is a multiple of len; all limbs equal width
        let padding = padded_bits - total_bits;

        // Precompute weights: w[j] = k^(j/L).
        let (fwd, inv_w) = build_weights(k, exp, len, b_lo, n_hi);


        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(len);
        let ifft = planner.plan_fft_inverse(len);
        let scratch_len = fft
            .get_inplace_scratch_len()
            .max(ifft.get_inplace_scratch_len());

        let signal = vec![Complex::new(0.0, 0.0); len];
        let scratch = vec![Complex::new(0.0, 0.0); scratch_len];

        let mut sq = DwtSquarer {
            len,
            k,
            exp,
            k_extra_bits,
            b_lo,
            n_hi,
            padding,
            fwd,
            inv_w,
            fft,
            ifft,
            signal,
            scratch,
        };
        sq.load(x);
        sq
    }

    /// Load `x` into transform domain.
    fn load(&mut self, x: &BigUint) {
        let bytes = x.to_bytes_le();
        limbs_from_bytes(&bytes, self.b_lo, self.n_hi, self.len, &mut self.signal);
        // Apply forward weights
        for (s, &w) in self.signal.iter_mut().zip(self.fwd.iter()) {
            s.re *= w;
            // s.im is 0 at load time; keep it 0
        }
        self.fft
            .process_with_scratch(&mut self.signal, &mut self.scratch);
    }

    /// One squaring: x ← x² mod (k·2^exp − 1).
    /// Operates entirely in transform domain; no BigUint allocation.
    pub fn square(&mut self) {
        // 1. Pointwise square (no FFT)
        for c in &mut self.signal {
            *c = *c * *c;
        }

        // 2. Inverse FFT
        self.ifft
            .process_with_scratch(&mut self.signal, &mut self.scratch);

        // 3. Scale + unweight → real coefficients
        let inv_n = 1.0 / self.len as f64;
        for (s, &iw) in self.signal.iter_mut().zip(self.inv_w.iter()) {
            s.re *= inv_n * iw;
            s.im = 0.0;
        }

        // 4. Round, carry-normalize, and reduce mod M.
        //    This modifies signal[j].re in place.
        carry_normalize(
            &mut self.signal,
            self.b_lo,
            self.n_hi,
            self.k,
            self.padding,
            self.len,
        );

        // 5. Re-apply forward weights + forward FFT
        for (s, &w) in self.signal.iter_mut().zip(self.fwd.iter()) {
            s.re *= w;
        }
        self.fft
            .process_with_scratch(&mut self.signal, &mut self.scratch);
    }

    /// Extract current value as a BigUint.
    pub fn to_biguint(&self) -> BigUint {
        // We need to IFFT + unweight + round + carry to read back the value.
        // Clone signal to avoid mutating state.
        let mut tmp = self.signal.clone();
        let mut scratch = self.scratch.clone();

        self.ifft.process_with_scratch(&mut tmp, &mut scratch);

        let inv_n = 1.0 / self.len as f64;
        for (s, &iw) in tmp.iter_mut().zip(self.inv_w.iter()) {
            s.re *= inv_n * iw;
            s.im = 0.0;
        }

        carry_normalize(&mut tmp, self.b_lo, self.n_hi, self.k, self.padding, self.len);

        biguint_from_signal(&tmp, self.b_lo, self.n_hi, self.len)
    }
}

// ---------------------------------------------------------------------------
// Weight construction
// ---------------------------------------------------------------------------

/// Build DWT weights for M = k·2^exp − 1 with transform length `len`.
///
/// For the cyclic DFT to represent multiplication in Z[X]/(X^len - k·2^exp),
/// limb j must be scaled by:
///
///   fwd[j] = k^(j/L) · 2^(j·exp/L − Σ_{i<j} b_i)
///
/// The second factor corrects for the difference between the nominal uniform
/// position j·b (b = exp/L) and the actual mixed-radix bit position
/// Σ_{i<j} b_i.  When L | exp this correction is always 1.
/// Build forward and inverse DWT weights.
///
/// The full weight for limb j is:
///
///   fwd[j] = k^(j/L) · 2^(j·exp/L − Σ_{i<j} b_i)
///
/// The second factor corrects for the difference between the uniform
/// fractional base 2^(exp/L) and the actual integer mixed-radix bit positions.
/// When L | exp this factor is always 1.0.
///
/// After unweighting the IFFT output, coefficient j represents a value in the
/// polynomial ring Z[X]/(X^L − k·2^exp), evaluated via the mixed-radix basis.
fn build_weights(k: u64, exp: u64, len: usize, b_lo: u64, n_hi: usize) -> (Vec<f64>, Vec<f64>) {
    let b = exp as f64 / len as f64; // uniform fractional limb width
    let kf = k as f64;
    let mut fwd = Vec::with_capacity(len);
    let mut bit_pos: u64 = 0; // Σ_{i<j} b_i
    for j in 0..len {
        let pos_correction = 2.0f64.powf(j as f64 * b - bit_pos as f64);
        let k_part = kf.powf(j as f64 / len as f64);
        fwd.push(pos_correction * k_part);
        bit_pos += if j < n_hi { b_lo + 1 } else { b_lo };
    }
    let inv_w: Vec<f64> = fwd.iter().map(|&w| 1.0 / w).collect();
    (fwd, inv_w)
}

// ---------------------------------------------------------------------------
// Limb decomposition / recomposition
// ---------------------------------------------------------------------------

/// Decompose `bytes` (little-endian) into `len` limbs stored in `out[j].re`.
/// Limb j has b_lo bits if j >= n_hi, else b_lo+1 bits.
fn limbs_from_bytes(bytes: &[u8], b_lo: u64, n_hi: usize, len: usize, out: &mut [Complex<f64>]) {
    // Iterate through the bit stream, extracting limbs.
    let mut bit_pos: u64 = 0;
    for j in 0..len {
        let width = if j < n_hi { b_lo + 1 } else { b_lo };
        out[j].re = extract_bits(bytes, bit_pos, width) as f64;
        out[j].im = 0.0;
        bit_pos += width;
    }
}

/// Extract `width` bits starting at bit `start` from little-endian `bytes`.
fn extract_bits(bytes: &[u8], start: u64, width: u64) -> u64 {
    if width == 0 {
        return 0;
    }
    let byte_start = (start / 8) as usize;
    let bit_off = start % 8;
    // Read up to 9 bytes (64 + 7 bits, worst case)
    let mut val: u64 = 0;
    for i in 0..9usize {
        let byte_idx = byte_start + i;
        let b = if byte_idx < bytes.len() {
            bytes[byte_idx] as u64
        } else {
            0
        };
        val |= b << (8 * i);
        if 8 * (i as u64 + 1) >= bit_off + width {
            break;
        }
    }
    (val >> bit_off) & ((1u64 << width) - 1)
}

/// Reconstruct BigUint from limbs in `signal[j].re` (already rounded/carried).
fn biguint_from_signal(signal: &[Complex<f64>], b_lo: u64, n_hi: usize, len: usize) -> BigUint {
    // Pack limbs into a bit stream, little-endian.
    let total_bits = n_hi as u64 * (b_lo + 1) + (len - n_hi) as u64 * b_lo;
    let total_bytes = (total_bits as usize + 7) / 8;
    let mut bytes = vec![0u8; total_bytes];

    let mut bit_pos: u64 = 0;
    for j in 0..len {
        let width = if j < n_hi { b_lo + 1 } else { b_lo };
        let v = signal[j].re as u64;
        write_bits(&mut bytes, bit_pos, width, v);
        bit_pos += width;
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

fn write_bits(bytes: &mut [u8], start: u64, width: u64, val: u64) {
    if width == 0 {
        return;
    }
    let byte_start = (start / 8) as usize;
    let bit_off = start % 8;
    let mut v = val << bit_off;
    let n_bytes = ((bit_off + width + 7) / 8) as usize;
    for i in 0..n_bytes {
        bytes[byte_start + i] |= (v & 0xFF) as u8;
        v >>= 8;
    }
}

// ---------------------------------------------------------------------------
// Carry normalization
// ---------------------------------------------------------------------------

/// Round signal coefficients to integers and carry-normalize in the mixed-radix
/// representation mod M = k·2^exp − 1.
///
/// After normalization each signal[j].re ∈ [0, 2^width_j) and signal[j].im = 0.
fn carry_normalize(
    signal: &mut [Complex<f64>],
    b_lo: u64,
    n_hi: usize,
    k: u64,
    padding: u64,
    len: usize,
) {
    // Step 1: round to nearest integer
    let mut a: Vec<i64> = signal.iter().map(|c| c.re.round() as i64).collect();

    // Step 2: left-to-right carry in mixed-radix base
    let mut carry: i64 = 0;
    for j in 0..len {
        let width = if j < n_hi { b_lo + 1 } else { b_lo };
        let mask = if width >= 64 {
            i64::MAX
        } else {
            (1i64 << width) - 1
        };
        let v = a[j] + carry;
        if width == 0 {
            carry = v;
            a[j] = 0;
        } else {
            // arithmetic right shift preserves sign for negative carries
            carry = v >> width;
            a[j] = v & mask;
        }
    }

    // Step 3: wrap overflow carry mod M = k·2^exp − 1.
    //
    // Overflow carry means: the integer value in the limbs exceeds 2^padded_bits,
    // where padded_bits = (exp + k_extra_bits + padding).
    //
    // 2^padded_bits mod M:  since k·2^exp = M+1, we have 2^exp ≡ 1/k (mod M),
    // so 2^padded_bits = 2^(exp + k_extra_bits + padding)
    //                  ≡ 2^(k_extra_bits + padding) / k  (mod M).
    //
    // Equivalently: scaled = carry * 2^padding, then carry*2^padded_bits ≡ scaled/k
    // (for the k_extra_bits contribution — see Task 2; k_extra_bits is already
    // baked into the limb representation).
    //
    // For k=1 (Mersenne), k_extra_bits=0: 2^padded_bits ≡ 2^padding (mod M).
    // So carry wraps as carry * 2^padding added to position 0.
    let mut wrap_iters = 0;
    while carry != 0 {
        wrap_iters += 1;
        debug_assert!(wrap_iters <= 8, "carry wrap not converging: carry={carry}");

        // Scale by 2^padding first, then divide by k.
        // For k=1, padding=0: scaled=carry, q=carry, r=0 (same as before).
        // For k=1, padding=1 (exp=31): scaled=2*carry, q=2*carry, r=0.
        let scaled = carry
            .checked_mul(1i64 << (padding as u32))
            .expect("carry overflow in wrap");
        let q = scaled.div_euclid(k as i64); // what goes to a[0]
        let r = scaled.rem_euclid(k as i64); // new wrap carry for next iteration

        // Add q to a[0] and propagate FORWARD through a[1..len-1].
        let mut add = q;
        for j in 0..len {
            if add == 0 {
                break;
            }
            let width = if j < n_hi { b_lo + 1 } else { b_lo };
            let mask = (1i64 << width) - 1;
            let v = a[j] + add;
            add = v >> width;
            a[j] = v & mask;
        }
        // If add is still non-zero after traversing all limbs, it wraps again.
        carry = add + r; // combine with remainder for next iteration
    }

    // Write back
    for (s, v) in signal.iter_mut().zip(a.iter()) {
        s.re = *v as f64;
        s.im = 0.0;
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
        // exp=32, len=8 → b_lo=4, n_hi=0 → weights should all be 1.0
        let (fwd, _) = build_weights(1, 32, 8, 4, 0);
        for (j, &w) in fwd.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-12, "weight[{j}] = {w}, expected 1.0");
        }
    }

    #[test]
    fn test_phase_a_l_divides_exp() {
        // k=1, exp=32, L=8 divides exp exactly → weights=1.0, should be correct
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
        // 3^2 mod 7 = 2
        let _m = mersenne(3); // 7
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
            if got != x_naive {
                eprintln!("iter {i}: got={got}, expected={x_naive}");
                // print the raw post-IFFT values
                let mut tmp = sq.signal.clone();
                let mut scratch2 = sq.scratch.clone();
                sq.ifft.process_with_scratch(&mut tmp, &mut scratch2);
                let inv_n = 1.0 / sq.len as f64;
                for (s, &iw) in tmp.iter_mut().zip(sq.inv_w.iter()) {
                    s.re *= inv_n * iw;
                }
                let rounded: Vec<i64> = tmp.iter().map(|c| c.re.round() as i64).collect();
                eprintln!("  rounded limbs: {rounded:?}");
            }
            assert_eq!(got, x_naive, "mismatch at iteration {i}");
        }
    }

    #[test]
    fn test_phase_b_single_square() {
        // 3^2 mod (3*2^4 - 1) = 9 mod 47 = 9
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
        // k=1005, smaller exp for speed
        let k = 1005u64;
        let exp = 64u64;
        let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
        let start = BigUint::from(3u64).modpow(&BigUint::from(k), &m);
        let mut x_naive = start.clone();
        let mut sq = DwtSquarer::new(k, exp, &start);
        for _ in 0..8 {
            x_naive = (&x_naive * &x_naive) % &m;
            sq.square();
        }
        assert_eq!(sq.to_biguint(), x_naive);
    }
}
