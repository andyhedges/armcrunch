use num_bigint::BigUint;

/// a * b mod p using u128 to avoid overflow.
fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128) * (b as u128) % (p as u128)) as u64
}

/// base^exp mod p via square-and-multiply.
fn mod_pow(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut result = 1u64;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp >>= 1;
    }
    result
}

/// Modular inverse: a^{p-2} mod p (p is prime — Fermat's little theorem).
fn mod_inv(a: u64, p: u64) -> u64 {
    mod_pow(a, p - 2, p)
}

/// In-place radix-2 DIT NTT with bit-reversal permutation.
/// `w` must be a primitive N-th root of unity mod p.
fn ntt(a: &mut [u64], w: u64, p: u64) {
    let n = a.len();
    let log_n = n.trailing_zeros() as usize;

    // Bit-reversal permutation
    for i in 0..n {
        let rev = bit_reverse(i, log_n);
        if i < rev {
            a.swap(i, rev);
        }
    }

    // Cooley-Tukey DIT butterflies
    let mut stride = 1usize;
    while stride < n {
        // w_stage = w^(n / (2*stride))
        let w_stage = mod_pow(w, (n / (2 * stride)) as u64, p);
        for start in (0..n).step_by(2 * stride) {
            let mut wj = 1u64;
            for j in 0..stride {
                let u = a[start + j];
                let v = mod_mul(a[start + j + stride], wj, p);
                a[start + j] = if u + v >= p { u + v - p } else { u + v };
                a[start + j + stride] = if u >= v { u - v } else { u + p - v };
                wj = mod_mul(wj, w_stage, p);
            }
        }
        stride *= 2;
    }
}

/// In-place inverse NTT. Does NOT include 1/N normalization.
fn intt(a: &mut [u64], w_inv: u64, p: u64) {
    ntt(a, w_inv, p);
}

pub struct FftSquarer {
    n: usize,
    p: u64,
    g_powers: Vec<u64>,
    g_inv_powers: Vec<u64>,
    w: u64,
    w_inv: u64,
    n_inv: u64,
}

impl FftSquarer {
    /// n must be a power of 2 and <= 2^27.
    pub fn new(n: usize) -> Self {
        let p = 2013265921u64;
        let g = mod_pow(31, (p - 1) / (2 * n as u64), p);
        let g_inv = mod_inv(g, p);
        let w = mod_mul(g, g, p);
        let w_inv = mod_inv(w, p);
        let n_inv = mod_inv(n as u64, p);

        let mut g_powers = Vec::with_capacity(n);
        let mut g_inv_powers = Vec::with_capacity(n);
        let mut gj = 1u64;
        let mut gij = 1u64;
        for _ in 0..n {
            g_powers.push(gj);
            g_inv_powers.push(gij);
            gj = mod_mul(gj, g, p);
            gij = mod_mul(gij, g_inv, p);
        }

        #[cfg(debug_assertions)]
        eprintln!(
            "NegacyclicSquarer(n={n}): max safe input = {n} bytes (single-prime limit)"
        );

        FftSquarer { n, p, g_powers, g_inv_powers, w, w_inv, n_inv }
    }

    /// Square x using negacyclic NTT. Returns x² (exact integer, not reduced mod anything).
    /// Panics if the input exceeds the single-prime dynamic range.
    pub fn fft_square(&mut self, x: &BigUint) -> BigUint {
        let b: u64 = 256;
        let bytes = x.to_bytes_le();
        assert!(
            bytes.len() <= self.n,
            "input too large: {} bytes > transform length {}. Use a larger n.",
            bytes.len(), self.n
        );

        let max_coeff = self.n as u128 * (b as u128 - 1) * (b as u128 - 1);
        assert!(
            max_coeff < self.p as u128,
            "dynamic range exceeded: N*(B-1)^2={max_coeff} >= p={}. \
             Use a larger NTT prime (Phase 2).",
            self.p
        );

        let mut a: Vec<u64> = bytes.into_iter().map(|b| b as u64).collect();
        a.resize(self.n, 0);

        for (j, x) in a.iter_mut().enumerate() {
            *x = mod_mul(*x, self.g_powers[j], self.p);
        }

        ntt(&mut a, self.w, self.p);

        for x in &mut a { *x = mod_mul(*x, *x, self.p); }

        intt(&mut a, self.w_inv, self.p);

        for (j, x) in a.iter_mut().enumerate() {
            *x = mod_mul(mod_mul(*x, self.g_inv_powers[j], self.p), self.n_inv, self.p);
        }

        let mut carry: u64 = 0;
        for x in &mut a {
            let v = *x + carry;
            *x = v % b;
            carry = v / b;
        }
        let mut extra = Vec::new();
        while carry > 0 {
            extra.push(carry % b);
            carry /= b;
        }
        a.extend(extra);

        let bytes: Vec<u8> = a.into_iter().map(|v| v as u8).collect();
        BigUint::from_bytes_le(&bytes)
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O(N^2) negacyclic convolution of `a` with itself, mod p.
    fn naive_negacyclic_square(a: &[u64], p: u64) -> Vec<u64> {
        let n = a.len();
        let mut c = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let ij = i + j;
                if ij < n {
                    c[ij] = (c[ij] + mod_mul(a[i], a[j], p)) % p;
                } else {
                    let k = ij - n;
                    c[k] = (c[k] + p - mod_mul(a[i], a[j], p)) % p;
                }
            }
        }
        c
    }

    #[test]
    fn test_mod_mul_no_overflow() {
        let p = 2013265921u64;
        assert_eq!(mod_mul(p - 1, p - 1, p), 1); // (p-1)^2 mod p = 1
    }

    #[test]
    fn test_mod_pow_known() {
        let p = 2013265921u64;
        assert_eq!(mod_pow(31, p - 1, p), 1); // Fermat's little theorem
        assert_eq!(mod_pow(2, 10, p), 1024);
    }

    #[test]
    fn test_mod_inv() {
        let p = 2013265921u64;
        let a = 123456789u64;
        assert_eq!(mod_mul(a, mod_inv(a, p), p), 1);
    }

    #[test]
    fn test_larger_square() {
        let x = BigUint::parse_bytes(b"DEADBEEFCAFEBABE1234567890ABCDEF", 16).unwrap();
        let mut sq = FftSquarer::new(256);
        let result = sq.fft_square(&x);
        assert_eq!(result, &x * &x);
    }

    #[test]
    fn test_small_square() {
        let x = BigUint::from(255u64);
        let mut sq = FftSquarer::new(64);
        assert_eq!(sq.fft_square(&x), &x * &x);
    }

    #[test]
    fn test_multi_limb_square() {
        let x = BigUint::from(123456789u64);
        let mut sq = FftSquarer::new(64);
        assert_eq!(sq.fft_square(&x), &x * &x);
    }

    #[test]
    fn test_zero() {
        let x = BigUint::from(0u64);
        let mut sq = FftSquarer::new(8);
        assert_eq!(sq.fft_square(&x), BigUint::from(0u64));
    }

    #[test]
    fn test_one() {
        let x = BigUint::from(1u64);
        let mut sq = FftSquarer::new(8);
        assert_eq!(sq.fft_square(&x), BigUint::from(1u64));
    }

    #[test]
    fn test_twist_correctness() {
        let p = 2013265921u64;
        let sq = FftSquarer::new(8);
        let a: Vec<u64> = vec![1, 2, 3, 4, 0, 0, 0, 0];

        let expected = naive_negacyclic_square(&a, p);

        let mut buf: Vec<u64> = a.iter().enumerate()
            .map(|(j, &x)| mod_mul(x, sq.g_powers[j], p))
            .collect();
        ntt(&mut buf, sq.w, p);
        for x in &mut buf { *x = mod_mul(*x, *x, p); }
        intt(&mut buf, sq.w_inv, p);
        let got: Vec<u64> = buf.iter().enumerate()
            .map(|(j, &x)| mod_mul(mod_mul(x, sq.g_inv_powers[j], p), sq.n_inv, p))
            .collect();

        assert_eq!(got, expected);
    }

    #[test]
    fn test_g_powers_cycle() {
        let p = 2013265921u64;
        let sq = FftSquarer::new(8);
        let g = sq.g_powers[1];
        let gn = mod_pow(g, sq.n as u64, p);
        assert_eq!(gn, p - 1);
    }

    fn root_of_unity(n: usize, p: u64) -> u64 {
        mod_pow(31, (p - 1) / n as u64, p)
    }

    #[test]
    fn test_ntt_round_trip() {
        let p = 2013265921u64;
        let n = 64usize;
        let w = root_of_unity(n, p);
        let w_inv = mod_inv(w, p);
        let n_inv = mod_inv(n as u64, p);

        let original: Vec<u64> = (1..=n as u64).map(|x| x % p).collect();
        let mut a = original.clone();

        ntt(&mut a, w, p);
        intt(&mut a, w_inv, p);
        for x in &mut a { *x = mod_mul(*x, n_inv, p); }

        assert_eq!(a, original);
    }

    #[test]
    fn test_ntt_known_convolution() {
        // NTT of [1, 0, 0, ...] is all-ones
        let p = 2013265921u64;
        let n = 8usize;
        let w = root_of_unity(n, p);
        let mut a = vec![0u64; n];
        a[0] = 1;
        ntt(&mut a, w, p);
        assert!(a.iter().all(|&x| x == 1));
    }
}
