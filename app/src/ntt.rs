use num_bigint::BigUint;
use num_traits::Zero;

const P1: u64 = 2_013_265_921; // 15 × 2²⁷ + 1, primitive root 31
const P2: u64 = 998_244_353;   // 119 × 2²³ + 1, primitive root 3
const G1_ROOT: u64 = 31;
const G2_ROOT: u64 = 3;

// --- Modular arithmetic helpers (used for setup; not on the hot path) ---

#[inline]
fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128) * (b as u128) % (p as u128)) as u64
}

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

fn mod_inv(a: u64, p: u64) -> u64 {
    mod_pow(a, p - 2, p)
}

// --- Barrett reduction (hot path) ---
//
// For p < 2³¹ and a, b < p: a*b < 2⁶², fits in u64.
// Barrett constant: p_inv = floor(2⁶⁴ / p).
// q = upper64(a*b * p_inv) ≈ floor(a*b / p).
// On ARM64 this compiles to: mul + umulh + mul + sub + csel  (~5 cycles).
// Replaces u128 division (__udivti3, ~30 cycles on AArch64).

#[derive(Clone)]
struct BarrettMod {
    p: u64,
    p_inv: u64, // floor(2⁶⁴ / p)
}

impl BarrettMod {
    fn new(p: u64) -> Self {
        // (1u128 << 64) / p: compute via shift to avoid overflow
        let p_inv = ((1u128 << 64) / p as u128) as u64;
        BarrettMod { p, p_inv }
    }

    /// Compute a * b mod p.  Requires a, b < p < 2³¹.
    #[inline]
    fn mul(&self, a: u64, b: u64) -> u64 {
        let n = a * b; // a, b < 2³¹ → n < 2⁶², no u64 overflow
        // Estimate quotient via upper 64 bits of (n * p_inv)
        let q = ((n as u128 * self.p_inv as u128) >> 64) as u64;
        let r = n - q * self.p; // q ≤ true quotient, so r ∈ [0, 2p)
        if r >= self.p { r - self.p } else { r }
    }
}

// --- Bit-reversal permutation ---

fn bit_reverse_copy(a: &mut [u64]) {
    let n = a.len();
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS as usize - log_n);
        if i < j {
            a.swap(i, j);
        }
    }
}

// --- Reference NTT (kept for unit tests; uses on-the-fly twiddles) ---

/// In-place radix-2 DIT NTT (reference scalar; used by unit tests).
#[cfg(test)]
pub(crate) fn ntt(a: &mut [u64], w: u64, p: u64) {
    let n = a.len();
    bit_reverse_copy(a);
    let mut half = 1usize;
    while half < n {
        let full = half * 2;
        let w_step = mod_pow(w, (n / full) as u64, p);
        let mut pos = 0;
        while pos < n {
            let mut wj = 1u64;
            for j in 0..half {
                let u = a[pos + j];
                let v = mod_mul(a[pos + half + j], wj, p);
                a[pos + j] = if u + v >= p { u + v - p } else { u + v };
                a[pos + half + j] = if u >= v { u - v } else { u + p - v };
                wj = mod_mul(wj, w_step, p);
            }
            pos += full;
        }
        half = full;
    }
}

/// In-place inverse NTT (reference scalar, no 1/N normalization; used by unit tests).
#[cfg(test)]
pub(crate) fn intt(a: &mut [u64], w_inv: u64, p: u64) {
    ntt(a, w_inv, p);
}

// --- Fast NTT with precomputed twiddles ---
//
// Precomputing twiddles removes the sequential dependency on `wj`, letting
// LLVM (with -C target-cpu=native) autovectorize the butterfly loop with NEON.
// Layout: stage with half=2^s uses twiddles[half-1 .. half-1+half].
// Total: n entries (last entry unused).

fn build_twiddles(n: usize, w: u64, p: u64) -> Vec<u64> {
    let mut twiddles = vec![0u64; n];
    let mut half = 1usize;
    while half < n {
        let full = half * 2;
        let w_step = mod_pow(w, (n / full) as u64, p);
        let base = half - 1;
        twiddles[base] = 1;
        for j in 1..half {
            twiddles[base + j] = mod_mul(twiddles[base + j - 1], w_step, p);
        }
        half = full;
    }
    twiddles
}

/// Fast in-place DIT NTT using precomputed twiddles and Barrett reduction.
///
/// `twiddles` must come from `build_twiddles(n, w, p)`.
/// With `-C target-cpu=native` the inner butterfly loop autovectorizes (NEON on AArch64).
fn ntt_fast(a: &mut [u64], twiddles: &[u64], p: u64, bm: &BarrettMod) {
    let n = a.len();
    bit_reverse_copy(a);
    let mut half = 1usize;
    while half < n {
        let full = half * 2;
        let t_base = half - 1;
        let mut pos = 0;
        while pos < n {
            // split_at_mut removes aliasing ambiguity, enabling autovectorization
            let (lo, hi) = a[pos..pos + full].split_at_mut(half);
            let tw = &twiddles[t_base..t_base + half];
            for j in 0..half {
                let u = lo[j];
                let v = bm.mul(hi[j], tw[j]);
                lo[j] = if u + v >= p { u + v - p } else { u + v };
                hi[j] = if u >= v { u - v } else { u + p - v };
            }
            pos += full;
        }
        half = full;
    }
}

fn intt_fast(a: &mut [u64], inv_twiddles: &[u64], p: u64, bm: &BarrettMod) {
    ntt_fast(a, inv_twiddles, p, bm);
}

// --- Naive negacyclic squaring (used only in unit tests) ---

#[cfg(test)]
pub(crate) fn naive_negacyclic_square(a: &[u64], p: u64) -> Vec<u64> {
    let n = a.len();
    let mut result = vec![0u64; n];
    for i in 0..n {
        if a[i] == 0 {
            continue;
        }
        for j in 0..n {
            if a[j] == 0 {
                continue;
            }
            let c = mod_mul(a[i], a[j], p);
            let k = i + j;
            if k < n {
                result[k] = (result[k] + c) % p;
            } else {
                let idx = k - n;
                result[idx] =
                    if result[idx] >= c { result[idx] - c } else { result[idx] + p - c };
            }
        }
    }
    result
}

// --- CRT ---

/// Reconstruct x ∈ [0, p1·p2) from residues via Garner's algorithm.
#[inline]
fn crt2(r1: u64, r2: u64, p1: u64, p2: u64, p1_inv_p2: u64) -> u64 {
    let r1_mod_p2 = r1 % p2;
    let diff = if r2 >= r1_mod_p2 { r2 - r1_mod_p2 } else { r2 + p2 - r1_mod_p2 };
    let t = mod_mul(diff, p1_inv_p2, p2);
    r1 + p1 * t
}

// --- Per-prime precomputed tables ---

struct PrimeCtx {
    p: u64,
    barrett: BarrettMod,
    g_powers: Vec<u64>,
    g_inv_powers: Vec<u64>,
    fwd_twiddles: Vec<u64>,
    inv_twiddles: Vec<u64>,
    n_inv: u64,
}

fn build_prime_ctx(n: usize, p: u64, g_root: u64) -> PrimeCtx {
    let g = mod_pow(g_root, (p - 1) / (2 * n as u64), p);
    let g_inv = mod_inv(g, p);
    let w = mod_mul(g, g, p);
    let w_inv = mod_inv(w, p);
    let n_inv = mod_inv(n as u64, p);

    let mut g_powers = vec![0u64; n];
    let mut g_inv_powers = vec![0u64; n];
    let mut gp = 1u64;
    let mut gip = 1u64;
    for j in 0..n {
        g_powers[j] = gp;
        g_inv_powers[j] = gip;
        gp = mod_mul(gp, g, p);
        gip = mod_mul(gip, g_inv, p);
    }

    let barrett = BarrettMod::new(p);
    let fwd_twiddles = build_twiddles(n, w, p);
    let inv_twiddles = build_twiddles(n, w_inv, p);

    PrimeCtx { p, barrett, g_powers, g_inv_powers, fwd_twiddles, inv_twiddles, n_inv }
}

// --- NttSquarer ---

/// NTT-based integer squarer using two-prime negacyclic CRT convolution.
///
/// Uses p1 = 2013265921 and p2 = 998244353. Combined dynamic range ≈ 2×10¹⁸.
/// n must be a power of 2 and ≤ 2²² (limit of p2 for negacyclic twist).
///
/// Hot path: Barrett reduction (no u128 division) + precomputed twiddles
/// (autovectorized to NEON on AArch64 with `-C target-cpu=native`).
pub struct NttSquarer {
    n: usize,
    ctx1: PrimeCtx,
    ctx2: PrimeCtx,
    p1_inv_p2: u64,
    buf1: Vec<u64>,
    buf2: Vec<u64>,
}

impl NttSquarer {
    /// Create for transform length `n` (power of 2, ≤ 2²²).
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(
            n <= 1 << 22,
            "n exceeds 2²² (limit set by P2 = 998244353 for negacyclic NTT)"
        );
        NttSquarer {
            n,
            ctx1: build_prime_ctx(n, P1, G1_ROOT),
            ctx2: build_prime_ctx(n, P2, G2_ROOT),
            p1_inv_p2: mod_inv(P1, P2),
            buf1: vec![0u64; n],
            buf2: vec![0u64; n],
        }
    }

    /// Square `x` and return x² as a BigUint (no modular reduction).
    pub fn ntt_square(&mut self, x: &BigUint) -> BigUint {
        let bytes = x.to_bytes_le();
        if bytes.is_empty() {
            return BigUint::zero();
        }
        assert!(
            bytes.len() * 2 <= self.n,
            "input ({} bytes) requires n >= {}, but NttSquarer has n = {}",
            bytes.len(),
            bytes.len() * 2,
            self.n
        );

        let n = self.n;

        for (j, &b) in bytes.iter().enumerate() {
            self.buf1[j] = b as u64;
            self.buf2[j] = b as u64;
        }
        self.buf1[bytes.len()..].fill(0);
        self.buf2[bytes.len()..].fill(0);

        // P1 pipeline
        {
            let p = self.ctx1.p;
            let bm = self.ctx1.barrett.clone();
            for j in 0..n {
                self.buf1[j] = bm.mul(self.buf1[j], self.ctx1.g_powers[j]);
            }
            ntt_fast(&mut self.buf1, &self.ctx1.fwd_twiddles, p, &bm);
            for j in 0..n {
                self.buf1[j] = bm.mul(self.buf1[j], self.buf1[j]);
            }
            intt_fast(&mut self.buf1, &self.ctx1.inv_twiddles, p, &bm);
            for j in 0..n {
                self.buf1[j] = bm.mul(bm.mul(self.buf1[j], self.ctx1.g_inv_powers[j]), self.ctx1.n_inv);
            }
        }

        // P2 pipeline
        {
            let p = self.ctx2.p;
            let bm = self.ctx2.barrett.clone();
            for j in 0..n {
                self.buf2[j] = bm.mul(self.buf2[j], self.ctx2.g_powers[j]);
            }
            ntt_fast(&mut self.buf2, &self.ctx2.fwd_twiddles, p, &bm);
            for j in 0..n {
                self.buf2[j] = bm.mul(self.buf2[j], self.buf2[j]);
            }
            intt_fast(&mut self.buf2, &self.ctx2.inv_twiddles, p, &bm);
            for j in 0..n {
                self.buf2[j] = bm.mul(bm.mul(self.buf2[j], self.ctx2.g_inv_powers[j]), self.ctx2.n_inv);
            }
        }

        // CRT + carry propagation
        let p1_inv_p2 = self.p1_inv_p2;
        let mut carry = 0u64;
        for j in 0..n {
            let c = crt2(self.buf1[j], self.buf2[j], P1, P2, p1_inv_p2) + carry;
            self.buf1[j] = c & 0xFF;
            carry = c >> 8;
        }
        assert_eq!(carry, 0, "carry overflow: input exceeds two-prime dynamic range");

        let mut result_bytes: Vec<u8> = self.buf1[..n].iter().map(|&v| v as u8).collect();
        while result_bytes.last() == Some(&0) {
            result_bytes.pop();
        }
        if result_bytes.is_empty() {
            BigUint::zero()
        } else {
            BigUint::from_bytes_le(&result_bytes)
        }
    }

    /// Square `x` in place mod N = k·2ⁿ − 1.
    pub fn square_kbn(&mut self, x: &mut BigUint, k: u64, n: u64, modulus: &BigUint) {
        let sq = self.ntt_square(x);
        let q = (&sq >> n as usize) / k;
        let r = sq + &q - ((&q * k) << n as usize);
        *x = if r >= *modulus { r - modulus } else { r };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tables(n: usize) -> (u64, u64, u64, u64, Vec<u64>, Vec<u64>) {
        let p = P1;
        let g = mod_pow(G1_ROOT, (p - 1) / (2 * n as u64), p);
        let g_inv = mod_inv(g, p);
        let w = mod_mul(g, g, p);
        let w_inv = mod_inv(w, p);
        let n_inv = mod_inv(n as u64, p);
        let mut g_powers = vec![0u64; n];
        let mut g_inv_powers = vec![0u64; n];
        let mut gp = 1u64;
        let mut gip = 1u64;
        for j in 0..n {
            g_powers[j] = gp;
            g_inv_powers[j] = gip;
            gp = mod_mul(gp, g, p);
            gip = mod_mul(gip, g_inv, p);
        }
        (w, w_inv, n_inv, p, g_powers, g_inv_powers)
    }

    #[test]
    fn test_ntt_round_trip() {
        let n = 8;
        let (w, w_inv, n_inv, p, _, _) = make_tables(n);
        let input = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut a = input;
        ntt(&mut a, w, p);
        intt(&mut a, w_inv, p);
        for (orig, &got) in input.iter().zip(a.iter()) {
            assert_eq!(got, mod_mul(*orig, n as u64, p));
        }
        for (orig, &got) in input.iter().zip(a.iter()) {
            assert_eq!(mod_mul(got, n_inv, p), *orig);
        }
    }

    #[test]
    fn test_twist_correctness() {
        let n = 8;
        let (w, w_inv, n_inv, p, g_powers, g_inv_powers) = make_tables(n);
        let input = [10u64, 20, 30, 40, 0, 0, 0, 0];
        let expected = naive_negacyclic_square(&input, p);
        let mut a = input;
        for j in 0..n {
            a[j] = mod_mul(a[j], g_powers[j], p);
        }
        ntt(&mut a, w, p);
        for j in 0..n {
            a[j] = mod_mul(a[j], a[j], p);
        }
        intt(&mut a, w_inv, p);
        for j in 0..n {
            a[j] = mod_mul(mod_mul(a[j], g_inv_powers[j], p), n_inv, p);
        }
        assert_eq!(a.to_vec(), expected);
    }

    #[test]
    fn test_crt2() {
        let p1_inv_p2 = mod_inv(P1, P2);
        let c: u64 = 12_345_678;
        assert_eq!(crt2(c % P1, c % P2, P1, P2, p1_inv_p2), c);
        assert_eq!(crt2(0, 0, P1, P2, p1_inv_p2), 0);
        let c: u64 = P1;
        assert_eq!(crt2(c % P1, c % P2, P1, P2, p1_inv_p2), c);
        let c = P1 * P2 - 1;
        assert_eq!(crt2(P1 - 1, P2 - 1, P1, P2, p1_inv_p2), c);
    }

    #[test]
    fn test_barrett_mul() {
        let bm = BarrettMod::new(P1);
        // Test against naive mod_mul
        let cases = [(0, 0), (1, 1), (255, 255), (P1 - 1, 1), (P1 - 1, P1 - 1),
                     (1_000_000, 1_000_001), (2_000_000, 1_000_000)];
        for (a, b) in cases {
            let expected = mod_mul(a, b, P1);
            assert_eq!(bm.mul(a, b), expected, "a={a}, b={b}");
        }
        let bm2 = BarrettMod::new(P2);
        for (a, b) in cases {
            let a = a % P2;
            let b = b % P2;
            let expected = mod_mul(a, b, P2);
            assert_eq!(bm2.mul(a, b), expected, "a={a}, b={b} mod P2");
        }
    }

    #[test]
    fn test_ntt_fast_matches_reference() {
        // ntt_fast with precomputed twiddles should match the reference ntt
        let n = 16;
        let p = P1;
        let g = mod_pow(G1_ROOT, (p - 1) / (2 * n as u64), p);
        let w = mod_mul(g, g, p);
        let bm = BarrettMod::new(p);
        let twiddles = build_twiddles(n, w, p);

        let input: Vec<u64> = (1..=n as u64).map(|x| x * 1000 % p).collect();
        let mut a_ref = input.clone();
        let mut a_fast = input.clone();

        ntt(&mut a_ref, w, p);
        ntt_fast(&mut a_fast, &twiddles, p, &bm);

        assert_eq!(a_ref, a_fast);
    }
}
