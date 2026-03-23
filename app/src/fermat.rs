use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::arithmetic::FftSquarer;
use crate::dwt::DwtSquarer;
use crate::ntt::NttSquarer;

/// Represents a number of the form k * 2^n - 1.
pub struct Kbn {
    pub k: u64,
    pub n: u64,
}

impl Kbn {
    pub fn new(k: u64, n: u64) -> Self {
        Self { k, n }
    }

    /// Compute the value N = k * 2^n - 1.
    pub fn value(&self) -> BigUint {
        (BigUint::from(self.k) << self.n) - BigUint::one()
    }
}

fn extended_gcd(
    a: &num_bigint::BigInt,
    b: &num_bigint::BigInt,
) -> (num_bigint::BigInt, num_bigint::BigInt, num_bigint::BigInt) {
    use num_bigint::BigInt;
    if a.is_zero() {
        return (b.clone(), BigInt::zero(), BigInt::one());
    }
    let (g, x, y) = extended_gcd(&(b % a), a);
    (g, y - (b / a) * &x, x)
}

/// Compute modular inverse of `a` mod `m`.  Returns None if gcd(a, m) != 1.
fn mod_inv(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    use num_bigint::BigInt;
    let a = BigInt::from(a.clone());
    let m = BigInt::from(m.clone());
    let (g, x, _) = extended_gcd(&a, &m);
    if g != BigInt::one() {
        return None;
    }
    let result = ((x % &m) + &m) % &m;
    result.try_into().ok()
}

/// Fermat probable prime test for N = k * 2^n - 1.
///
/// Returns true if N is a probable prime to base `a`.
///
/// Uses the smooth decomposition:
///   a^(N-1) = a^(k*2^n - 2) = (a^k)^(2^n) * a^(-2) mod N
pub fn fermat_test(kbn: &Kbn, a: u64, progress: impl Fn(f64)) -> BigUint {
    let modulus = kbn.value();

    if modulus.is_zero() || modulus == BigUint::one() {
        return BigUint::zero();
    }

    let a_big = BigUint::from(a);

    // Step 1: x = a^k mod modulus
    let mut x = a_big.modpow(&BigUint::from(kbn.k), &modulus);

    // Step 2: square x `n` times → x = a^(k * 2^n) mod modulus
    let mut squarer = FftSquarer::new(&modulus);
    let mut last_report = std::time::Instant::now();
    let report_interval = std::time::Duration::from_secs(1);
    for i in 0..kbn.n {
        squarer.square_kbn(&mut x, kbn.k, kbn.n, &modulus);
        if last_report.elapsed() >= report_interval {
            progress(i as f64 * 100.0 / kbn.n as f64);
            last_report = std::time::Instant::now();
        }
    }

    // Step 3: multiply by a^(-2) mod modulus → x = a^(k*2^n - 2) = a^(N-1) mod modulus
    let a_sq = (&a_big * &a_big) % &modulus;
    let a_sq_inv = match mod_inv(&a_sq, &modulus) {
        Some(inv) => inv,
        None => return BigUint::zero(), // gcd(a^2, modulus) != 1, so N is composite
    };
    x = (x * a_sq_inv) % &modulus;

    x
}

/// Fermat probable prime test for N = k * 2^n - 1, using DWT-based squaring.
///
/// Stores x in FFT domain throughout the squaring loop; one FFT pair per iteration.
pub fn fermat_test_dwt(kbn: &Kbn, a: u64, progress: impl Fn(f64)) -> BigUint {
    let modulus = kbn.value();

    if modulus.is_zero() || modulus == BigUint::one() {
        return BigUint::zero();
    }

    let a_big = BigUint::from(a);

    // Step 1: x = a^k mod modulus
    let x = a_big.modpow(&BigUint::from(kbn.k), &modulus);

    // Step 2: square x `n` times using DWT squarer
    let mut squarer = DwtSquarer::new(kbn.k, kbn.n, &x);
    let mut last_report = std::time::Instant::now();
    let report_interval = std::time::Duration::from_secs(1);
    for i in 0..kbn.n {
        squarer.square();
        if last_report.elapsed() >= report_interval {
            progress(i as f64 * 100.0 / kbn.n as f64);
            last_report = std::time::Instant::now();
        }
    }

    let mut x = squarer.to_biguint();

    // Step 3: multiply by a^(-2) mod modulus
    let a_sq = (&a_big * &a_big) % &modulus;
    let a_sq_inv = match mod_inv(&a_sq, &modulus) {
        Some(inv) => inv,
        None => return BigUint::zero(),
    };
    x = (x * a_sq_inv) % &modulus;

    x
}

/// Fermat probable prime test for N = k * 2^n - 1, using NTT-based squaring.
///
/// Returns true if N is a probable prime to base `a`.
pub fn fermat_test_ntt(kbn: &Kbn, a: u64, progress: impl Fn(f64)) -> BigUint {
    let modulus = kbn.value();

    if modulus.is_zero() || modulus == BigUint::one() {
        return BigUint::zero();
    }

    let a_big = BigUint::from(a);

    // Step 1: x = a^k mod modulus
    let mut x = a_big.modpow(&BigUint::from(kbn.k), &modulus);

    // Step 2: square x `n` times using NTT squarer
    let n_bytes = (modulus.bits() as usize + 7) / 8;
    let ntt_len = (2 * n_bytes).next_power_of_two();
    let mut squarer = NttSquarer::new(ntt_len);
    let mut last_report = std::time::Instant::now();
    let report_interval = std::time::Duration::from_secs(1);
    for i in 0..kbn.n {
        squarer.square_kbn(&mut x, kbn.k, kbn.n, &modulus);
        if last_report.elapsed() >= report_interval {
            progress(i as f64 * 100.0 / kbn.n as f64);
            last_report = std::time::Instant::now();
        }
    }

    // Step 3: multiply by a^(-2) mod modulus
    let a_sq = (&a_big * &a_big) % &modulus;
    let a_sq_inv = match mod_inv(&a_sq, &modulus) {
        Some(inv) => inv,
        None => return BigUint::zero(),
    };
    x = (x * a_sq_inv) % &modulus;

    x
}
