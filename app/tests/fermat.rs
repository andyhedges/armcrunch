use armcrunch::{fermat_test, Kbn};
use num_bigint::BigUint;
use num_traits::One;

fn is_prime(kbn: &Kbn) -> bool {
    fermat_test(kbn, 2, |_| {}) == BigUint::one()
}

#[test]
fn test_known_primes() {
    let primes: &[(u64, u64)] = &[
        (1, 2),  // 3
        (1, 3),  // 7
        (1, 5),  // 31
        (3, 2),  // 11
        (3, 4),  // 47
        (1, 7),  // 127
        (1, 13), // 8191
    ];
    for &(k, n) in primes {
        assert!(is_prime(&Kbn::new(k, n)), "{}*2^{}-1 should be probable prime", k, n);
    }
}

#[test]
fn test_known_composites() {
    let composites: &[(u64, u64)] = &[
        (5, 1),  // 9 = 3^2
        (25, 1), // 49 = 7^2
        (1, 1),  // 1 (not prime by definition)
    ];
    for &(k, n) in composites {
        assert!(!is_prime(&Kbn::new(k, n)), "{}*2^{}-1 should NOT be probable prime", k, n);
    }
}

#[test]
#[ignore = "slow: run with --release"]
fn test_large_prime() {
    assert!(is_prime(&Kbn::new(1005, 33144)));
}

#[test]
#[ignore = "slow: run with --release"]
fn test_large_composite() {
    assert!(!is_prime(&Kbn::new(1007, 83036)));
}
