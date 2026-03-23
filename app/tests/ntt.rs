use armcrunch::NttSquarer;
use num_bigint::BigUint;

#[test]
fn test_small_square() {
    let x = BigUint::from(255u64);
    let mut sq = NttSquarer::new(64);
    let result = sq.ntt_square(&x);
    assert_eq!(result, &x * &x);
}

#[test]
fn test_multi_limb_square() {
    let x = BigUint::from(123456789u64);
    let mut sq = NttSquarer::new(64);
    let result = sq.ntt_square(&x);
    assert_eq!(result, &x * &x);
}

#[test]
fn test_larger_square() {
    let x = BigUint::parse_bytes(b"DEADBEEFCAFEBABE1234567890ABCDEF", 16).unwrap();
    let mut sq = NttSquarer::new(256);
    let result = sq.ntt_square(&x);
    assert_eq!(result, &x * &x);
}
