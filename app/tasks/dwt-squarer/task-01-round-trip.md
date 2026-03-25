# Task 1 of 5 — Load / to_biguint round-trip

Before fixing anything, establish the most basic invariant: loading a value and immediately
reading it back gives the same value. If this fails it means `limbs_from_bytes`,
`biguint_from_signal`, or the weight application in `load()` is wrong.

## Tests to add (in `dwt.rs`)

```rust
#[test]
fn test_round_trip_k1_l_divides_exp() {
    // L | exp → weights = 1, simplest case
    let x = BigUint::from(123456789u64);
    let sq = DwtSquarer::new(1, 32, &x);
    assert_eq!(sq.to_biguint(), x);
}

#[test]
fn test_round_trip_k1_l_not_divides_exp() {
    // L ∤ exp → fractional weights, the problematic case
    let x = BigUint::from(123456789u64);
    let sq = DwtSquarer::new(1, 31, &x);
    assert_eq!(sq.to_biguint(), x);
}

#[test]
fn test_round_trip_k_gt_1() {
    // k=1005, exp=64: x can be wider than exp bits
    let k = 1005u64;
    let exp = 64u64;
    let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
    // Use a value that exercises the full range (> 2^exp)
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
```

## What to look for if tests fail

- `test_round_trip_k1_l_divides_exp` fails → bug in `limbs_from_bytes` or `biguint_from_signal`
- `test_round_trip_k1_l_not_divides_exp` fails → bug in how fractional weights are applied/inverted
- `test_round_trip_k_gt_1` fails → representation is too narrow (see Task 2)

Add diagnostic prints to `to_biguint` if needed:
```rust
let raw: Vec<f64> = tmp.iter().map(|c| c.re).collect();
eprintln!("raw limbs after IFFT+unweight: {raw:?}");
let rounded: Vec<i64> = tmp.iter().map(|c| c.re.round() as i64).collect();
eprintln!("rounded limbs: {rounded:?}");
```

## Done when

```
cargo test test_round_trip
```

4 tests green.
