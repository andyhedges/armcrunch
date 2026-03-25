# Task 5 of 5 — Integration and lib.rs wiring

Wire `negacyclic.rs` into `lib.rs`, add the final integration test, and add a debug dynamic-range hint. Depends on Tasks 1–4.

## lib.rs changes

```rust
mod negacyclic;
pub use negacyclic::FftSquarer as NegacyclicSquarer;
```

## Debug hint in FftSquarer::new

Add inside `new()` after building the struct (debug builds only):

```rust
#[cfg(debug_assertions)]
eprintln!(
    "NegacyclicSquarer(n={n}): max safe input = {} bytes (single-prime limit)",
    n  // base-256, so n bytes = n*8 bits
);
```

## Final integration test

```rust
#[test]
fn test_larger_square() {
    use num_bigint::BigUint;
    let x = BigUint::parse_bytes(b"DEADBEEFCAFEBABE1234567890ABCDEF", 16).unwrap();
    let mut sq = FftSquarer::new(256);
    let result = sq.fft_square(&x);
    assert_eq!(result, &x * &x);
}
```

## Done when

```
cargo test
```

All 10 tests pass:

```
test test_mod_mul_no_overflow ... ok
test test_mod_pow_known ... ok
test test_mod_inv ... ok
test test_ntt_round_trip ... ok
test test_ntt_known_convolution ... ok
test test_twist_correctness ... ok
test test_g_powers_cycle ... ok
test test_small_square ... ok
test test_multi_limb_square ... ok
test test_zero ... ok
test test_one ... ok
test test_larger_square ... ok
```
