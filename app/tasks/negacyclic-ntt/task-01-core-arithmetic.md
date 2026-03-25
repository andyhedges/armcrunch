# Task 1 of 5 — Core modular arithmetic

Implement three free functions in a new file `armcrunch/src/negacyclic.rs`.

## Functions

```rust
/// a * b mod p using u128 to avoid overflow.
fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128) * (b as u128) % (p as u128)) as u64
}

/// base^exp mod p via square-and-multiply.
fn mod_pow(base: u64, exp: u64, p: u64) -> u64 { ... }

/// Modular inverse: a^{p-2} mod p (p is prime — Fermat's little theorem).
fn mod_inv(a: u64, p: u64) -> u64 {
    mod_pow(a, p - 2, p)
}
```

## Tests

```rust
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
```

## Done when

```
cargo test test_mod_
```

3 tests green.
