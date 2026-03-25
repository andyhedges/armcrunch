# Task 2 of 5 — NTT and INTT

Add two in-place transform functions to `armcrunch/src/negacyclic.rs`. Depends on Task 1.

## Functions

```rust
/// In-place radix-2 DIT NTT with bit-reversal permutation.
/// `w` must be a primitive N-th root of unity mod p.
fn ntt(a: &mut [u64], w: u64, p: u64) { ... }

/// In-place inverse NTT. Does NOT include 1/N normalization.
/// `w_inv` = modular inverse of w.
fn intt(a: &mut [u64], w_inv: u64, p: u64) { ... }
```

## Implementation notes

- **Bit-reversal**: for each index `i`, compute `rev = bit_reverse(i, log2(n))`, swap `a[i]` and `a[rev]` when `i < rev`.
- **Cooley-Tukey DIT butterfly**: iterative, stages from `stride = 1` up to `stride = n/2`. At each stage, for each butterfly pair `(u, v)`:
  ```
  t = w_stage * v mod p
  u' = (u + t) mod p
  v' = (u - t + p) mod p
  ```
  where `w_stage` cycles through powers of `w^(n / stride / 2)`.
- `intt` is the same algorithm but using `w_inv` in place of `w`.

## Tests

```rust
// Helper: w = 31^((p-1)/n) mod p
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
```

## Done when

```
cargo test test_ntt_
```

2 tests green.
