# Task 3 of 5 — Negacyclic twist and FftSquarer::new

Add the `FftSquarer` struct, its constructor, and the naive reference function. Depends on Tasks 1–2.

## Struct

```rust
pub struct FftSquarer {
    n: usize,               // transform length (power of 2)
    p: u64,                 // NTT prime = 2013265921
    g_powers: Vec<u64>,     // g_powers[j] = g^j mod p  (twist)
    g_inv_powers: Vec<u64>, // g^{-j} mod p             (untwist)
    w: u64,                 // primitive N-th root of unity mod p
    w_inv: u64,             // inverse of w
    n_inv: u64,             // N^{-1} mod p
}
```

## Constructor

```rust
impl FftSquarer {
    /// n must be a power of 2 and <= 2^27.
    pub fn new(n: usize) -> Self {
        let p = 2013265921u64;
        // g = 31^((p-1)/(2*n)) — primitive 2N-th root of unity
        let g = mod_pow(31, (p - 1) / (2 * n as u64), p);
        let g_inv = mod_inv(g, p);
        let w = mod_mul(g, g, p);   // g^2 = primitive N-th root of unity
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

        FftSquarer { n, p, g_powers, g_inv_powers, w, w_inv, n_inv }
    }
}
```

## Naive reference (for testing only)

```rust
/// O(N^2) negacyclic convolution of `a` with itself, mod p.
/// c[k] = Σ_{i+j=k} a[i]*a[j]  −  Σ_{i+j=k+N} a[i]*a[j]  (all mod p)
fn naive_negacyclic_square(a: &[u64], p: u64) -> Vec<u64> {
    let n = a.len();
    let mut c = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            let ij = i + j;
            if ij < n {
                c[ij] = (c[ij] + mod_mul(a[i], a[j], p)) % p;
            } else {
                // negacyclic wrap: subtract (sign flip)
                let k = ij - n;
                c[k] = (c[k] + p - mod_mul(a[i], a[j], p)) % p;
            }
        }
    }
    c
}
```

## Tests

```rust
#[test]
fn test_twist_correctness() {
    // Verify twist + NTT + pointwise square + INTT + untwist
    // matches naive_negacyclic_square on a small raw limb array.
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
    // g^N = −1 mod p (g is a primitive 2N-th root of unity)
    let p = 2013265921u64;
    let sq = FftSquarer::new(8);
    let g = sq.g_powers[1];
    let gn = mod_pow(g, sq.n as u64, p);
    assert_eq!(gn, p - 1);
}
```

## Done when

```
cargo test test_twist_ test_g_powers_
```

2 tests green.
