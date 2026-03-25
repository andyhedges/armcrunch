# Task 4 of 5 — fft_square: BigUint integration

Implement `FftSquarer::fft_square`. This is the top-level squaring API. Depends on Tasks 1–3.

## Implementation

```rust
impl FftSquarer {
    /// Square x using negacyclic NTT. Returns x² (exact integer, not reduced mod anything).
    /// Panics if the input exceeds the single-prime dynamic range.
    pub fn fft_square(&mut self, x: &BigUint) -> BigUint {
        // Step 1: decompose x into n base-256 limbs (little-endian, zero-padded)
        let b: u64 = 256;
        let bytes = x.to_bytes_le();
        assert!(
            bytes.len() <= self.n,
            "input too large: {} bytes > transform length {}. Use a larger n.",
            bytes.len(), self.n
        );

        // Validate dynamic range: N * (B-1)^2 < p
        let max_coeff = self.n as u128 * (b as u128 - 1) * (b as u128 - 1);
        assert!(
            max_coeff < self.p as u128,
            "dynamic range exceeded: N*(B-1)^2={max_coeff} >= p={}. \
             Use a larger NTT prime (Phase 2).",
            self.p
        );

        let mut a: Vec<u64> = bytes.into_iter().map(|b| b as u64).collect();
        a.resize(self.n, 0);

        // Step 2: twist
        for (j, x) in a.iter_mut().enumerate() {
            *x = mod_mul(*x, self.g_powers[j], self.p);
        }

        // Step 3: forward NTT
        ntt(&mut a, self.w, self.p);

        // Step 4: pointwise square
        for x in &mut a { *x = mod_mul(*x, *x, self.p); }

        // Step 5: inverse NTT
        intt(&mut a, self.w_inv, self.p);

        // Step 6: untwist + normalize by 1/N
        for (j, x) in a.iter_mut().enumerate() {
            *x = mod_mul(mod_mul(*x, self.g_inv_powers[j], self.p), self.n_inv, self.p);
        }

        // Step 7: carry propagation — push limbs back into base 256
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

        // Step 8: reassemble into BigUint
        let bytes: Vec<u8> = a.into_iter().map(|v| v as u8).collect();
        BigUint::from_bytes_le(&bytes)
    }
}
```

## Tests

```rust
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
```

## Done when

```
cargo test test_small_square test_multi_limb_square test_zero test_one
```

4 tests green.
