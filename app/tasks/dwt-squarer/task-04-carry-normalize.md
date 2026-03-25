# Task 4 of 5 — Fix carry normalization for k > 1 (test_phase_b_chain)

## Context

After Tasks 2 and 3, the limb representation is correct (wide enough for k>1) and b
is an integer (weights are exact). The remaining failure for `test_phase_b_chain`
(k=1005, exp=64) is in `carry_normalize`: the wrap arithmetic for k>1 must correctly
reduce `carry · 2^exp mod (k·2^exp − 1)`.

## The wrap identity

When the left-to-right carry propagation produces a non-zero carry out of the last limb,
the integer value held in the limbs exceeds `k·2^exp − 1` by `carry · 2^total_bits`.

With `total_bits = exp + k_extra_bits` and the padded-b approach from Task 3,
`2^total_bits = 2^(exp + k_extra_bits)`. Reducing mod M = k·2^exp − 1:

```
carry · 2^total_bits  mod M
= carry · 2^k_extra_bits · 2^exp  mod M
= carry · 2^k_extra_bits · (M+1)/k  mod M
≡ carry · 2^k_extra_bits / k  mod M
```

Since `2^k_extra_bits ≥ k` (by definition of k_extra_bits), `2^k_extra_bits / k` is an
integer. Let `scale = 2^k_extra_bits / k`. Then the wrap adds `carry · scale` to position 0.

For k=1005, k_extra_bits=10: scale = 1024/1005 — NOT an integer! This is the subtlety.

**Better formulation**: use the fact that `k · 2^exp ≡ 1 (mod M)`:

```
carry · 2^total_bits = carry · 2^k_extra_bits · 2^exp
                     ≡ carry · 2^k_extra_bits / k  (mod M)
```

Split `carry · 2^k_extra_bits` by k:

```rust
let scaled = carry * (1i64 << k_extra_bits);
let q = scaled.div_euclid(k as i64);  // goes to position 0
let r = scaled.rem_euclid(k as i64);  // leftover: r · 2^exp needs further reduction
// r · 2^exp ≡ r/k mod M  → but r < k, so r · 2^exp / k < 2^exp; fits in L limbs
// In practice: add q to position 0, then re-enter the wrap loop with carry = r
```

The loop converges because `r < k` and `k` is small.

## Updated `carry_normalize` signature

```rust
fn carry_normalize(
    signal: &mut [Complex<f64>],
    b_lo: u64,
    n_hi: usize,
    k: u64,
    k_extra_bits: u64,   // ← new: was unused _exp before
    len: usize,
)
```

## Updated wrap loop

```rust
// carry represents: carry · 2^total_bits worth of excess
// total_bits = b_lo * len (since n_hi=0 after Task 3 padding)
// 2^total_bits = 2^(exp + k_extra_bits) ≡ 2^k_extra_bits / k  (mod M)
let mut wrap_iters = 0;
while carry != 0 {
    wrap_iters += 1;
    debug_assert!(wrap_iters <= 8, "carry wrap not converging: carry={carry}");

    let scaled = carry.checked_mul(1i64 << k_extra_bits)
        .expect("carry too large");
    let q = scaled.div_euclid(k as i64);
    let r = scaled.rem_euclid(k as i64);

    // Add q to limbs starting at position 0, propagate forward
    carry = 0;
    let mut add = q;
    for j in 0..len {
        if add == 0 { break; }
        let width = if j < n_hi { b_lo + 1 } else { b_lo };
        let mask = (1i64 << width) - 1;
        let v = a[j] + add;
        add = v >> width;
        a[j] = v & mask;
    }
    carry = add;

    // r · 2^exp is the remaining excess; feed it back as carry for next iteration
    // (it will be scaled again by 2^k_extra_bits / k in the next loop pass)
    // Simpler: treat r as a new wrap carry of r (representing r · 2^total_bits / 2^k_extra_bits)
    // This requires a second sub-loop or a reformulation.
    // Easiest: recurse with carry += r (r is small, ≤ k-1)
    carry += r;
}
```

**Note**: The `r` term represents `r · 2^exp` of excess. On the next loop iteration,
it gets scaled by `2^k_extra_bits / k` again, which gives `r · 2^k_extra_bits / k < k · 2^k_extra_bits / k = 2^k_extra_bits`. This converges rapidly since r < k.

## Verify k=1 still works

For k=1, k_extra_bits=0: `scaled = carry * 1 = carry`, q=carry, r=0. Same as current
behaviour. ✓

## Tests

The existing `test_phase_b_chain` is the target test:

```rust
// k=1005, exp=64, 8 squarings — must match naive BigUint
test dwt::tests::test_phase_b_chain ... ok
```

Also add a small controlled k>1 chain:

```rust
#[test]
fn test_phase_b_chain_small() {
    // k=3, exp=8: M = 3*256-1 = 767. Easy to compute naively.
    let k = 3u64;
    let exp = 8u64;
    let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
    let start = BigUint::from(100u64);
    let mut x_naive = start.clone();
    let mut sq = DwtSquarer::new(k, exp, &start);
    for i in 0..6 {
        x_naive = (&x_naive * &x_naive) % &m;
        sq.square();
        assert_eq!(sq.to_biguint(), x_naive, "mismatch at iteration {i}");
    }
}
```

## Done when

```
cargo test test_phase_b
```

Both `test_phase_b_single_square`, `test_phase_b_chain`, and `test_phase_b_chain_small` green.
