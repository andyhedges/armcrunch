# Task 2 of 5 — Widen limb representation for k > 1

## The bug

`DwtSquarer::new` computes:

```rust
let n_bytes = (exp as usize + 7) / 8;
let len = (2 * n_bytes).next_power_of_two();
let b_lo = exp / len as u64;
let n_hi = (exp % len as u64) as usize;  // ← only covers exp bits
```

`n_hi` limbs are one bit wider, giving a total of exactly `exp` bits of
representation. But M = k·2^exp − 1 has `exp + ⌈log₂(k)⌉` bits. Values x ∈ [0, M)
can be that wide. When k=1005 and exp=64: M ≈ 1005·2^64 needs ~74 bits; the
representation holds only 64, so `limbs_from_bytes` silently drops the top 10 bits.

## The fix

Add `⌈log₂(k)⌉` to `n_hi` so the extra bits from k fit in the high limbs:

```rust
let k_bits = if k <= 1 { 0u64 } else { 64 - (k - 1).leading_zeros() as u64 };
let total_bits = exp + k_bits;
let b_lo = exp / len as u64;               // base limb width unchanged
let n_hi = (total_bits % (len as u64 * b_lo + len as u64 - exp + exp)) as usize;
```

Simpler to think about it as: total bits needed = exp + k_bits.
High limbs cover the extra: n_hi = (total_bits - b_lo * len as u64) as usize.

```rust
let k_bits = (u64::BITS - k.leading_zeros()) as u64; // ⌈log₂(k+1)⌉ ≈ ⌈log₂(k)⌉
let total_bits = exp + k_bits;
let b_lo = total_bits / len as u64;
let n_hi = (total_bits % len as u64) as usize;
```

**Note**: this changes `b_lo` too. For k=1 (k_bits=1 for k=1... careful): use k_bits=0
when k=1 (no extra bits needed). A clean formula:

```rust
let k_extra_bits = if k <= 1 { 0u64 } else { u64::BITS as u64 - k.leading_zeros() as u64 };
let total_bits = exp + k_extra_bits;
let b_lo = total_bits / len as u64;
let n_hi = (total_bits % len as u64) as usize;
```

Verify: for k=1, exp=31, len=8: total_bits=31, b_lo=3, n_hi=7 (unchanged from current)
For k=1005, exp=64, len=16: k_extra_bits=10, total_bits=74, b_lo=4, n_hi=10

## Also update `build_weights`

`build_weights` takes `b_lo` and `n_hi` and already uses them correctly — no change needed there, but rerun the `test_weights_unity_when_l_divides_exp` test to confirm it still holds for k=1.

## Tests

After this change, `test_round_trip_k_gt_1` from Task 1 should pass. Also add:

```rust
#[test]
fn test_limb_width_covers_modulus() {
    // The total bit width must be >= log2(M+1) = log2(k) + exp
    let k = 1005u64;
    let exp = 64u64;
    let m = BigUint::from(k) * (BigUint::from(1u64) << exp as usize) - 1u64;
    let sq = DwtSquarer::new(k, exp, &m);
    // M itself should round-trip (it reduces to 0, but M-1 should work)
    let x = &m - BigUint::from(1u64);
    let sq = DwtSquarer::new(k, exp, &x);
    assert_eq!(sq.to_biguint(), x);
}
```

## Done when

```
cargo test test_round_trip test_limb_width
```

All round-trip tests green, including the k=1005 case.
