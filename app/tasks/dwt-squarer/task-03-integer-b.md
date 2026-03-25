# Task 3 of 5 — Guarantee integer b (fix test_phase_a_chain)

## The bug

For k=1, exp=31, L=8: b = exp/L = 3.875 (non-integer). The weight formula
`fwd[j] = 2^(j·b − bit_pos_j)` is supposed to convert integer mixed-radix limbs
into a uniform fractional-base representation so that cyclic convolution computes
multiplication mod M. However for the wrap terms (i+j = k+L), the unweighted
IFFT coefficient contains a factor of `2^(bit_pos_k − bit_pos_i − bit_pos_j)` which
equals 1 only when all limb widths are equal (i.e. when L | exp, making b an integer).

When L∤exp, limb widths alternate between b_lo and b_lo+1, so
`bit_pos_k ≠ bit_pos_i + bit_pos_j` in general, and the coefficient is off by up to
`2^(n_hi/L)`. For exp=31: this is `2^(7/8) ≈ 1.83` — large enough to cause wrong
rounding after the first squaring that mixes high and low limbs.

## The fix: ensure L | total_bits

Choose `len` large enough that `total_bits % len == 0` — i.e., b is exactly an integer.

```rust
// Find the smallest power-of-2 len >= 2*n_bytes such that len | total_bits.
// total_bits = exp (for k=1) or exp + k_extra_bits (for k>1).
let min_len = (2 * n_bytes).next_power_of_two();
let len = {
    let mut l = min_len;
    while total_bits % l as u64 != 0 {
        l *= 2;
    }
    l
};
```

For k=1, exp=31:
- min_len = 8, 31 % 8 = 7 ≠ 0
- l=16: 31 % 16 = 15 ≠ 0
- l=32: 31 % 32 = 31 ≠ 0
- l=64: 31 % 64 = 31 ≠ 0

31 is prime — no power-of-2 divides it. In this case, **pad `exp` up to the next
multiple of `len`** and widen the representation:

```rust
let min_len = (2 * n_bytes).next_power_of_two();
// Round total_bits up to a multiple of min_len so b is an integer.
let padded_bits = min_len as u64 * ((total_bits + min_len as u64 - 1) / min_len as u64);
let len = min_len; // keep the same FFT length
let b_lo = padded_bits / len as u64;
let n_hi = 0usize; // padded_bits is a multiple of len, so all limbs equal width
```

For k=1, exp=31, min_len=8: padded_bits = 8 * ceil(31/8) = 8*4 = 32.
b_lo = 32/8 = 4, n_hi = 0. Total bits = 32 ≥ 31 ✓. The extra bit means we
represent x with 32 bits (a[0..7] each 4 bits), where the true value is < 2^31
so a[7] always ≤ 7 (top limb is at most 2^3-1 since x < M < 2^31).

**Important**: The modulus check in `carry_normalize` must use `exp` (not `padded_bits`)
for the wrap: overflow from position L-1 represents `2^padded_bits`, and we need to
reduce `2^padded_bits mod M = 2^(padded_bits - exp) mod M = 2^(padding)`. For the
padded case, wrap carries contribute `2^padding` to position 0. Since `padding` =
`padded_bits - exp` is small (< 8 bits for typical cases), this is just a small
integer multiplicative correction in the carry wrap.

For k=1, exp=31, padded_bits=32, padding=1: carry·2^32 ≡ carry·2 (mod 2^31−1),
so the wrap multiplies by 2 instead of 1.

Update `carry_normalize` to take `padding = padded_bits - exp` and apply the wrap as:

```rust
// Overflow carry from position L-1 represents carry·2^padded_bits.
// Reduce: 2^padded_bits = 2^(exp+padding) ≡ 2^padding (mod 2^exp−1)  [k=1 case]
// So add carry · (1 << padding) to the low limbs.
```

## Tests

```rust
#[test]
fn test_phase_a_chain_prime_exp() {
    // exp=31 is prime — requires padding to guarantee integer b
    let exp = 31u64;
    let m = (BigUint::from(1u64) << 31usize) - 1u64;
    let start = BigUint::from(3u64);
    let mut x_naive = start.clone();
    let mut sq = DwtSquarer::new(1, exp, &start);
    for i in 0..10 {
        x_naive = (&x_naive * &x_naive) % &m;
        sq.square();
        assert_eq!(sq.to_biguint(), x_naive, "mismatch at iteration {i}");
    }
}
```

This is the existing `test_phase_a_chain` test — it should now pass.

## Done when

```
cargo test test_phase_a
```

Both `test_phase_a_single_square` and `test_phase_a_chain` green.
