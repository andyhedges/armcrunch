# Task 5 of 5 — Integration: fermat_test_dwt and benchmarks

Wire everything up. Depends on Tasks 1–4.

## Verify all DWT tests pass

```
cargo test dwt
```

Expected:
```
test dwt::tests::test_weights_unity_when_l_divides_exp ... ok
test dwt::tests::test_phase_a_single_square ... ok
test dwt::tests::test_phase_a_l_divides_exp ... ok
test dwt::tests::test_phase_a_chain ... ok
test dwt::tests::test_phase_b_single_square ... ok
test dwt::tests::test_phase_b_chain ... ok
test dwt::tests::test_phase_b_chain_small ... ok
test dwt::tests::test_round_trip_k1_l_divides_exp ... ok
test dwt::tests::test_round_trip_k1_l_not_divides_exp ... ok
test dwt::tests::test_round_trip_k_gt_1 ... ok
test dwt::tests::test_round_trip_zero_and_one ... ok
test dwt::tests::test_limb_width_covers_modulus ... ok
```

## Check fermat_test_dwt correctness

`src/fermat.rs` contains `fermat_test_dwt`. Verify it agrees with `fermat_test` on a
small case:

```rust
// In tests/fermat.rs or src/fermat.rs tests:
#[test]
fn test_dwt_matches_naive() {
    use crate::{fermat_test, fermat_test_dwt, Kbn};
    // Small modulus: k=3, n=8 → M=767, run 8 squarings
    let kbn = Kbn::new(3, 8);
    let naive = fermat_test(&kbn, 2, |_| {});
    let dwt   = fermat_test_dwt(&kbn, 2, |_| {});
    assert_eq!(naive, dwt);
}
```

## Add DWT to benchmarks

In `benches/square_mod.rs`:

```rust
use armcrunch::{..., DwtSquarer};

// In bench_square_mod():
let mut squarer_dwt = DwtSquarer::new(kbn.k, kbn.n, &x);

group.bench_function("dwt", |b| {
    b.iter(|| squarer_dwt.square())
});
```

Note: `DwtSquarer::square()` does NOT return a value — it mutates internal state.
It also does NOT reduce mod M externally (reduction is part of the carry step).
The benchmark measures raw squaring throughput. Add a separate extraction bench if useful:

```rust
group.bench_function("dwt_extract", |b| {
    b.iter(|| squarer_dwt.to_biguint())
});
```

## Remove pub(crate) on signal field

`signal` was made `pub(crate)` for diagnostic prints in `test_phase_a_chain`. Once all
tests pass, make it private:

```rust
// In DwtSquarer struct:
signal: Vec<Complex<f64>>,   // remove pub(crate)
```

And remove the diagnostic block from `test_phase_a_chain`.

## Done when

```
cargo test
cargo bench --bench square_mod -- dwt
```

- All tests green (no failures)
- Benchmark output includes a `dwt` row
- `dwt` throughput is meaningfully faster than `ntt` (target: ~2× faster, since one FFT pair vs two)
