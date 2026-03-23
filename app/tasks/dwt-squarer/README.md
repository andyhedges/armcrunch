# DWT Squarer — fixing the failing chain tests

The `DwtSquarer` in `src/dwt.rs` keeps `x` in FFT domain between squarings, so the inner loop of the Fermat test costs only one FFT pair per iteration (no BigUint allocation). This is the performance goal.

## Current state

4 tests pass, 2 fail:

```
test dwt::tests::test_weights_unity_when_l_divides_exp ... ok
test dwt::tests::test_phase_a_l_divides_exp ... ok   ← k=1, exp=32, L|exp
test dwt::tests::test_phase_a_single_square ... ok   ← single square only
test dwt::tests::test_phase_b_single_square ... ok   ← single square only
test dwt::tests::test_phase_a_chain ... FAILED       ← k=1, exp=31, L∤exp
test dwt::tests::test_phase_b_chain ... FAILED       ← k=1005, exp=64
```

## Known root causes

**Bug 1 — representation too narrow for k>1 (`test_phase_b_chain`)**

`DwtSquarer::new` sizes its limb array to hold `exp` bits. But M = k·2^exp − 1 has
`exp + ⌈log₂(k)⌉` bits, and values x ∈ [0, M) can be that wide. For k=1005 and
exp=64: M needs ~74 bits, but the representation only covers 64. Loading x loses the
top ~10 bits.

Fix: add `⌈log₂(k)⌉` bits to `n_hi` so the limb array can hold the full range.

**Bug 2 — weight formula breaks for L∤exp (`test_phase_a_chain`)**

For M = 2^31 − 1 with transform length L=8: b = exp/L = 3.875 (non-integer).
The forward weight `fwd[j] = k^(j/L) · 2^(j·exp/L − bit_pos_j)` is supposed to
convert integer mixed-radix limbs into a "fractional base 2^b" representation that
makes cyclic convolution compute multiplication mod M. However, for the wrap terms
(i+j = k+L), the unweighted coefficient contains an extra factor of
`2^(bit_pos_k − bit_pos_i − bit_pos_j)` which is 1 only when L | exp.

When L∤exp, the rounded IFFT coefficients are not the true polynomial product
coefficients, so carry normalization produces wrong limbs.

Fix: choose the transform length so that L | exp (guaranteed integer b), OR derive
and implement the correct two-pass balanced weight approach from GWnum.

## Key files

| File | Role |
|------|------|
| `src/dwt.rs` | All DWT code — struct, weights, carry, load, square, to_biguint |
| `src/fermat.rs` | `fermat_test_dwt` — uses DwtSquarer for the Fermat loop |
| `benches/square_mod.rs` | Benchmarks — needs a `dwt` entry |

## Shared constants / invariants

```
M = k·2^exp − 1
transform length L = power of 2
b_lo = floor(exp / L), n_hi = exp % L   (before k fix)
total limb bits = n_hi·(b_lo+1) + (L−n_hi)·b_lo = exp
```

After fix: `total limb bits = exp + ⌈log₂(k)⌉` when k > 1.

## Task order

| Status | File | What it fixes | Done when |
|--------|------|--------------|-----------|
| [x] | [task-01-round-trip.md](task-01-round-trip.md) | Verify load → to_biguint = identity | 3/4 pass; k_gt_1 fails → Task 2 |
| [x] | [task-02-k-gt-1-repr.md](task-02-k-gt-1-repr.md) | Widen limb array for k>1 | round-trip holds for k=1005 |
| [ ] | [task-03-integer-b.md](task-03-integer-b.md) | Guarantee L\|exp so b is integer | `test_phase_a_chain` green |
| [ ] | [task-04-carry-normalize.md](task-04-carry-normalize.md) | Correct carry wrap for k>1 | `test_phase_b_chain` green |
| [ ] | [task-05-integration.md](task-05-integration.md) | `fermat_test_dwt` + benchmarks | all DWT tests green, bench runs |

## Marking a task complete

Change `[ ]` to `[x]` in the table above when the task's "Done when" criterion is met.

This file is the source of truth for progress.
