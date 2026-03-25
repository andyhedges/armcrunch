# DWT Squarer — fixing the failing chain tests

The `DwtSquarer` in `src/dwt.rs` keeps `x` in FFT domain between squarings, so the inner loop of the Fermat test costs only one FFT pair per iteration (no BigUint allocation). This is the performance goal.

## Current state

All tests pass.

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

Fix: pad total_bits to a multiple of L so b is always an integer. Add final
reduction step (compare against M, subtract if needed) to handle values in
[M, 2^padded_bits).

**Bug 3 — carry wrap for k>1 didn't converge**

The original carry wrap used a division-by-k approach that didn't converge for
k=1005. Fixed by precomputing `wrap_factor = 2^padded_bits mod M` as limbs and
using it directly in the carry wrap loop.

## Key files

| File | Role |
|------|------|
| `src/dwt.rs` | All DWT code — struct, weights, carry, load, square, to_biguint |
| `src/fermat.rs` | `fermat_test_dwt` — uses DwtSquarer for the Fermat loop |
| `benches/square_mod.rs` | Benchmarks — includes `dwt` entry |

## Shared constants / invariants

```
M = k·2^exp − 1
transform length L = power of 2
b_lo = padded_bits / L (uniform, integer)
n_hi = 0 (all limbs same width after padding)
padded_bits = ceil(total_bits / L) * L
total_bits = exp + k_extra_bits
```

## Task order

| Status | File | What it fixes | Done when |
|--------|------|--------------|-----------|
| [x] | [task-01-round-trip.md](task-01-round-trip.md) | Verify load → to_biguint = identity | 3/4 pass; k_gt_1 fails → Task 2 |
| [x] | [task-02-k-gt-1-repr.md](task-02-k-gt-1-repr.md) | Widen limb array for k>1 | round-trip holds for k=1005 |
| [x] | [task-03-integer-b.md](task-03-integer-b.md) | Guarantee L\|exp so b is integer | `test_phase_a_chain` green |
| [x] | [task-04-carry-normalize.md](task-04-carry-normalize.md) | Correct carry wrap for k>1 | `test_phase_b_chain` green |
| [x] | [task-05-integration.md](task-05-integration.md) | `fermat_test_dwt` + benchmarks | all DWT tests green, bench runs |

## Marking a task complete

Change `[ ]` to `[x]` in the table above when the task's "Done when" criterion is met.

This file is the source of truth for progress.