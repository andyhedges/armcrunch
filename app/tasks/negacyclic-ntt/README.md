# Negacyclic NTT — Phase 1 (scalar reference)

Modular squaring of large integers for primality testing (LLR-style repeated squaring of numbers with millions of digits). Target platform is AWS Graviton 3/4 (ARM64). Phase 1 is a correct, readable scalar implementation; SIMD optimization comes later.

All code lives in `armcrunch/src/negacyclic.rs` (new file), wired into `armcrunch/src/lib.rs`.

## Phase 1 constraints

DO:
- `num-bigint` and `num-traits` are already in `Cargo.toml`
- All NTT arithmetic in `u64`, use `u128` intermediates for `mod_mul`
- Radix-2 DIT NTT with bit-reversal permutation
- Precomputed tables in a struct
- Tests comparing against naive BigUint multiplication

DO NOT:
- SIMD intrinsics or `std::arch` — Phase 3
- Multi-prime CRT — Phase 2
- Threading, rayon, or async
- Any optimization beyond basic clean code

## Shared constants

```
p = 2013265921   (= 15 × 2^27 + 1, fits in 31 bits)
primitive root mod p = 31
p − 1 = 15 × 2^27  →  supports NTTs up to length 2^27
```

## Task order

| Status | File | What it builds | Done when |
|--------|------|---------------|-----------|
| [x] | [task-01-core-arithmetic.md](task-01-core-arithmetic.md) | `mod_mul`, `mod_pow`, `mod_inv` | `cargo test test_mod_` → 3 green |
| [x] | [task-02-ntt-intt.md](task-02-ntt-intt.md) | `ntt`, `intt` with bit-reversal | `cargo test test_ntt_` → 2 green |
| [x] | [task-03-twist-and-struct.md](task-03-twist-and-struct.md) | `FftSquarer::new`, `naive_negacyclic_square` | `cargo test test_twist_ test_g_powers_` → 2 green |
| [x] | [task-04-fft-square.md](task-04-fft-square.md) | `FftSquarer::fft_square` | 4 new tests green |
| [x] | [task-05-integration.md](task-05-integration.md) | `lib.rs` wiring + final test | all 10 tests green |

## Marking a task complete

When a task's tests pass, check it off in the table above by changing `[ ]` to `[x]`:

```
| [x] | [task-01-core-arithmetic.md](task-01-core-arithmetic.md) | ...
```

This file is the source of truth for progress — update it at the end of each session so the next session can see exactly where to pick up.
