# Task 6 of 6 — Mixed-radix FFT (radix-3, radix-5)

## Goal

Support non-power-of-2 transform lengths like 3×2^n, 5×2^n, 7×2^n.
This allows choosing a smaller FFT length that still satisfies the dynamic
range constraint, reducing the total work.

## Background

gwnum uses FFT length 196608 = 3×2^16 for 1003·2^2499999−1. Our current
implementation must use a power-of-2 length, which may be larger than needed.

For example, if the optimal transform length is 192K but we must round up
to 2^18 = 262144, we're doing 37% more work than necessary.

## Design

Implement radix-3 and radix-5 butterflies:

**Radix-3 butterfly:**
```
a' = a + b + c
b' = a + w3·b + w3²·c     where w3 = e^{-2πi/3}
c' = a + w3²·b + w3·c
```

**Radix-5 butterfly:**
```
(Similar 5-point DFT using 5th roots of unity)
```

These are combined with the radix-2/4 passes in a mixed-radix decomposition:
1. Factor N = n1 × n2 × ... where factors are 2, 3, 4, 5
2. Apply radix-n butterflies in succession
3. Apply twiddle factors between stages

## Implementation

1. Add radix-3 NEON butterfly
2. Add radix-5 NEON butterfly (optional, lower priority)
3. Add FFT length selection logic: choose smallest N = 2^a × 3^b × 5^c
   such that N ≥ min_length and b_max < (53 - log2(N)) / 2
4. Update DwtSquarer::new() to use mixed-radix length selection

## Expected impact

If we can use N = 3×2^16 = 196608 instead of N = 2^18 = 262144, that's
a 25% reduction in FFT work. Combined with NEON acceleration, this could
bring us to ~1.5-2.0 ms/iter, approaching gwnum's Rosetta performance.

## Done when

Mixed-radix FFT supports at least radix-3 × power-of-2 lengths.
DwtSquarer selects optimal mixed-radix length automatically.
`fermat_bench` shows improvement from smaller transform length.