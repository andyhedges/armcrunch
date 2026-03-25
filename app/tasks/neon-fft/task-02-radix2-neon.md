# Task 2 of 6 — NEON radix-2 DIT butterfly

## Goal

Implement a radix-2 decimation-in-time (DIT) FFT using ARM64 NEON intrinsics
for the complex butterfly operations. This replaces the inner loop of the FFT
with SIMD-accelerated code.

## Background

A radix-2 DIT butterfly computes:
```
a' = a + w·b
b' = a − w·b
```
where `a`, `b`, `w` are complex numbers.

With NEON (2×f64 per vector register):
```rust
use std::arch::aarch64::*;

// a = (a_re, a_im), b = (b_re, b_im), w = (w_re, w_im)
// wb_re = w_re*b_re - w_im*b_im
// wb_im = w_re*b_im + w_im*b_re
// a' = (a_re + wb_re, a_im + wb_im)
// b' = (a_re - wb_re, a_im - wb_im)

unsafe {
    let a = vld1q_f64(a_ptr);           // [a_re, a_im]
    let b = vld1q_f64(b_ptr);           // [b_re, b_im]
    let w_re = vdupq_n_f64(w.re);       // [w_re, w_re]
    let w_im = vdupq_n_f64(w.im);       // [w_im, w_im]

    // Complex multiply: wb = w * b
    let b_flip = vextq_f64(b, b, 1);    // [b_im, b_re]
    let wb_1 = vmulq_f64(w_re, b);      // [w_re*b_re, w_re*b_im]
    let wb_2 = vmulq_f64(w_im, b_flip); // [w_im*b_im, w_im*b_re]
    // wb = [w_re*b_re - w_im*b_im, w_re*b_im + w_im*b_re]
    let wb = vfmsq_f64(wb_1, w_im, b_flip); // Actually need careful sign handling

    let a_out = vaddq_f64(a, wb);
    let b_out = vsubq_f64(a, wb);
    vst1q_f64(a_ptr, a_out);
    vst1q_f64(b_ptr, b_out);
}
```

Note: the complex multiply needs care with signs. The `vfmsq_f64` (fused
multiply-subtract) and `vfmaq_f64` (fused multiply-add) intrinsics are key.

## Implementation

### File: `src/neon_fft.rs`

```rust
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON-accelerated radix-2 DIT FFT (in-place, complex input).
    /// `data` is interleaved [re0, im0, re1, im1, ...] of length 2*n.
    /// `n` must be a power of 2.
    pub fn fft_radix2(data: &mut [f64], n: usize, inverse: bool) {
        // Bit-reversal permutation
        bit_reverse_permute(data, n);

        // Butterfly passes
        let mut half = 1;
        while half < n {
            let step = half * 2;
            let angle = if inverse { std::f64::consts::PI / half as f64 }
                       else { -std::f64::consts::PI / half as f64 };

            for k in 0..half {
                let theta = angle * k as f64;
                let w_re = theta.cos();
                let w_im = theta.sin();

                let mut j = k;
                while j < n {
                    let i1 = j * 2;
                    let i2 = (j + half) * 2;
                    unsafe {
                        neon_butterfly(&mut data[i1..], &mut data[i2..], w_re, w_im);
                    }
                    j += step;
                }
            }
            half *= 2;
        }

        if inverse {
            let inv_n = 1.0 / n as f64;
            for v in data.iter_mut() { *v *= inv_n; }
        }
    }

    #[inline(always)]
    unsafe fn neon_butterfly(a: &mut [f64], b: &mut [f64], w_re: f64, w_im: f64) {
        // Load a and b as [re, im] pairs
        let va = vld1q_f64(a.as_ptr());
        let vb = vld1q_f64(b.as_ptr());

        // Complex multiply: wb = w * b
        let vw_re = vdupq_n_f64(w_re);
        let vw_im = vdupq_n_f64(w_im);
        let vb_swap = vextq_f64(vb, vb, 1); // [b_im, b_re]

        // wb_re = w_re*b_re - w_im*b_im
        // wb_im = w_re*b_im + w_im*b_re
        let prod1 = vmulq_f64(vw_re, vb);       // [w_re*b_re, w_re*b_im]
        let prod2 = vmulq_f64(vw_im, vb_swap);  // [w_im*b_im, w_im*b_re]
        // Negate first element of prod2: [-w_im*b_im, w_im*b_re]
        let neg_mask = vsetq_lane_f64(-1.0, vdupq_n_f64(1.0), 0);
        let prod2_signed = vmulq_f64(prod2, neg_mask);
        let wb = vaddq_f64(prod1, prod2_signed);

        // Butterfly
        let a_out = vaddq_f64(va, wb);
        let b_out = vsubq_f64(va, wb);

        vst1q_f64(a.as_mut_ptr(), a_out);
        vst1q_f64(b.as_mut_ptr(), b_out);
    }
}
```

### Real FFT wrapper

The custom NEON FFT operates on complex data. To get a real FFT:
1. Pack N real values into N/2 complex values: `z[k] = x[2k] + i·x[2k+1]`
2. Run complex FFT of length N/2
3. Post-process to extract the N/2+1 complex spectral values (Hermitian unpack)

This is the standard real FFT trick and gives the same 2× savings as realfft.

## Tests

```rust
#[test]
fn test_neon_fft_matches_realfft() {
    // Compare NEON FFT output against realfft for random input
    let n = 1024;
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

    let neon_result = neon_fft_forward(&input);
    let realfft_result = realfft_forward(&input);

    for i in 0..n/2+1 {
        assert!((neon_result[i].re - realfft_result[i].re).abs() < 1e-10);
        assert!((neon_result[i].im - realfft_result[i].im).abs() < 1e-10);
    }
}
```

## Done when

NEON radix-2 FFT produces identical results to realfft (within f64 precision)
for all power-of-2 sizes used by DwtSquarer.