#include "arm64_asm_data.h"
#include "gwtables.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__aarch64__) || defined(ARM64)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct arm64_complex {
	double r;
	double i;
} arm64_complex;

static inline arm64_complex arm64_c_load(const double *data, size_t idx) {
	arm64_complex z;
	z.r = data[idx * 2u];
	z.i = data[idx * 2u + 1u];
	return z;
}

static inline void arm64_c_store(double *data, size_t idx, arm64_complex z) {
	data[idx * 2u] = z.r;
	data[idx * 2u + 1u] = z.i;
}

static inline arm64_complex arm64_c_add(arm64_complex a, arm64_complex b) {
	arm64_complex z;
	z.r = a.r + b.r;
	z.i = a.i + b.i;
	return z;
}

static inline arm64_complex arm64_c_sub(arm64_complex a, arm64_complex b) {
	arm64_complex z;
	z.r = a.r - b.r;
	z.i = a.i - b.i;
	return z;
}

static inline arm64_complex arm64_c_mul(arm64_complex a, arm64_complex b) {
	arm64_complex z;
	z.r = a.r * b.r - a.i * b.i;
	z.i = a.r * b.i + a.i * b.r;
	return z;
}

static int arm64_is_power_of_two(size_t n) {
	return (n != 0u) && ((n & (n - 1u)) == 0u);
}

static int arm64_log2_u32(uint32_t v) {
	int n = 0;
	while (v > 1u) { v >>= 1u; ++n; }
	return n;
}

static uint32_t arm64_reverse_bits(uint32_t x, unsigned bits) {
	uint32_t y = 0;
	unsigned i;
	for (i = 0; i < bits; ++i) {
		y = (y << 1u) | (x & 1u);
		x >>= 1u;
	}
	return y;
}

static void arm64_bit_reverse_permute(double *data, size_t n) {
	unsigned bits = (unsigned)arm64_log2_u32((uint32_t)n);
	size_t i;
	for (i = 0; i < n; ++i) {
		size_t j = (size_t)arm64_reverse_bits((uint32_t)i, bits);
		if (j > i) {
			arm64_complex a = arm64_c_load(data, i);
			arm64_complex b = arm64_c_load(data, j);
			arm64_c_store(data, i, b);
			arm64_c_store(data, j, a);
		}
	}
}

static void arm64_scale_inverse(double *data, size_t complex_len) {
	size_t words = complex_len * 2u;
	double inv_n = 1.0 / (double)complex_len;
	size_t i;
#if defined(__aarch64__) || defined(ARM64)
	{
		float64x2_t inv = vdupq_n_f64(inv_n);
		for (i = 0; i + 1u < words; i += 2u) {
			float64x2_t v = vld1q_f64(&data[i]);
			v = vmulq_f64(v, inv);
			vst1q_f64(&data[i], v);
		}
		if (i < words) data[i] *= inv_n;
	}
#else
	for (i = 0; i < words; ++i) data[i] *= inv_n;
#endif
}

/* Radix-2 Cooley-Tukey DIT FFT (forward) or inverse. */
static void arm64_fft(double *data, size_t n, int inverse) {
	size_t m;

	if (data == NULL || n < 2u || !arm64_is_power_of_two(n)) return;

	arm64_bit_reverse_permute(data, n);

	for (m = 2u; m <= n; m *= 2u) {
		size_t half = m / 2u;
		double angle_base = (inverse ? 2.0 : -2.0) * M_PI / (double)m;
		size_t k;

		for (k = 0; k < n; k += m) {
			size_t j;
			for (j = 0; j < half; ++j) {
				double angle = angle_base * (double)j;
				arm64_complex w;
				size_t i0 = k + j;
				size_t i1 = i0 + half;
				arm64_complex a = arm64_c_load(data, i0);
				arm64_complex b = arm64_c_load(data, i1);
				arm64_complex bw;

				w.r = cos(angle);
				w.i = sin(angle);
				bw = arm64_c_mul(b, w);

				arm64_c_store(data, i0, arm64_c_add(a, bw));
				arm64_c_store(data, i1, arm64_c_sub(a, bw));
			}
		}
	}

	if (inverse) arm64_scale_inverse(data, n);
}

static inline size_t arm64_word_offset_bytes(const struct gwasm_data *ad, size_t word) {
	if (ad != NULL && ad->gwdata != NULL)
		return (size_t)addr_offset(ad->gwdata, (unsigned long)word);
	return word * sizeof(double);
}

static inline double arm64_load_scrambled_word(const struct gwasm_data *ad, const double *base, size_t word) {
	return *(const double *)((const char *)base + arm64_word_offset_bytes(ad, word));
}

static inline void arm64_store_scrambled_word(const struct gwasm_data *ad, double *base, size_t word, double value) {
	*(double *)((char *)base + arm64_word_offset_bytes(ad, word)) = value;
}

/*
 * Pack gwnum's FFTLEN scrambled doubles into FFTLEN/2 complex numbers.
 *
 * In gwnum's SSE2 HG one-pass layout, FFTLEN doubles represent FFTLEN/2
 * complex numbers: word k is the real part and word k+FFTLEN/2 is the
 * imaginary part of complex element k.
 *
 * Output: dst_complex has FFTLEN/2 complex numbers = FFTLEN doubles
 *         stored as interleaved (re0, im0, re1, im1, ...)
 */
static int arm64_pack_scrambled_to_complex(const struct gwasm_data *ad, const double *src, double *dst_complex) {
	size_t words, half, k;
	if (ad == NULL || src == NULL || dst_complex == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;
	half = words / 2u;

	for (k = 0; k < half; ++k) {
		double re = arm64_load_scrambled_word(ad, src, k);
		double im = arm64_load_scrambled_word(ad, src, k + half);
		dst_complex[2u * k] = re;
		dst_complex[2u * k + 1u] = im;
	}
	return 1;
}

/*
 * Unpack FFTLEN/2 complex numbers back into gwnum's FFTLEN scrambled doubles.
 *
 * Splits each complex number: real part → word k, imaginary part → word k+FFTLEN/2.
 */
static int arm64_unpack_complex_to_scrambled(const struct gwasm_data *ad, const double *src_complex, double *dst) {
	size_t words, half, k;
	if (ad == NULL || src_complex == NULL || dst == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;
	half = words / 2u;

	for (k = 0; k < half; ++k) {
		arm64_store_scrambled_word(ad, dst, k, src_complex[2u * k]);
		arm64_store_scrambled_word(ad, dst, k + half, src_complex[2u * k + 1u]);
	}
	return 1;
}

static void arm64_pointwise_mul(double *dst, const double *a, const double *b, size_t complex_len) {
	size_t i = 0;
#if defined(__aarch64__) || defined(ARM64)
	for (; i + 1u < complex_len; i += 2u) {
		float64x2x2_t va = vld2q_f64(&a[i * 2u]);
		float64x2x2_t vb = vld2q_f64(&b[i * 2u]);
		float64x2_t re = vmulq_f64(va.val[0], vb.val[0]);
		float64x2_t im = vmulq_f64(va.val[0], vb.val[1]);
		float64x2x2_t out;
		re = vfmsq_f64(re, va.val[1], vb.val[1]);
		im = vfmaq_f64(im, va.val[1], vb.val[0]);
		out.val[0] = re;
		out.val[1] = im;
		vst2q_f64(&dst[i * 2u], out);
	}
#endif
	for (; i < complex_len; ++i) {
		arm64_complex x = arm64_c_load(a, i);
		arm64_complex y = arm64_c_load(b, i);
		arm64_c_store(dst, i, arm64_c_mul(x, y));
	}
}

static void arm64_pointwise_square(double *dst, const double *src, size_t complex_len) {
	size_t i = 0;
#if defined(__aarch64__) || defined(ARM64)
	for (; i + 1u < complex_len; i += 2u) {
		float64x2x2_t v = vld2q_f64(&src[i * 2u]);
		float64x2_t rr = vmulq_f64(v.val[0], v.val[0]);
		float64x2_t ii = vmulq_f64(v.val[1], v.val[1]);
		float64x2_t ri = vmulq_f64(v.val[0], v.val[1]);
		float64x2x2_t out;
		out.val[0] = vsubq_f64(rr, ii);
		out.val[1] = vaddq_f64(ri, ri);
		vst2q_f64(&dst[i * 2u], out);
	}
#endif
	for (; i < complex_len; ++i) {
		arm64_complex x = arm64_c_load(src, i);
		arm64_complex y;
		y.r = x.r * x.r - x.i * x.i;
		y.i = 2.0 * x.r * x.i;
		arm64_c_store(dst, i, y);
	}
}

static void arm64_normalize(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	if (ad != NULL && ad->NORMRTN != NULL)
		((void (*)(struct gwasm_data *))ad->NORMRTN)(asm_data);
	else
		arm64_norm_plain(asm_data);
}

void arm64_fft_entry(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	double *s1;
	double *s2;
	size_t words;
	size_t complex_len;
	unsigned int ffttype;
	double *tmp1 = NULL;
	double *tmp2 = NULL;
	int ok = 1;

	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;

	/* Disable gwmul3_carefully at runtime as a belt-and-suspenders measure. */
	if (ad->gwdata != NULL)
		ad->gwdata->careful_count = 0;

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return;

	/* gwnum's HG one-pass SSE2 layout: FFTLEN doubles = FFTLEN/2 complex pairs.
	   Word k is the real part, word k+FFTLEN/2 is the imaginary part. */
	complex_len = words / 2u;
	if (!arm64_is_power_of_two(complex_len)) return;

	s1 = arm64_fftsrc_ptr(ad);
	s2 = arm64_mulsrc_ptr(ad);
	if (s1 == NULL) s1 = dest;
	if (s2 == NULL) s2 = dest;

	ffttype = (unsigned int)(unsigned char)ad->ffttype;

	/* ffttype=1 (forward FFT only): no-op. All multiply/square operations
	   do the full pipeline internally using temp buffers. */
	if (ffttype == 1u) return;

	/* Allocate temp buffer: FFTLEN/2 complex numbers = FFTLEN doubles */
	tmp1 = (double *)malloc(words * sizeof(double));
	if (tmp1 == NULL) return;

	if (ffttype == 3u || ffttype == 4u) {
		tmp2 = (double *)malloc(words * sizeof(double));
		if (tmp2 == NULL) {
			free(tmp1);
			return;
		}
	}

	switch (ffttype) {
	case 2:	/* forward + square + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_pointwise_square(tmp1, tmp1, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 3:	/* forward s1 + mul by s2 + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, s2, tmp2);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_fft(tmp2, complex_len, 0);
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 4:	/* mul two operands + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, s2, tmp2);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_fft(tmp2, complex_len, 0);
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 5:	/* normalize only */
		arm64_normalize(asm_data);
		break;

	default:
		break;
	}

	free(tmp2);
	free(tmp1);
}