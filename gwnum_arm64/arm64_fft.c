#include "arm64_asm_data.h"
#include "gwtables.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

static inline arm64_complex arm64_c_mul_i(arm64_complex a) {
	arm64_complex z;
	z.r = -a.i;
	z.i = a.r;
	return z;
}

static inline arm64_complex arm64_c_mul_minus_i(arm64_complex a) {
	arm64_complex z;
	z.r = a.i;
	z.i = -a.r;
	return z;
}

static int arm64_log2_u32(uint32_t v) {
	int n = 0;
	while (v > 1u) {
		v >>= 1u;
		++n;
	}
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

static arm64_complex arm64_twiddle(
	const struct gwasm_data *ad,
	unsigned tw_mul,
	size_t j,
	size_t m,
	int inverse_sign)
{
	const double *table = NULL;

	if (ad != NULL) {
		if (tw_mul == 1u) table = (const double *)ad->sincos1;
		else if (tw_mul == 2u) table = (const double *)ad->sincos2;
		else table = (const double *)ad->sincos3;
	}

	if (table != NULL && ad != NULL) {
		size_t n = arm64_complex_len(ad);
		size_t period = n != 0 ? n : m;
		size_t idx = period != 0 ? ((j * (size_t)tw_mul) % period) : 0;
		double wr = table[idx * 2u];
		double wi = table[idx * 2u + 1u];
		if (inverse_sign) wi = -wi;
		if ((wr != 0.0 || wi != 0.0) || idx == 0) {
			arm64_complex w = { wr, wi };
			return w;
		}
	}

	{
		double angle = (inverse_sign ? 2.0 : -2.0) * M_PI * (double)(tw_mul * j) / (double)m;
		arm64_complex w = { cos(angle), sin(angle) };
		return w;
	}
}

static void arm64_apply_ibdwt_preweights(const struct gwasm_data *ad, double *data, size_t n) {
	if (ad == NULL) return;
	if (ad->norm_col_mults == NULL && ad->norm_grp_mults == NULL) return;

#if defined(__aarch64__) || defined(ARM64)
	{
		size_t i = 0;
		for (; i + 1u < n; i += 2u) {
			double wbuf[2];
			float64x2_t w;
			float64x2x2_t z;

			wbuf[0] = arm64_forward_weight_at(ad, i);
			wbuf[1] = arm64_forward_weight_at(ad, i + 1u);

			w = vld1q_f64(wbuf);
			z = vld2q_f64(&data[i * 2u]);
			z.val[0] = vmulq_f64(z.val[0], w);
			z.val[1] = vmulq_f64(z.val[1], w);
			vst2q_f64(&data[i * 2u], z);
		}
		if (i < n) {
			double w = arm64_forward_weight_at(ad, i);
			data[i * 2u] *= w;
			data[i * 2u + 1u] *= w;
		}
	}
#else
	{
		size_t i;
		for (i = 0; i < n; ++i) {
			double w = arm64_forward_weight_at(ad, i);
			data[i * 2u] *= w;
			data[i * 2u + 1u] *= w;
		}
	}
#endif
}

static void arm64_forward_radix2_stage(const struct gwasm_data *ad, double *data, size_t n, size_t m) {
	size_t half = m / 2u;
	size_t k;

	for (k = 0; k < n; k += m) {
		size_t j;
		for (j = 0; j < half; ++j) {
			size_t i0 = k + j;
			size_t i1 = i0 + half;
			arm64_complex a = arm64_c_load(data, i0);
			arm64_complex b = arm64_c_load(data, i1);
			arm64_complex w = arm64_twiddle(ad, 1u, j, m, 0);
			arm64_complex bw = arm64_c_mul(b, w);

			arm64_c_store(data, i0, arm64_c_add(a, bw));
			arm64_c_store(data, i1, arm64_c_sub(a, bw));
		}
	}
}

static void arm64_forward_radix4_stage(const struct gwasm_data *ad, double *data, size_t n, size_t m) {
	size_t quarter = m / 4u;
	size_t k;

	for (k = 0; k < n; k += m) {
		size_t j;
		for (j = 0; j < quarter; ++j) {
			size_t i0 = k + j;
			size_t i1 = i0 + quarter;
			size_t i2 = i1 + quarter;
			size_t i3 = i2 + quarter;

			arm64_complex a0 = arm64_c_load(data, i0);
			arm64_complex a1 = arm64_c_load(data, i1);
			arm64_complex a2 = arm64_c_load(data, i2);
			arm64_complex a3 = arm64_c_load(data, i3);

			arm64_complex w1 = arm64_twiddle(ad, 1u, j, m, 0);
			arm64_complex w2 = arm64_twiddle(ad, 2u, j, m, 0);
			arm64_complex w3 = arm64_twiddle(ad, 3u, j, m, 0);

			arm64_complex t0, t1, t2, t3;
			arm64_complex b0, b1, b2, b3;

			a1 = arm64_c_mul(a1, w1);
			a2 = arm64_c_mul(a2, w2);
			a3 = arm64_c_mul(a3, w3);

			t0 = arm64_c_add(a0, a2);
			t1 = arm64_c_sub(a0, a2);
			t2 = arm64_c_add(a1, a3);
			t3 = arm64_c_mul_minus_i(arm64_c_sub(a1, a3));

			b0 = arm64_c_add(t0, t2);
			b2 = arm64_c_sub(t0, t2);
			b1 = arm64_c_add(t1, t3);
			b3 = arm64_c_sub(t1, t3);

			arm64_c_store(data, i0, b0);
			arm64_c_store(data, i1, b1);
			arm64_c_store(data, i2, b2);
			arm64_c_store(data, i3, b3);
		}
	}
}

static void arm64_inverse_radix2_final_stage(double *data, size_t n) {
	size_t i;
	for (i = 0; i < n; i += 2u) {
		arm64_complex a = arm64_c_load(data, i);
		arm64_complex b = arm64_c_load(data, i + 1u);
		arm64_c_store(data, i, arm64_c_add(a, b));
		arm64_c_store(data, i + 1u, arm64_c_sub(a, b));
	}
}

static void arm64_inverse_radix4_stage(const struct gwasm_data *ad, double *data, size_t n, size_t m) {
	size_t quarter = m / 4u;
	size_t k;
	(void)n;

	for (k = 0; k < n; k += m) {
		size_t j;
		for (j = 0; j < quarter; ++j) {
			size_t i0 = k + j;
			size_t i1 = i0 + quarter;
			size_t i2 = i1 + quarter;
			size_t i3 = i2 + quarter;

			arm64_complex a0 = arm64_c_load(data, i0);
			arm64_complex a1 = arm64_c_load(data, i1);
			arm64_complex a2 = arm64_c_load(data, i2);
			arm64_complex a3 = arm64_c_load(data, i3);

			arm64_complex t0 = arm64_c_add(a0, a2);
			arm64_complex t1 = arm64_c_sub(a0, a2);
			arm64_complex t2 = arm64_c_add(a1, a3);
			arm64_complex t3 = arm64_c_sub(a1, a3);
			arm64_complex t3i = arm64_c_mul_i(t3);

			arm64_complex y0 = arm64_c_add(t0, t2);
			arm64_complex y1 = arm64_c_add(t1, t3i);
			arm64_complex y2 = arm64_c_sub(t0, t2);
			arm64_complex y3 = arm64_c_sub(t1, t3i);

			arm64_complex w1 = arm64_twiddle(ad, 1u, j, m, 1);
			arm64_complex w2 = arm64_twiddle(ad, 2u, j, m, 1);
			arm64_complex w3 = arm64_twiddle(ad, 3u, j, m, 1);

			y1 = arm64_c_mul(y1, w1);
			y2 = arm64_c_mul(y2, w2);
			y3 = arm64_c_mul(y3, w3);

			arm64_c_store(data, i0, y0);
			arm64_c_store(data, i1, y1);
			arm64_c_store(data, i2, y2);
			arm64_c_store(data, i3, y3);
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

static void arm64_forward_fft(const struct gwasm_data *ad, double *data) {
	size_t n;
	int log2_n;
	size_t m;

	if (ad == NULL) return;
	n = arm64_complex_len(ad);
	if (n < 2u) return;
	if ((n & (n - 1u)) != 0u) return;	/* power-of-two only */

	arm64_apply_ibdwt_preweights(ad, data, n);
	arm64_bit_reverse_permute(data, n);

	log2_n = arm64_log2_u32((uint32_t)n);

	if ((log2_n & 1) != 0) {
		arm64_forward_radix2_stage(ad, data, n, 2u);
		m = 8u;
	} else {
		m = 4u;
	}

	for (; m <= n; m *= 4u) {
		arm64_forward_radix4_stage(ad, data, n, m);
	}
}

static void arm64_inverse_fft(const struct gwasm_data *ad, double *data) {
	size_t n;
	int log2_n;
	size_t m;

	if (ad == NULL) return;
	n = arm64_complex_len(ad);
	if (n < 2u) return;
	if ((n & (n - 1u)) != 0u) return;

	log2_n = arm64_log2_u32((uint32_t)n);

	for (m = n; m >= 4u; m /= 4u) {
		arm64_inverse_radix4_stage(ad, data, n, m);
		if (m == 4u) break;
	}

	if ((log2_n & 1) != 0) {
		arm64_inverse_radix2_final_stage(data, n);
	}

	arm64_bit_reverse_permute(data, n);
	arm64_scale_inverse(data, n);
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
		float64x2_t re = vsubq_f64(rr, ii);
		float64x2_t im = vaddq_f64(ri, ri);
		float64x2x2_t out;

		out.val[0] = re;
		out.val[1] = im;
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
	if (ad != NULL && ad->NORMRTN != NULL) {
		((void (*)(struct gwasm_data *))ad->NORMRTN)(asm_data);
	} else {
		arm64_norm_plain(asm_data);
	}
}

void arm64_fft_entry(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	double *s1;
	double *s2;
	size_t n;
	size_t words;

	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;

	s1 = arm64_fftsrc_ptr(ad);
	s2 = arm64_mulsrc_ptr(ad);
	if (s1 == NULL) s1 = dest;
	if (s2 == NULL) s2 = dest;

	n = arm64_complex_len(ad);
	if (n == 0) return;
	words = arm64_data_words(ad);
	if (words == 0) words = n * 2u;

	switch ((unsigned int)(unsigned char)ad->ffttype) {
	case 1:	/* forward FFT only */
		if (dest != s1) memmove(dest, s1, words * sizeof(double));
		arm64_forward_fft(ad, dest);
		break;

	case 2:	/* forward + square + inverse + normalize */
		if (dest != s1) memmove(dest, s1, words * sizeof(double));
		arm64_forward_fft(ad, dest);
		arm64_pointwise_square(dest, dest, n);
		arm64_inverse_fft(ad, dest);
		arm64_normalize(asm_data);
		break;

	case 3:	/* forward s1 + mul by s2 + inverse + normalize */
		if (dest != s1) memmove(dest, s1, words * sizeof(double));
		arm64_forward_fft(ad, dest);
		arm64_pointwise_mul(dest, dest, s2, n);
		arm64_inverse_fft(ad, dest);
		arm64_normalize(asm_data);
		break;

	case 4:	/* pointwise mul + inverse + normalize (already FFTed) */
		if (dest != s1) memmove(dest, s1, words * sizeof(double));
		arm64_pointwise_mul(dest, dest, s2, n);
		arm64_inverse_fft(ad, dest);
		arm64_normalize(asm_data);
		break;

	case 5:	/* inverse + normalize only */
		if (dest != s1) memmove(dest, s1, words * sizeof(double));
		arm64_inverse_fft(ad, dest);
		arm64_normalize(asm_data);
		break;

	default:
		if (dest != s1) memmove(dest, s1, words * sizeof(double));
		arm64_forward_fft(ad, dest);
		break;
	}
}