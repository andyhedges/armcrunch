#include "arm64_asm_data.h"
#include "gwtables.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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

static int arm64_is_power_of_two(size_t n) {
	return (n != 0u) && ((n & (n - 1u)) == 0u);
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

static arm64_complex arm64_twiddle(unsigned tw_mul, size_t j, size_t m, int inverse_sign) {
	double angle;
	arm64_complex w;

	if (m == 0u) {
		w.r = 1.0;
		w.i = 0.0;
		return w;
	}

	angle = (inverse_sign ? 2.0 : -2.0) * M_PI * (double)tw_mul * (double)j / (double)m;
	w.r = cos(angle);
	w.i = sin(angle);
	return w;
}

static void arm64_forward_radix2_stage(double *data, size_t n, size_t m) {
	size_t half = m / 2u;
	size_t k;

	for (k = 0; k < n; k += m) {
		size_t j;
		for (j = 0; j < half; ++j) {
			size_t i0 = k + j;
			size_t i1 = i0 + half;
			arm64_complex a = arm64_c_load(data, i0);
			arm64_complex b = arm64_c_load(data, i1);
			arm64_complex w = arm64_twiddle(1u, j, m, 0);
			arm64_complex bw = arm64_c_mul(b, w);

			arm64_c_store(data, i0, arm64_c_add(a, bw));
			arm64_c_store(data, i1, arm64_c_sub(a, bw));
		}
	}
}

static void arm64_forward_radix4_stage(double *data, size_t n, size_t m) {
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

			arm64_complex w1 = arm64_twiddle(1u, j, m, 0);
			arm64_complex w2 = arm64_twiddle(2u, j, m, 0);
			arm64_complex w3 = arm64_twiddle(3u, j, m, 0);

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

static void arm64_inverse_radix4_stage(double *data, size_t n, size_t m) {
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

			arm64_complex w1 = arm64_twiddle(1u, j, m, 1);
			arm64_complex w2 = arm64_twiddle(2u, j, m, 1);
			arm64_complex w3 = arm64_twiddle(3u, j, m, 1);

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

static void arm64_forward_fft(double *data, size_t n) {
	int log2_n;
	size_t m;

	if (data == NULL) return;
	if (n < 2u) return;
	if (!arm64_is_power_of_two(n)) return;

	arm64_bit_reverse_permute(data, n);
	log2_n = arm64_log2_u32((uint32_t)n);

	if ((log2_n & 1) != 0) {
		arm64_forward_radix2_stage(data, n, 2u);
		m = 8u;
	} else {
		m = 4u;
	}

	for (; m <= n; m *= 4u) {
		arm64_forward_radix4_stage(data, n, m);
	}
}

static void arm64_inverse_fft(double *data, size_t n) {
	int log2_n;
	size_t m;

	if (data == NULL) return;
	if (n < 2u) return;
	if (!arm64_is_power_of_two(n)) return;

	log2_n = arm64_log2_u32((uint32_t)n);

	for (m = n; m >= 4u; m /= 4u) {
		arm64_inverse_radix4_stage(data, n, m);
		if (m == 4u) break;
	}

	if ((log2_n & 1) != 0) {
		arm64_inverse_radix2_final_stage(data, n);
	}

	arm64_bit_reverse_permute(data, n);
	arm64_scale_inverse(data, n);
}

static inline size_t arm64_word_offset_bytes(const struct gwasm_data *ad, size_t word) {
	if (ad != NULL && ad->gwdata != NULL) {
		return (size_t)addr_offset(ad->gwdata, (unsigned long)word);
	}
	return word * sizeof(double);
}

static inline double arm64_load_scrambled_word(const struct gwasm_data *ad, const double *base, size_t word) {
	const char *ptr = (const char *)base + arm64_word_offset_bytes(ad, word);
	return *(const double *)ptr;
}

static inline void arm64_store_scrambled_word(const struct gwasm_data *ad, double *base, size_t word, double value) {
	char *ptr = (char *)base + arm64_word_offset_bytes(ad, word);
	*(double *)ptr = value;
}

static int arm64_pack_scrambled_to_complex(
	const struct gwasm_data *ad,
	const double *src,
	double *dst_complex,
	int apply_weights)
{
	size_t words;
	size_t half;
	size_t k;

	if (ad == NULL || src == NULL || dst_complex == NULL) return 0;

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;
	half = words / 2u;

	for (k = 0; k < half; ++k) {
		size_t imag_word = k + half;
		double re = arm64_load_scrambled_word(ad, src, k);
		double im = arm64_load_scrambled_word(ad, src, imag_word);

		if (apply_weights) {
			re *= arm64_forward_weight_at(ad, k);
			im *= arm64_forward_weight_at(ad, imag_word);
		}

		dst_complex[2u * k] = re;
		dst_complex[2u * k + 1u] = im;
	}

	return 1;
}

static int arm64_unpack_complex_to_scrambled(
	const struct gwasm_data *ad,
	const double *src_complex,
	double *dst)
{
	size_t words;
	size_t half;
	size_t k;

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
	static int debug_count = 0;
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

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return;

	complex_len = words / 2u;

	if (debug_count < 3) {
		size_t i;
		fprintf(stderr, "[ARM64 FFT] call #%d ffttype=%d FFTLEN=%u words=%zu complex_len=%zu\n",
			debug_count, (int)(unsigned char)ad->ffttype, ad->FFTLEN, words, complex_len);
		if (ad->gwdata) {
			fprintf(stderr, "[ARM64 FFT] FOURKBGAPSIZE=%ld RATIONAL=%d B_IS_2=%d\n",
				(long)ad->gwdata->FOURKBGAPSIZE, (int)ad->RATIONAL_FFT, (int)ad->B_IS_2);
		}
		fprintf(stderr, "[ARM64 FFT] Raw doubles[0..7]: ");
		for (i = 0; i < 8 && i < words; i++)
			fprintf(stderr, "%.6g ", dest[i]);
		fprintf(stderr, "\n");
		fprintf(stderr, "[ARM64 FFT] Scrambled words[0..7]: ");
		for (i = 0; i < 8 && i < words; i++)
			fprintf(stderr, "%.6g ", arm64_load_scrambled_word(ad, dest, i));
		fprintf(stderr, "\n");
		if (ad->gwdata) {
			fprintf(stderr, "[ARM64 FFT] addr_offset[0..7]: ");
			for (i = 0; i < 8 && i < words; i++)
				fprintf(stderr, "%lu ", addr_offset(ad->gwdata, (unsigned long)i));
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "[ARM64 FFT] fwd_weight[0..3]: ");
		for (i = 0; i < 4; i++)
			fprintf(stderr, "%.10g ", arm64_forward_weight_at(ad, i));
		fprintf(stderr, "\n");
		fprintf(stderr, "[ARM64 FFT] inv_weight[0..3]: ");
		for (i = 0; i < 4; i++)
			fprintf(stderr, "%.10g ", arm64_inverse_weight_at(ad, i));
		fprintf(stderr, "\n");
		debug_count++;
	}
	if (!arm64_is_power_of_two(complex_len)) return;

	s1 = arm64_fftsrc_ptr(ad);
	s2 = arm64_mulsrc_ptr(ad);
	if (s1 == NULL) s1 = dest;
	if (s2 == NULL) s2 = dest;

	ffttype = (unsigned int)(unsigned char)ad->ffttype;

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
	case 1:	/* forward FFT only (s1 is time-domain, apply forward weights) */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1, 1);
		if (!ok) break;
		arm64_forward_fft(tmp1, complex_len);
		(void)arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		break;

	case 2:	/* forward + square + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1, 1);
		if (!ok) break;
		arm64_forward_fft(tmp1, complex_len);
		arm64_pointwise_square(tmp1, tmp1, complex_len);
		arm64_inverse_fft(tmp1, complex_len);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 3:	/* forward s1 + mul by already-FFTed s2 + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1, 1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, s2, tmp2, 0);
		if (!ok) break;
		arm64_forward_fft(tmp1, complex_len);
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_inverse_fft(tmp1, complex_len);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 4:	/* mul two already-FFTed operands + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1, 0);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, s2, tmp2, 0);
		if (!ok) break;
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_inverse_fft(tmp1, complex_len);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 5:	/* inverse + normalize only (input already FFTed) */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1, 0);
		if (!ok) break;
		arm64_inverse_fft(tmp1, complex_len);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	default:
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1, 1);
		if (!ok) break;
		arm64_forward_fft(tmp1, complex_len);
		(void)arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		break;
	}

	free(tmp2);
	free(tmp1);
}