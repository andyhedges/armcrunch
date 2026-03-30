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

static void arm64_fft(double *data, size_t n, int inverse) {
	size_t m;

	if (data == NULL) return;
	if (n < 2u) return;
	if (!arm64_is_power_of_two(n)) return;

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

	if (inverse) {
		arm64_scale_inverse(data, n);
	}
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

static int arm64_unscramble_to_linear(
	const struct gwasm_data *ad,
	const double *src,
	double *dst_linear)
{
	size_t words;
	size_t j;

	if (ad == NULL || src == NULL || dst_linear == NULL) return 0;

	words = arm64_data_words(ad);
	if (words == 0u) return 0;

	for (j = 0; j < words; ++j) {
		dst_linear[j] = arm64_load_scrambled_word(ad, src, j);
	}

	return 1;
}

static int arm64_rescramble_from_linear(
	const struct gwasm_data *ad,
	const double *src_linear,
	double *dst)
{
	size_t words;
	size_t j;

	if (ad == NULL || src_linear == NULL || dst == NULL) return 0;

	words = arm64_data_words(ad);
	if (words == 0u) return 0;

	for (j = 0; j < words; ++j) {
		arm64_store_scrambled_word(ad, dst, j, src_linear[j]);
	}

	return 1;
}

static int arm64_pack_scrambled_to_complex(
	const struct gwasm_data *ad,
	const double *src,
	double *dst_complex)
{
	size_t words;
	size_t j;

	if (ad == NULL || src == NULL || dst_complex == NULL) return 0;

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;

	if (!arm64_unscramble_to_linear(ad, src, dst_complex)) return 0;

	for (j = words; j > 0u; --j) {
		double v = dst_complex[j - 1u];
		dst_complex[(j - 1u) * 2u] = v;
		dst_complex[(j - 1u) * 2u + 1u] = 0.0;
	}

	return 1;
}

static int arm64_unpack_complex_to_scrambled(
	const struct gwasm_data *ad,
	const double *src_complex,
	double *dst)
{
	size_t words;
	size_t j;
	double *linear;
	int ok;

	if (ad == NULL || src_complex == NULL || dst == NULL) return 0;

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;

	linear = (double *)malloc(words * sizeof(double));
	if (linear == NULL) return 0;

	for (j = 0; j < words; ++j) {
		linear[j] = src_complex[2u * j];
	}

	ok = arm64_rescramble_from_linear(ad, linear, dst);
	free(linear);
	return ok;
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
	static int fft_call_count = 0;
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

	complex_len = words;

	fft_call_count++;
	if (fft_call_count <= 5) {
		size_t k;
		fprintf(stderr, "[ARM64 FFT #%d] ffttype=%d FFTLEN=%u words=%zu\n",
			fft_call_count, (int)(unsigned char)ad->ffttype, ad->FFTLEN, words);
		fprintf(stderr, "[ARM64 FFT #%d] DESTARG=%p FFTSRC dist=%ld MULSRC dist=%ld\n",
			fft_call_count, (void*)dest, (long)ad->DIST_TO_FFTSRCARG, (long)ad->DIST_TO_MULSRCARG);
		fprintf(stderr, "[ARM64 FFT #%d] input scrambled[0..7]: ", fft_call_count);
		for (k = 0; k < 8 && k < words; k++)
			fprintf(stderr, "%.6g ", arm64_load_scrambled_word(ad, dest, k));
		fprintf(stderr, "\n");
	}

	if (!arm64_is_power_of_two(complex_len)) return;

	s1 = arm64_fftsrc_ptr(ad);
	s2 = arm64_mulsrc_ptr(ad);
	if (s1 == NULL) s1 = dest;
	if (s2 == NULL) s2 = dest;

	ffttype = (unsigned int)(unsigned char)ad->ffttype;

	tmp1 = (double *)malloc(2u * words * sizeof(double));
	if (tmp1 == NULL) return;

	if (ffttype == 3u || ffttype == 4u) {
		tmp2 = (double *)malloc(2u * words * sizeof(double));
		if (tmp2 == NULL) {
			free(tmp1);
			return;
		}
	}

	switch (ffttype) {
	case 1:	/* forward FFT only using full N-point complex FFT */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		(void)arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		break;

	case 2:	/* forward + square + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		if (fft_call_count <= 2) {
			fprintf(stderr, "[ARM64 FFT #%d] post-fwd Z[0..3]: (%.6g,%.6g) (%.6g,%.6g) (%.6g,%.6g) (%.6g,%.6g)\n",
				fft_call_count, tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5], tmp1[6], tmp1[7]);
		}
		arm64_pointwise_square(tmp1, tmp1, complex_len);
		if (fft_call_count <= 2) {
			fprintf(stderr, "[ARM64 FFT #%d] post-square Z2[0..3]: (%.6g,%.6g) (%.6g,%.6g) (%.6g,%.6g) (%.6g,%.6g)\n",
				fft_call_count, tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5], tmp1[6], tmp1[7]);
		}
		arm64_fft(tmp1, complex_len, 1);
		if (fft_call_count <= 2) {
			size_t k;
			fprintf(stderr, "[ARM64 FFT #%d] post-ifft linear[0..7]: ", fft_call_count);
			for (k = 0; k < 8u && k < words; ++k) {
				fprintf(stderr, "%.6g ", tmp1[2u * k]);
			}
			fprintf(stderr, "\n");
		}
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 3:	/* forward s1 + mul by already-FFTed s2 + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, s2, tmp2);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 4:	/* mul two already-FFTed operands + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, s2, tmp2);
		if (!ok) break;
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 5:	/* inverse + normalize only (input already FFTed) */
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	default:
		ok = arm64_pack_scrambled_to_complex(ad, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		(void)arm64_unpack_complex_to_scrambled(ad, tmp1, dest);
		break;
	}

	free(tmp2);
	free(tmp1);
}