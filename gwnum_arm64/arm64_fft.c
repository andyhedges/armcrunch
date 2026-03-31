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

enum { ARM64_TWIDDLE_CACHE_SLOTS = 32 };
static double *arm64_twiddle_cache[ARM64_TWIDDLE_CACHE_SLOTS];

#if defined(__GNUC__) || defined(__clang__)
#define ARM64_THREAD_LOCAL __thread
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define ARM64_THREAD_LOCAL _Thread_local
#else
#define ARM64_THREAD_LOCAL
#endif

static ARM64_THREAD_LOCAL double *arm64_fft_tmp1_cache;
static ARM64_THREAD_LOCAL size_t arm64_fft_tmp1_capacity;
static ARM64_THREAD_LOCAL double *arm64_fft_tmp2_cache;
static ARM64_THREAD_LOCAL size_t arm64_fft_tmp2_capacity;

static int arm64_ensure_tmp_capacity(double **buffer, size_t *capacity, size_t words) {
	double *new_buffer;

	if (buffer == NULL || capacity == NULL || words == 0u) return 0;
	if (*buffer != NULL && *capacity >= words) return 1;

	new_buffer = (double *)realloc(*buffer, words * sizeof(double));
	if (new_buffer == NULL) return 0;

	*buffer = new_buffer;
	*capacity = words;
	return 1;
}

static int arm64_get_fft_tmp_buffers(size_t words, int need_tmp2, double **tmp1, double **tmp2) {
	if (tmp1 == NULL || tmp2 == NULL || words == 0u) return 0;

	if (!arm64_ensure_tmp_capacity(&arm64_fft_tmp1_cache, &arm64_fft_tmp1_capacity, words))
		return 0;

	if (need_tmp2) {
		if (!arm64_ensure_tmp_capacity(&arm64_fft_tmp2_cache, &arm64_fft_tmp2_capacity, words))
			return 0;
		*tmp2 = arm64_fft_tmp2_cache;
	} else {
		*tmp2 = NULL;
	}

	*tmp1 = arm64_fft_tmp1_cache;
	return 1;
}

static int arm64_log2_u32(uint32_t v) {
	int n = 0;
	while (v > 1u) { v >>= 1u; ++n; }
	return n;
}

static const double *arm64_get_twiddle_table(size_t n) {
	size_t j;
	size_t half;
	int log2_n;
	double *table;
	double angle_base;

	if (n < 2u || !arm64_is_power_of_two(n) || n > (size_t)UINT32_MAX) return NULL;

	log2_n = arm64_log2_u32((uint32_t)n);
	if (log2_n < 0 || log2_n >= ARM64_TWIDDLE_CACHE_SLOTS) return NULL;

	table = arm64_twiddle_cache[log2_n];
	if (table != NULL) return table;

	table = (double *)malloc(n * sizeof(double));
	if (table == NULL) return NULL;

	half = n / 2u;
	angle_base = -2.0 * M_PI / (double)n;
	for (j = 0; j < half; ++j) {
		double angle = angle_base * (double)j;
		table[2u * j] = cos(angle);
		table[2u * j + 1u] = sin(angle);
	}

	arm64_twiddle_cache[log2_n] = table;
	return table;
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

static void arm64_fft_stage(double *data, size_t n, size_t m, const double *twiddles, int inverse) {
	size_t half = m / 2u;
	size_t tw_step = n / m;
	double imag_sign = inverse ? -1.0 : 1.0;
	size_t k;

	for (k = 0; k < n; k += m) {
		size_t j = 0u;
		size_t tw_idx = 0u;
#if defined(__aarch64__) || defined(ARM64)
		for (; j + 1u < half; j += 2u, tw_idx += 2u * tw_step) {
			size_t t0 = tw_idx;
			size_t t1 = tw_idx + tw_step;
			double w_re_pair[2];
			double w_im_pair[2];
			float64x2_t w_re;
			float64x2_t w_im;
			float64x2x2_t va;
			float64x2x2_t vb;
			float64x2_t bw_re;
			float64x2_t bw_im;
			float64x2x2_t out0;
			float64x2x2_t out1;

			w_re_pair[0] = twiddles[2u * t0];
			w_re_pair[1] = twiddles[2u * t1];
			w_im_pair[0] = imag_sign * twiddles[2u * t0 + 1u];
			w_im_pair[1] = imag_sign * twiddles[2u * t1 + 1u];

			w_re = vld1q_f64(w_re_pair);
			w_im = vld1q_f64(w_im_pair);

			va = vld2q_f64(&data[(k + j) * 2u]);
			vb = vld2q_f64(&data[(k + j + half) * 2u]);

			bw_re = vmulq_f64(vb.val[0], w_re);
			bw_im = vmulq_f64(vb.val[0], w_im);
			bw_re = vfmsq_f64(bw_re, vb.val[1], w_im);
			bw_im = vfmaq_f64(bw_im, vb.val[1], w_re);

			out0.val[0] = vaddq_f64(va.val[0], bw_re);
			out0.val[1] = vaddq_f64(va.val[1], bw_im);
			out1.val[0] = vsubq_f64(va.val[0], bw_re);
			out1.val[1] = vsubq_f64(va.val[1], bw_im);

			vst2q_f64(&data[(k + j) * 2u], out0);
			vst2q_f64(&data[(k + j + half) * 2u], out1);
		}
#endif
		for (; j < half; ++j, tw_idx += tw_step) {
			size_t i0 = k + j;
			size_t i1 = i0 + half;
			arm64_complex a = arm64_c_load(data, i0);
			arm64_complex b = arm64_c_load(data, i1);
			arm64_complex w;
			arm64_complex bw;

			w.r = twiddles[2u * tw_idx];
			w.i = imag_sign * twiddles[2u * tw_idx + 1u];
			bw = arm64_c_mul(b, w);

			arm64_c_store(data, i0, arm64_c_add(a, bw));
			arm64_c_store(data, i1, arm64_c_sub(a, bw));
		}
	}
}

/* Radix-2 Cooley-Tukey DIT FFT (forward) or inverse. */
static void arm64_fft(double *data, size_t n, int inverse) {
	size_t m;
	const double *twiddles;

	if (data == NULL || n < 2u || !arm64_is_power_of_two(n)) return;

	twiddles = arm64_get_twiddle_table(n);
	if (twiddles == NULL) return;

	arm64_bit_reverse_permute(data, n);

	for (m = 2u; m <= n; m *= 2u)
		arm64_fft_stage(data, n, m, twiddles, inverse);

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

/* Pack FFTLEN real words from gwnum layout into FFTLEN/2 complex values:
   word k is the real part, word k+FFTLEN/2 is the imaginary part. */
static int arm64_pack_scrambled_to_complex(const struct gwasm_data *ad, const double *src, double *dst_complex) {
	size_t words, half, k;
	if (ad == NULL || src == NULL || dst_complex == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;

	half = words / 2u;
	for (k = 0; k < half; ++k) {
		dst_complex[2u * k] = arm64_load_scrambled_word(ad, src, k);
		dst_complex[2u * k + 1u] = arm64_load_scrambled_word(ad, src, k + half);
	}
	return 1;
}

/* Unpack FFTLEN/2 complex values into FFTLEN real words in gwnum layout:
   real part -> word k, imaginary part -> word k+FFTLEN/2. */
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
	int need_tmp2;
	int ok = 1;

	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;

	/* Disable gwmul3_carefully at runtime as a belt-and-suspenders measure.
	   The compile-time guard in the patched gwnum.c should prevent it, but
	   this catches any case where the patch didn't apply. */
	if (ad->gwdata != NULL)
		ad->gwdata->careful_count = 0;

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return;

	complex_len = words / 2u;
	if (!arm64_is_power_of_two(complex_len)) return;

	s1 = arm64_fftsrc_ptr(ad);
	s2 = arm64_mulsrc_ptr(ad);
	if (s1 == NULL) s1 = dest;
	if (s2 == NULL) s2 = dest;

	ffttype = (unsigned int)(unsigned char)ad->ffttype;

	/* ffttype=1 (forward FFT only): no-op. Full pipelines are handled
	   internally by ffttype=2/3/4 using scratch buffers. */
	if (ffttype == 1u) return;

	need_tmp2 = (ffttype == 3u || ffttype == 4u) ? 1 : 0;
	if (!arm64_get_fft_tmp_buffers(words, need_tmp2, &tmp1, &tmp2)) return;

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

	case 3:	/* forward s1 + mul by s2 + inverse + normalize
		   (s2 may be marked as "FFTed" by gwnum but we treat it as raw data
		    since our ffttype=1 is a no-op) */
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

	case 4:	/* mul two operands + inverse + normalize
		   (both may be "FFTed" but we FFT from scratch) */
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
}