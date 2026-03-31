#include "arm64_asm_data.h"
#include "gwdbldbl.h"
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

#if defined(_MSC_VER)
#define ARM64_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ARM64_THREAD_LOCAL _Thread_local
#else
#define ARM64_THREAD_LOCAL __thread
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ARM64_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#else
#define ARM64_PREFETCH_READ(addr) ((void)0)
#endif

/* Prefetch 4 cache lines (256 bytes) ahead = 16 complex values. */
enum {
	ARM64_CACHE_SLOTS = 32,
	ARM64_PREFETCH_COMPLEX_AHEAD = 16
};

typedef struct arm64_complex {
	double r;
	double i;
} arm64_complex;

typedef struct arm64_word_cache {
	size_t fftlen;              /* key: FFTLEN this cache was built for */
	size_t *byte_offsets;       /* addr_offset(gwdata, word) */
	double *fwd_weights;        /* gwfft_weight_sloppy(dd_data, word) */
	double *inv_weights;        /* gwfft_weight_inverse_sloppy(dd_data, word) */
	int *big_words;             /* is_big_word(gwdata, word) */
} arm64_word_cache;

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

static double *arm64_twiddle_cache[ARM64_CACHE_SLOTS];
static uint32_t *arm64_bitrev_cache[ARM64_CACHE_SLOTS];
static arm64_word_cache arm64_global_word_cache = {0u, NULL, NULL, NULL, NULL};

static ARM64_THREAD_LOCAL double *arm64_tls_tmp1 = NULL;
static ARM64_THREAD_LOCAL double *arm64_tls_tmp2 = NULL;
static ARM64_THREAD_LOCAL size_t arm64_tls_tmp1_capacity = 0u;
static ARM64_THREAD_LOCAL size_t arm64_tls_tmp2_capacity = 0u;

static void arm64_free_word_cache_arrays(arm64_word_cache *cache) {
	if (cache == NULL) return;
	free(cache->byte_offsets);
	free(cache->fwd_weights);
	free(cache->inv_weights);
	free(cache->big_words);
	cache->byte_offsets = NULL;
	cache->fwd_weights = NULL;
	cache->inv_weights = NULL;
	cache->big_words = NULL;
	cache->fftlen = 0u;
}

const arm64_word_cache *arm64_get_word_cache(void) {
	return &arm64_global_word_cache;
}

int arm64_ensure_word_cache(const struct gwasm_data *ad) {
	size_t fftlen, word;
	size_t *byte_offsets;
	double *fwd_weights;
	double *inv_weights;
	int *big_words;
	int rational_fft;

	if (ad == NULL || ad->gwdata == NULL) return 0;

	fftlen = arm64_data_words(ad);
	if (fftlen == 0u) return 0;

	if (arm64_global_word_cache.fftlen == fftlen &&
		arm64_global_word_cache.byte_offsets != NULL &&
		arm64_global_word_cache.fwd_weights != NULL &&
		arm64_global_word_cache.inv_weights != NULL &&
		arm64_global_word_cache.big_words != NULL) {
		return 1;
	}

	byte_offsets = (size_t *)malloc(fftlen * sizeof(size_t));
	fwd_weights = (double *)malloc(fftlen * sizeof(double));
	inv_weights = (double *)malloc(fftlen * sizeof(double));
	big_words = (int *)malloc(fftlen * sizeof(int));
	if (byte_offsets == NULL || fwd_weights == NULL || inv_weights == NULL || big_words == NULL) {
		free(byte_offsets);
		free(fwd_weights);
		free(inv_weights);
		free(big_words);
		return 0;
	}

	rational_fft = (ad->RATIONAL_FFT != 0);
	for (word = 0; word < fftlen; ++word) {
		byte_offsets[word] = (size_t)addr_offset(ad->gwdata, (unsigned long)word);
		big_words[word] = arm64_is_big_word(ad, word);

		if (rational_fft) {
			fwd_weights[word] = 1.0;
			inv_weights[word] = 1.0;
		} else if (ad->gwdata->dd_data != NULL) {
			fwd_weights[word] = gwfft_weight_sloppy(ad->gwdata->dd_data, (unsigned long)word);
			inv_weights[word] = gwfft_weight_inverse_sloppy(ad->gwdata->dd_data, (unsigned long)word);
		} else {
			fwd_weights[word] = arm64_forward_weight_at(ad, word);
			inv_weights[word] = arm64_inverse_weight_at(ad, word);
		}
	}

	arm64_free_word_cache_arrays(&arm64_global_word_cache);
	arm64_global_word_cache.fftlen = fftlen;
	arm64_global_word_cache.byte_offsets = byte_offsets;
	arm64_global_word_cache.fwd_weights = fwd_weights;
	arm64_global_word_cache.inv_weights = inv_weights;
	arm64_global_word_cache.big_words = big_words;
	return 1;
}

static int arm64_reserve_tls_buffer(double **buffer, size_t *capacity, size_t required_doubles) {
	double *new_buffer;

	if (required_doubles == 0u) return 0;
	if (*buffer != NULL && *capacity >= required_doubles) return 1;

	new_buffer = (double *)realloc(*buffer, required_doubles * sizeof(double));
	if (new_buffer == NULL) return 0;

	*buffer = new_buffer;
	*capacity = required_doubles;
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
	if (log2_n < 0 || log2_n >= ARM64_CACHE_SLOTS) return NULL;

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

/* Cached bit-reverse permutation table, keyed by log2(n). */
static const uint32_t *arm64_get_bitrev_table(size_t n) {
	int log2_n;
	unsigned bits;
	size_t i;
	uint32_t *table;

	if (n < 2u || !arm64_is_power_of_two(n) || n > (size_t)UINT32_MAX) return NULL;

	log2_n = arm64_log2_u32((uint32_t)n);
	if (log2_n < 0 || log2_n >= ARM64_CACHE_SLOTS) return NULL;

	table = arm64_bitrev_cache[log2_n];
	if (table != NULL) return table;

	table = (uint32_t *)malloc(n * sizeof(uint32_t));
	if (table == NULL) return NULL;

	bits = (unsigned)log2_n;
	for (i = 0; i < n; ++i)
		table[i] = arm64_reverse_bits((uint32_t)i, bits);

	arm64_bitrev_cache[log2_n] = table;
	return table;
}

static void arm64_bit_reverse_permute(double *data, size_t n) {
	const uint32_t *bitrev;
	size_t i;

	if (data == NULL || n < 2u) return;

	bitrev = arm64_get_bitrev_table(n);
	if (bitrev != NULL) {
		for (i = 0; i < n; ++i) {
			size_t j = (size_t)bitrev[i];
			if (j > i) {
				arm64_complex a = arm64_c_load(data, i);
				arm64_complex b = arm64_c_load(data, j);
				arm64_c_store(data, i, b);
				arm64_c_store(data, j, a);
			}
		}
		return;
	}

	/* Fallback if cache allocation fails. */
	{
		unsigned bits = (unsigned)arm64_log2_u32((uint32_t)n);
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
		/* 4-wide NEON butterfly: process 4 complex elements per iteration */
		for (; j + 3u < half; j += 4u, tw_idx += 4u * tw_step) {
			size_t t0 = tw_idx;
			size_t t1 = t0 + tw_step;
			size_t t2 = t1 + tw_step;
			size_t t3 = t2 + tw_step;
			double *a0_ptr = &data[(k + j) * 2u];
			double *b0_ptr = &data[(k + j + half) * 2u];
			double *a1_ptr = a0_ptr + 4u;   /* +4 doubles = +2 complex */
			double *b1_ptr = b0_ptr + 4u;
			float64x2_t w_re0, w_im0, w_re1, w_im1;
			float64x2x2_t va0, vb0, va1, vb1;
			float64x2_t bw_re0, bw_im0, bw_re1, bw_im1;
			float64x2x2_t out_top, out_bot;
			double w_re_pair0[2], w_im_pair0[2];
			double w_re_pair1[2], w_im_pair1[2];

			/* Prefetch ahead */
			{
				size_t pf_j = j + (size_t)ARM64_PREFETCH_COMPLEX_AHEAD;
				if (pf_j < half) {
					ARM64_PREFETCH_READ(&data[(k + pf_j) * 2u]);
					ARM64_PREFETCH_READ(&data[(k + pf_j + half) * 2u]);
				}
			}

			w_re_pair0[0] = twiddles[2u * t0];
			w_re_pair0[1] = twiddles[2u * t1];
			w_im_pair0[0] = imag_sign * twiddles[2u * t0 + 1u];
			w_im_pair0[1] = imag_sign * twiddles[2u * t1 + 1u];
			w_re_pair1[0] = twiddles[2u * t2];
			w_re_pair1[1] = twiddles[2u * t3];
			w_im_pair1[0] = imag_sign * twiddles[2u * t2 + 1u];
			w_im_pair1[1] = imag_sign * twiddles[2u * t3 + 1u];

			w_re0 = vld1q_f64(w_re_pair0);
			w_im0 = vld1q_f64(w_im_pair0);
			w_re1 = vld1q_f64(w_re_pair1);
			w_im1 = vld1q_f64(w_im_pair1);

			va0 = vld2q_f64(a0_ptr);
			vb0 = vld2q_f64(b0_ptr);
			va1 = vld2q_f64(a1_ptr);
			vb1 = vld2q_f64(b1_ptr);

			/* Complex multiply b0 * w0 */
			bw_re0 = vmulq_f64(vb0.val[0], w_re0);
			bw_im0 = vmulq_f64(vb0.val[0], w_im0);
			/* Complex multiply b1 * w1 (interleaved for ILP) */
			bw_re1 = vmulq_f64(vb1.val[0], w_re1);
			bw_im1 = vmulq_f64(vb1.val[0], w_im1);

			bw_re0 = vfmsq_f64(bw_re0, vb0.val[1], w_im0);
			bw_im0 = vfmaq_f64(bw_im0, vb0.val[1], w_re0);
			bw_re1 = vfmsq_f64(bw_re1, vb1.val[1], w_im1);
			bw_im1 = vfmaq_f64(bw_im1, vb1.val[1], w_re1);

			/* Butterfly pair 0 */
			out_top.val[0] = vaddq_f64(va0.val[0], bw_re0);
			out_top.val[1] = vaddq_f64(va0.val[1], bw_im0);
			out_bot.val[0] = vsubq_f64(va0.val[0], bw_re0);
			out_bot.val[1] = vsubq_f64(va0.val[1], bw_im0);
			vst2q_f64(a0_ptr, out_top);
			vst2q_f64(b0_ptr, out_bot);

			/* Butterfly pair 1 */
			out_top.val[0] = vaddq_f64(va1.val[0], bw_re1);
			out_top.val[1] = vaddq_f64(va1.val[1], bw_im1);
			out_bot.val[0] = vsubq_f64(va1.val[0], bw_re1);
			out_bot.val[1] = vsubq_f64(va1.val[1], bw_im1);
			vst2q_f64(a1_ptr, out_top);
			vst2q_f64(b1_ptr, out_bot);
		}

		/* 2-wide NEON tail */
		for (; j + 1u < half; j += 2u, tw_idx += 2u * tw_step) {
			size_t t0 = tw_idx;
			size_t t1 = tw_idx + tw_step;
			double w_re_pair[2], w_im_pair[2];
			float64x2_t w_re, w_im;
			float64x2x2_t va, vb;
			float64x2_t bw_re, bw_im;
			float64x2x2_t out0, out1;

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
		/* Scalar tail for remaining elements */
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

/* Unscramble FFTLEN real words from gwnum layout into a contiguous complex array
   (each word becomes a complex number with zero imaginary part). */
static int arm64_pack_scrambled_to_complex(const struct gwasm_data *ad, const arm64_word_cache *cache, const double *src, double *dst_complex) {
	size_t words, j;
	const char *src_bytes;

	if (ad == NULL || cache == NULL || src == NULL || dst_complex == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;
	if (cache->fftlen != words || cache->byte_offsets == NULL) return 0;

	src_bytes = (const char *)src;

	/* First unscramble into temporary contiguous order at the start of dst_complex */
	for (j = 0; j < words; ++j)
		dst_complex[j] = *(const double *)(src_bytes + cache->byte_offsets[j]);

	/* Expand in-place from real to complex (backwards to avoid overwrite) */
	for (j = words; j > 0u; --j) {
		double v = dst_complex[j - 1u];
		dst_complex[(j - 1u) * 2u] = v;
		dst_complex[(j - 1u) * 2u + 1u] = 0.0;
	}
	return 1;
}

/* Extract real parts from complex array and rescramble into gwnum layout. */
static int arm64_unpack_complex_to_scrambled(const struct gwasm_data *ad, const arm64_word_cache *cache, const double *src_complex, double *dst) {
	size_t words, j;
	char *dst_bytes;

	if (ad == NULL || cache == NULL || src_complex == NULL || dst == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;
	if (cache->fftlen != words || cache->byte_offsets == NULL) return 0;

	dst_bytes = (char *)dst;
	for (j = 0; j < words; ++j)
		*(double *)(dst_bytes + cache->byte_offsets[j]) = src_complex[2u * j];

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
	const arm64_word_cache *word_cache;
	double *dest;
	double *s1;
	double *s2;
	size_t words;
	size_t complex_len;
	unsigned int ffttype;
	size_t required_doubles;
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

	if (!arm64_ensure_word_cache(ad)) return;
	word_cache = arm64_get_word_cache();
	if (word_cache == NULL ||
		word_cache->fftlen != words ||
		word_cache->byte_offsets == NULL ||
		word_cache->fwd_weights == NULL ||
		word_cache->inv_weights == NULL ||
		word_cache->big_words == NULL) {
		return;
	}

	complex_len = words;
	if (!arm64_is_power_of_two(complex_len)) return;

	s1 = arm64_fftsrc_ptr(ad);
	s2 = arm64_mulsrc_ptr(ad);
	if (s1 == NULL) s1 = dest;
	if (s2 == NULL) s2 = dest;

	ffttype = (unsigned int)(unsigned char)ad->ffttype;

	/* ffttype=1 (forward FFT only): no-op. */
	if (ffttype == 1u) return;

	if (ffttype == 2u || ffttype == 3u || ffttype == 4u) {
		required_doubles = 2u * words;
		if (!arm64_reserve_tls_buffer(&arm64_tls_tmp1, &arm64_tls_tmp1_capacity, required_doubles)) return;
		tmp1 = arm64_tls_tmp1;

		if (ffttype == 3u || ffttype == 4u) {
			if (!arm64_reserve_tls_buffer(&arm64_tls_tmp2, &arm64_tls_tmp2_capacity, required_doubles)) return;
			tmp2 = arm64_tls_tmp2;
		}
	}

	switch (ffttype) {
	case 2:	/* forward + square + inverse + normalize */
		ok = arm64_pack_scrambled_to_complex(ad, word_cache, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_pointwise_square(tmp1, tmp1, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, word_cache, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 3:
		ok = arm64_pack_scrambled_to_complex(ad, word_cache, s1, tmp1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, word_cache, s2, tmp2);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_fft(tmp2, complex_len, 0);
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, word_cache, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 4:
		ok = arm64_pack_scrambled_to_complex(ad, word_cache, s1, tmp1);
		if (!ok) break;
		ok = arm64_pack_scrambled_to_complex(ad, word_cache, s2, tmp2);
		if (!ok) break;
		arm64_fft(tmp1, complex_len, 0);
		arm64_fft(tmp2, complex_len, 0);
		arm64_pointwise_mul(tmp1, tmp1, tmp2, complex_len);
		arm64_fft(tmp1, complex_len, 1);
		ok = arm64_unpack_complex_to_scrambled(ad, word_cache, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;

	case 5:
		arm64_normalize(asm_data);
		break;

	default:
		break;
	}
}