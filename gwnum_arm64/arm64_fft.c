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

enum {
	ARM64_CACHE_SLOTS = 32,
	ARM64_PREFETCH_AHEAD = 16
};

typedef struct arm64_complex {
	double r;
	double i;
} arm64_complex;

typedef struct arm64_word_cache {
	size_t fftlen;
	size_t *byte_offsets;
	double *fwd_weights;
	double *inv_weights;
	int *big_words;
} arm64_word_cache;

/* Per-stage contiguous twiddle tables for sequential memory access. */
typedef struct arm64_stage_twiddles {
	size_t n;
	size_t num_stages;
	double **stage_tables;  /* stage_tables[s] has half*2 doubles for stage s (m=2^(s+1)) */
} arm64_stage_twiddles;

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
	arm64_complex z; z.r = a.r + b.r; z.i = a.i + b.i; return z;
}

static inline arm64_complex arm64_c_sub(arm64_complex a, arm64_complex b) {
	arm64_complex z; z.r = a.r - b.r; z.i = a.i - b.i; return z;
}

static inline arm64_complex arm64_c_mul(arm64_complex a, arm64_complex b) {
	arm64_complex z;
	z.r = a.r * b.r - a.i * b.i;
	z.i = a.r * b.i + a.i * b.r;
	return z;
}

static inline arm64_complex arm64_c_conj(arm64_complex a) {
	arm64_complex z; z.r = a.r; z.i = -a.i; return z;
}

static int arm64_is_power_of_two(size_t n) {
	return (n != 0u) && ((n & (n - 1u)) == 0u);
}

static int arm64_log2_u32(uint32_t v) {
	int n = 0;
	while (v > 1u) { v >>= 1u; ++n; }
	return n;
}

/* --- Caches --- */

static double *arm64_twiddle_cache[ARM64_CACHE_SLOTS];
static uint32_t *arm64_bitrev_cache[ARM64_CACHE_SLOTS];
static arm64_stage_twiddles *arm64_staged_tw_cache[ARM64_CACHE_SLOTS];

typedef struct arm64_real_fft_twiddles {
	size_t n;
	double *twiddles;
} arm64_real_fft_twiddles;

static arm64_real_fft_twiddles *arm64_real_fft_tw_cache[ARM64_CACHE_SLOTS];
static arm64_word_cache arm64_global_word_cache = {0u, NULL, NULL, NULL, NULL};

static ARM64_THREAD_LOCAL double *arm64_tls_tmp1 = NULL;
static ARM64_THREAD_LOCAL double *arm64_tls_tmp2 = NULL;
static ARM64_THREAD_LOCAL size_t arm64_tls_tmp1_capacity = 0u;
static ARM64_THREAD_LOCAL size_t arm64_tls_tmp2_capacity = 0u;

/* --- Word cache --- */

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
	if (!byte_offsets || !fwd_weights || !inv_weights || !big_words) {
		free(byte_offsets); free(fwd_weights); free(inv_weights); free(big_words);
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

/* --- TLS buffer management --- */

static int arm64_reserve_tls_buffer(double **buffer, size_t *capacity, size_t required) {
	double *nb;
	if (required == 0u) return 0;
	if (*buffer != NULL && *capacity >= required) return 1;
	nb = (double *)realloc(*buffer, required * sizeof(double));
	if (nb == NULL) return 0;
	*buffer = nb;
	*capacity = required;
	return 1;
}

/* --- Base twiddle table (used to build per-stage tables) --- */

static const double *arm64_get_twiddle_table(size_t n) {
	int log2_n;
	size_t j, half;
	double *table, angle_base;

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

/* --- Per-stage contiguous twiddle tables --- */

static const arm64_stage_twiddles *arm64_get_staged_twiddles(size_t n) {
	int log2_n;
	arm64_stage_twiddles *stw;
	size_t s, m, half, j, tw_step;
	const double *base_table;

	if (n < 2u || !arm64_is_power_of_two(n) || n > (size_t)UINT32_MAX) return NULL;
	log2_n = arm64_log2_u32((uint32_t)n);
	if (log2_n < 1 || log2_n >= ARM64_CACHE_SLOTS) return NULL;

	if (arm64_staged_tw_cache[log2_n] != NULL)
		return arm64_staged_tw_cache[log2_n];

	base_table = arm64_get_twiddle_table(n);
	if (base_table == NULL) return NULL;

	stw = (arm64_stage_twiddles *)malloc(sizeof(arm64_stage_twiddles));
	if (stw == NULL) return NULL;

	stw->n = n;
	stw->num_stages = (size_t)log2_n;
	stw->stage_tables = (double **)calloc((size_t)log2_n, sizeof(double *));
	if (stw->stage_tables == NULL) { free(stw); return NULL; }

	for (s = 0, m = 2u; m <= n; m *= 2u, ++s) {
		half = m / 2u;
		tw_step = n / m;
		stw->stage_tables[s] = (double *)malloc(half * 2u * sizeof(double));
		if (stw->stage_tables[s] == NULL) {
			size_t c;
			for (c = 0; c < s; ++c) free(stw->stage_tables[c]);
			free(stw->stage_tables);
			free(stw);
			return NULL;
		}
		for (j = 0; j < half; ++j) {
			size_t ti = j * tw_step;
			stw->stage_tables[s][2u * j]     = base_table[2u * ti];
			stw->stage_tables[s][2u * j + 1u] = base_table[2u * ti + 1u];
		}
	}

	arm64_staged_tw_cache[log2_n] = stw;
	return stw;
}

/* --- Real FFT split/merge twiddle tables --- */

static const arm64_real_fft_twiddles *arm64_get_real_fft_twiddles(size_t n_real) {
	int log2_n;
	size_t k, half;
	double angle_base;
	arm64_real_fft_twiddles *entry;

	if (n_real < 2u || (n_real & 1u) != 0u || !arm64_is_power_of_two(n_real) || n_real > (size_t)UINT32_MAX) return NULL;

	log2_n = arm64_log2_u32((uint32_t)n_real);
	if (log2_n < 1 || log2_n >= ARM64_CACHE_SLOTS) return NULL;

	entry = arm64_real_fft_tw_cache[log2_n];
	if (entry != NULL && entry->n == n_real && entry->twiddles != NULL)
		return entry;

	if (entry != NULL) {
		free(entry->twiddles);
		free(entry);
		arm64_real_fft_tw_cache[log2_n] = NULL;
	}

	entry = (arm64_real_fft_twiddles *)malloc(sizeof(arm64_real_fft_twiddles));
	if (entry == NULL) return NULL;

	half = n_real / 2u;
	entry->twiddles = (double *)malloc((half + 1u) * 2u * sizeof(double));
	if (entry->twiddles == NULL) {
		free(entry);
		return NULL;
	}

	entry->n = n_real;
	angle_base = -2.0 * M_PI / (double)n_real;
	for (k = 0; k <= half; ++k) {
		double angle = angle_base * (double)k;
		entry->twiddles[2u * k] = cos(angle);
		entry->twiddles[2u * k + 1u] = sin(angle);
	}

	arm64_real_fft_tw_cache[log2_n] = entry;
	return entry;
}

/* --- Bit-reverse permutation --- */

static uint32_t arm64_reverse_bits(uint32_t x, unsigned bits) {
	uint32_t y = 0;
	unsigned i;
	for (i = 0; i < bits; ++i) { y = (y << 1u) | (x & 1u); x >>= 1u; }
	return y;
}

static const uint32_t *arm64_get_bitrev_table(size_t n) {
	int log2_n;
	size_t i;
	uint32_t *table;

	if (n < 2u || !arm64_is_power_of_two(n) || n > (size_t)UINT32_MAX) return NULL;
	log2_n = arm64_log2_u32((uint32_t)n);
	if (log2_n < 0 || log2_n >= ARM64_CACHE_SLOTS) return NULL;

	table = arm64_bitrev_cache[log2_n];
	if (table != NULL) return table;

	table = (uint32_t *)malloc(n * sizeof(uint32_t));
	if (table == NULL) return NULL;

	for (i = 0; i < n; ++i)
		table[i] = arm64_reverse_bits((uint32_t)i, (unsigned)log2_n);

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

	/* Fallback */
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

/* --- Inverse scaling --- */

static void arm64_scale_inverse(double *data, size_t complex_len) {
	size_t words = complex_len * 2u;
	double inv_n = 1.0 / (double)complex_len;
	size_t i;
#if defined(__aarch64__) || defined(ARM64)
	{
		float64x2_t inv = vdupq_n_f64(inv_n);
		for (i = 0; i + 1u < words; i += 2u) {
			float64x2_t v = vld1q_f64(&data[i]);
			vst1q_f64(&data[i], vmulq_f64(v, inv));
		}
		if (i < words) data[i] *= inv_n;
	}
#else
	for (i = 0; i < words; ++i) data[i] *= inv_n;
#endif
}

/* --- FFT butterfly with contiguous per-stage twiddle access --- */

static void arm64_fft_stage_contiguous(double *data, size_t n, size_t m, const double *stage_tw, int inverse) {
	size_t half = m / 2u;
	double imag_sign = inverse ? -1.0 : 1.0;
	size_t k;

	for (k = 0; k < n; k += m) {
		size_t j = 0u;
#if defined(__aarch64__) || defined(ARM64)
		/* 4-wide NEON butterfly with contiguous twiddle loads via vld2q */
		for (; j + 3u < half; j += 4u) {
			double *a0_ptr = &data[(k + j) * 2u];
			double *b0_ptr = &data[(k + j + half) * 2u];
			double *a1_ptr = a0_ptr + 4u;
			double *b1_ptr = b0_ptr + 4u;
			const double *tw0_ptr = &stage_tw[j * 2u];
			const double *tw1_ptr = tw0_ptr + 4u;
			float64x2x2_t tw0_pair, tw1_pair;
			float64x2_t w_re0, w_im0, w_re1, w_im1;
			float64x2x2_t va0, vb0, va1, vb1;
			float64x2_t bw_re0, bw_im0, bw_re1, bw_im1;
			float64x2x2_t out_top, out_bot;

			{
				size_t pf = j + (size_t)ARM64_PREFETCH_AHEAD;
				if (pf < half) {
					ARM64_PREFETCH_READ(&data[(k + pf) * 2u]);
					ARM64_PREFETCH_READ(&data[(k + pf + half) * 2u]);
					ARM64_PREFETCH_READ(&stage_tw[pf * 2u]);
				}
			}

			/* Contiguous twiddle load: vld2q deinterleaves re/im */
			tw0_pair = vld2q_f64(tw0_ptr);
			tw1_pair = vld2q_f64(tw1_ptr);
			w_re0 = tw0_pair.val[0];
			w_im0 = vmulq_n_f64(tw0_pair.val[1], imag_sign);
			w_re1 = tw1_pair.val[0];
			w_im1 = vmulq_n_f64(tw1_pair.val[1], imag_sign);

			va0 = vld2q_f64(a0_ptr);
			vb0 = vld2q_f64(b0_ptr);
			va1 = vld2q_f64(a1_ptr);
			vb1 = vld2q_f64(b1_ptr);

			/* Interleaved complex multiply for ILP */
			bw_re0 = vmulq_f64(vb0.val[0], w_re0);
			bw_im0 = vmulq_f64(vb0.val[0], w_im0);
			bw_re1 = vmulq_f64(vb1.val[0], w_re1);
			bw_im1 = vmulq_f64(vb1.val[0], w_im1);

			bw_re0 = vfmsq_f64(bw_re0, vb0.val[1], w_im0);
			bw_im0 = vfmaq_f64(bw_im0, vb0.val[1], w_re0);
			bw_re1 = vfmsq_f64(bw_re1, vb1.val[1], w_im1);
			bw_im1 = vfmaq_f64(bw_im1, vb1.val[1], w_re1);

			out_top.val[0] = vaddq_f64(va0.val[0], bw_re0);
			out_top.val[1] = vaddq_f64(va0.val[1], bw_im0);
			out_bot.val[0] = vsubq_f64(va0.val[0], bw_re0);
			out_bot.val[1] = vsubq_f64(va0.val[1], bw_im0);
			vst2q_f64(a0_ptr, out_top);
			vst2q_f64(b0_ptr, out_bot);

			out_top.val[0] = vaddq_f64(va1.val[0], bw_re1);
			out_top.val[1] = vaddq_f64(va1.val[1], bw_im1);
			out_bot.val[0] = vsubq_f64(va1.val[0], bw_re1);
			out_bot.val[1] = vsubq_f64(va1.val[1], bw_im1);
			vst2q_f64(a1_ptr, out_top);
			vst2q_f64(b1_ptr, out_bot);
		}

		/* 2-wide NEON tail */
		for (; j + 1u < half; j += 2u) {
			const double *tw_ptr = &stage_tw[j * 2u];
			float64x2x2_t tw_pair;
			float64x2_t w_re, w_im;
			float64x2x2_t va, vb;
			float64x2_t bw_re, bw_im;
			float64x2x2_t out0, out1;

			tw_pair = vld2q_f64(tw_ptr);
			w_re = tw_pair.val[0];
			w_im = vmulq_n_f64(tw_pair.val[1], imag_sign);

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
		/* Scalar tail */
		for (; j < half; ++j) {
			size_t i0 = k + j;
			size_t i1 = i0 + half;
			arm64_complex a = arm64_c_load(data, i0);
			arm64_complex b = arm64_c_load(data, i1);
			arm64_complex w, bw;

			w.r = stage_tw[2u * j];
			w.i = imag_sign * stage_tw[2u * j + 1u];
			bw = arm64_c_mul(b, w);

			arm64_c_store(data, i0, arm64_c_add(a, bw));
			arm64_c_store(data, i1, arm64_c_sub(a, bw));
		}
	}
}

/* Radix-2 Cooley-Tukey DIT FFT using per-stage contiguous twiddle tables. */
static void arm64_fft(double *data, size_t n, int inverse) {
	size_t m, s;
	const arm64_stage_twiddles *stw;

	if (data == NULL || n < 2u || !arm64_is_power_of_two(n)) return;

	stw = arm64_get_staged_twiddles(n);
	if (stw == NULL) return;

	arm64_bit_reverse_permute(data, n);

	for (s = 0, m = 2u; m <= n; m *= 2u, ++s)
		arm64_fft_stage_contiguous(data, n, m, stw->stage_tables[s], inverse);

	if (inverse) arm64_scale_inverse(data, n);
}

/* --- Pack/unpack between gwnum scrambled layout and contiguous complex --- */

static int arm64_pack_scrambled_to_complex(const struct gwasm_data *ad, const arm64_word_cache *cache, const double *src, double *dst_complex) {
	size_t words, j;
	const char *src_bytes;

	if (ad == NULL || cache == NULL || src == NULL || dst_complex == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return 0;
	if (cache->fftlen != words || cache->byte_offsets == NULL) return 0;

	src_bytes = (const char *)src;
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

/* --- Real FFT split/merge and layout helpers --- */

static int arm64_unscramble(const struct gwasm_data *ad, const arm64_word_cache *cache, const double *src, double *dst_linear) {
	size_t words, j;
	const char *src_bytes;

	if (ad == NULL || cache == NULL || src == NULL || dst_linear == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u) return 0;
	if (cache->fftlen != words || cache->byte_offsets == NULL) return 0;

	src_bytes = (const char *)src;
	for (j = 0; j < words; ++j)
		dst_linear[j] = *(const double *)(src_bytes + cache->byte_offsets[j]);

	return 1;
}

static int arm64_rescramble(const struct gwasm_data *ad, const arm64_word_cache *cache, const double *src_linear, double *dst) {
	size_t words, j;
	char *dst_bytes;

	if (ad == NULL || cache == NULL || src_linear == NULL || dst == NULL) return 0;
	words = arm64_data_words(ad);
	if (words == 0u) return 0;
	if (cache->fftlen != words || cache->byte_offsets == NULL) return 0;

	dst_bytes = (char *)dst;
	for (j = 0; j < words; ++j)
		*(double *)(dst_bytes + cache->byte_offsets[j]) = src_linear[j];

	return 1;
}

static int arm64_pack_real_to_complex(const double *linear, double *complex_buf, size_t n) {
	size_t j;

	if (linear == NULL || complex_buf == NULL || n == 0u) return 0;

	for (j = n; j > 0u; --j) {
		double v = linear[j - 1u];
		complex_buf[(j - 1u) * 2u] = v;
		complex_buf[(j - 1u) * 2u + 1u] = 0.0;
	}
	return 1;
}

static int arm64_extract_real(const double *complex_buf, double *linear, size_t n) {
	size_t j;

	if (complex_buf == NULL || linear == NULL || n == 0u) return 0;

	for (j = 0; j < n; ++j)
		linear[j] = complex_buf[2u * j];

	return 1;
}

static void arm64_real_fft_split(const double *Z, double *X, size_t half, const double *tw) {
	size_t k, mid;
	arm64_complex z0, xk;

	if (Z == NULL || X == NULL || tw == NULL || half == 0u) return;

	z0 = arm64_c_load(Z, 0u);

	xk.r = z0.r + z0.i;
	xk.i = 0.0;
	arm64_c_store(X, 0u, xk);

	xk.r = z0.r - z0.i;
	xk.i = 0.0;
	arm64_c_store(X, half, xk);

	if (half == 1u) return;

	mid = half / 2u;

	for (k = 1u; k < mid; ++k) {
		arm64_complex a = arm64_c_load(Z, k);
		arm64_complex b = arm64_c_conj(arm64_c_load(Z, half - k));
		arm64_complex sum = arm64_c_add(a, b);
		arm64_complex diff = arm64_c_sub(a, b);
		arm64_complex e, o, w;

		e.r = 0.5 * sum.r;
		e.i = 0.5 * sum.i;
		o.r = 0.5 * diff.i;
		o.i = -0.5 * diff.r;
		w.r = tw[2u * k];
		w.i = tw[2u * k + 1u];

		xk = arm64_c_add(e, arm64_c_mul(w, o));
		arm64_c_store(X, k, xk);
		arm64_c_store(X, half - k, arm64_c_conj(xk));
	}

	if ((half & 1u) == 0u) {
		k = mid;
		{
			arm64_complex a = arm64_c_load(Z, k);
			arm64_complex b = arm64_c_conj(arm64_c_load(Z, half - k));
			arm64_complex sum = arm64_c_add(a, b);
			arm64_complex diff = arm64_c_sub(a, b);
			arm64_complex e, o, w;

			e.r = 0.5 * sum.r;
			e.i = 0.5 * sum.i;
			o.r = 0.5 * diff.i;
			o.i = -0.5 * diff.r;
			w.r = tw[2u * k];
			w.i = tw[2u * k + 1u];

			xk = arm64_c_add(e, arm64_c_mul(w, o));
		}
		xk.i = 0.0;
		arm64_c_store(X, k, xk);
	} else if (mid > 0u) {
		k = mid;
		{
			arm64_complex a = arm64_c_load(Z, k);
			arm64_complex b = arm64_c_conj(arm64_c_load(Z, half - k));
			arm64_complex sum = arm64_c_add(a, b);
			arm64_complex diff = arm64_c_sub(a, b);
			arm64_complex e, o, w;

			e.r = 0.5 * sum.r;
			e.i = 0.5 * sum.i;
			o.r = 0.5 * diff.i;
			o.i = -0.5 * diff.r;
			w.r = tw[2u * k];
			w.i = tw[2u * k + 1u];

			xk = arm64_c_add(e, arm64_c_mul(w, o));
		}
		arm64_c_store(X, k, xk);
		arm64_c_store(X, half - k, arm64_c_conj(xk));
	}
}

static void arm64_enforce_hermitian(double *X, size_t half) {
	size_t k, mid;
	arm64_complex xk;

	if (X == NULL || half == 0u) return;

	xk = arm64_c_load(X, 0u);
	xk.i = 0.0;
	arm64_c_store(X, 0u, xk);

	xk = arm64_c_load(X, half);
	xk.i = 0.0;
	arm64_c_store(X, half, xk);

	mid = half / 2u;

	if ((half & 1u) == 0u) {
		for (k = 1u; k < mid; ++k) {
			xk = arm64_c_load(X, k);
			arm64_c_store(X, half - k, arm64_c_conj(xk));
		}
		if (mid > 0u) {
			xk = arm64_c_load(X, mid);
			xk.i = 0.0;
			arm64_c_store(X, mid, xk);
		}
	} else {
		for (k = 1u; k <= mid; ++k) {
			xk = arm64_c_load(X, k);
			arm64_c_store(X, half - k, arm64_c_conj(xk));
		}
	}
}

static void arm64_real_fft_merge(const double *X, double *Z, size_t half, const double *tw) {
	size_t k, mid;
	arm64_complex x0, xh, z0;

	if (X == NULL || Z == NULL || tw == NULL || half == 0u) return;

	x0 = arm64_c_load(X, 0u);
	xh = arm64_c_load(X, half);
	z0.r = 0.5 * (x0.r + xh.r);
	z0.i = 0.5 * (x0.r - xh.r);
	arm64_c_store(Z, 0u, z0);

	if (half == 1u) return;

	mid = half / 2u;

	for (k = 1u; k < mid; ++k) {
		arm64_complex a = arm64_c_load(X, k);
		arm64_complex b = arm64_c_conj(arm64_c_load(X, half - k));
		arm64_complex sum = arm64_c_add(a, b);
		arm64_complex diff = arm64_c_sub(a, b);
		arm64_complex w_conj, t, it, zk, zmk_conj;

		sum.r *= 0.5;
		sum.i *= 0.5;
		diff.r *= 0.5;
		diff.i *= 0.5;

		w_conj.r = tw[2u * k];
		w_conj.i = -tw[2u * k + 1u];

		t = arm64_c_mul(diff, w_conj);
		it.r = -t.i;
		it.i = t.r;

		zk = arm64_c_add(sum, it);
		zmk_conj = arm64_c_sub(sum, it);

		arm64_c_store(Z, k, zk);
		arm64_c_store(Z, half - k, arm64_c_conj(zmk_conj));
	}

	if ((half & 1u) == 0u) {
		k = mid;
		{
			arm64_complex a = arm64_c_load(X, k);
			arm64_complex b = arm64_c_conj(arm64_c_load(X, half - k));
			arm64_complex sum = arm64_c_add(a, b);
			arm64_complex diff = arm64_c_sub(a, b);
			arm64_complex w_conj, t, it, zk;

			sum.r *= 0.5;
			sum.i *= 0.5;
			diff.r *= 0.5;
			diff.i *= 0.5;

			w_conj.r = tw[2u * k];
			w_conj.i = -tw[2u * k + 1u];

			t = arm64_c_mul(diff, w_conj);
			it.r = -t.i;
			it.i = t.r;

			zk = arm64_c_add(sum, it);
			arm64_c_store(Z, k, zk);
		}
	} else if (mid > 0u) {
		k = mid;
		{
			arm64_complex a = arm64_c_load(X, k);
			arm64_complex b = arm64_c_conj(arm64_c_load(X, half - k));
			arm64_complex sum = arm64_c_add(a, b);
			arm64_complex diff = arm64_c_sub(a, b);
			arm64_complex w_conj, t, it, zk, zmk_conj;

			sum.r *= 0.5;
			sum.i *= 0.5;
			diff.r *= 0.5;
			diff.i *= 0.5;

			w_conj.r = tw[2u * k];
			w_conj.i = -tw[2u * k + 1u];

			t = arm64_c_mul(diff, w_conj);
			it.r = -t.i;
			it.i = t.r;

			zk = arm64_c_add(sum, it);
			zmk_conj = arm64_c_sub(sum, it);

			arm64_c_store(Z, k, zk);
			arm64_c_store(Z, half - k, arm64_c_conj(zmk_conj));
		}
	}
}

/* --- Pointwise operations --- */

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
		out.val[0] = re; out.val[1] = im;
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

/* --- Normalization dispatch --- */

static void arm64_normalize(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	if (ad != NULL && ad->NORMRTN != NULL)
		((void (*)(struct gwasm_data *))ad->NORMRTN)(asm_data);
	else
		arm64_norm_plain(asm_data);
}

/* --- Main FFT entry point --- */

void arm64_fft_entry(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	const arm64_word_cache *word_cache;
	double *dest, *s1, *s2;
	size_t words, complex_len, required_doubles;
	unsigned int ffttype;
	double *tmp1 = NULL, *tmp2 = NULL;
	int ok = 1;

	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;

	/* Disable gwmul3_carefully at runtime (belt-and-suspenders). */
	if (ad->gwdata != NULL)
		ad->gwdata->careful_count = 0;

	words = arm64_data_words(ad);
	if (words == 0u || (words & 1u) != 0u) return;

	if (!arm64_ensure_word_cache(ad)) return;
	word_cache = arm64_get_word_cache();
	if (word_cache == NULL || word_cache->fftlen != words ||
		!word_cache->byte_offsets || !word_cache->fwd_weights ||
		!word_cache->inv_weights || !word_cache->big_words)
		return;

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
		required_doubles = 2u * words + 4u;
		if (!arm64_reserve_tls_buffer(&arm64_tls_tmp1, &arm64_tls_tmp1_capacity, required_doubles)) return;
		tmp1 = arm64_tls_tmp1;
		if (!arm64_reserve_tls_buffer(&arm64_tls_tmp2, &arm64_tls_tmp2_capacity, required_doubles)) return;
		tmp2 = arm64_tls_tmp2;
	}

	switch (ffttype) {
	case 2:
	{
		size_t half;
		const arm64_real_fft_twiddles *real_tw;

		half = words / 2u;
		if (half == 0u || !arm64_is_power_of_two(half)) { ok = 0; break; }

		real_tw = arm64_get_real_fft_twiddles(words);
		if (real_tw == NULL || real_tw->twiddles == NULL) { ok = 0; break; }

		ok = arm64_unscramble(ad, word_cache, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, half, 0);
		arm64_real_fft_split(tmp1, tmp2, half, real_tw->twiddles);
		arm64_pointwise_square(tmp2, tmp2, half + 1u);
		arm64_enforce_hermitian(tmp2, half);
		arm64_real_fft_merge(tmp2, tmp1, half, real_tw->twiddles);
		arm64_fft(tmp1, half, 1);
		ok = arm64_rescramble(ad, word_cache, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;
	}

	case 3:
	{
		size_t half;
		size_t spec_len;
		double *spec1, *spec2;
		const arm64_real_fft_twiddles *real_tw;

		half = words / 2u;
		if (half == 0u || !arm64_is_power_of_two(half)) { ok = 0; break; }

		real_tw = arm64_get_real_fft_twiddles(words);
		if (real_tw == NULL || real_tw->twiddles == NULL) { ok = 0; break; }

		spec_len = (half + 1u) * 2u;
		spec1 = tmp2;
		spec2 = tmp2 + spec_len;

		ok = arm64_unscramble(ad, word_cache, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, half, 0);
		arm64_real_fft_split(tmp1, spec1, half, real_tw->twiddles);

		ok = arm64_unscramble(ad, word_cache, s2, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, half, 0);
		arm64_real_fft_split(tmp1, spec2, half, real_tw->twiddles);

		arm64_pointwise_mul(spec1, spec1, spec2, half + 1u);
		arm64_enforce_hermitian(spec1, half);
		arm64_real_fft_merge(spec1, tmp1, half, real_tw->twiddles);
		arm64_fft(tmp1, half, 1);
		ok = arm64_rescramble(ad, word_cache, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;
	}

	case 4:
	{
		size_t half;
		size_t spec_len;
		double *spec1, *spec2;
		const arm64_real_fft_twiddles *real_tw;

		half = words / 2u;
		if (half == 0u || !arm64_is_power_of_two(half)) { ok = 0; break; }

		real_tw = arm64_get_real_fft_twiddles(words);
		if (real_tw == NULL || real_tw->twiddles == NULL) { ok = 0; break; }

		spec_len = (half + 1u) * 2u;
		spec1 = tmp2;
		spec2 = tmp2 + spec_len;

		ok = arm64_unscramble(ad, word_cache, s1, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, half, 0);
		arm64_real_fft_split(tmp1, spec1, half, real_tw->twiddles);

		ok = arm64_unscramble(ad, word_cache, s2, tmp1);
		if (!ok) break;
		arm64_fft(tmp1, half, 0);
		arm64_real_fft_split(tmp1, spec2, half, real_tw->twiddles);

		arm64_pointwise_mul(spec1, spec1, spec2, half + 1u);
		arm64_enforce_hermitian(spec1, half);
		arm64_real_fft_merge(spec1, tmp1, half, real_tw->twiddles);
		arm64_fft(tmp1, half, 1);
		ok = arm64_rescramble(ad, word_cache, tmp1, dest);
		if (!ok) break;
		arm64_normalize(asm_data);
		break;
	}

	case 5:
		arm64_normalize(asm_data);
		break;

	default:
		break;
	}
}