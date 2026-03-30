#include "arm64_asm_data.h"

#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) || defined(ARM64)
#include <arm_neon.h>
#endif

static inline double *arm64_aux_source(const arm64_gwasm_data_view *ad) {
	if (ad == NULL) return NULL;
	if (ad->SRCARG != NULL) return ad->SRCARG;
	return arm64_fftsrc_ptr(ad);
}

static inline size_t arm64_aux_words(const arm64_gwasm_data_view *ad) {
	return arm64_data_words(ad);
}

#if defined(__aarch64__) || defined(ARM64)
static void arm64_vec_add(double *dst, const double *src, size_t words) {
	size_t i;
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t vd = vld1q_f64(&dst[i]);
		float64x2_t vs = vld1q_f64(&src[i]);
		vst1q_f64(&dst[i], vaddq_f64(vd, vs));
	}
	if (i < words) dst[i] += src[i];
}

static void arm64_vec_sub(double *dst, const double *src, size_t words) {
	size_t i;
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t vd = vld1q_f64(&dst[i]);
		float64x2_t vs = vld1q_f64(&src[i]);
		vst1q_f64(&dst[i], vsubq_f64(vd, vs));
	}
	if (i < words) dst[i] -= src[i];
}
#else
static void arm64_vec_add(double *dst, const double *src, size_t words) {
	size_t i;
	for (i = 0; i < words; ++i) dst[i] += src[i];
}

static void arm64_vec_sub(double *dst, const double *src, size_t words) {
	size_t i;
	for (i = 0; i < words; ++i) dst[i] -= src[i];
}
#endif

void arm64_gw_addq(struct gwasm_data *asm_data) {
	arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
	double *dst;
	double *src;
	size_t words;

	if (ad == NULL) return;
	dst = ad->DESTARG;
	src = arm64_aux_source(ad);
	words = arm64_aux_words(ad);
	if (dst == NULL || src == NULL || words == 0) return;

	arm64_vec_add(dst, src, words);
}

void arm64_gw_add(struct gwasm_data *asm_data) {
	arm64_gw_addq(asm_data);
	{
		arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
		if (ad != NULL && ad->DESTARG != NULL) arm64_normalize_buffer(asm_data, ad->DESTARG, 0, 0);
	}
}

void arm64_gw_subq(struct gwasm_data *asm_data) {
	arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
	double *dst;
	double *src;
	size_t words;

	if (ad == NULL) return;
	dst = ad->DESTARG;
	src = arm64_aux_source(ad);
	words = arm64_aux_words(ad);
	if (dst == NULL || src == NULL || words == 0) return;

	arm64_vec_sub(dst, src, words);
}

void arm64_gw_sub(struct gwasm_data *asm_data) {
	arm64_gw_subq(asm_data);
	{
		arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
		if (ad != NULL && ad->DESTARG != NULL) arm64_normalize_buffer(asm_data, ad->DESTARG, 0, 0);
	}
}

void arm64_gw_addsubq(struct gwasm_data *asm_data) {
	arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
	double *dst;
	double *src;
	size_t words;
	size_t i;

	if (ad == NULL) return;
	dst = ad->DESTARG;
	src = arm64_aux_source(ad);
	words = arm64_aux_words(ad);
	if (dst == NULL || src == NULL || words == 0) return;

#if defined(__aarch64__) || defined(ARM64)
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t vd = vld1q_f64(&dst[i]);
		float64x2_t vs = vld1q_f64(&src[i]);
		float64x2_t vsum = vaddq_f64(vd, vs);
		float64x2_t vdiff = vsubq_f64(vd, vs);
		vst1q_f64(&dst[i], vsum);
		vst1q_f64(&src[i], vdiff);
	}
	if (i < words) {
		double a = dst[i];
		double b = src[i];
		dst[i] = a + b;
		src[i] = a - b;
	}
#else
	for (i = 0; i < words; ++i) {
		double a = dst[i];
		double b = src[i];
		dst[i] = a + b;
		src[i] = a - b;
	}
#endif
}

void arm64_gw_addsub(struct gwasm_data *asm_data) {
	arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
	double *src;
	if (ad == NULL) return;

	arm64_gw_addsubq(asm_data);

	if (ad->DESTARG != NULL) {
		arm64_normalize_buffer(asm_data, ad->DESTARG, 0, 0);
	}

	src = arm64_aux_source(ad);
	if (src != NULL && src != ad->DESTARG) {
		arm64_normalize_buffer(asm_data, src, 0, 0);
	}
}

void arm64_gw_copy4kb(struct gwasm_data *asm_data) {
	arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
	double *dst;
	double *src;
	size_t words = 4096u / sizeof(double);
	size_t i;

	if (ad == NULL) return;
	dst = ad->DESTARG;
	src = arm64_aux_source(ad);
	if (dst == NULL || src == NULL) return;

#if defined(__aarch64__) || defined(ARM64)
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t v = vld1q_f64(&src[i]);
		vst1q_f64(&dst[i], v);
	}
	if (i < words) dst[i] = src[i];
#else
	memcpy(dst, src, 4096u);
#endif
}

void arm64_gw_muls(struct gwasm_data *asm_data) {
	arm64_gwasm_data_view *ad = arm64_asm_data_view(asm_data);
	double *dst;
	double *src;
	size_t words;
	double mul;

	if (ad == NULL) return;
	dst = ad->DESTARG;
	src = arm64_aux_source(ad);
	words = arm64_aux_words(ad);
	if (dst == NULL || words == 0) return;

	if (src != NULL && src != dst) {
		memmove(dst, src, words * sizeof(double));
	}

	mul = arm64_mulconst(ad);

#if defined(__aarch64__) || defined(ARM64)
	{
		size_t i = 0;
		float64x2_t vm = vdupq_n_f64(mul);
		for (; i + 1u < words; i += 2u) {
			float64x2_t vd = vld1q_f64(&dst[i]);
			vd = vmulq_f64(vd, vm);
			vst1q_f64(&dst[i], vd);
		}
		if (i < words) dst[i] *= mul;
	}
#else
	{
		size_t i;
		for (i = 0; i < words; ++i) dst[i] *= mul;
	}
#endif

	arm64_normalize_buffer(asm_data, dst, 0, 0);
}