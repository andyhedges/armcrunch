#include "arm64_asm_data.h"
#include "gwtables.h"

#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) || defined(ARM64)
#include <arm_neon.h>
#endif

/*
 * gwnum asm_data field conventions for add/sub/muls:
 *
 * gwadd3o(s1, s2, d):   SRCARG=s1, SRC2ARG=s2, DESTARG=d
 *   gw_add:  d = s1 + s2  (normalized)
 *   gw_addq: d = s1 + s2  (unnormalized)
 *
 * gwsub3o(s1, s2, d):   SRCARG=s1, SRC2ARG=s2, DESTARG=d
 *   gw_sub:  d = s1 - s2  (normalized)
 *   gw_subq: d = s1 - s2  (unnormalized)
 *
 * gwaddsub4o(s1, s2, d1, d2): SRCARG=s1, SRC2ARG=s2, DESTARG=d1, DEST2ARG=d2
 *   gw_addsub:  d1 = s1 + s2, d2 = s1 - s2  (normalized)
 *   gw_addsubq: d1 = s1 + s2, d2 = s1 - s2  (unnormalized)
 *
 * gwsmallmul(mult, g):  SRCARG=g, DESTARG=g, DBLARG=mult
 *   gw_muls: g = g * DBLARG  (normalized)
 */

static inline size_t arm64_aux_words(const struct gwasm_data *ad) {
	return arm64_data_words(ad);
}

/* Operate on raw double arrays (physical memory layout, not logical word order).
   gwnum's add/sub/addsub operate element-by-element on the physical doubles. */

#if defined(__aarch64__) || defined(ARM64)
static void arm64_vec_add3(double *dst, const double *s1, const double *s2, size_t count) {
	size_t i;
	for (i = 0; i + 1u < count; i += 2u) {
		float64x2_t v1 = vld1q_f64(&s1[i]);
		float64x2_t v2 = vld1q_f64(&s2[i]);
		vst1q_f64(&dst[i], vaddq_f64(v1, v2));
	}
	if (i < count) dst[i] = s1[i] + s2[i];
}

static void arm64_vec_sub3(double *dst, const double *s1, const double *s2, size_t count) {
	size_t i;
	for (i = 0; i + 1u < count; i += 2u) {
		float64x2_t v1 = vld1q_f64(&s1[i]);
		float64x2_t v2 = vld1q_f64(&s2[i]);
		vst1q_f64(&dst[i], vsubq_f64(v1, v2));
	}
	if (i < count) dst[i] = s1[i] - s2[i];
}
#else
static void arm64_vec_add3(double *dst, const double *s1, const double *s2, size_t count) {
	size_t i;
	for (i = 0; i < count; ++i) dst[i] = s1[i] + s2[i];
}

static void arm64_vec_sub3(double *dst, const double *s1, const double *s2, size_t count) {
	size_t i;
	for (i = 0; i < count; ++i) dst[i] = s1[i] - s2[i];
}
#endif

void arm64_gw_addq(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dst;
	double *s1;
	double *s2;
	size_t words;

	if (ad == NULL) return;
	dst = (double *)ad->DESTARG;
	s1 = (double *)ad->SRCARG;
	s2 = (double *)ad->SRC2ARG;
	words = arm64_aux_words(ad);
	if (dst == NULL || s1 == NULL || s2 == NULL || words == 0) return;

	/* d = s1 + s2 (physical double arrays) */
	arm64_vec_add3(dst, s1, s2, words);
}

void arm64_gw_add(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;

	arm64_gw_addq(asm_data);

	if (ad != NULL && ad->DESTARG != NULL) {
		arm64_normalize_buffer(asm_data, (double *)ad->DESTARG, 0, 0, 0);
	}
}

void arm64_gw_subq(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dst;
	double *s1;
	double *s2;
	size_t words;

	if (ad == NULL) return;
	dst = (double *)ad->DESTARG;
	s1 = (double *)ad->SRCARG;
	s2 = (double *)ad->SRC2ARG;
	words = arm64_aux_words(ad);
	if (dst == NULL || s1 == NULL || s2 == NULL || words == 0) return;

	/* d = s1 - s2 (physical double arrays) */
	arm64_vec_sub3(dst, s1, s2, words);
}

void arm64_gw_sub(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;

	arm64_gw_subq(asm_data);

	if (ad != NULL && ad->DESTARG != NULL) {
		arm64_normalize_buffer(asm_data, (double *)ad->DESTARG, 0, 0, 0);
	}
}

void arm64_gw_addsubq(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *s1;
	double *s2;
	double *d1;
	double *d2;
	size_t words;
	size_t i;

	if (ad == NULL) return;
	s1 = (double *)ad->SRCARG;
	s2 = (double *)ad->SRC2ARG;
	d1 = (double *)ad->DESTARG;
	d2 = (double *)ad->DEST2ARG;
	words = arm64_aux_words(ad);
	if (s1 == NULL || s2 == NULL || d1 == NULL || d2 == NULL || words == 0) return;

	/* d1 = s1 + s2, d2 = s1 - s2 */
#if defined(__aarch64__) || defined(ARM64)
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t v1 = vld1q_f64(&s1[i]);
		float64x2_t v2 = vld1q_f64(&s2[i]);
		vst1q_f64(&d1[i], vaddq_f64(v1, v2));
		vst1q_f64(&d2[i], vsubq_f64(v1, v2));
	}
	if (i < words) {
		d1[i] = s1[i] + s2[i];
		d2[i] = s1[i] - s2[i];
	}
#else
	for (i = 0; i < words; ++i) {
		double a = s1[i];
		double b = s2[i];
		d1[i] = a + b;
		d2[i] = a - b;
	}
#endif
}

void arm64_gw_addsub(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;

	if (ad == NULL) return;

	arm64_gw_addsubq(asm_data);

	if (ad->DESTARG != NULL) {
		arm64_normalize_buffer(asm_data, (double *)ad->DESTARG, 0, 0, 0);
	}

	if (ad->DEST2ARG != NULL && ad->DEST2ARG != ad->DESTARG) {
		arm64_normalize_buffer(asm_data, (double *)ad->DEST2ARG, 0, 0, 0);
	}
}

void arm64_gw_copy4kb(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dst;
	double *src;

	if (ad == NULL) return;
	dst = (double *)ad->DESTARG;
	src = (double *)ad->SRCARG;
	if (dst == NULL || src == NULL) return;

#if defined(__aarch64__) || defined(ARM64)
	{
		size_t words = 4096u / sizeof(double);
		size_t i;
		for (i = 0; i + 1u < words; i += 2u) {
			float64x2_t v = vld1q_f64(&src[i]);
			vst1q_f64(&dst[i], v);
		}
		if (i < words) dst[i] = src[i];
	}
#else
	memcpy(dst, src, 4096u);
#endif
}

void arm64_gw_muls(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dst;
	double *src;
	size_t words;
	double mul;

	if (ad == NULL) return;
	dst = (double *)ad->DESTARG;
	src = (double *)ad->SRCARG;
	words = arm64_aux_words(ad);
	if (dst == NULL || words == 0) return;

	if (src != NULL && src != dst) {
		memmove(dst, src, words * sizeof(double));
	}

	/* gwsmallmul stores the multiplier in DBLARG, not in XMM_MULCONST */
	mul = ad->DBLARG;

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

	arm64_normalize_buffer(asm_data, dst, 0, 0, 0);
}