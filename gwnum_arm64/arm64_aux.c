#include "arm64_asm_data.h"
#include "gwtables.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if defined(__aarch64__) || defined(ARM64)
#include <arm_neon.h>
#endif

static inline size_t arm64_aux_words(const struct gwasm_data *ad) {
	return arm64_data_words(ad);
}

void arm64_gw_addq(struct gwasm_data *asm_data) {
	static int addq_count = 0;
	struct gwasm_data *ad = asm_data;
	double *dst;
	const double *s1;
	const double *s2;
	size_t words;
	size_t i;

	if (ad == NULL) return;
	/* gwnum sets SRCARG=s1, SRC2ARG=s2, DESTARG=d; compute d = s1 + s2 */
	dst = (double *)ad->DESTARG;
	s1 = (const double *)ad->SRCARG;
	s2 = (const double *)ad->SRC2ARG;
	words = arm64_aux_words(ad);
	if (dst == NULL || s1 == NULL || s2 == NULL || words == 0) return;

	addq_count++;
	if (addq_count <= 3) fprintf(stderr, "[ARM64 ADDQ #%d] words=%zu dst=%p s1=%p s2=%p\n",
		addq_count, words, (void*)dst, (void*)s1, (void*)s2);

#if defined(__aarch64__) || defined(ARM64)
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t v1 = vld1q_f64(&s1[i]);
		float64x2_t v2 = vld1q_f64(&s2[i]);
		vst1q_f64(&dst[i], vaddq_f64(v1, v2));
	}
	if (i < words) dst[i] = s1[i] + s2[i];
#else
	for (i = 0; i < words; ++i) dst[i] = s1[i] + s2[i];
#endif
}

void arm64_gw_add(struct gwasm_data *asm_data) {
	static int add_count = 0;
	struct gwasm_data *ad = asm_data;

	if (ad == NULL) return;

	add_count++;
	if (add_count <= 3) fprintf(stderr, "[ARM64 ADD #%d] dst=%p\n", add_count, (void*)ad->DESTARG);

	arm64_gw_addq(asm_data);

	if (ad != NULL && ad->DESTARG != NULL) {
		arm64_normalize_buffer(asm_data, (double *)ad->DESTARG, 0, 0);
	}
}

void arm64_gw_subq(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dst;
	const double *s1;
	const double *s2;
	size_t words;
	size_t i;

	if (ad == NULL) return;
	/* gwnum sets SRCARG=s1, SRC2ARG=s2, DESTARG=d; compute d = s1 - s2 */
	dst = (double *)ad->DESTARG;
	s1 = (const double *)ad->SRCARG;
	s2 = (const double *)ad->SRC2ARG;
	words = arm64_aux_words(ad);
	if (dst == NULL || s1 == NULL || s2 == NULL || words == 0) return;

#if defined(__aarch64__) || defined(ARM64)
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t v1 = vld1q_f64(&s1[i]);
		float64x2_t v2 = vld1q_f64(&s2[i]);
		vst1q_f64(&dst[i], vsubq_f64(v1, v2));
	}
	if (i < words) dst[i] = s1[i] - s2[i];
#else
	for (i = 0; i < words; ++i) dst[i] = s1[i] - s2[i];
#endif
}

void arm64_gw_sub(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;

	arm64_gw_subq(asm_data);

	if (ad != NULL && ad->DESTARG != NULL) {
		arm64_normalize_buffer(asm_data, (double *)ad->DESTARG, 0, 0);
	}
}

void arm64_gw_addsubq(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *d1;
	double *d2;
	const double *s1;
	const double *s2;
	size_t words;
	size_t i;

	if (ad == NULL) return;
	/* gwnum sets SRCARG=s1, SRC2ARG=s2, DESTARG=d1, DEST2ARG=d2 */
	/* Compute d1 = s1 + s2, d2 = s1 - s2 */
	s1 = (const double *)ad->SRCARG;
	s2 = (const double *)ad->SRC2ARG;
	d1 = (double *)ad->DESTARG;
	d2 = (double *)ad->DEST2ARG;
	words = arm64_aux_words(ad);
	if (d1 == NULL || d2 == NULL || s1 == NULL || s2 == NULL || words == 0) return;

#if defined(__aarch64__) || defined(ARM64)
	for (i = 0; i + 1u < words; i += 2u) {
		float64x2_t v1 = vld1q_f64(&s1[i]);
		float64x2_t v2 = vld1q_f64(&s2[i]);
		vst1q_f64(&d1[i], vaddq_f64(v1, v2));
		vst1q_f64(&d2[i], vsubq_f64(v1, v2));
	}
	if (i < words) {
		double a = s1[i];
		double b = s2[i];
		d1[i] = a + b;
		d2[i] = a - b;
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
		arm64_normalize_buffer(asm_data, (double *)ad->DESTARG, 0, 0);
	}

	if (ad->DEST2ARG != NULL && ad->DEST2ARG != ad->DESTARG) {
		arm64_normalize_buffer(asm_data, (double *)ad->DEST2ARG, 0, 0);
	}
}

void arm64_gw_copy4kb(struct gwasm_data *asm_data) {
	static int copy_count = 0;
	struct gwasm_data *ad = asm_data;
	double *dst;
	const double *src;

	if (ad == NULL) return;
	dst = (double *)ad->DESTARG;
	src = (const double *)ad->SRCARG;
	if (dst == NULL || src == NULL) return;

	copy_count++;
	if (copy_count <= 3) fprintf(stderr, "[ARM64 COPY4KB #%d]\n", copy_count);

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
	static int muls_count = 0;
	struct gwasm_data *ad = asm_data;
	double *dst;
	const double *src;
	size_t words;
	double mul;

	if (ad == NULL) return;
	dst = (double *)ad->DESTARG;
	src = (const double *)ad->SRCARG;
	words = arm64_aux_words(ad);
	if (dst == NULL || words == 0) return;

	/* gwsmallmul passes the multiplier in asm_data->DBLARG */
	mul = ad->DBLARG;

	muls_count++;
	if (muls_count <= 3) fprintf(stderr, "[ARM64 MULS #%d] words=%zu mul=%.6g dst=%p\n",
		muls_count, words, mul, (void*)dst);

	if (src != NULL && src != dst) {
		memmove(dst, src, words * sizeof(double));
	}

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