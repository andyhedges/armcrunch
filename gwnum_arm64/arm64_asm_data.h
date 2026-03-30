#ifndef ARMCRUNCH_GWNUM_ARM64_ASM_DATA_H
#define ARMCRUNCH_GWNUM_ARM64_ASM_DATA_H

#include <stddef.h>
#include <stdint.h>
#include "gwtables.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*arm64_gwproc_routine)(struct gwasm_data *);

enum { ARM64_WORD_SMALL = 0, ARM64_WORD_BIG = 1 };

typedef struct arm64_asm_constants {
	double NEON_LARGE_BASE;
	double NEON_SMALL_BASE;
	double NEON_LARGE_BASE_INV;
	double NEON_SMALL_BASE_INV;
	double NEON_BIGVAL;				/* 3.0 * 2^51 */
	double NEON_LIMIT_INVERSE[2];		/* [small, big] */
	double NEON_LIMIT_BIGMAX[2];		/* [small, big] */
	double NEON_K_HI;
	double NEON_K_LO;
	double NEON_MINUS_C;
	double NEON_MULCONST;
	double NEON_NORM012_FF;			/* FFTLEN/2 or FFTLEN/2/k */
	double NEON_MAXERR;
	double NEON_B;
	double NEON_ONE_OVER_B;
	arm64_gwproc_routine NEON_CARRIES_ROUTINE;
	arm64_gwproc_routine NEON_PASS2_ROUTINE;
} arm64_asm_constants;

/* Single active constants block populated by arm64_gwsetup_hook(). */
extern arm64_asm_constants *arm64_active_asm_constants;

#define ARM64_DEFAULT_SMALL_BASE	134217728.0
#define ARM64_DEFAULT_LARGE_BASE	268435456.0
#define ARM64_DEFAULT_BIGVAL		6755399441055744.0	/* 3.0 * 2^51 */

static inline size_t arm64_complex_len(const struct gwasm_data *ad) {
	if (ad == NULL) return 0;
	return (size_t)ad->FFTLEN;
}

static inline size_t arm64_data_words(const struct gwasm_data *ad) {
	return arm64_complex_len(ad) * 2u;
}

static inline double *arm64_fftsrc_ptr(const struct gwasm_data *ad) {
	if (ad == NULL || ad->DESTARG == NULL) return NULL;
	return (double *)((char *)ad->DESTARG + ad->DIST_TO_FFTSRCARG);
}

static inline double *arm64_mulsrc_ptr(const struct gwasm_data *ad) {
	if (ad == NULL || ad->DESTARG == NULL) return NULL;
	return (double *)((char *)ad->DESTARG + ad->DIST_TO_MULSRCARG);
}

static inline const arm64_asm_constants *arm64_constants(const struct gwasm_data *ad) {
	(void)ad;
	return arm64_active_asm_constants;
}

static inline double arm64_bigval(const struct gwasm_data *ad) {
	const arm64_asm_constants *ac = arm64_constants(ad);
	if (ac != NULL && ac->NEON_BIGVAL != 0.0) return ac->NEON_BIGVAL;
	if (ad != NULL && ad->u.xmm.XMM_BIGVAL[0] != 0.0) return ad->u.xmm.XMM_BIGVAL[0];
	return ARM64_DEFAULT_BIGVAL;
}

static inline double arm64_word_base(const struct gwasm_data *ad, int big_word) {
	const arm64_asm_constants *ac = arm64_constants(ad);
	int idx = big_word ? ARM64_WORD_BIG : ARM64_WORD_SMALL;
	double base = 0.0;

	if (ac != NULL) {
		base = big_word ? ac->NEON_LARGE_BASE : ac->NEON_SMALL_BASE;
	}
	if (base == 0.0 && ad != NULL) {
		double inv = ad->u.xmm.XMM_LIMIT_INVERSE[idx];
		if (inv != 0.0) base = 1.0 / inv;
	}
	if (base == 0.0) {
		base = big_word ? ARM64_DEFAULT_LARGE_BASE : ARM64_DEFAULT_SMALL_BASE;
	}
	return base;
}

static inline double arm64_word_base_inverse(const struct gwasm_data *ad, int big_word) {
	const arm64_asm_constants *ac = arm64_constants(ad);
	int idx = big_word ? ARM64_WORD_BIG : ARM64_WORD_SMALL;
	double inv = 0.0;

	if (ac != NULL) {
		inv = ac->NEON_LIMIT_INVERSE[idx];
		if (inv == 0.0) inv = big_word ? ac->NEON_LARGE_BASE_INV : ac->NEON_SMALL_BASE_INV;
	}
	if (inv == 0.0 && ad != NULL) {
		inv = ad->u.xmm.XMM_LIMIT_INVERSE[idx];
	}
	if (inv == 0.0) {
		double base = arm64_word_base(ad, big_word);
		inv = (base != 0.0) ? (1.0 / base) : 0.0;
	}
	return inv;
}

static inline double arm64_word_limit(const struct gwasm_data *ad, int big_word) {
	const arm64_asm_constants *ac = arm64_constants(ad);
	int idx = big_word ? ARM64_WORD_BIG : ARM64_WORD_SMALL;
	double lim = 0.0;

	if (ac != NULL) lim = ac->NEON_LIMIT_BIGMAX[idx];
	if (lim == 0.0 && ad != NULL) lim = ad->u.xmm.XMM_LIMIT_BIGMAX[idx];
	if (lim == 0.0) {
		double base = arm64_word_base(ad, big_word);
		double bigval = arm64_bigval(ad);
		lim = base * bigval - bigval;
	}
	return lim;
}

static inline double arm64_mulconst(const struct gwasm_data *ad) {
	const arm64_asm_constants *ac = arm64_constants(ad);
	if (ac != NULL && ac->NEON_MULCONST != 0.0) return ac->NEON_MULCONST;
	if (ad != NULL && ad->u.xmm.XMM_MULCONST[0] != 0.0) return ad->u.xmm.XMM_MULCONST[0];
	return 1.0;
}

static inline double arm64_inverse_weight_at(const struct gwasm_data *ad, size_t complex_index) {
	size_t n;
	double inv_weight = 1.0;
	const double *col_mults;
	const double *grp_mults;

	if (ad == NULL) return 1.0;
	n = arm64_complex_len(ad);
	if (n == 0) return 1.0;

	col_mults = (const double *)ad->norm_col_mults;
	grp_mults = (const double *)ad->norm_grp_mults;

	if (col_mults != NULL) {
		inv_weight *= col_mults[complex_index % n];
	}
	if (grp_mults != NULL) {
		size_t group_index = 0;
		inv_weight *= grp_mults[group_index];
	}
	if (inv_weight == 0.0) return 1.0;
	return inv_weight;
}

static inline double arm64_forward_weight_at(const struct gwasm_data *ad, size_t complex_index) {
	double inv_weight = arm64_inverse_weight_at(ad, complex_index);
	return inv_weight == 0.0 ? 1.0 : 1.0 / inv_weight;
}

/* FFT entry point and auxiliary GWPROCPTRS routines */
void arm64_fft_entry(struct gwasm_data *asm_data);
void arm64_gw_add(struct gwasm_data *asm_data);
void arm64_gw_addq(struct gwasm_data *asm_data);
void arm64_gw_sub(struct gwasm_data *asm_data);
void arm64_gw_subq(struct gwasm_data *asm_data);
void arm64_gw_addsub(struct gwasm_data *asm_data);
void arm64_gw_addsubq(struct gwasm_data *asm_data);
void arm64_gw_copy4kb(struct gwasm_data *asm_data);
void arm64_gw_muls(struct gwasm_data *asm_data);

/* Normalization GWPROCPTRS routines */
void arm64_norm_plain(struct gwasm_data *asm_data);
void arm64_norm_errchk(struct gwasm_data *asm_data);
void arm64_norm_mulconst(struct gwasm_data *asm_data);
void arm64_norm_errchk_mulconst(struct gwasm_data *asm_data);

/* Shared normalizer used by FFT and aux paths */
void arm64_normalize_buffer(struct gwasm_data *asm_data, double *buffer, int errchk, int mulconst_mode);

/* gwinfo/jmptab */
const struct gwasm_jmptab *arm64_gwinfo1(int negacyclic);
const char *arm64_gwinfo_backend_version(void);
void arm64_install_gwprocptrs(void (**procptrs)(void *));

#ifdef __cplusplus
}
#endif

#endif