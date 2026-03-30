#ifndef ARMCRUNCH_GWNUM_ARM64_ASM_DATA_H
#define ARMCRUNCH_GWNUM_ARM64_ASM_DATA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gwasm_data;
struct gwasm_jmptab;

typedef void (*arm64_gwproc_routine)(struct gwasm_data *);

enum { ARM64_WORD_SMALL = 0, ARM64_WORD_BIG = 1 };

typedef struct arm64_asm_constants {
	double NEON_LARGE_BASE;
	double NEON_SMALL_BASE;
	double NEON_LARGE_BASE_INV;
	double NEON_SMALL_BASE_INV;
	double NEON_BIGVAL;			/* 3.0 * 2^51 */
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

/*
 * ARM64 mirror of the subset of asm_data consumed by the C/NEON backend.
 * The gwnum_arm64.patch file shows where gwnum.c wires this in.
 */
typedef struct arm64_gwasm_data_view {
	double *DESTARG;
	double *SRCARG;
	intptr_t DIST_TO_FFTSRCARG;
	intptr_t DIST_TO_MULSRCARG;
	uint32_t ffttype;
	uint32_t const_fft;
	uint32_t FFTLEN;
	uint32_t PASS1_SIZE;
	uint32_t PASS2_SIZE;
	uint32_t ADDIN_OFFSET;
	double ADDIN_VALUE;
	double POSTADDIN_VALUE;
	arm64_gwproc_routine NORMRTN;
	double *sincos1;
	double *sincos2;
	double *sincos3;
	double *norm_col_mults;
	double *norm_grp_mults;
	double *carries;
	double MAXERR;
	arm64_asm_constants arm64;
} arm64_gwasm_data_view;

#define ARM64_DEFAULT_SMALL_BASE	134217728.0
#define ARM64_DEFAULT_LARGE_BASE	268435456.0
#define ARM64_DEFAULT_BIGVAL		6755399441055744.0	/* 3.0 * 2^51 */

static inline arm64_gwasm_data_view *arm64_asm_data_view(struct gwasm_data *asm_data) {
	return (arm64_gwasm_data_view *)(void *)asm_data;
}

static inline const arm64_gwasm_data_view *arm64_asm_data_const_view(const struct gwasm_data *asm_data) {
	return (const arm64_gwasm_data_view *)(const void *)asm_data;
}

static inline size_t arm64_complex_len(const arm64_gwasm_data_view *ad) {
	if (ad == NULL) return 0;
	if (ad->FFTLEN != 0u) return (size_t)ad->FFTLEN;
	if (ad->PASS1_SIZE != 0u) return (size_t)ad->PASS1_SIZE;
	return 0;
}

static inline size_t arm64_data_words(const arm64_gwasm_data_view *ad) {
	return arm64_complex_len(ad) * 2u;
}

static inline double *arm64_fftsrc_ptr(const arm64_gwasm_data_view *ad) {
	if (ad == NULL || ad->DESTARG == NULL) return NULL;
	if (ad->DIST_TO_FFTSRCARG == 0) return ad->DESTARG;
	return (double *)((char *)ad->DESTARG + ad->DIST_TO_FFTSRCARG);
}

static inline double *arm64_mulsrc_ptr(const arm64_gwasm_data_view *ad) {
	if (ad == NULL || ad->DESTARG == NULL) return NULL;
	if (ad->DIST_TO_MULSRCARG == 0) return ad->DESTARG;
	return (double *)((char *)ad->DESTARG + ad->DIST_TO_MULSRCARG);
}

static inline double arm64_word_base(const arm64_gwasm_data_view *ad, int big_word) {
	if (ad == NULL) return big_word ? ARM64_DEFAULT_LARGE_BASE : ARM64_DEFAULT_SMALL_BASE;
	if (big_word) {
		return ad->arm64.NEON_LARGE_BASE != 0.0 ? ad->arm64.NEON_LARGE_BASE : ARM64_DEFAULT_LARGE_BASE;
	}
	return ad->arm64.NEON_SMALL_BASE != 0.0 ? ad->arm64.NEON_SMALL_BASE : ARM64_DEFAULT_SMALL_BASE;
}

static inline double arm64_word_base_inverse(const arm64_gwasm_data_view *ad, int big_word) {
	if (ad != NULL) {
		double inv = ad->arm64.NEON_LIMIT_INVERSE[big_word ? ARM64_WORD_BIG : ARM64_WORD_SMALL];
		if (inv != 0.0) return inv;
		inv = big_word ? ad->arm64.NEON_LARGE_BASE_INV : ad->arm64.NEON_SMALL_BASE_INV;
		if (inv != 0.0) return inv;
	}
	return 1.0 / arm64_word_base(ad, big_word);
}

static inline double arm64_word_limit(const arm64_gwasm_data_view *ad, int big_word) {
	if (ad != NULL) {
		double lim = ad->arm64.NEON_LIMIT_BIGMAX[big_word ? ARM64_WORD_BIG : ARM64_WORD_SMALL];
		if (lim != 0.0) return lim;
	}
	return arm64_word_base(ad, big_word) * 0.5;
}

static inline double arm64_mulconst(const arm64_gwasm_data_view *ad) {
	if (ad == NULL) return 1.0;
	return ad->arm64.NEON_MULCONST != 0.0 ? ad->arm64.NEON_MULCONST : 1.0;
}

static inline double arm64_inverse_weight_at(const arm64_gwasm_data_view *ad, size_t complex_index) {
	size_t n;
	double inv_weight = 1.0;

	if (ad == NULL) return 1.0;
	n = arm64_complex_len(ad);
	if (n == 0) return 1.0;

	if (ad->norm_col_mults != NULL) {
		inv_weight *= ad->norm_col_mults[complex_index % n];
	}
	if (ad->norm_grp_mults != NULL) {
		size_t group_span = ad->PASS2_SIZE != 0 ? (size_t)ad->PASS2_SIZE : n;
		size_t group_index = group_span != 0 ? (complex_index / group_span) : 0;
		inv_weight *= ad->norm_grp_mults[group_index];
	}
	if (inv_weight == 0.0) return 1.0;
	return inv_weight;
}

static inline double arm64_forward_weight_at(const arm64_gwasm_data_view *ad, size_t complex_index) {
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