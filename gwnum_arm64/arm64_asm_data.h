#ifndef ARMCRUNCH_GWNUM_ARM64_ASM_DATA_H
#define ARMCRUNCH_GWNUM_ARM64_ASM_DATA_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "gwnum.h"
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
	unsigned long NEON_N;
	unsigned long NEON_NUM_B_PER_SMALL_WORD;
} arm64_asm_constants;

/* Single active constants block populated by arm64_gwsetup_hook(). */
extern arm64_asm_constants *arm64_active_asm_constants;

#define ARM64_DEFAULT_SMALL_BASE	134217728.0
#define ARM64_DEFAULT_LARGE_BASE	268435456.0
#define ARM64_DEFAULT_BIGVAL		6755399441055744.0	/* 3.0 * 2^51 */

/*
 * In gwnum data format FFTLEN is the total count of real doubles stored for one value.
 */
static inline size_t arm64_complex_len(const struct gwasm_data *ad) {
	if (ad == NULL) return 0;
	return (size_t)ad->FFTLEN;
}

static inline size_t arm64_data_words(const struct gwasm_data *ad) {
	return arm64_complex_len(ad);
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

static inline double arm64_inverse_weight_at(const struct gwasm_data *ad, size_t word_index) {
	size_t fftlen;
	const arm64_asm_constants *ac;
	unsigned long n_exp;
	double base;
	size_t j;
	unsigned long n_mod;
	size_t frac_num;
	double frac;
	double weight;

	if (ad == NULL) return 1.0;
	if (ad->RATIONAL_FFT != 0) return 1.0;

	fftlen = arm64_complex_len(ad);
	if (fftlen == 0) return 1.0;

	ac = arm64_constants(ad);
	n_exp = (ac != NULL) ? ac->NEON_N : 0ul;
	if (n_exp == 0ul) return 1.0;

	if (ad->B_IS_2 != 0) base = 2.0;
	else if (ac != NULL && ac->NEON_B > 0.0 && ac->NEON_B != 1.0) base = ac->NEON_B;
	else return 1.0;

	j = word_index % fftlen;
	n_mod = n_exp % (unsigned long)fftlen;

#if defined(__SIZEOF_INT128__)
	frac_num = (size_t)(((__uint128_t)j * (__uint128_t)n_mod) % (__uint128_t)fftlen);
#else
	frac_num = (size_t)fmodl((long double)j * (long double)n_mod, (long double)fftlen);
#endif

	frac = (double)frac_num / (double)fftlen;

	/* gwnum's map_to_weight_power_sloppy negates the fractional part:
	   tmp2 = 0 - tmp2, computing ceil(j*n/N) - j*n/N instead of
	   j*n/N - floor(j*n/N). Match this by using (1 - frac). */
	if (frac > 0.0) frac = 1.0 - frac;

	weight = pow(base, frac);

	if (weight == 0.0) return 1.0;
	return 1.0 / weight;
}

static inline double arm64_forward_weight_at(const struct gwasm_data *ad, size_t word_index) {
	double inv_weight = arm64_inverse_weight_at(ad, word_index);
	return inv_weight == 0.0 ? 1.0 : 1.0 / inv_weight;
}

static inline int arm64_is_big_word(const struct gwasm_data *ad, size_t word_index) {
	const arm64_asm_constants *ac;
	size_t fftlen;
	unsigned long n_exp;
	unsigned long num_b_small;

	if (ad == NULL) return 0;
	if (ad->RATIONAL_FFT != 0) return 0;

	fftlen = arm64_complex_len(ad);
	if (fftlen == 0) return 0;

	ac = arm64_constants(ad);
	n_exp = (ac != NULL) ? ac->NEON_N : 0ul;
	if (n_exp == 0ul) {
		return (int)((word_index >> 1u) & 1u);
	}

	num_b_small = (ac != NULL) ? ac->NEON_NUM_B_PER_SMALL_WORD : 0ul;
	if (num_b_small == 0ul) {
		num_b_small = n_exp / (unsigned long)fftlen;
	}

	/* gwfft_base(j) returns ceil(j*n/FFTLEN), not floor.
	   num_b_in_word = ceil((j+1)*n/FFTLEN) - ceil(j*n/FFTLEN)
	   Use the identity: ceil(a/b) = (a + b - 1) / b for positive integers. */
#if defined(__SIZEOF_INT128__)
	{
		__uint128_t fn = (__uint128_t)fftlen;
		__uint128_t base_lo = ((__uint128_t)word_index * (__uint128_t)n_exp + fn - 1u) / fn;
		__uint128_t base_hi = ((__uint128_t)(word_index + 1u) * (__uint128_t)n_exp + fn - 1u) / fn;
		__uint128_t num_b = base_hi - base_lo;
		return (num_b > (__uint128_t)num_b_small) ? 1 : 0;
	}
#else
	{
		long double fn = (long double)fftlen;
		long double base_lo = ceill(((long double)word_index * (long double)n_exp) / fn);
		long double base_hi = ceill(((long double)(word_index + 1u) * (long double)n_exp) / fn);
		long double num_b = base_hi - base_lo;
		return (num_b > (long double)num_b_small) ? 1 : 0;
	}
#endif
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