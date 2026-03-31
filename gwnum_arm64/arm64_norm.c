#include "arm64_asm_data.h"
#include "gwnum.h"
#include "gwdbldbl.h"
#include "gwtables.h"

#include <math.h>
#include <stddef.h>

static inline size_t arm64_word_offset_bytes(const struct gwasm_data *ad, size_t word) {
	if (ad != NULL && ad->gwdata != NULL) {
		return (size_t)addr_offset(ad->gwdata, (unsigned long)word);
	}
	return word * sizeof(double);
}

static inline double arm64_load_scrambled_word(const struct gwasm_data *ad, const double *buffer, size_t word) {
	const char *ptr = (const char *)buffer + arm64_word_offset_bytes(ad, word);
	return *(const double *)ptr;
}

static inline void arm64_store_scrambled_word(const struct gwasm_data *ad, double *buffer, size_t word, double value) {
	char *ptr = (char *)buffer + arm64_word_offset_bytes(ad, word);
	*(double *)ptr = value;
}

/* Use gwnum's own double-double precision weight functions for exact compatibility
   with set_fft_value/get_fft_value. Our first-principles pow() computation has
   insufficient precision, causing errors that compound over iterations. */

static inline double arm64_gw_forward_weight(const struct gwasm_data *ad, size_t word) {
	if (ad != NULL && ad->gwdata != NULL && ad->gwdata->dd_data != NULL) {
		if (ad->RATIONAL_FFT) return 1.0;
		return gwfft_weight_sloppy(ad->gwdata->dd_data, (unsigned long)word);
	}
	return arm64_forward_weight_at(ad, word);
}

static inline double arm64_gw_inverse_weight(const struct gwasm_data *ad, size_t word) {
	if (ad != NULL && ad->gwdata != NULL && ad->gwdata->dd_data != NULL) {
		if (ad->RATIONAL_FFT) return 1.0;
		return gwfft_weight_inverse_sloppy(ad->gwdata->dd_data, (unsigned long)word);
	}
	return arm64_inverse_weight_at(ad, word);
}

void arm64_normalize_buffer(struct gwasm_data *asm_data, double *buffer, int errchk, int mulconst_mode) {
	struct gwasm_data *ad = asm_data;
	size_t words;
	size_t word;
	double carry = 0.0;
	double maxerr;
	int use_mulconst;
	double mulconst;
	size_t addin_offset;

	if (ad == NULL || buffer == NULL) return;

	words = arm64_data_words(ad);
	if (words == 0) return;

	maxerr = ad->MAXERR;
	use_mulconst = (mulconst_mode != 0) || (ad->const_fft != 0);
	mulconst = use_mulconst ? arm64_mulconst(ad) : 1.0;
	addin_offset = (size_t)ad->ADDIN_OFFSET;

	/* Apply ADDIN_VALUE directly to the physical FFT buffer at ADDIN_OFFSET,
	   before any per-word unweighting and carry propagation. */
	if (ad->ADDIN_VALUE != 0.0 || ad->POSTADDIN_VALUE != 0.0) {
		double *addin_ptr = (double *)((char *)buffer + addin_offset);
		*addin_ptr += ad->ADDIN_VALUE;
		*addin_ptr += ad->POSTADDIN_VALUE;
	}

	for (word = 0; word < words; ++word) {
		int big_word = arm64_is_big_word(ad, word);
		double base = arm64_word_base(ad, big_word);
		double inv_base = arm64_word_base_inverse(ad, big_word);
		double value = arm64_load_scrambled_word(ad, buffer, word);
		double rounded;
		double carry_out;
		double digit;
		double stored;

		/* Undo IBDWT weight using gwnum's own double-double precision function. */
		value *= arm64_gw_inverse_weight(ad, word);

		/* Optional mulconst path. */
		if (use_mulconst) value *= mulconst;

		/* Round to nearest integer and track roundoff error. */
		rounded = nearbyint(value);
		if (errchk) {
			double err = fabs(value - rounded);
			if (err > maxerr) maxerr = err;
		}

		/* Add incoming carry and ALWAYS extract outgoing carry. */
		rounded += carry;
		if (base != 0.0) {
			carry_out = nearbyint(rounded * inv_base);
		} else {
			carry_out = 0.0;
		}

		/* Balanced digit after carry extraction. */
		digit = rounded - carry_out * base;

		/* Re-apply forward weight using gwnum's own function for exact match
		   with what set_fft_value/get_fft_value expect. */
		stored = digit * arm64_gw_forward_weight(ad, word);
		arm64_store_scrambled_word(ad, buffer, word, stored);

		carry = carry_out;
	}

	/* Wraparound carry: multiply by -c and add into word 0 (unweighted domain). */
	if (carry != 0.0) {
		const arm64_asm_constants *ac = arm64_constants(ad);
		double minus_c = 1.0;
		double wrap_carry;
		double w0;

		if (ac != NULL && ac->NEON_MINUS_C != 0.0) {
			minus_c = ac->NEON_MINUS_C;
		} else if (ad->gwdata != NULL && ad->gwdata->c != 0) {
			minus_c = (double)(-ad->gwdata->c);
		}

		wrap_carry = carry * minus_c;

		w0 = arm64_load_scrambled_word(ad, buffer, 0u);
		w0 *= arm64_gw_inverse_weight(ad, 0u);
		w0 += wrap_carry;
		w0 *= arm64_gw_forward_weight(ad, 0u);
		arm64_store_scrambled_word(ad, buffer, 0u, w0);
	}

	if (errchk && maxerr > ad->MAXERR) {
		ad->MAXERR = maxerr;
		if (arm64_active_asm_constants != NULL) {
			arm64_active_asm_constants->NEON_MAXERR = maxerr;
		}
	}
}

void arm64_norm_plain(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 0, 0);
}

void arm64_norm_errchk(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 1, 0);
}

void arm64_norm_mulconst(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 0, 1);
}

void arm64_norm_errchk_mulconst(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 1, 1);
}