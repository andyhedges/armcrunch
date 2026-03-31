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
   with set_fft_value/get_fft_value. */

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

/* Convert ADDIN_OFFSET (byte offset into FFT buffer) to a logical word index
   by finding the word whose addr_offset matches. Returns FFTLEN (invalid) if
   no match is found. */
static size_t arm64_addin_offset_to_word(const struct gwasm_data *ad, uint32_t byte_offset) {
	size_t words, w;
	if (ad == NULL || ad->gwdata == NULL) return 0;
	words = arm64_data_words(ad);
	for (w = 0; w < words; ++w) {
		if ((uint32_t)addr_offset(ad->gwdata, (unsigned long)w) == byte_offset)
			return w;
	}
	return words; /* not found */
}

void arm64_normalize_buffer(struct gwasm_data *asm_data, double *buffer, int errchk, int mulconst_mode) {
	struct gwasm_data *ad = asm_data;
	size_t words;
	size_t word;
	double carry = 0.0;
	double maxerr;
	int use_mulconst;
	double mulconst;
	size_t addin_word;
	double addin_integer;
	double postaddin_integer;

	if (ad == NULL || buffer == NULL) return;

	words = arm64_data_words(ad);
	if (words == 0) return;

	maxerr = ad->MAXERR;
	use_mulconst = (mulconst_mode != 0) || (ad->const_fft != 0);
	mulconst = use_mulconst ? arm64_mulconst(ad) : 1.0;

	/* Convert ADDIN_OFFSET from byte offset to logical word index. */
	addin_word = arm64_addin_offset_to_word(ad, ad->ADDIN_OFFSET);

	/* raw_gwsetaddin pre-multiplies ADDIN_VALUE by weight(word) * FFTLEN/2 / k
	   for the x86 assembly normalization which operates on raw FFT output
	   (still weighted and scaled by FFTLEN/2). Our normalization works in the
	   unweighted integer domain after IFFT scaling and weight removal, so we
	   undo this pre-scaling to recover the actual integer addin value.
	   Formula: integer_addin = ADDIN_VALUE / (weight(addin_word) * FFTLEN/2 / k) */
	addin_integer = 0.0;
	postaddin_integer = ad->POSTADDIN_VALUE;  /* POSTADDIN is not pre-scaled by weight*FFTLEN/2/k */
	if (ad->ADDIN_VALUE != 0.0 && ad->gwdata != NULL && addin_word < words) {
		double k = ad->gwdata->k;
		double fftlen_half = (double)ad->gwdata->FFTLEN * 0.5;
		double weight_at_addin = arm64_gw_forward_weight(ad, addin_word);
		double prescale;
		if (k == 0.0) k = 1.0;  /* guard against divide-by-zero */
		prescale = weight_at_addin * fftlen_half / k;
		if (prescale != 0.0)
			addin_integer = ad->ADDIN_VALUE / prescale;
		else
			addin_integer = ad->ADDIN_VALUE;
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

		/* Optional addin at the configured word (in unweighted integer domain). */
		if (word == addin_word) {
			value += addin_integer;
			value += postaddin_integer;
		}

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