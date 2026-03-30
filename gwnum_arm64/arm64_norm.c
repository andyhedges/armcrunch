#include "arm64_asm_data.h"
#include "gwtables.h"

#include <math.h>
#include <stddef.h>

static double arm64_carry_quotient(double value, double base, double inv_base) {
	if (base == 0.0) return 0.0;
	if (value >= 0.0) return floor(value * inv_base);
	return ceil(value * inv_base);
}

void arm64_normalize_buffer(struct gwasm_data *asm_data, double *buffer, int errchk, int mulconst_mode) {
	struct gwasm_data *ad = asm_data;
	size_t complex_len;
	size_t words;
	size_t word;
	double carry = 0.0;
	double maxerr;
	int use_mulconst;
	double mulconst;
	size_t addin_offset;
	double *carries;

	if (ad == NULL || buffer == NULL) return;

	complex_len = arm64_complex_len(ad);
	if (complex_len == 0) return;
	words = complex_len * 2u;

	maxerr = ad->MAXERR;
	use_mulconst = (mulconst_mode != 0) || (ad->const_fft != 0);
	mulconst = use_mulconst ? arm64_mulconst(ad) : 1.0;
	addin_offset = (size_t)ad->ADDIN_OFFSET;
	carries = (double *)ad->carries;

	for (word = 0; word < words; ++word) {
		size_t complex_index = word >> 1u;
		int big_word = (int)(complex_index & 1u);
		double base = arm64_word_base(ad, big_word);
		double inv_base = arm64_word_base_inverse(ad, big_word);
		double limit = arm64_word_limit(ad, big_word);
		double value = buffer[word];
		double rounded;
		double carry_out = 0.0;

		/* Undo IBDWT weight (inverse weight table). */
		value *= arm64_inverse_weight_at(ad, complex_index);

		if (use_mulconst) value *= mulconst;

		if (word == addin_offset) {
			value += ad->ADDIN_VALUE;
			value += ad->POSTADDIN_VALUE;
		}

		rounded = nearbyint(value);

		if (errchk) {
			double err = fabs(value - rounded);
			if (err > maxerr) maxerr = err;
		}

		rounded += carry;

		if (rounded > limit || rounded < -limit) {
			carry_out = arm64_carry_quotient(rounded, base, inv_base);
			rounded -= carry_out * base;
		}

		carry = carry_out;
		buffer[word] = rounded;

		if (carries != NULL) {
			carries[word] = carry;
		}
	}

	if (carry != 0.0) {
		buffer[0] += carry;
		if (carries != NULL) carries[0] += carry;
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