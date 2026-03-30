#include "arm64_asm_data.h"
#include "gwtables.h"

#include <math.h>
#include <stdio.h>
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

static double arm64_carry_quotient(double value, double base, double inv_base) {
	if (base == 0.0) return 0.0;
	if (value >= 0.0) return floor(value * inv_base);
	return ceil(value * inv_base);
}

void arm64_normalize_buffer(struct gwasm_data *asm_data, double *buffer, int errchk, int mulconst_mode) {
	static int norm_debug = 0;
	int do_debug = (norm_debug < 2);
	struct gwasm_data *ad = asm_data;
	size_t words;
	size_t word;
	double carry = 0.0;
	double maxerr;
	int use_mulconst;
	double mulconst;
	size_t addin_offset;
	double *carries;

	if (ad == NULL || buffer == NULL) return;

	words = arm64_data_words(ad);
	if (words == 0) return;

	maxerr = ad->MAXERR;
	use_mulconst = (mulconst_mode != 0) || (ad->const_fft != 0);
	mulconst = use_mulconst ? arm64_mulconst(ad) : 1.0;
	addin_offset = (size_t)ad->ADDIN_OFFSET;
	carries = (double *)ad->carries;

	if (do_debug) {
		size_t k;
		fprintf(stderr, "[ARM64 NORM] call #%d words=%zu mulconst=%d addin_off=%zu base[s]=%.4g base[b]=%.4g limit[s]=%.4g limit[b]=%.4g\n",
			norm_debug, words, use_mulconst, addin_offset,
			arm64_word_base(ad, 0), arm64_word_base(ad, 1),
			arm64_word_limit(ad, 0), arm64_word_limit(ad, 1));
		fprintf(stderr, "[ARM64 NORM] post-FFT scrambled[0..7]: ");
		for (k = 0; k < 8 && k < words; k++)
			fprintf(stderr, "%.6f ", arm64_load_scrambled_word(ad, buffer, k));
		fprintf(stderr, "\n[ARM64 NORM] inv-weighted[0..7]:      ");
		for (k = 0; k < 8 && k < words; k++) {
			double v = arm64_load_scrambled_word(ad, buffer, k) * arm64_inverse_weight_at(ad, k);
			fprintf(stderr, "%.6f ", v);
		}
		fprintf(stderr, "\n[ARM64 NORM] is_big[0..7]:            ");
		for (k = 0; k < 8 && k < words; k++)
			fprintf(stderr, "%d ", arm64_is_big_word(ad, k));
		fprintf(stderr, "\n");
		norm_debug++;
	}

	for (word = 0; word < words; ++word) {
		int big_word = arm64_is_big_word(ad, word);
		double base = arm64_word_base(ad, big_word);
		double inv_base = arm64_word_base_inverse(ad, big_word);
		double limit = arm64_word_limit(ad, big_word);
		double value = arm64_load_scrambled_word(ad, buffer, word);
		double rounded;
		double carry_out = 0.0;

		value *= arm64_inverse_weight_at(ad, word);

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

		if (do_debug && word < 8) {
			fprintf(stderr, "[ARM64 NORM]  w%zu: raw=%.6f inv_w=%.6f rounded=%.1f carry_in=%.1f",
				word, arm64_load_scrambled_word(ad, buffer, word),
				value, rounded, carry);
		}

		rounded += carry;

		if (rounded > limit || rounded < -limit) {
			carry_out = arm64_carry_quotient(rounded, base, inv_base);
			rounded -= carry_out * base;
		}

		if (do_debug && word < 8) {
			fprintf(stderr, " after_carry=%.1f carry_out=%.1f big=%d\n",
				rounded, carry_out, big_word);
		}

		carry = carry_out;
		arm64_store_scrambled_word(ad, buffer, word, rounded);

		if (carries != NULL) {
			carries[word] = carry;
		}
	}

	if (carry != 0.0) {
		double wrapped = arm64_load_scrambled_word(ad, buffer, 0u) + carry;
		arm64_store_scrambled_word(ad, buffer, 0u, wrapped);
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