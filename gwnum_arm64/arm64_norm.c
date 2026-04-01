#include "arm64_asm_data.h"
#include "gwnum.h"
#include "gwdbldbl.h"
#include "gwtables.h"

#include <math.h>
#include <stdio.h>
#include <stddef.h>

typedef struct arm64_word_cache {
	size_t fftlen;
	size_t *byte_offsets;
	double *fwd_weights;
	double *inv_weights;
	int *big_words;
} arm64_word_cache;

extern int arm64_ensure_word_cache(const struct gwasm_data *ad);
extern const arm64_word_cache *arm64_get_word_cache(void);

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

/* Convert ADDIN_OFFSET (byte offset into FFT buffer) to a logical word index.
   Returns FFTLEN (invalid) if no match is found. */
static size_t arm64_addin_offset_to_word(const struct gwasm_data *ad, const arm64_word_cache *cache, uint32_t byte_offset) {
	size_t words, w;
	uint32_t candidate_offset;

	if (ad == NULL) return 0;
	words = arm64_data_words(ad);

	if (cache != NULL && cache->fftlen == words && cache->byte_offsets != NULL) {
		for (w = 0; w < words; ++w) {
			if ((uint32_t)cache->byte_offsets[w] == byte_offset)
				return w;
		}

		if (byte_offset >= 8) {
			candidate_offset = byte_offset - 8;
			for (w = 0; w < words; ++w) {
				if ((uint32_t)cache->byte_offsets[w] == candidate_offset)
					return w;
			}
		}

		candidate_offset = byte_offset + 8;
		if (candidate_offset > byte_offset) {
			for (w = 0; w < words; ++w) {
				if ((uint32_t)cache->byte_offsets[w] == candidate_offset)
					return w;
			}
		}

		return words;
	}

	if (ad->gwdata == NULL) return words;
	for (w = 0; w < words; ++w) {
		if ((uint32_t)addr_offset(ad->gwdata, (unsigned long)w) == byte_offset)
			return w;
	}

	if (byte_offset >= 8) {
		candidate_offset = byte_offset - 8;
		for (w = 0; w < words; ++w) {
			if ((uint32_t)addr_offset(ad->gwdata, (unsigned long)w) == candidate_offset)
				return w;
		}
	}

	candidate_offset = byte_offset + 8;
	if (candidate_offset > byte_offset) {
		for (w = 0; w < words; ++w) {
			if ((uint32_t)addr_offset(ad->gwdata, (unsigned long)w) == candidate_offset)
				return w;
		}
	}

	return words;
}

void arm64_normalize_buffer(struct gwasm_data *asm_data, double *buffer, int errchk, int mulconst_mode, int post_fft) {
	struct gwasm_data *ad = asm_data;
	const arm64_word_cache *cache = NULL;
	const size_t *byte_offsets = NULL;
	const double *fwd_weights = NULL;
	const double *inv_weights = NULL;
	const int *big_words = NULL;
	char *buffer_bytes = NULL;
	int use_cached_tables = 0;
	size_t words;
	size_t word;
	double carry = 0.0;
	double maxerr;
	int use_mulconst;
	double mulconst;
	double k_factor;
	size_t addin_word;
	double addin_integer;
	double postaddin_integer;

	if (ad == NULL || buffer == NULL) return;

	words = arm64_data_words(ad);
	if (words == 0) return;

	if (arm64_ensure_word_cache(ad)) {
		cache = arm64_get_word_cache();
		if (cache != NULL &&
			cache->fftlen == words &&
			cache->byte_offsets != NULL &&
			cache->fwd_weights != NULL &&
			cache->inv_weights != NULL &&
			cache->big_words != NULL) {
			byte_offsets = cache->byte_offsets;
			fwd_weights = cache->fwd_weights;
			inv_weights = cache->inv_weights;
			big_words = cache->big_words;
			buffer_bytes = (char *)buffer;
			use_cached_tables = 1;
		}
	}

	maxerr = ad->MAXERR;
	use_mulconst = (mulconst_mode != 0) || (ad->const_fft != 0);
	mulconst = use_mulconst ? arm64_mulconst(ad) : 1.0;
	k_factor = (post_fft && ad->gwdata != NULL && ad->gwdata->k > 1.0) ? ad->gwdata->k : 1.0;

	/* Convert ADDIN_OFFSET from byte offset to logical word index. */
	addin_word = arm64_addin_offset_to_word(ad, use_cached_tables ? cache : NULL, ad->ADDIN_OFFSET);

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
		double weight_at_addin = use_cached_tables ? fwd_weights[addin_word] : arm64_gw_forward_weight(ad, addin_word);
		double prescale;
		if (k == 0.0) k = 1.0;  /* guard against divide-by-zero */
		prescale = weight_at_addin * fftlen_half / k;
		if (prescale != 0.0)
			addin_integer = ad->ADDIN_VALUE / prescale;
		else
			addin_integer = ad->ADDIN_VALUE;
	}

	if (ad->ADDIN_VALUE != 0.0 && addin_word >= words) {
		static int addin_warned = 0;
		if (!addin_warned) {
			fprintf(stderr, "[ARM64 NORM] ADDIN LOST: ADDIN_OFFSET=%u not found in %zu words, ADDIN_VALUE=%.6g k=%.1f\n",
				(unsigned)ad->ADDIN_OFFSET, words, ad->ADDIN_VALUE,
				ad->gwdata ? ad->gwdata->k : 0.0);
			addin_warned = 1;
		}
	}

	if (use_cached_tables) {
		double small_base = arm64_word_base(ad, 0);
		double big_base_val = arm64_word_base(ad, 1);
		double small_inv_base = arm64_word_base_inverse(ad, 0);
		double big_inv_base = arm64_word_base_inverse(ad, 1);

		for (word = 0; word < words; ++word) {
			double base = big_words[word] ? big_base_val : small_base;
			double inv_base = big_words[word] ? big_inv_base : small_inv_base;
			double value = *(double *)(buffer_bytes + byte_offsets[word]);
			double rounded;
			double carry_out;
			double digit;
			double stored;

			value *= inv_weights[word];

			if (use_mulconst) value *= mulconst;
			value *= k_factor;

			if (word == addin_word) {
				value += addin_integer;
				value += postaddin_integer;
			}

			rounded = nearbyint(value);
			if (errchk) {
				double err = fabs(value - rounded);
				if (err > maxerr) maxerr = err;
			}

			rounded += carry;
			if (base != 0.0) {
				carry_out = nearbyint(rounded * inv_base);
			} else {
				carry_out = 0.0;
			}

			digit = rounded - carry_out * base;
			stored = digit * fwd_weights[word];
			*(double *)(buffer_bytes + byte_offsets[word]) = stored;

			carry = carry_out;
		}
	} else {
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
			value *= k_factor;

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
	}

	/* Wraparound carry for k*b^n+c:
	   carry at b^n maps through b^n ≡ -c/k (mod k*b^n+c).
	   Split carry*(-c) into quotient*k + remainder, inject quotient at word 0
	   with wrap propagation, then inject accumulated remainder at top word. */
	if (carry != 0.0) {
		const arm64_asm_constants *ac = arm64_constants(ad);
		double minus_c = 1.0;
		double k_val = 1.0;
		double carry_times_minus_c;
		double quotient_carry;
		double remainder;
		double wrap_carry;
		double remainder_accum;
		int wrap_pass;
		const int max_wrap_passes = 10;

		if (ad->gwdata != NULL && ad->gwdata->c != 0) {
			minus_c = (double)(-ad->gwdata->c);
		} else if (ac != NULL && ac->NEON_MINUS_C != 0.0) {
			minus_c = ac->NEON_MINUS_C;
		}

		if (ad->gwdata != NULL && ad->gwdata->k != 0.0) {
			k_val = ad->gwdata->k;
		}
		if (k_val <= 1.0) k_val = 1.0;

		carry_times_minus_c = carry * minus_c;
		if (k_val == 1.0) {
			quotient_carry = carry_times_minus_c;
			remainder = 0.0;
		} else {
			quotient_carry = floor(carry_times_minus_c / k_val);
			remainder = carry_times_minus_c - quotient_carry * k_val;
		}

		wrap_carry = quotient_carry;
		remainder_accum = remainder;

		for (wrap_pass = 0; wrap_pass < max_wrap_passes && wrap_carry != 0.0; ++wrap_pass) {
			double pass_carry = wrap_carry;
			size_t wrap_word;

			if (use_cached_tables) {
				double small_base = arm64_word_base(ad, 0);
				double big_base_val = arm64_word_base(ad, 1);
				double small_inv_base = arm64_word_base_inverse(ad, 0);
				double big_inv_base = arm64_word_base_inverse(ad, 1);

				for (wrap_word = 0; wrap_word < words && pass_carry != 0.0; ++wrap_word) {
					double base = big_words[wrap_word] ? big_base_val : small_base;
					double inv_base = big_words[wrap_word] ? big_inv_base : small_inv_base;
					double value = *(double *)(buffer_bytes + byte_offsets[wrap_word]);
					double rounded;
					double carry_out;
					double digit;
					double stored;

					value *= inv_weights[wrap_word];
					rounded = nearbyint(value);
					if (errchk) {
						double err = fabs(value - rounded);
						if (err > maxerr) maxerr = err;
					}
					rounded += pass_carry;
					if (base != 0.0) {
						carry_out = nearbyint(rounded * inv_base);
					} else {
						carry_out = 0.0;
					}

					digit = rounded - carry_out * base;
					stored = digit * fwd_weights[wrap_word];
					*(double *)(buffer_bytes + byte_offsets[wrap_word]) = stored;

					pass_carry = carry_out;
				}
			} else {
				for (wrap_word = 0; wrap_word < words && pass_carry != 0.0; ++wrap_word) {
					int big_word = arm64_is_big_word(ad, wrap_word);
					double base = arm64_word_base(ad, big_word);
					double inv_base = arm64_word_base_inverse(ad, big_word);
					double value = arm64_load_scrambled_word(ad, buffer, wrap_word);
					double rounded;
					double carry_out;
					double digit;
					double stored;

					value *= arm64_gw_inverse_weight(ad, wrap_word);
					rounded = nearbyint(value);
					if (errchk) {
						double err = fabs(value - rounded);
						if (err > maxerr) maxerr = err;
					}
					rounded += pass_carry;
					if (base != 0.0) {
						carry_out = nearbyint(rounded * inv_base);
					} else {
						carry_out = 0.0;
					}

					digit = rounded - carry_out * base;
					stored = digit * arm64_gw_forward_weight(ad, wrap_word);
					arm64_store_scrambled_word(ad, buffer, wrap_word, stored);

					pass_carry = carry_out;
				}
			}

			if (pass_carry == 0.0) {
				wrap_carry = 0.0;
			} else {
				carry_times_minus_c = pass_carry * minus_c;
				if (k_val == 1.0) {
					wrap_carry = carry_times_minus_c;
				} else {
					double pass_remainder;
					wrap_carry = floor(carry_times_minus_c / k_val);
					pass_remainder = carry_times_minus_c - wrap_carry * k_val;
					remainder_accum += pass_remainder;
				}
			}
		}

		if (remainder_accum != 0.0) {
			size_t top_word = words - 1;

			if (use_cached_tables) {
				double small_base = arm64_word_base(ad, 0);
				double big_base_val = arm64_word_base(ad, 1);
				double small_inv_base = arm64_word_base_inverse(ad, 0);
				double big_inv_base = arm64_word_base_inverse(ad, 1);
				double base = big_words[top_word] ? big_base_val : small_base;
				double inv_base = big_words[top_word] ? big_inv_base : small_inv_base;
				double value = *(double *)(buffer_bytes + byte_offsets[top_word]);
				double rounded;
				double carry_out;
				double digit;
				double stored;

				value *= inv_weights[top_word];
				rounded = nearbyint(value);
				if (errchk) {
					double err = fabs(value - rounded);
					if (err > maxerr) maxerr = err;
				}
				rounded += remainder_accum;
				if (base != 0.0) {
					carry_out = nearbyint(rounded * inv_base);
				} else {
					carry_out = 0.0;
				}

				digit = rounded - carry_out * base;
				stored = digit * fwd_weights[top_word];
				*(double *)(buffer_bytes + byte_offsets[top_word]) = stored;
			} else {
				int big_word = arm64_is_big_word(ad, top_word);
				double base = arm64_word_base(ad, big_word);
				double inv_base = arm64_word_base_inverse(ad, big_word);
				double value = arm64_load_scrambled_word(ad, buffer, top_word);
				double rounded;
				double carry_out;
				double digit;
				double stored;

				value *= arm64_gw_inverse_weight(ad, top_word);
				rounded = nearbyint(value);
				if (errchk) {
					double err = fabs(value - rounded);
					if (err > maxerr) maxerr = err;
				}
				rounded += remainder_accum;
				if (base != 0.0) {
					carry_out = nearbyint(rounded * inv_base);
				} else {
					carry_out = 0.0;
				}

				digit = rounded - carry_out * base;
				stored = digit * arm64_gw_forward_weight(ad, top_word);
				arm64_store_scrambled_word(ad, buffer, top_word, stored);
			}
		}
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
	arm64_normalize_buffer(asm_data, dest, 0, 0, 1);
}

void arm64_norm_errchk(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 1, 0, 1);
}

void arm64_norm_mulconst(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 0, 1, 1);
}

void arm64_norm_errchk_mulconst(struct gwasm_data *asm_data) {
	struct gwasm_data *ad = asm_data;
	double *dest;
	if (ad == NULL) return;
	dest = (double *)ad->DESTARG;
	if (dest == NULL) return;
	arm64_normalize_buffer(asm_data, dest, 1, 1, 1);
}