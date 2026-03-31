/*
 * gwnum_arm64_integration.c
 *
 * ARM64 integration hooks for gwnum.c.
 *
 *   int arm64_gwinfo_hook(gwdata, negacyclic)
 *     Called from the patched gwinfo() on ARM64 (see patch_gwnum_for_arm64.sh
 *     iteration 8 which declares: extern int arm64_gwinfo_hook(...) and does
 *     'return 0' immediately after the hook succeeds).
 *     Completely replaces the x86 jmptab traversal by directly setting all
 *     gwdata fields that gwinfo() would normally populate.
 *     Returns 0 on success, GWERROR_TOO_LARGE or GWERROR_VERSION on failure.
 *
 *   void arm64_gwsetup_hook(gwdata)
 *     Called from internal_gwsetup() after asm_data allocation. Installs
 *     all 13 ARM64 entry points and initializes asm_data constants.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "cpuid.h"
#include "gwnum.h"
#include "gwdbldbl.h"
#include "gwtables.h"
#include "arm64_asm_data.h"

/* ARM64 one-pass FFT sizes we support */
static const uint32_t arm64_fft_sizes[] = { 1024, 2048, 4096, 8192, 16384, 32768, 0 };

/* Active ARM64 normalization constants block used by backend routines. */
static arm64_asm_constants arm64_constants_storage;
arm64_asm_constants *arm64_active_asm_constants = NULL;

/* Bits per FFT word for our ARM64 radix-4 implementation */
static double arm64_bits_per_word(uint32_t fftlen, int negacyclic) {
	double base = negacyclic ? 19.35 : 19.60;
	int log2n = 0;
	uint32_t v = fftlen;
	while (v > 1) { v >>= 1; log2n++; }
	return base - (log2n - 10) * 0.05;
}

int arm64_gwinfo_hook(gwhandle *gwdata, int negacyclic)
{
	double k, log2b, n_bits, safety_margin;
	unsigned long b, n;
	int i;
	uint32_t fftlen;
	const char *backend_version;

	if (gwdata == NULL) return GWERROR_INTERNAL;

	backend_version = arm64_gwinfo_backend_version();
	if (strcmp(backend_version, GWNUM_VERSION) != 0) {
		return GWERROR_VERSION;
	}

	k = gwdata->k;
	b = gwdata->b;
	n = gwdata->n;

	log2b = log2((double)b);
	n_bits = log2(k) + log2b * (double)n;

	safety_margin = gwdata->safety_margin + gwdata->polymult_safety_margin;

	fftlen = 0;
	for (i = 0; arm64_fft_sizes[i] != 0; i++) {
		double bpw = arm64_bits_per_word(arm64_fft_sizes[i], negacyclic) - safety_margin;
		if ((double)arm64_fft_sizes[i] * bpw >= n_bits) {
			fftlen = arm64_fft_sizes[i];
			break;
		}
	}

	{
		int skip;
		for (skip = 0; skip < gwdata->larger_fftlen_count && arm64_fft_sizes[i] != 0; skip++) {
			i++;
			if (arm64_fft_sizes[i] != 0) fftlen = arm64_fft_sizes[i];
		}
	}

	if (gwdata->minimum_fftlen > 0) {
		int j;
		for (j = 0; arm64_fft_sizes[j] != 0; j++) {
			if (arm64_fft_sizes[j] >= gwdata->minimum_fftlen) {
				if (arm64_fft_sizes[j] > fftlen)
					fftlen = arm64_fft_sizes[j];
				break;
			}
		}
	}

	if (fftlen == 0) return GWERROR_TOO_LARGE;

	gwdata->cpu_flags = CPU_SSE2;

	gwdata->FFTLEN = fftlen;
	gwdata->PASS2_SIZE = 0;
	gwdata->PASS1_SIZE = 0;
	gwdata->FFT_TYPE = FFT_TYPE_HOME_GROWN;
	gwdata->ARCH = 0;
	gwdata->NO_PREFETCH_FFT = 0;
	gwdata->IN_PLACE_FFT = 0;
	gwdata->PASS1_CACHE_LINES = 2;
	gwdata->FOURKBGAPSIZE = 0;
	gwdata->PASS2GAPSIZE = 0;
	gwdata->SCRATCH_SIZE = 0;
	gwdata->GW_ALIGNMENT = 128;
	gwdata->GW_ALIGNMENT_MOD = 0;
	gwdata->mem_needed = fftlen * (unsigned long)(sizeof(double) * 12);

	{
		double bpw = arm64_bits_per_word(fftlen, negacyclic);
		double actual = n_bits / (double)fftlen;
		double extra = (bpw - actual) * 2.0;
		if (extra < 0.5) extra = 0.5;
		gwdata->EXTRA_BITS = (float)extra;
	}

	gwdata->avg_num_b_per_word = (double)n / (double)fftlen;
	if (b == 2)
		gwdata->NUM_B_PER_SMALL_WORD = (unsigned long)floor((double)n / (double)fftlen);
	else
		gwdata->NUM_B_PER_SMALL_WORD = (unsigned long)floor(log2b * (double)n / (double)fftlen / log2(2.0));

	gwdata->bit_length = n_bits;
	gwdata->fft_max_bits_per_word = arm64_bits_per_word(fftlen, negacyclic);

	gwdata->RATIONAL_FFT = (b == 2 && n % fftlen == 0) ? 1 : 0;

	gwdata->ZERO_PADDED_FFT = 0;
	gwdata->NEGACYCLIC_FFT = negacyclic ? 1 : 0;

	gwdata->datasize = fftlen * sizeof(double);
	gwdata->jmptab = arm64_gwinfo1(negacyclic);

	return 0;
}

void arm64_gwsetup_hook(gwhandle *gwdata)
{
	struct gwasm_data *ad;
	arm64_asm_constants *ac;
	double small_word = 0.0;
	double big_word = 0.0;

	if (gwdata == NULL || gwdata->asm_data == NULL) return;

	arm64_install_gwprocptrs(gwdata->GWPROCPTRS);

	ad = (struct gwasm_data *)gwdata->asm_data;

	ad->gwdata = gwdata;

	ad->FFTLEN = (uint32_t)gwdata->FFTLEN;
	ad->const_fft = 0;
	ad->ADDIN_VALUE = gwdata->asm_addin_value;
	ad->POSTADDIN_VALUE = gwdata->asm_postaddin_value;

	ac = &arm64_constants_storage;
	memset(ac, 0, sizeof(*ac));
	arm64_active_asm_constants = ac;

	ac->NEON_BIGVAL = ad->u.xmm.XMM_BIGVAL[0];
	if (ac->NEON_BIGVAL == 0.0) ac->NEON_BIGVAL = ARM64_DEFAULT_BIGVAL;

	if (gwdata->b == 2) {
		small_word = ldexp(1.0, (int)gwdata->NUM_B_PER_SMALL_WORD);
		big_word   = small_word * 2.0;
	} else {
		small_word = pow((double)gwdata->b, (double)gwdata->NUM_B_PER_SMALL_WORD);
		big_word   = small_word * (double)gwdata->b;
	}

	ac->NEON_SMALL_BASE = small_word;
	ac->NEON_LARGE_BASE = big_word;
	ac->NEON_SMALL_BASE_INV = 1.0 / small_word;
	ac->NEON_LARGE_BASE_INV = 1.0 / big_word;

	ac->NEON_LIMIT_INVERSE[ARM64_WORD_SMALL] = ac->NEON_SMALL_BASE_INV;
	ac->NEON_LIMIT_INVERSE[ARM64_WORD_BIG]   = ac->NEON_LARGE_BASE_INV;

	ac->NEON_LIMIT_BIGMAX[ARM64_WORD_SMALL] = small_word * ac->NEON_BIGVAL - ac->NEON_BIGVAL;
	ac->NEON_LIMIT_BIGMAX[ARM64_WORD_BIG]   = big_word   * ac->NEON_BIGVAL - ac->NEON_BIGVAL;

	ac->NEON_K_HI = ad->u.xmm.XMM_K_HI[0];
	ac->NEON_K_LO = ad->u.xmm.XMM_K_LO[0];
	if (ac->NEON_K_HI == 0.0 && ac->NEON_K_LO == 0.0) {
		double k_hi = floor(gwdata->k / 4294967296.0) * 4294967296.0;
		ac->NEON_K_HI = k_hi;
		ac->NEON_K_LO = gwdata->k - k_hi;
	}

	ac->NEON_MINUS_C = ad->u.xmm.XMM_MINUS_C[0];
	if (ac->NEON_MINUS_C == 0.0 && gwdata->c != 0) {
		ac->NEON_MINUS_C = (double)(-gwdata->c);
	}

	ac->NEON_MULCONST = ad->u.xmm.XMM_MULCONST[0];
	if (ac->NEON_MULCONST == 0.0) {
		ac->NEON_MULCONST = (double)gwdata->mulbyconst;
	}

	ac->NEON_NORM012_FF = ad->u.xmm.XMM_NORM012_FF[0];
	if (ac->NEON_NORM012_FF == 0.0) {
		if (gwdata->k == 1.0)
			ac->NEON_NORM012_FF = (double)gwdata->FFTLEN * 0.5;
		else
			ac->NEON_NORM012_FF = (double)gwdata->FFTLEN * 0.5 / gwdata->k;
	}

	ac->NEON_MAXERR = ad->u.xmm.XMM_MAXERR[0];
	ad->MAXERR = ac->NEON_MAXERR;

	ac->NEON_B = (double)gwdata->b;
	ac->NEON_ONE_OVER_B = (gwdata->b != 0) ? 1.0 / (double)gwdata->b : 0.0;

	ac->NEON_CARRIES_ROUTINE = NULL;
	ac->NEON_PASS2_ROUTINE = NULL;

	ac->NEON_N = gwdata->n;
	ac->NEON_NUM_B_PER_SMALL_WORD = gwdata->NUM_B_PER_SMALL_WORD;

	/* Temporary diagnostic output */
	{
		long val_out = 0;
		int err;
		gwnum test_g;

		fprintf(stderr, "[ARM64 SETUP] FFTLEN=%lu b=%lu n=%lu c=%ld k=%.1f\n",
			gwdata->FFTLEN, gwdata->b, gwdata->n, gwdata->c, gwdata->k);
		fprintf(stderr, "[ARM64 SETUP] FFT_TYPE=%d RATIONAL=%d NEGACYCLIC=%d ZERO_PADDED=%d\n",
			gwdata->FFT_TYPE, gwdata->RATIONAL_FFT, gwdata->NEGACYCLIC_FFT, gwdata->ZERO_PADDED_FFT);
		fprintf(stderr, "[ARM64 SETUP] cpu_flags=0x%x NUM_B_PER_SMALL=%lu avg_b_per_word=%.6f\n",
			gwdata->cpu_flags, gwdata->NUM_B_PER_SMALL_WORD, gwdata->avg_num_b_per_word);
		fprintf(stderr, "[ARM64 SETUP] datasize=%lu EXTRA_BITS=%.4f bit_length=%.2f\n",
			gwdata->datasize, (double)gwdata->EXTRA_BITS, gwdata->bit_length);
		fprintf(stderr, "[ARM64 SETUP] small_base=%.1f big_base=%.1f\n", ac->NEON_SMALL_BASE, ac->NEON_LARGE_BASE);
		fprintf(stderr, "[ARM64 SETUP] PASS1_SIZE=%lu PASS2_SIZE=%lu FOURKBGAPSIZE=%ld\n",
			gwdata->PASS1_SIZE, gwdata->PASS2_SIZE, gwdata->FOURKBGAPSIZE);
		fprintf(stderr, "[ARM64 SETUP] asm_data->FFTLEN=%u B_IS_2=%d RATIONAL_FFT=%d\n",
			ad->FFTLEN, ad->B_IS_2, ad->RATIONAL_FFT);
		fprintf(stderr, "[ARM64 SETUP] norm_col_mults=%p norm_grp_mults=%p sincos1=%p\n",
			(void*)ad->norm_col_mults, (void*)ad->norm_grp_mults, (void*)ad->sincos1);
		fprintf(stderr, "[ARM64 SETUP] XMM_BIGVAL=%.6g XMM_LIMIT_INV[0]=%.10g XMM_LIMIT_INV[1]=%.10g\n",
			ad->u.xmm.XMM_BIGVAL[0], ad->u.xmm.XMM_LIMIT_INVERSE[0], ad->u.xmm.XMM_LIMIT_INVERSE[1]);
		fprintf(stderr, "[ARM64 SETUP] XMM_LIMIT_BIGMAX[0]=%.6g XMM_LIMIT_BIGMAX[1]=%.6g\n",
			ad->u.xmm.XMM_LIMIT_BIGMAX[0], ad->u.xmm.XMM_LIMIT_BIGMAX[1]);

		/* addr_offset test */
		fprintf(stderr, "[ARM64 SETUP] addr_offset[0..7]: ");
		{
			unsigned long j;
			for (j = 0; j < 8 && j < gwdata->FFTLEN; j++)
				fprintf(stderr, "%lu ", addr_offset(gwdata, j));
		}
		fprintf(stderr, "\n");

		/* Round-trip test */
		test_g = gwalloc(gwdata);
		if (test_g != NULL) {
			set_fft_value(gwdata, test_g, 0, 3);
			err = get_fft_value(gwdata, test_g, 0, &val_out);
			fprintf(stderr, "[ARM64 SETUP] Round-trip: set_fft_value(0,3) -> get_fft_value(0) = %ld (err=%d)\n",
				val_out, err);
			if (val_out != 3) {
				double raw = *addr(gwdata, test_g, 0);
				double weight = gwfft_weight_sloppy(gwdata->dd_data, 0);
				double inv_weight = gwfft_weight_inverse_sloppy(gwdata->dd_data, 0);
				fprintf(stderr, "[ARM64 SETUP] MISMATCH: raw=%.15g weight=%.15g inv_weight=%.15g raw*inv=%.15g\n",
					raw, weight, inv_weight, raw * inv_weight);
			}
			set_fft_value(gwdata, test_g, 1, 1);
			err = get_fft_value(gwdata, test_g, 1, &val_out);
			fprintf(stderr, "[ARM64 SETUP] Round-trip: set_fft_value(1,1) -> get_fft_value(1) = %ld (err=%d)\n",
				val_out, err);
			if (val_out != 1) {
				double raw = *addr(gwdata, test_g, 1);
				double weight = gwfft_weight_sloppy(gwdata->dd_data, 1);
				double inv_weight = gwfft_weight_inverse_sloppy(gwdata->dd_data, 1);
				fprintf(stderr, "[ARM64 SETUP] MISMATCH w1: raw=%.15g weight=%.15g inv_weight=%.15g raw*inv=%.15g\n",
					raw, weight, inv_weight, raw * inv_weight);
			}

			/* Weight comparison: gwnum vs ARM64 for first 4 words */
			{
				unsigned long j;
				fprintf(stderr, "[ARM64 SETUP] Weight comparison (gwnum vs arm64):\n");
				for (j = 0; j < 4; j++) {
					double gw_fwd = gwfft_weight_sloppy(gwdata->dd_data, j);
					double gw_inv = gwfft_weight_inverse_sloppy(gwdata->dd_data, j);
					double arm_fwd = arm64_forward_weight_at(ad, j);
					double arm_inv = arm64_inverse_weight_at(ad, j);
					fprintf(stderr, "  word %lu: gw_fwd=%.10g arm_fwd=%.10g gw_inv=%.10g arm_inv=%.10g\n",
						j, gw_fwd, arm_fwd, gw_inv, arm_inv);
				}
			}

			/* Big-word comparison: gwnum vs ARM64 for first 8 words */
			{
				unsigned long j;
				fprintf(stderr, "[ARM64 SETUP] Big-word comparison (gwnum vs arm64): ");
				for (j = 0; j < 8 && j < gwdata->FFTLEN; j++) {
					int gw_big = is_big_word(gwdata, j);
					int arm_big = arm64_is_big_word(ad, j);
					fprintf(stderr, "w%lu(%d/%d) ", j, gw_big, arm_big);
				}
				fprintf(stderr, "\n");
			}

			/* gwsquare2 functional test: store 3, square, read back, expect 9 */
			{
				gwnum sq_g = gwalloc(gwdata);
				if (sq_g != NULL) {
					long sq_val = 0;
					int sq_err;
					dbltogw(gwdata, 3.0, sq_g);
					gwsquare2(gwdata, sq_g, sq_g, 0);
					sq_err = gw_test_for_error(gwdata);
					{
						giant tmp_giant = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						int conv_err = gwtogiant(gwdata, sq_g, tmp_giant);
						if (conv_err >= 0 && tmp_giant->sign == 1) {
							sq_val = (long)tmp_giant->n[0];
						} else if (conv_err >= 0 && tmp_giant->sign == 0) {
							sq_val = 0;
						} else {
							sq_val = -999;
						}
						fprintf(stderr, "[ARM64 SETUP] gwsquare2 test: 3^2 = %ld (gwerror=%d, conv=%d, sign=%d)\n",
							sq_val, sq_err, conv_err, tmp_giant->sign);
						if (sq_val != 9) {
							unsigned long j;
							fprintf(stderr, "[ARM64 SETUP] squared result words[0..7]: ");
							for (j = 0; j < 8 && j < gwdata->FFTLEN; j++) {
								long wv = 0;
								get_fft_value(gwdata, sq_g, j, &wv);
								fprintf(stderr, "%ld ", wv);
							}
							fprintf(stderr, "\n");
						}
						pushg(&gwdata->gdata, 1);
					}

					/* gwsmallmul(1.0) round-trip test: store 3, multiply by 1, expect 3 */
					dbltogw(gwdata, 3.0, sq_g);
					gwsmallmul(gwdata, 1.0, sq_g);
					{
						giant tmp_giant = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						int conv_err = gwtogiant(gwdata, sq_g, tmp_giant);
						long muls_val = -999;
						if (conv_err >= 0 && tmp_giant->sign >= 1) muls_val = (long)tmp_giant->n[0];
						else if (conv_err >= 0 && tmp_giant->sign == 0) muls_val = 0;
						fprintf(stderr, "[ARM64 SETUP] gwsmallmul(1) test: 3*1 = %ld\n", muls_val);
						pushg(&gwdata->gdata, 1);
					}

					/* gwadd3o test: store 3+3, expect 6 */
					{
						gwnum g2 = gwalloc(gwdata);
						if (g2 != NULL) {
							dbltogw(gwdata, 3.0, sq_g);
							dbltogw(gwdata, 3.0, g2);
							gwadd3o(gwdata, sq_g, g2, sq_g, GWADD_FORCE_NORMALIZE);
							{
								giant tmp_giant = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
								int conv_err = gwtogiant(gwdata, sq_g, tmp_giant);
								long add_val = -999;
								if (conv_err >= 0 && tmp_giant->sign >= 1) add_val = (long)tmp_giant->n[0];
								else if (conv_err >= 0 && tmp_giant->sign == 0) add_val = 0;
								fprintf(stderr, "[ARM64 SETUP] gwadd3o test: 3+3 = %ld\n", add_val);
								pushg(&gwdata->gdata, 1);
							}
							gwfree(gwdata, g2);
						}
					}

					/* 5-iteration squaring test: 3 -> 9 -> 81 -> 6561 -> 43046721 -> 1853020188851841 */
					{
						int iter;
						dbltogw(gwdata, 3.0, sq_g);
						for (iter = 0; iter < 5; iter++) {
							gwsquare2(gwdata, sq_g, sq_g, 0);
						}
						{
							giant tmp_giant = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
							int conv_err = gwtogiant(gwdata, sq_g, tmp_giant);
							uint64_t result = 0;
							if (conv_err >= 0 && tmp_giant->sign >= 1) {
								result = (uint64_t)tmp_giant->n[0];
								if (tmp_giant->sign >= 2) result |= ((uint64_t)tmp_giant->n[1]) << 32;
							}
							fprintf(stderr, "[ARM64 SETUP] 5-iter square test: 3^32 = %llu (expected 1853020188851841, sign=%d)\n",
								(unsigned long long)result, tmp_giant->sign);
							pushg(&gwdata->gdata, 1);
						}
					}

					gwfree(gwdata, sq_g);
				}
			}

			/* Disable careful_count to prevent gwmul3_carefully from being called.
			   gwmul3_carefully uses gwaddsub4o + gwmuladd4 with FMA state that
			   our ARM64 backend does not yet fully support. */
			gwdata->careful_count = 0;

			gwfree(gwdata, test_g);
		} else {
			fprintf(stderr, "[ARM64 SETUP] gwalloc returned NULL!\n");
		}
	}
}