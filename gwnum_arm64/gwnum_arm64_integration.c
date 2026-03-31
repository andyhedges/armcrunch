/*
 * gwnum_arm64_integration.c
 *
 * ARM64 integration hooks for gwnum.c.
 *
 *   int arm64_gwinfo_hook(gwdata, negacyclic)
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
#include <stdint.h>
#ifdef ARM64_DIAGNOSTICS
#include <stdio.h>
#endif
#include "cpuid.h"
#include "gwnum.h"
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
	if (strcmp(backend_version, GWNUM_VERSION) != 0)
		return GWERROR_VERSION;

	k = gwdata->k;
	b = gwdata->b;
	n = gwdata->n;

	log2b = log2((double)b);
	n_bits = log2(k) + log2b * (double)n;

	safety_margin = gwdata->safety_margin + gwdata->polymult_safety_margin;

	/* Find smallest FFT length that can handle this number */
	fftlen = 0;
	for (i = 0; arm64_fft_sizes[i] != 0; i++) {
		double bpw = arm64_bits_per_word(arm64_fft_sizes[i], negacyclic) - safety_margin;
		if ((double)arm64_fft_sizes[i] * bpw >= n_bits) {
			fftlen = arm64_fft_sizes[i];
			break;
		}
	}

	/* Honor larger_fftlen_count */
	{
		int skip;
		for (skip = 0; skip < gwdata->larger_fftlen_count && arm64_fft_sizes[i] != 0; skip++) {
			i++;
			if (arm64_fft_sizes[i] != 0) fftlen = arm64_fft_sizes[i];
		}
	}

	/* Honor minimum_fftlen */
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

	/* Set CPU_SSE2 so internal_gwsetup() follows the SSE2 code path for
	   table building (weight tables, sin/cos tables). arm64_gwsetup_hook
	   then overwrites the SSE2 GWPROCPTRS with ARM64 implementations. */
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

	/* Set the gwdata back-pointer so addr_offset() works. */
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

	/* Always compute word bases from first principles. The SSE2
	   XMM_LIMIT_INVERSE/XMM_LIMIT_BIGMAX arrays use a complex interleaved
	   layout specific to the HG one-pass FFT that does NOT map to simple
	   [small, big] indexing. */
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
	ac->NEON_LIMIT_BIGMAX[ARM64_WORD_SMALL]  = small_word * ac->NEON_BIGVAL - ac->NEON_BIGVAL;
	ac->NEON_LIMIT_BIGMAX[ARM64_WORD_BIG]    = big_word   * ac->NEON_BIGVAL - ac->NEON_BIGVAL;

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

	/* Exponent and word-size metadata for first-principles IBDWT weight
	   computation and big/small word classification. */
	ac->NEON_N = gwdata->n;
	ac->NEON_NUM_B_PER_SMALL_WORD = gwdata->NUM_B_PER_SMALL_WORD;

	/* Disable gwmul3_carefully — our backend doesn't support the FMA-based
	   gwaddsub4o + gwmuladd4 path that gwmul3_carefully uses. */
	gwdata->careful_count = 0;

#ifdef ARM64_DIAGNOSTICS
	/* Diagnostic tests gated by -DARM64_DIAGNOSTICS compile flag.
	   Tests the exact round-trip paths that PRST uses for init checks. */
	{
		gwnum test_g;
		giant test_giant;
		int conv_err;

		test_g = gwalloc(gwdata);
		if (test_g != NULL) {
			/* Test 1: gianttogw(3) -> gwtogiant round-trip */
			test_giant = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
			itog(3, test_giant);
			gianttogw(gwdata, test_giant, test_g);
			{
				giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
				conv_err = gwtogiant(gwdata, test_g, result);
				if (conv_err >= 0 && result->sign == 1 && result->n[0] == 3) {
					fprintf(stderr, "[ARM64 DIAG] gianttogw(3)->gwtogiant = 3 OK\n");
				} else {
					fprintf(stderr, "[ARM64 DIAG] gianttogw(3)->gwtogiant FAILED: conv=%d sign=%d",
						conv_err, result->sign);
					if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
					if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
					fprintf(stderr, "\n");
					{
						unsigned long j;
						fprintf(stderr, "[ARM64 DIAG] raw fft words[0..7]: ");
						for (j = 0; j < 8 && j < gwdata->FFTLEN; j++) {
							long wv = 0;
							get_fft_value(gwdata, test_g, j, &wv);
							fprintf(stderr, "%ld ", wv);
						}
						fprintf(stderr, "\n");
					}
				}
				pushg(&gwdata->gdata, 1);
			}
			pushg(&gwdata->gdata, 1);

			/* Test 2: dbltogw(3) -> gwtogiant */
			dbltogw(gwdata, 3.0, test_g);
			{
				giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
				conv_err = gwtogiant(gwdata, test_g, result);
				if (conv_err >= 0 && result->sign == 1 && result->n[0] == 3) {
					fprintf(stderr, "[ARM64 DIAG] dbltogw(3)->gwtogiant = 3 OK\n");
				} else {
					fprintf(stderr, "[ARM64 DIAG] dbltogw(3)->gwtogiant FAILED: conv=%d sign=%d",
						conv_err, result->sign);
					if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
					fprintf(stderr, "\n");
				}
				pushg(&gwdata->gdata, 1);
			}

			/* Test 3: gwsquare2(3) = 9 */
			dbltogw(gwdata, 3.0, test_g);
			gwsquare2(gwdata, test_g, test_g, 0);
			{
				giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
				conv_err = gwtogiant(gwdata, test_g, result);
				if (conv_err >= 0 && result->sign == 1 && result->n[0] == 9) {
					fprintf(stderr, "[ARM64 DIAG] gwsquare2(3) = 9 OK\n");
				} else {
					fprintf(stderr, "[ARM64 DIAG] gwsquare2(3) FAILED: conv=%d sign=%d",
						conv_err, result->sign);
					if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
					if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
					fprintf(stderr, "\n");
				}
				pushg(&gwdata->gdata, 1);
			}

			/* Test 4: Lucas V doubling: V(2n) = V(n)^2 - 2
			   Start with V=4, apply V -> V^2 - 2 three times:
			   V_2=4 -> V_4=14 -> V_8=194 -> V_16=37634 */
			{
				gwnum lucas_g = gwalloc(gwdata);
				if (lucas_g != NULL) {
					gwsetaddin(gwdata, -2);
					dbltogw(gwdata, 4.0, lucas_g);
					gwsquare2(gwdata, lucas_g, lucas_g, GWMUL_ADDINCONST);
					gwsquare2(gwdata, lucas_g, lucas_g, GWMUL_ADDINCONST);
					gwsquare2(gwdata, lucas_g, lucas_g, GWMUL_ADDINCONST);
					gwsetaddin(gwdata, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, lucas_g, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 37634) {
							fprintf(stderr, "[ARM64 DIAG] Lucas V_16(P=4) = 37634 OK\n");
						} else {
							fprintf(stderr, "[ARM64 DIAG] Lucas V_16(P=4) FAILED: conv=%d sign=%d",
								conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
							fprintf(stderr, "\n");
						}
						pushg(&gwdata->gdata, 1);
					}
					gwfree(gwdata, lucas_g);
				}
			}

			/* Test 5: 100-iteration squaring with checkpoints at 5, 10, 20, 50, 100. */
			{
				gwnum iter_g = gwalloc(gwdata);
				if (iter_g != NULL) {
					int iter;
					int checkpoints[] = {5, 10, 20, 50, 100};
					int num_checkpoints = 5;
					int cp_idx = 0;

					dbltogw(gwdata, 3.0, iter_g);
					gw_clear_maxerr(gwdata);

					for (iter = 1; iter <= 100 && cp_idx < num_checkpoints; iter++) {
						gwsquare2(gwdata, iter_g, iter_g, 0);

						if (iter == checkpoints[cp_idx]) {
							giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
							conv_err = gwtogiant(gwdata, iter_g, result);
							fprintf(stderr, "[ARM64 DIAG] iter=%d: conv=%d sign=%d",
								iter, conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
							fprintf(stderr, " maxerr=%.6f\n", gw_get_maxerr(gwdata));
							pushg(&gwdata->gdata, 1);
							cp_idx++;
						}
					}
					gwfree(gwdata, iter_g);
				}
			}

			gwfree(gwdata, test_g);
		}
	}
#endif /* ARM64_DIAGNOSTICS */
}