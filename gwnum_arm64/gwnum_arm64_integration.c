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
#include <stdio.h>
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
	gwdata->mem_needed = fftlen * (unsigned long)(sizeof(double) * 20);

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
	if (ac->NEON_MINUS_C == 0.0 && gwdata->c != 0)
		ac->NEON_MINUS_C = (double)(-gwdata->c);

	ac->NEON_MULCONST = ad->u.xmm.XMM_MULCONST[0];
	if (ac->NEON_MULCONST == 0.0)
		ac->NEON_MULCONST = (double)gwdata->mulbyconst;

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

	gwdata->careful_count = 0;

	/* Quick k>1 round-trip test: dbltogw(3) -> gwtogiant should return 3 for any k.
	   This fires once per gwsetup and helps diagnose Riesel number failures. */
	if (gwdata->k > 1.0) {
		gwnum rt_g = gwalloc(gwdata);
		if (rt_g != NULL) {
			giant rt_result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
			int rt_err;
			dbltogw(gwdata, 3.0, rt_g);
			rt_err = gwtogiant(gwdata, rt_g, rt_result);
			if (rt_err >= 0 && rt_result->sign == 1 && rt_result->n[0] == 3) {
				fprintf(stderr, "[ARM64 K>1 RT] dbltogw(3)->gwtogiant = 3 OK (k=%.1f)\n", gwdata->k);
			} else {
				fprintf(stderr, "[ARM64 K>1 RT] dbltogw(3)->gwtogiant FAILED: err=%d sign=%d", rt_err, rt_result->sign);
				if (rt_result->sign >= 1) fprintf(stderr, " n[0]=%u", rt_result->n[0]);
				if (rt_result->sign >= 2) fprintf(stderr, " n[1]=%u", rt_result->n[1]);
				fprintf(stderr, " (k=%.1f)\n", gwdata->k);
			}
			pushg(&gwdata->gdata, 1);
			gwfree(gwdata, rt_g);
		}
	}

#ifdef ARM64_DIAGNOSTICS
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
				if (conv_err >= 0 && result->sign == 1 && result->n[0] == 3)
					fprintf(stderr, "[ARM64 DIAG] gianttogw(3)->gwtogiant = 3 OK\n");
				else {
					fprintf(stderr, "[ARM64 DIAG] gianttogw(3)->gwtogiant FAILED: conv=%d sign=%d", conv_err, result->sign);
					if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
					fprintf(stderr, "\n");
				}
				pushg(&gwdata->gdata, 1);
			}
			pushg(&gwdata->gdata, 1);

			/* Test 2: dbltogw(3) -> gwtogiant */
			dbltogw(gwdata, 3.0, test_g);
			{
				giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
				conv_err = gwtogiant(gwdata, test_g, result);
				if (conv_err >= 0 && result->sign == 1 && result->n[0] == 3)
					fprintf(stderr, "[ARM64 DIAG] dbltogw(3)->gwtogiant = 3 OK\n");
				else {
					fprintf(stderr, "[ARM64 DIAG] dbltogw(3)->gwtogiant FAILED: conv=%d sign=%d", conv_err, result->sign);
					if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
					fprintf(stderr, "\n");
				}
				pushg(&gwdata->gdata, 1);
			}

			/* Test 3: gwsquare2(3) = 9 (in-place squaring) */
			dbltogw(gwdata, 3.0, test_g);
			gwsquare2(gwdata, test_g, test_g, 0);
			{
				giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
				conv_err = gwtogiant(gwdata, test_g, result);
				if (conv_err >= 0 && result->sign == 1 && result->n[0] == 9)
					fprintf(stderr, "[ARM64 DIAG] gwsquare2(3) = 9 OK\n");
				else {
					fprintf(stderr, "[ARM64 DIAG] gwsquare2(3) FAILED: conv=%d sign=%d", conv_err, result->sign);
					if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
					fprintf(stderr, "\n");
				}
				pushg(&gwdata->gdata, 1);
			}

			/* Test 4a: gwsquare2(4) without ADDINCONST = 16 (baseline) */
			{
				gwnum t4a = gwalloc(gwdata);
				if (t4a != NULL) {
					dbltogw(gwdata, 4.0, t4a);
					gwsquare2(gwdata, t4a, t4a, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, t4a, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 16)
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(4) = 16 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(4) FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							fprintf(stderr, " (expected 16)\n");
						}
						pushg(&gwdata->gdata, 1);
					}
					gwfree(gwdata, t4a);
				}
			}

			/* Test 4b: gwsquare2(4, GWMUL_ADDINCONST) with addin=-2 = 14 */
			{
				gwnum t4b = gwalloc(gwdata);
				if (t4b != NULL) {
					gwsetaddin(gwdata, -2);
					dbltogw(gwdata, 4.0, t4b);
					gwsquare2(gwdata, t4b, t4b, GWMUL_ADDINCONST);
					gwsetaddin(gwdata, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, t4b, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 14)
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(4)+addin(-2) = 14 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(4)+addin(-2) FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
							fprintf(stderr, " (expected 14)\n");
						}
						pushg(&gwdata->gdata, 1);
					}
					gwfree(gwdata, t4b);
				}
			}

			/* Test 4c: fresh gwnum gwsquare2(4, GWMUL_ADDINCONST) addin=-2 = 14
			   (tests whether addin state leaks between allocations) */
			{
				gwnum t4c = gwalloc(gwdata);
				if (t4c != NULL) {
					gwsetaddin(gwdata, -2);
					dbltogw(gwdata, 4.0, t4c);
					gwsquare2(gwdata, t4c, t4c, GWMUL_ADDINCONST);
					gwsetaddin(gwdata, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, t4c, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 14)
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(4)+addin(-2) fresh = 14 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(4)+addin(-2) fresh FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
							fprintf(stderr, " (expected 14)\n");
						}
						pushg(&gwdata->gdata, 1);
					}
					gwfree(gwdata, t4c);
				}
			}

			/* Test 4d: non-in-place gwsquare2 where source != destination
			   Store 5 in src, square into separate dst, expect 25 */
			{
				gwnum t4d_src = gwalloc(gwdata);
				gwnum t4d_dst = gwalloc(gwdata);
				if (t4d_src != NULL && t4d_dst != NULL) {
					dbltogw(gwdata, 5.0, t4d_src);
					gwsquare2(gwdata, t4d_src, t4d_dst, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, t4d_dst, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 25)
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(5,dst) = 25 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] gwsquare2(5,dst) FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							fprintf(stderr, " (expected 25)\n");
						}
						pushg(&gwdata->gdata, 1);
					}
				}
				if (t4d_dst != NULL) gwfree(gwdata, t4d_dst);
				if (t4d_src != NULL) gwfree(gwdata, t4d_src);
			}

			/* Test 4e: gwmul3(3, 5) = 15 (non-squaring multiply, ffttype=3 path) */
			{
				gwnum t4e_a = gwalloc(gwdata);
				gwnum t4e_b = gwalloc(gwdata);
				if (t4e_a != NULL && t4e_b != NULL) {
					dbltogw(gwdata, 3.0, t4e_a);
					dbltogw(gwdata, 5.0, t4e_b);
					gwmul3(gwdata, t4e_a, t4e_b, t4e_a, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, t4e_a, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 15)
							fprintf(stderr, "[ARM64 DIAG] gwmul3(3,5) = 15 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] gwmul3(3,5) FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							fprintf(stderr, " (expected 15)\n");
						}
						pushg(&gwdata->gdata, 1);
					}
				}
				if (t4e_b != NULL) gwfree(gwdata, t4e_b);
				if (t4e_a != NULL) gwfree(gwdata, t4e_a);
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

			/* Test 6: identify target number and re-check core operations after stress squaring. */
			fprintf(stderr, "[ARM64 DIAG] k=%.1f b=%lu n=%lu c=%ld\n",
				gwdata->k, (unsigned long)gwdata->b, (unsigned long)gwdata->n, (long)gwdata->c);
			{
				gwnum diag_g = gwalloc(gwdata);
				if (diag_g != NULL) {
					int word;

					dbltogw(gwdata, 3.0, diag_g);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, diag_g, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 3)
							fprintf(stderr, "[ARM64 DIAG] post5 dbltogw(3)->gwtogiant = 3 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] post5 dbltogw(3)->gwtogiant FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
							fprintf(stderr, "\n");
							dbltogw(gwdata, 3.0, diag_g);
							for (word = 0; word < 8; word++) {
								long wv = 0;
								get_fft_value(gwdata, diag_g, (unsigned long)word, &wv);
								fprintf(stderr, "[ARM64 DIAG] raw_fft[%d]=%ld\n", word, wv);
							}
						}
						pushg(&gwdata->gdata, 1);
					}

					dbltogw(gwdata, 3.0, diag_g);
					gwsquare2(gwdata, diag_g, diag_g, 0);
					{
						giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
						conv_err = gwtogiant(gwdata, diag_g, result);
						if (conv_err >= 0 && result->sign == 1 && result->n[0] == 9)
							fprintf(stderr, "[ARM64 DIAG] post5 gwsquare2(3) = 9 OK\n");
						else {
							fprintf(stderr, "[ARM64 DIAG] post5 gwsquare2(3) FAILED: conv=%d sign=%d", conv_err, result->sign);
							if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
							if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
							fprintf(stderr, " (expected 9)\n");
						}
						pushg(&gwdata->gdata, 1);
					}

					if (gwdata->k > 1.0) {
						intptr_t addin_offset = (intptr_t)ad->ADDIN_OFFSET;
						intptr_t addin_word = addin_offset / (intptr_t)sizeof(double);
						fprintf(stderr, "[ARM64 DIAG] ADDIN_OFFSET=%lld addin_word=%lld\n",
							(long long)addin_offset, (long long)addin_word);
					}

					if (gwdata->k > 1.0 && gwdata->c == -1) {
						gwsetaddin(gwdata, -2);
						dbltogw(gwdata, 4.0, diag_g);
						gwsquare2(gwdata, diag_g, diag_g, GWMUL_ADDINCONST);
						gwsetaddin(gwdata, 0);
						{
							giant result = popg(&gwdata->gdata, ((unsigned long)gwdata->bit_length >> 5) + 5);
							conv_err = gwtogiant(gwdata, diag_g, result);
							if (conv_err >= 0 && result->sign == 1 && result->n[0] == 14)
								fprintf(stderr, "[ARM64 DIAG] Lucas V dbl: gwsquare2(4)+addin(-2) = 14 OK\n");
							else {
								fprintf(stderr, "[ARM64 DIAG] Lucas V dbl FAILED: conv=%d sign=%d", conv_err, result->sign);
								if (result->sign >= 1) fprintf(stderr, " n[0]=%u", result->n[0]);
								if (result->sign >= 2) fprintf(stderr, " n[1]=%u", result->n[1]);
								fprintf(stderr, " (expected 14)\n");
							}
							pushg(&gwdata->gdata, 1);
						}
					}

					gwsetaddin(gwdata, 0);
					gwfree(gwdata, diag_g);
				}
			}

			gwfree(gwdata, test_g);
		}
	}
#endif /* ARM64_DIAGNOSTICS */
}