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
#include <stdint.h>
#include "gwnum.h"
#include "arm64_asm_data.h"

/* ARM64 one-pass FFT sizes we support */
static const uint32_t arm64_fft_sizes[] = { 1024, 2048, 4096, 8192, 16384, 32768, 0 };

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

	/* Verify ARM64 backend version matches gwnum.h at compile time. */
	backend_version = arm64_gwinfo_backend_version();
	if (strcmp(backend_version, GWNUM_VERSION) != 0) {
		return GWERROR_VERSION;
	}

	k = gwdata->k;
	b = gwdata->b;
	n = gwdata->n;

	/* Calculate number of bits in k*b^n */
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

	/* Populate all gwdata fields that gwinfo() normally sets. */

	gwdata->FFTLEN = fftlen;
	gwdata->PASS2_SIZE = 0;
	gwdata->PASS1_SIZE = 0;
	gwdata->FFT_TYPE = FFT_TYPE_RADIX_4;
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
	gwdata->GWPROCPTRS[0] = (void (*)(void *))arm64_fft_entry;

	/* EXTRA_BITS: headroom before roundoff errors become dangerous */
	{
		double bpw = arm64_bits_per_word(fftlen, negacyclic);
		double actual = n_bits / (double)fftlen;
		double extra = (bpw - actual) * 2.0;
		if (extra < 0.5) extra = 0.5;
		gwdata->EXTRA_BITS = (float)extra;
	}

	/* avg_num_b_per_word and NUM_B_PER_SMALL_WORD */
	gwdata->avg_num_b_per_word = (double)n / (double)fftlen;
	if (b == 2)
		gwdata->NUM_B_PER_SMALL_WORD = (unsigned long)floor((double)n / (double)fftlen);
	else
		gwdata->NUM_B_PER_SMALL_WORD = (unsigned long)floor(log2b * (double)n / (double)fftlen / log2(2.0));

	gwdata->bit_length = n_bits;
	gwdata->fft_max_bits_per_word = arm64_bits_per_word(fftlen, negacyclic);

	/* RATIONAL_FFT: all weights == 1 when n divides FFTLEN evenly (base-2 only) */
	gwdata->RATIONAL_FFT = (b == 2 && n % fftlen == 0) ? 1 : 0;

	/* Use the negacyclic parameter provided by gwinfo()'s caller. */
	gwdata->ZERO_PADDED_FFT = 0;
	gwdata->NEGACYCLIC_FFT = negacyclic ? 1 : 0;

	gwdata->datasize = fftlen * sizeof(double);
	gwdata->jmptab = arm64_gwinfo1(negacyclic);

	return 0;
}

void arm64_gwsetup_hook(gwhandle *gwdata)
{
	arm64_gwasm_data_view *ad;

	if (gwdata == NULL || gwdata->asm_data == NULL) return;

	arm64_install_gwprocptrs(gwdata->GWPROCPTRS);

	ad = (arm64_gwasm_data_view *)gwdata->asm_data;

	ad->FFTLEN     = (uint32_t)gwdata->FFTLEN;
	ad->PASS1_SIZE = (uint32_t)gwdata->PASS1_SIZE;
	ad->PASS2_SIZE = (uint32_t)gwdata->PASS2_SIZE;
	ad->const_fft  = 0;
	ad->ADDIN_VALUE    = gwdata->asm_addin_value;
	ad->POSTADDIN_VALUE = gwdata->asm_postaddin_value;

	{
		double big_word, small_word;

		if (gwdata->b == 2) {
			small_word = ldexp(1.0, (int)gwdata->NUM_B_PER_SMALL_WORD);
			big_word   = small_word * 2.0;
		} else {
			small_word = pow((double)gwdata->b, (double)gwdata->NUM_B_PER_SMALL_WORD);
			big_word   = small_word * (double)gwdata->b;
		}

		ad->arm64.NEON_LARGE_BASE     = big_word;
		ad->arm64.NEON_SMALL_BASE     = small_word;
		ad->arm64.NEON_LARGE_BASE_INV = 1.0 / big_word;
		ad->arm64.NEON_SMALL_BASE_INV = 1.0 / small_word;
		ad->arm64.NEON_BIGVAL = 3.0 * ldexp(1.0, 51);
		ad->arm64.NEON_LIMIT_INVERSE[ARM64_WORD_SMALL] = 1.0 / small_word;
		ad->arm64.NEON_LIMIT_INVERSE[ARM64_WORD_BIG]   = 1.0 / big_word;
		ad->arm64.NEON_LIMIT_BIGMAX[ARM64_WORD_SMALL]  = small_word * ad->arm64.NEON_BIGVAL - ad->arm64.NEON_BIGVAL;
		ad->arm64.NEON_LIMIT_BIGMAX[ARM64_WORD_BIG]    = big_word   * ad->arm64.NEON_BIGVAL - ad->arm64.NEON_BIGVAL;
	}

	{
		double k_hi = floor(gwdata->k / 4294967296.0) * 4294967296.0;
		ad->arm64.NEON_K_HI      = k_hi;
		ad->arm64.NEON_K_LO      = gwdata->k - k_hi;
		ad->arm64.NEON_MINUS_C   = (double)(-gwdata->c);
		ad->arm64.NEON_MULCONST  = (double)gwdata->mulbyconst;
		ad->arm64.NEON_B         = (double)gwdata->b;
		ad->arm64.NEON_ONE_OVER_B = (gwdata->b != 0) ? 1.0 / (double)gwdata->b : 0.0;
	}

	if (gwdata->k == 1.0)
		ad->arm64.NEON_NORM012_FF = (double)gwdata->FFTLEN * 0.5;
	else
		ad->arm64.NEON_NORM012_FF = (double)gwdata->FFTLEN * 0.5 / gwdata->k;

	ad->arm64.NEON_MAXERR = 0.0;
	ad->MAXERR = 0.0;
}