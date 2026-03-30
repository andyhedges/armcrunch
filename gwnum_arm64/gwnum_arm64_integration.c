/*
 * gwnum_arm64_integration.c
 *
 * ARM64 integration hooks for gwnum.c.  This file is compiled into gwnum.a
 * on ARM64 targets and provides two entry points that gwnum.c calls when
 * built with -DARM64 (or when __aarch64__ is defined):
 *
 *   arm64_gwinfo_hook(gwdata, negacyclic)
 *     Called from gwinfo() after computing NEGACYCLIC_FFT.
 *     Returns the appropriate jmptab pointer for ARM64 FFT selection.
 *     Replaces the x86 gwinfo1() assembly call and ISA-based table selection.
 *
 *   arm64_gwsetup_hook(gwdata)
 *     Called from internal_gwsetup() after asm_data allocation and before
 *     the first use of GWPROCPTRS.  Installs all 13 ARM64 entry points
 *     and initializes the ARM64-specific asm_data constants.
 *
 * INTEGRATION INSTRUCTIONS FOR gwnum.c:
 *
 * 1. Add near the top of gwnum.c, after existing includes:
 *
 *      #if defined(ARM64) || defined(__aarch64__)
 *      #include "arm64_asm_data.h"
 *      extern const struct gwasm_jmptab *arm64_gwinfo_hook(gwhandle *gwdata, int negacyclic);
 *      extern void arm64_gwsetup_hook(gwhandle *gwdata);
 *      #endif
 *
 * 2. In gwinfo(), replace the gwinfo1() call and ISA table selection with:
 *
 *      #if defined(ARM64) || defined(__aarch64__)
 *          gwdata->jmptab = arm64_gwinfo_hook(gwdata, gwdata->NEGACYCLIC_FFT);
 *      #else
 *          // ... existing x86 gwinfo1() call and table selection ...
 *      #endif
 *
 * 3. In internal_gwsetup(), after asm_data is allocated and zeroed, before
 *    the ISA-specific GWPROCPTRS assignment block, add:
 *
 *      #if defined(ARM64) || defined(__aarch64__)
 *          arm64_gwsetup_hook(gwdata);
 *      #else
 *          // ... existing x86 GWPROCPTRS assignment ...
 *      #endif
 *
 * 4. In gwinit2(), after CPU_FLAGS is copied into gwdata->cpu_flags, add:
 *
 *      #if defined(ARM64) || defined(__aarch64__)
 *          gwdata->cpu_flags = 0;  // No x86 SIMD flags on ARM64
 *      #endif
 *
 * 5. Guard all x86 assembly extern declarations and prctab arrays:
 *
 *      #if !defined(ARM64) && !defined(__aarch64__)
 *          // ... avx512_aux_prctab, avx_aux_prctab, sse2_aux_prctab, etc. ...
 *      #endif
 *
 * 6. Stub out x86-only utilities on ARM64:
 *
 *      #if defined(ARM64) || defined(__aarch64__)
 *      static inline void fpu_init(void) {}
 *      #endif
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include "gwnum.h"
#include "arm64_asm_data.h"

const struct gwasm_jmptab *arm64_gwinfo_hook(gwhandle *gwdata, int negacyclic)
{
	const char *backend_version;

	(void)gwdata;

	backend_version = arm64_gwinfo_backend_version();
	if (strcmp(backend_version, GWNUM_VERSION) != 0) {
		/* Version mismatch between ARM64 backend and gwnum.h.
		 * Return NULL to trigger GWERROR_VERSION in gwinfo(). */
		return NULL;
	}

	return arm64_gwinfo1(negacyclic);
}

void arm64_gwsetup_hook(gwhandle *gwdata)
{
	arm64_gwasm_data_view *ad;

	if (gwdata == NULL || gwdata->asm_data == NULL) return;

	/* Install all 13 ARM64 entry points into GWPROCPTRS. */
	arm64_install_gwprocptrs(gwdata->GWPROCPTRS);

	/* Initialize the ARM64-specific asm_data constants. */
	ad = (arm64_gwasm_data_view *)gwdata->asm_data;

	ad->FFTLEN     = (uint32_t)gwdata->FFTLEN;
	ad->PASS1_SIZE = (uint32_t)gwdata->PASS1_SIZE;
	ad->PASS2_SIZE = (uint32_t)gwdata->PASS2_SIZE;
	ad->const_fft  = 0;
	ad->ADDIN_VALUE    = gwdata->asm_addin_value;
	ad->POSTADDIN_VALUE = gwdata->asm_postaddin_value;

	/* NEON-specific constants for normalization and carry propagation. */
	{
		double big_word, small_word;

		/* Compute word sizes from NUM_B_PER_SMALL_WORD and base b. */
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

	/* k, c, b constants for modular reduction. */
	{
		double k_hi = floor(gwdata->k / 4294967296.0) * 4294967296.0;
		ad->arm64.NEON_K_HI      = k_hi;
		ad->arm64.NEON_K_LO      = gwdata->k - k_hi;
		ad->arm64.NEON_MINUS_C   = (double)(-gwdata->c);
		ad->arm64.NEON_MULCONST  = (double)gwdata->mulbyconst;
		ad->arm64.NEON_B         = (double)gwdata->b;
		ad->arm64.NEON_ONE_OVER_B = (gwdata->b != 0) ? 1.0 / (double)gwdata->b : 0.0;
	}

	/* Normalization scaling factor. */
	if (gwdata->k == 1.0)
		ad->arm64.NEON_NORM012_FF = (double)gwdata->FFTLEN * 0.5;
	else
		ad->arm64.NEON_NORM012_FF = (double)gwdata->FFTLEN * 0.5 / gwdata->k;

	ad->arm64.NEON_MAXERR = 0.0;
	ad->MAXERR = 0.0;
}