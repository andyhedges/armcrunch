#include <stdint.h>
#include <string.h>

#include "gwnum.h"
#include "arm64_asm_data.h"

#define ARM64_JMPTAB_FLAG_ARCH_ARM64	0xA6400000u
#define ARM64_JMPTAB_FLAG_ONE_PASS	0x00000001u
#define ARM64_JMPTAB_FLAG_NEGACYCLIC	0x00000002u

static const uint32_t ARM64_ONE_PASS_FFTS[] = { 1024u, 2048u, 4096u, 8192u, 16384u, 32768u };
#define ARM64_NUM_ONE_PASS_FFTS (sizeof(ARM64_ONE_PASS_FFTS) / sizeof(ARM64_ONE_PASS_FFTS[0]))

static struct gwasm_jmptab arm64_cyclic_tab[ARM64_NUM_ONE_PASS_FFTS + 1];
static struct gwasm_jmptab arm64_negacyclic_tab[ARM64_NUM_ONE_PASS_FFTS + 1];
static int arm64_tables_initialized = 0;

static int arm64_log2_u32(uint32_t v) {
	int n = 0;
	while (v > 1u) {
		v >>= 1u;
		++n;
	}
	return n;
}

static double arm64_bits_per_fft_word(uint32_t fftlen, int negacyclic) {
	double base = negacyclic ? 19.35 : 19.60;
	double penalty = ((double)arm64_log2_u32(fftlen) - 10.0) * 0.05;
	double bits = base - penalty;
	if (bits < 16.0) bits = 16.0;
	return bits;
}

static uint32_t arm64_calc_max_exp(uint32_t fftlen, int negacyclic) {
	double bits = arm64_bits_per_fft_word(fftlen, negacyclic);
	double max_exp = (double)fftlen * bits;
	if (max_exp < 0.0) max_exp = 0.0;
	if (max_exp > 4294967294.0) max_exp = 4294967294.0;
	return (uint32_t)max_exp;
}

static float arm64_timing_estimate(uint32_t fftlen) {
	double log2n = (double)arm64_log2_u32(fftlen);
	return (float)(((double)fftlen * log2n) / 4096.0);
}

static void arm64_fill_entry(struct gwasm_jmptab *entry, uint32_t fftlen, int negacyclic) {
	int stages = arm64_log2_u32(fftlen);

	memset(entry, 0, sizeof(*entry));
	entry->max_exp = arm64_calc_max_exp(fftlen, negacyclic);
	entry->fftlen = fftlen;
	entry->timing = arm64_timing_estimate(fftlen);
	entry->flags = ARM64_JMPTAB_FLAG_ARCH_ARM64 |
		       ARM64_JMPTAB_FLAG_ONE_PASS |
		       (negacyclic ? ARM64_JMPTAB_FLAG_NEGACYCLIC : 0u);
	entry->proc_ptr = (void *)arm64_fft_entry;
	entry->mem_needed = fftlen * (uint32_t)(sizeof(double) * 12u);
	entry->counts[0] = stages;
	entry->counts[1] = 4;	/* radix-4 core */
	entry->counts[2] = 1;	/* one-pass */
}

static void arm64_init_tables_once(void) {
	size_t i;

	if (arm64_tables_initialized) return;

	for (i = 0; i < ARM64_NUM_ONE_PASS_FFTS; ++i) {
		arm64_fill_entry(&arm64_cyclic_tab[i], ARM64_ONE_PASS_FFTS[i], 0);
		arm64_fill_entry(&arm64_negacyclic_tab[i], ARM64_ONE_PASS_FFTS[i], 1);
	}

	memset(&arm64_cyclic_tab[ARM64_NUM_ONE_PASS_FFTS], 0, sizeof(struct gwasm_jmptab));
	memset(&arm64_negacyclic_tab[ARM64_NUM_ONE_PASS_FFTS], 0, sizeof(struct gwasm_jmptab));

	arm64_tables_initialized = 1;
}

const struct gwasm_jmptab *arm64_gwinfo1(int negacyclic) {
	arm64_init_tables_once();
	return negacyclic ? arm64_negacyclic_tab : arm64_cyclic_tab;
}

const struct gwasm_jmptab *gwinfo1(void) {
	return arm64_gwinfo1(0);
}

const char *arm64_gwinfo_backend_version(void) {
	return GWNUM_VERSION;
}

#define ARM64_PROC_CAST(fn) ((void (*)(void *))(fn))

void arm64_install_gwprocptrs(void (**procptrs)(void *)) {
	if (procptrs == NULL) return;

	procptrs[0] = ARM64_PROC_CAST(arm64_fft_entry);
	procptrs[1] = ARM64_PROC_CAST(arm64_gw_add);
	procptrs[2] = ARM64_PROC_CAST(arm64_gw_addq);
	procptrs[3] = ARM64_PROC_CAST(arm64_gw_sub);
	procptrs[4] = ARM64_PROC_CAST(arm64_gw_subq);
	procptrs[5] = ARM64_PROC_CAST(arm64_gw_addsub);
	procptrs[6] = ARM64_PROC_CAST(arm64_gw_addsubq);
	procptrs[7] = ARM64_PROC_CAST(arm64_gw_copy4kb);
	procptrs[8] = ARM64_PROC_CAST(arm64_gw_muls);
	procptrs[9] = ARM64_PROC_CAST(arm64_norm_plain);
	procptrs[10] = ARM64_PROC_CAST(arm64_norm_errchk);
	procptrs[11] = ARM64_PROC_CAST(arm64_norm_mulconst);
	procptrs[12] = ARM64_PROC_CAST(arm64_norm_errchk_mulconst);
}