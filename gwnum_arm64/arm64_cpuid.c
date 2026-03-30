#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#endif

#if defined(__linux__)
#include <sys/auxv.h>
#include <time.h>
#endif

#ifndef CPU_ARCHITECTURE_ARM64
#define CPU_ARCHITECTURE_ARM64 64
#endif

int CPU_FLAGS = 0;
int CPU_ARCHITECTURE = CPU_ARCHITECTURE_ARM64;
int CPU_SPEED = 0;
int CPU_FAMILY = 8;
int CPU_MODEL = 0;
int CPU_STEPPING = 0;
int CPU_NUM_CORES = 1;
int CPU_NUM_LOGICAL_CORES = 1;
int NUM_CPUS = 1;
int NUM_CORES = 1;
int NUM_THREADS = 1;
int HYPERTHREADING = 0;
int CPU_L2_CACHE_SIZE = 0;
int CPU_TOTAL_L2_CACHE_SIZE = 0;
int CPU_L3_CACHE_SIZE = 0;
char CPU_BRAND[128] = "ARM64";

/* Compatibility globals used elsewhere in gwnum/Prime95 code paths. */
unsigned int CPU_CORES = 1;
unsigned int CPU_HYPERTHREADS = 1;
unsigned int CPU_SIGNATURE = 0;

static void arm64_copy_string(char *dst, size_t dst_len, const char *src) {
	if (dst == NULL || dst_len == 0) return;
	if (src == NULL) {
		dst[0] = '\0';
		return;
	}
	strncpy(dst, src, dst_len - 1);
	dst[dst_len - 1] = '\0';
	{
		size_t len = strlen(dst);
		while (len > 0 && (dst[len - 1] == '\n' || dst[len - 1] == '\r')) {
			dst[len - 1] = '\0';
			--len;
		}
	}
}

#if defined(__APPLE__)
static int arm64_sysctl_int(const char *name, int *value) {
	size_t len = sizeof(*value);
	if (sysctlbyname(name, value, &len, NULL, 0) != 0) return 0;
	return 1;
}

static int arm64_sysctl_u64(const char *name, uint64_t *value) {
	size_t len = sizeof(*value);
	if (sysctlbyname(name, value, &len, NULL, 0) != 0) return 0;
	return 1;
}

static int arm64_sysctl_string(const char *name, char *buf, size_t buflen) {
	size_t len = buflen;
	if (buf == NULL || buflen == 0) return 0;
	if (sysctlbyname(name, buf, &len, NULL, 0) != 0) return 0;
	buf[buflen - 1] = '\0';
	return 1;
}
#endif

#if defined(__linux__)
static int arm64_read_first_line(const char *path, char *buf, size_t buflen) {
	FILE *f;
	if (buf == NULL || buflen == 0) return 0;
	f = fopen(path, "r");
	if (f == NULL) return 0;
	if (fgets(buf, (int)buflen, f) == NULL) {
		fclose(f);
		return 0;
	}
	fclose(f);
	arm64_copy_string(buf, buflen, buf);
	return 1;
}

static int arm64_parse_cache_size(const char *text) {
	unsigned long value = 0;
	char suffix = '\0';
	int fields;

	if (text == NULL) return 0;
	fields = sscanf(text, "%lu%c", &value, &suffix);
	if (fields <= 0) return 0;
	if (fields == 1) return (int)value;

	if (suffix == 'K' || suffix == 'k') return (int)(value * 1024UL);
	if (suffix == 'M' || suffix == 'm') return (int)(value * 1024UL * 1024UL);
	return (int)value;
}

static void arm64_read_brand_linux(char *brand, size_t brand_len) {
	FILE *f = fopen("/proc/cpuinfo", "r");
	char line[512];

	if (f == NULL) {
		arm64_copy_string(brand, brand_len, "ARM64 Linux");
		return;
	}

	while (fgets(line, sizeof(line), f) != NULL) {
		if (strncmp(line, "model name", 10) == 0 ||
		    strncmp(line, "Hardware", 8) == 0 ||
		    strncmp(line, "Processor", 9) == 0) {
			char *colon = strchr(line, ':');
			if (colon != NULL) {
				++colon;
				while (*colon == ' ' || *colon == '\t') ++colon;
				arm64_copy_string(brand, brand_len, colon);
				fclose(f);
				return;
			}
		}
	}

	fclose(f);
	arm64_copy_string(brand, brand_len, "ARM64 Linux");
}
#endif

static void arm64_detect_cpu(void) {
	static int initialized = 0;

	if (initialized) return;
	initialized = 1;

	CPU_FLAGS = 0;
	CPU_ARCHITECTURE = CPU_ARCHITECTURE_ARM64;
	HYPERTHREADING = 0;

#if defined(__APPLE__)
	{
		int tmp = 0;
		uint64_t l2 = 0;
		char brand[128];

		if (arm64_sysctl_int("hw.physicalcpu", &tmp) && tmp > 0) CPU_NUM_CORES = tmp;
		if (arm64_sysctl_int("hw.logicalcpu", &tmp) && tmp > 0) CPU_NUM_LOGICAL_CORES = tmp;
		if (arm64_sysctl_u64("hw.l2cachesize", &l2) && l2 > 0) CPU_L2_CACHE_SIZE = (int)l2;

		if (!arm64_sysctl_string("machdep.cpu.brand_string", brand, sizeof(brand))) {
			if (!arm64_sysctl_string("hw.model", brand, sizeof(brand))) {
				arm64_copy_string(brand, sizeof(brand), "Apple Silicon");
			}
		}
		arm64_copy_string(CPU_BRAND, sizeof(CPU_BRAND), brand);
	}
#elif defined(__linux__)
	{
		long logical = sysconf(_SC_NPROCESSORS_ONLN);
		char line[128];

		if (logical > 0) CPU_NUM_LOGICAL_CORES = (int)logical;
		CPU_NUM_CORES = CPU_NUM_LOGICAL_CORES;

		arm64_read_brand_linux(CPU_BRAND, sizeof(CPU_BRAND));

		if (arm64_read_first_line("/sys/devices/system/cpu/cpu0/cache/index2/size", line, sizeof(line))) {
			CPU_L2_CACHE_SIZE = arm64_parse_cache_size(line);
		} else if (arm64_read_first_line("/sys/devices/system/cpu/cpu0/cache/index3/size", line, sizeof(line))) {
			CPU_L2_CACHE_SIZE = arm64_parse_cache_size(line);
		}

#if defined(AT_HWCAP)
		{
			unsigned long hwcap = getauxval(AT_HWCAP);
			(void)hwcap;	/* informational only; x86 SIMD flags remain disabled */
		}
#endif
	}
#else
	arm64_copy_string(CPU_BRAND, sizeof(CPU_BRAND), "ARM64");
#endif

	if (CPU_NUM_CORES < 1) CPU_NUM_CORES = 1;
	if (CPU_NUM_LOGICAL_CORES < 1) CPU_NUM_LOGICAL_CORES = CPU_NUM_CORES;
	if (CPU_L2_CACHE_SIZE < 0) CPU_L2_CACHE_SIZE = 0;

	NUM_CORES = CPU_NUM_CORES;
	NUM_THREADS = CPU_NUM_LOGICAL_CORES;
	NUM_CPUS = CPU_NUM_LOGICAL_CORES;
	CPU_TOTAL_L2_CACHE_SIZE = CPU_L2_CACHE_SIZE;

	CPU_CORES = (unsigned int)CPU_NUM_CORES;
	CPU_HYPERTHREADS = 1U;
	CPU_SIGNATURE = 0U;
}

void guessCpuType(void) {
	arm64_detect_cpu();
}

void getCpuInfo(void) {
	arm64_detect_cpu();
}

void cpuid_init(void) {
	arm64_detect_cpu();
}

int max_cores_for_work_prefetching(void) {
	arm64_detect_cpu();
	return CPU_NUM_CORES > 1 ? CPU_NUM_CORES - 1 : 1;
}

int num_cpus(void) {
	arm64_detect_cpu();
	return NUM_CPUS;
}

const char *cpu_brand_string(void) {
	arm64_detect_cpu();
	return CPU_BRAND;
}

void guessCpuSpeed(void) {
	arm64_detect_cpu();
}

void fpu_init(void) {
}

/* High-resolution timers */

double getHighResTimer(void) {
#if defined(__APPLE__)
	return (double)mach_absolute_time();
#elif defined(__linux__)
	struct timespec ts;
#if defined(CLOCK_MONOTONIC)
	if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
	return (double)ts.tv_sec * 1000000000.0 + (double)ts.tv_nsec;
#else
	return 0.0;
#endif
#else
	return 0.0;
#endif
}

double getHighResTimerFrequency(void) {
#if defined(__APPLE__)
	static double frequency = 0.0;
	if (frequency == 0.0) {
		mach_timebase_info_data_t tb;
		if (mach_timebase_info(&tb) != 0 || tb.numer == 0) {
			frequency = 1000000000.0;
		} else {
			frequency = (1000000000.0 * (double)tb.denom) / (double)tb.numer;
		}
	}
	return frequency;
#elif defined(__linux__)
	return 1000000000.0;
#else
	return 1.0;
#endif
}

/* Low-level multi-precision helpers */

void addhlp(uint32_t *res, uint32_t *carry, uint32_t val) {
	uint64_t r;
	uint64_t c;
	uint64_t sum;

	if (res == NULL || carry == NULL) return;

	r = (uint64_t)(*res);
	c = (uint64_t)(*carry);
	sum = r + c + (uint64_t)val;

	*res = (uint32_t)sum;
	*carry = (uint32_t)(sum >> 32);
}

void subhlp(uint32_t *res, uint32_t *carry, uint32_t val) {
	uint64_t r;
	uint64_t c;
	uint64_t sub;

	if (res == NULL || carry == NULL) return;

	r = (uint64_t)(*res);
	c = (uint64_t)(*carry);
	sub = (uint64_t)val + c;

	*res = (uint32_t)(r - sub);
	*carry = (r < sub) ? 1U : 0U;
}

void muladdhlp(uint32_t *res, uint32_t *carryl, uint32_t *carryh, uint32_t val1, uint32_t val2) {
	uint64_t r;
	uint64_t cl;
	uint64_t ch;
	uint64_t prod;
	uint64_t t;

	if (res == NULL || carryl == NULL || carryh == NULL) return;

	r = (uint64_t)(*res);
	cl = (uint64_t)(*carryl);
	ch = (uint64_t)(*carryh);
	prod = (uint64_t)val1 * (uint64_t)val2;

	t = r + (uint64_t)(uint32_t)prod;
	*res = (uint32_t)t;

	t = cl + (prod >> 32) + (t >> 32);
	*carryl = (uint32_t)t;

	ch += (t >> 32);
	*carryh = (uint32_t)ch;
}

void muladd2hlp(uint32_t *res, uint32_t *carryl, uint32_t *carryh, uint32_t val1, uint32_t val2) {
	muladdhlp(res, carryl, carryh, val1, val2);
	muladdhlp(res, carryl, carryh, val1, val2);
}

void mulsubhlp(uint32_t *res, uint32_t *carryl, uint32_t *carryh, uint32_t val1, uint32_t val2) {
	uint64_t r;
	uint64_t cl;
	uint64_t ch;
	uint64_t prod;
	uint64_t lo_sub;
	uint64_t borrow;
	uint64_t mid_sub;
	uint64_t mid_borrow;

	if (res == NULL || carryl == NULL || carryh == NULL) return;

	r = (uint64_t)(*res);
	cl = (uint64_t)(*carryl);
	ch = (uint64_t)(*carryh);
	prod = (uint64_t)val1 * (uint64_t)val2;

	lo_sub = (uint64_t)(uint32_t)prod;
	borrow = (r < lo_sub) ? 1ULL : 0ULL;
	*res = (uint32_t)(r - lo_sub);

	mid_sub = (prod >> 32) + borrow;
	mid_borrow = (cl < mid_sub) ? 1ULL : 0ULL;
	*carryl = (uint32_t)(cl - mid_sub);

	*carryh = (uint32_t)(ch - mid_borrow);
}

/* Lehmer GCD helper */

static unsigned int arm64_clz32(uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
	if (value == 0) return 32U;
	return (unsigned int)__builtin_clz(value);
#else
	unsigned int n = 0U;
	if (value == 0) return 32U;
	while ((value & 0x80000000U) == 0U) {
		value <<= 1;
		++n;
	}
	return n;
#endif
}

static int arm64_newd_overflow(uint32_t b, uint32_t d, uint64_t q) {
	if (d == 0) return 0;
	return q > (((uint64_t)UINT32_MAX - (uint64_t)b) / (uint64_t)d);
}

static void arm64_lehmer_step(
	uint64_t q,
	uint64_t *u,
	uint64_t *v,
	uint32_t *a,
	uint32_t *b,
	uint32_t *c,
	uint32_t *d
) {
	uint32_t old_a = *a;
	uint32_t old_b = *b;
	uint32_t old_c = *c;
	uint32_t old_d = *d;
	uint64_t new_v = *u - q * (*v);
	uint64_t new_c = (uint64_t)old_a + q * (uint64_t)old_c;
	uint64_t new_d = (uint64_t)old_b + q * (uint64_t)old_d;

	*a = old_c;
	*b = old_d;
	*c = (uint32_t)new_c;
	*d = (uint32_t)new_d;

	*u = *v;
	*v = new_v;
}

int gcdhlp(uint32_t usize, uint32_t *udata, uint32_t vsize, uint32_t *vdata, void *retptr) {
	uint64_t U = 0;
	uint64_t V = 0;
	uint32_t A = 1, B = 0, C = 0, D = 1, ODD = 0;
	int half2_terminated = 0;
	uint32_t *ret;

	if (usize == 0 || vsize == 0) return 0;
	if (udata == NULL || vdata == NULL || retptr == NULL) return 0;
	if (usize < vsize) return 0;
	if ((usize - vsize) > 1) return 0;

	/* Exact short path: one or two limbs only. */
	if (usize <= 2) {
		if (usize == 1) {
			U = (uint64_t)udata[0];
			V = (uint64_t)vdata[0];
		} else {
			U = ((uint64_t)udata[1] << 32) | (uint64_t)udata[0];
			if (vsize == 2) {
				V = ((uint64_t)vdata[1] << 32) | (uint64_t)vdata[0];
			} else {
				/* usize > vsize: V top word is 0, second word is vdata[vsize-1]. */
				V = (uint64_t)vdata[0];
			}
		}

		while (V != 0) {
			uint64_t q = U / V;
			if (arm64_newd_overflow(B, D, q)) break;
			arm64_lehmer_step(q, &U, &V, &A, &B, &C, &D);
			ODD ^= 1U;
		}
	}

	/* Lehmer accelerated path using top normalized ~63 bits. */
	else {
		uint32_t u0, u1, u2;
		uint32_t v0, v1, v2;
		unsigned int shift;
		uint64_t top_u, top_v;

		u0 = udata[usize - 1];
		u1 = udata[usize - 2];
		u2 = udata[usize - 3];

		if (usize == vsize) {
			v0 = vdata[vsize - 1];
			v1 = vdata[vsize - 2];
			v2 = vdata[vsize - 3];
		} else {
			/* usize == vsize + 1 */
			v0 = 0;
			v1 = vdata[vsize - 1];
			v2 = (vsize >= 2) ? vdata[vsize - 2] : 0;
		}

		if (u0 == 0) return 0;

		shift = arm64_clz32(u0);

		top_u = ((uint64_t)u0 << 32) | (uint64_t)u1;
		top_v = ((uint64_t)v0 << 32) | (uint64_t)v1;

		if (shift != 0) {
			top_u = (top_u << shift) | ((uint64_t)u2 >> (32 - shift));
			top_v = (top_v << shift) | ((uint64_t)v2 >> (32 - shift));
		}

		/* Keep values signed-safe for Lehmer arithmetic. */
		U = top_u >> 1;
		V = top_v >> 1;

		while (V != 0) {
			uint64_t q;
			uint64_t denom;
			uint64_t span;
			uint64_t rhs;
			uint64_t factor;

			/* Half-step 1:
			   q = (U - B) / (V + D)
			   require (q+1)*(V-C) > U+A
			   require B + q*D < 2^32
			*/
			if (U < (uint64_t)B) break;
			denom = V + (uint64_t)D;
			if (denom == 0) break;

			q = (U - (uint64_t)B) / denom;

			if (V <= (uint64_t)C) break;
			span = V - (uint64_t)C;
			factor = q + 1;
			if (factor == 0) break; /* defensive */

			rhs = U + (uint64_t)A;
			if (factor <= (rhs / span)) break;

			if (arm64_newd_overflow(B, D, q)) break;

			arm64_lehmer_step(q, &U, &V, &A, &B, &C, &D);
			ODD ^= 1U;

			if (V == 0) break;

			/* Half-step 2:
			   q = (U - A) / (V + C)
			   require (q+1)*(V-D) > U+B
			   require B + q*D < 2^32
			*/
			if (U < (uint64_t)A) {
				half2_terminated = 1;
				break;
			}
			denom = V + (uint64_t)C;
			if (denom == 0) {
				half2_terminated = 1;
				break;
			}

			q = (U - (uint64_t)A) / denom;

			if (V <= (uint64_t)D) {
				half2_terminated = 1;
				break;
			}
			span = V - (uint64_t)D;
			factor = q + 1;
			if (factor == 0) {
				half2_terminated = 1;
				break;
			}

			rhs = U + (uint64_t)B;
			if (factor <= (rhs / span)) {
				half2_terminated = 1;
				break;
			}

			if (arm64_newd_overflow(B, D, q)) {
				half2_terminated = 1;
				break;
			}

			arm64_lehmer_step(q, &U, &V, &A, &B, &C, &D);
			ODD ^= 1U;
		}
	}

	/* x86 lpdone1 behavior: if half-step 2 terminated, force ODD=1. */
	if (half2_terminated) ODD = 1U;

	if (B == 0) return 0;

	ret = (uint32_t *)retptr;
	ret[0] = A;
	ret[1] = B;
	ret[2] = C;
	ret[3] = D;
	ret[4] = ODD;
	return 1;
}

#if defined(__GNUC__)
__attribute__((constructor))
static void arm64_cpuid_constructor(void) {
	arm64_detect_cpu();
}
#endif