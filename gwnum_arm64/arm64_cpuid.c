#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <sys/auxv.h>
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

#if defined(__GNUC__)
__attribute__((constructor))
static void arm64_cpuid_constructor(void) {
	arm64_detect_cpu();
}
#endif