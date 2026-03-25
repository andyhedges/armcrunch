/*
 * gwnum_bench.c — Benchmark gwnum modular squaring for k*2^n-1.
 *
 * Provides a performance reference for comparison with the Rust
 * implementations in armcrunch.
 *
 * Build:
 *   make -f Makefile.gwnum
 *
 * Usage:
 *   ./gwnum_bench [--k K] [--n N] [--iters ITERS]
 *
 * Defaults: k=1003, n=2499999, iters=1000
 *
 * Requires: gwnum.a built from Prime95 source (see Makefile.gwnum).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gwnum.h"

int main(int argc, char *argv[]) {
    double k = 1003.0;
    unsigned long n = 2499999;
    int iters = 1000;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = atof(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = atol(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--k K] [--n N] [--iters ITERS]\n", argv[0]);
            return 1;
        }
    }

    /* Validate arguments */
    if (k <= 0.0) {
        fprintf(stderr, "Error: k must be a positive number (got %.0f)\n", k);
        return 1;
    }
    if (n == 0) {
        fprintf(stderr, "Error: n must be a positive integer\n");
        return 1;
    }
    if (iters <= 0) {
        fprintf(stderr, "Error: iters must be a positive integer (got %d)\n", iters);
        return 1;
    }

    printf("gwnum benchmark: %.0f*2^%lu-1  (%d squarings)\n", k, n, iters);
    printf("gwnum version: %s\n", GWNUM_VERSION);

    /* Initialize gwnum for k*2^n-1 */
    gwhandle gwdata;
    gwinit(&gwdata);
    gwset_num_threads(&gwdata, 1);  /* Single-threaded for fair comparison */

    int err = gwsetup(&gwdata, k, 2, n, -1);
    if (err) {
        fprintf(stderr, "gwsetup failed with error %d\n", err);
        return 1;
    }

    printf("FFT length: %lu\n\n", (unsigned long)gwdata.FFTLEN);

    /* Allocate and initialize x = 3 */
    gwnum x = gwalloc(&gwdata);
    if (!x) {
        fprintf(stderr, "gwalloc failed\n");
        gwdone(&gwdata);
        return 1;
    }
    dbltogw(&gwdata, 3.0, x);

    /* Warm up */
    for (int i = 0; i < 10; i++) {
        gwsquare2(&gwdata, x, x, 0);
    }

    /* Timed benchmark */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < iters; i++) {
        gwsquare2(&gwdata, x, x, 0);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ms_per = elapsed * 1000.0 / iters;
    double projected_secs = (ms_per / 1000.0) * (double)n;

    printf("  Iterations:  %d\n", iters);
    printf("  Wall time:   %.3f s\n", elapsed);
    printf("  Per iter:    %.3f ms\n", ms_per);

    if (projected_secs < 3600.0)
        printf("  Projected:   %.1f minutes\n", projected_secs / 60.0);
    else if (projected_secs < 86400.0)
        printf("  Projected:   %.1f hours\n", projected_secs / 3600.0);
    else
        printf("  Projected:   %.1f days\n", projected_secs / 86400.0);

    /* Machine-parseable result line for scripted comparisons */
    printf("\nGWNUM_RESULT %.6f\n", ms_per);

    /* Cleanup */
    gwfree(&gwdata, x);
    gwdone(&gwdata);

    return 0;
}