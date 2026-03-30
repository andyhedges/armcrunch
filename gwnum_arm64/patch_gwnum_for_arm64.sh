#!/bin/bash
# patch_gwnum_for_arm64.sh
# Applies ARM64 modifications to gwnum.c and cpuid.c for building on AArch64.
# Usage: ./patch_gwnum_for_arm64.sh <gwnum_src_dir> <output_dir>
#
# This script reads the original gwnum sources and writes patched copies
# to the output directory. The originals are not modified.

set -euo pipefail

GWNUM_SRC="${1:?Usage: $0 <gwnum_src_dir> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <gwnum_src_dir> <output_dir>}"

mkdir -p "$OUT_DIR"

echo "Patching gwnum.c for ARM64..."

# Patch gwnum.c
sed \
  -e '/#include "gwbench.h"/a\
\
#if defined(ARM64) || defined(__aarch64__)\
#include "arm64_asm_data.h"\
extern const struct gwasm_jmptab *arm64_gwinfo_hook(gwhandle *gwdata, int negacyclic);\
extern void arm64_gwsetup_hook(gwhandle *gwdata);\
#endif' \
  -e '/^#define extern_decl/i\
#if !defined(ARM64) && !defined(__aarch64__)' \
  -e '/^void pass2_aux_entry_point/a\
#endif /* !ARM64 */' \
  -e 's/^\([ \t]*\)gwinfo1 (&asm_info);/\
#if defined(ARM64) || defined(__aarch64__)\
\1{ const struct gwasm_jmptab *arm64_tab = arm64_gwinfo_hook(gwdata, gwdata->NEGACYCLIC_FFT);\
\1  if (arm64_tab == NULL) return (GWERROR_VERSION);\
\1  jmptab = arm64_tab; goto arm64_skip_version_check; }\
#else\
\1gwinfo1 (\&asm_info);\
#endif/' \
  -e "/sprintf (buf, \"%d.%d\", asm_info.version/i\\
#if defined(ARM64) || defined(__aarch64__)\\
arm64_skip_version_check:;\\
#endif" \
  -e 's/^\([ \t]*\)fpu_init ();/\
#if !defined(ARM64) \&\& !defined(__aarch64__)\
\1fpu_init ();\
#endif/' \
  -e '/Set the procedure pointers from the proc tables/,/^$/{
/Set the procedure pointers from the proc tables/a\
\
#if defined(ARM64) || defined(__aarch64__)\
\tarm64_gwsetup_hook(gwdata);\
#else
}' \
  -e '/gwdata->GWPROCPTRS\[norm_routines\+3\] = x87_prctab/a\
#endif /* !ARM64 */' \
  -e 's/^\([ \t]*\)void pass1_aux_entry_point/\
#if !defined(ARM64) \&\& !defined(__aarch64__)\
\1void pass1_aux_entry_point/' \
  -e '/pass2_aux_entry_point (asm_data);/{N;s/$/\
/}' \
  "$GWNUM_SRC/gwnum.c" > "$OUT_DIR/gwnum.c"

echo "Patching cpuid.c for ARM64..."

# For cpuid.c, we simply use arm64_cpuid.c instead
cp "$GWNUM_SRC/../gwnum_arm64/arm64_cpuid.c" "$OUT_DIR/cpuid.c" 2>/dev/null || true

echo "Copying other unmodified sources..."
for f in gwtables.c gwthread.cpp gwini.c gwbench.c gwutil.c gwdbldbl.cpp giants.c radix.c; do
    if [ -f "$GWNUM_SRC/$f" ]; then
        cp "$GWNUM_SRC/$f" "$OUT_DIR/$f"
    fi
done

echo "Done. Patched sources in $OUT_DIR"
