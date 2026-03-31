#!/bin/bash
# patch_gwnum_for_arm64.sh
# Applies ARM64 modifications to gwnum.c for building on AArch64.
# Usage: ./patch_gwnum_for_arm64.sh <gwnum_src_dir> <output_dir>
#
# This script reads the original gwnum sources and writes patched copies
# to the output directory. The originals are not modified.
#
# Patches applied to gwnum.c:
#   1. Add ARM64 includes and extern declarations after gwbench.h include
#   2. Guard x86 assembly extern declarations (lines 113-440) with #if !ARM64
#   3. Replace gwinfo1() call with arm64_gwinfo_hook() that returns immediately on ARM64
#   4. Insert arm64_gwsetup_hook before x86 GWPROCPTRS block, with #else/#endif
#   5. Guard pass1/pass2_aux_entry_point declarations

set -euo pipefail

GWNUM_SRC="${1:?Usage: $0 <gwnum_src_dir> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <gwnum_src_dir> <output_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"

echo "Patching gwnum.c for ARM64..."

# We apply patches via a Python script for more reliable multi-line editing
python3 - "$GWNUM_SRC/gwnum.c" "$OUT_DIR/gwnum.c" << 'PYSCRIPT'
import sys
import re

src_path = sys.argv[1]
dst_path = sys.argv[2]

with open(src_path, 'r') as f:
    lines = f.readlines()

out = []
i = 0
n = len(lines)

# State tracking
in_x86_externs = False
x86_extern_start = False
gwprocptrs_else_open = False

while i < n:
    line = lines[i]

    # 1. After #include "gwbench.h", add ARM64 includes
    if '#include "gwbench.h"' in line:
        out.append(line)
        out.append('\n')
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append('#include "arm64_asm_data.h"\n')
        out.append('extern int arm64_gwinfo_hook(gwhandle *gwdata, int negacyclic);\n')
        out.append('extern void arm64_gwsetup_hook(gwhandle *gwdata);\n')
        out.append('#endif\n')
        i += 1
        continue

    # 2. Guard x86 assembly extern declarations block
    # Start: #define extern_decl(name)
    if line.startswith('#define extern_decl(name)') and not in_x86_externs:
        out.append('#if !defined(ARM64) && !defined(__aarch64__)\n')
        out.append(line)
        in_x86_externs = True
        i += 1
        continue

    # End: "/* Helper macros */" comment (after all prctab code)
    if in_x86_externs and '/* Helper macros */' in line:
        out.append('#endif /* !ARM64 - x86 assembly externs */\n')
        out.append('\n')
        # Add ARM64 no-op stubs for x86 assembly functions called from unguarded code
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append('/* ARM64 stubs for x86 assembly-only functions */\n')
        out.append('static inline void gwz3_apply_carries(void *d) { (void)d; }\n')
        out.append('static inline void gwy3_apply_carries(void *d) { (void)d; }\n')
        out.append('static inline void prefetchL2(void *addr, int count) { (void)addr; (void)count; }\n')
        out.append('static inline void pause_for_count(int count) { (void)count; }\n')
        out.append('#endif\n')
        out.append('\n')
        out.append(line)
        in_x86_externs = False
        i += 1
        continue

    # 3. Guard the original gwinfo1 declaration
    if line.strip().startswith('void gwinfo1 (struct gwinfo1_data'):
        out.append('#if !defined(ARM64) && !defined(__aarch64__)\n')
        out.append(line)
        out.append('#endif\n')
        i += 1
        continue

    # 3b. Guard prefetchL2 and pause_for_count declarations (they get ARM64 stubs above)
    if line.strip().startswith('void prefetchL2 (') or line.strip().startswith('void pause_for_count ('):
        out.append('#if !defined(ARM64) && !defined(__aarch64__)\n')
        out.append(line)
        out.append('#endif\n')
        i += 1
        continue

    # 3c. Guard pass1_aux_entry_point and pass2_aux_entry_point forward declarations
    if line.strip().startswith('void pass1_aux_entry_point'):
        out.append('#if !defined(ARM64) && !defined(__aarch64__)\n')
        out.append(line)
        i += 1
        if i < n and 'pass2_aux_entry_point' in lines[i]:
            out.append(lines[i])
            i += 1
        out.append('#else\n')
        out.append('static inline void pass1_aux_entry_point(void *d) { (void)d; }\n')
        out.append('static inline void pass2_aux_entry_point(void *d) { (void)d; }\n')
        out.append('#endif\n')
        continue

    # 4. Replace gwinfo1(&asm_info) call with ARM64 hook that returns immediately
    if 'gwinfo1 (&asm_info);' in line and 'gwinfo1' in line:
        indent = line[:len(line) - len(line.lstrip())]
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append(indent + '{ int arm64_rc = arm64_gwinfo_hook(gwdata, gwdata->NEGACYCLIC_FFT);\n')
        out.append(indent + '  if (arm64_rc) return arm64_rc;\n')
        out.append(indent + '  return 0; }\n')
        out.append('#else\n')
        out.append(line)
        out.append('#endif\n')
        i += 1
        continue

    # 5. Insert arm64_gwsetup_hook before x86 GWPROCPTRS assignment block
    if '/* Set the procedure pointers from the proc tables */' in line:
        out.append(line)
        out.append('\n')
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append('\tarm64_gwsetup_hook(gwdata);\n')
        out.append('#else\n')
        gwprocptrs_else_open = True
        i += 1
        continue

    # 6. Skip gwmul3_carefully on ARM64 by guarding the careful_count check
    if 'if (gwdata->careful_count > 0)' in line and 'careful_count--' not in line:
        out.append('#if !defined(ARM64) && !defined(__aarch64__)\n')
        out.append(line)
        i += 1
        while i < n:
            out.append(lines[i])
            if lines[i].strip() == '}':
                i += 1
                break
            i += 1
        out.append('#endif\n')
        continue

    # Close the ARM64 #else block before "Default normalization"
    if gwprocptrs_else_open and '/* Default normalization routines and behaviors */' in line:
        out.append('#endif /* !ARM64 - GWPROCPTRS */\n')
        out.append('\n')
        out.append(line)
        gwprocptrs_else_open = False
        i += 1
        continue

    # Default: pass through unchanged
    out.append(line)
    i += 1

with open(dst_path, 'w') as f:
    f.writelines(out)

print(f"  Patched {len(lines)} -> {len(out)} lines")
PYSCRIPT

echo "Copying cpuid replacement..."
cp "$SCRIPT_DIR/arm64_cpuid.c" "$OUT_DIR/cpuid.c" 2>/dev/null || \
    echo "  Warning: arm64_cpuid.c not found at $SCRIPT_DIR, cpuid.c not replaced"

echo "Copying and patching other sources..."
for f in gwtables.c gwini.c gwbench.c gwutil.c giants.c radix.c; do
    if [ -f "$GWNUM_SRC/$f" ]; then
        cp "$GWNUM_SRC/$f" "$OUT_DIR/$f"
    fi
done

# Patch gwdbldbl.cpp: prevent #define x86 on ARM64 (disables x87 FPU control word asm)
if [ -f "$GWNUM_SRC/gwdbldbl.cpp" ]; then
    sed \
      -e 's/^#ifndef X86_64$/#if !defined(X86_64) \&\& !defined(ARM64) \&\& !defined(__aarch64__)/' \
      "$GWNUM_SRC/gwdbldbl.cpp" > "$OUT_DIR/gwdbldbl.cpp"
    echo "  Patched gwdbldbl.cpp (x86 FPU guard for ARM64)"
fi

# Patch gwthread.cpp: replace __builtin_ia32_pause() with ARM64 yield
if [ -f "$GWNUM_SRC/gwthread.cpp" ]; then
    sed \
      -e 's/__builtin_ia32_pause();/__asm__ volatile("yield");/' \
      "$GWNUM_SRC/gwthread.cpp" > "$OUT_DIR/gwthread.cpp"
    echo "  Patched gwthread.cpp (ia32_pause -> yield)"
fi

echo "Done. Patched sources in $OUT_DIR"