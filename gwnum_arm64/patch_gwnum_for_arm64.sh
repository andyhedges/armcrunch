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
#   3. Replace gwinfo1() call with arm64_gwinfo_hook() on ARM64
#   4. Insert arm64_skip_version_check label before version sprintf
#   5. Guard fpu_init() call
#   6. Insert arm64_gwsetup_hook before x86 GWPROCPTRS block, with #else/#endif
#   7. Guard pass1/pass2_aux_entry_point declarations

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
        out.append('extern const struct gwasm_jmptab *arm64_gwinfo_hook(gwhandle *gwdata, int negacyclic);\n')
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

    # End: void pass2_aux_entry_point
    if in_x86_externs and 'pass2_aux_entry_point' in line:
        out.append(line)
        out.append('#endif /* !ARM64 - x86 assembly externs */\n')
        in_x86_externs = False
        i += 1
        continue

    # 3. Replace gwinfo1(&asm_info) with ARM64 hook
    if 'gwinfo1 (&asm_info);' in line and 'gwinfo1' in line:
        indent = line[:len(line) - len(line.lstrip())]
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append(indent + '{ const struct gwasm_jmptab *arm64_tab = arm64_gwinfo_hook(gwdata, gwdata->NEGACYCLIC_FFT);\n')
        out.append(indent + '  if (arm64_tab == NULL) return (GWERROR_VERSION);\n')
        out.append(indent + '  jmptab = arm64_tab; goto arm64_skip_version_check; }\n')
        out.append('#else\n')
        out.append(line)
        out.append('#endif\n')
        i += 1
        continue

    # 4. Insert arm64_skip_version_check label before version sprintf
    if 'sprintf (buf, "%d.%d", asm_info.version' in line:
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append('arm64_skip_version_check:;\n')
        out.append('#endif\n')
        out.append(line)
        i += 1
        continue

    # 5. Guard fpu_init()
    if 'fpu_init ()' in line and '#' not in line:
        indent = line[:len(line) - len(line.lstrip())]
        out.append('#if !defined(ARM64) && !defined(__aarch64__)\n')
        out.append(line)
        out.append('#endif\n')
        i += 1
        continue

    # 6. Insert arm64_gwsetup_hook before x86 GWPROCPTRS assignment block
    if '/* Set the procedure pointers from the proc tables */' in line:
        out.append(line)
        out.append('\n')
        out.append('#if defined(ARM64) || defined(__aarch64__)\n')
        out.append('\tarm64_gwsetup_hook(gwdata);\n')
        out.append('#else\n')
        gwprocptrs_else_open = True
        i += 1
        continue

    # Close the ARM64 #else block after the x87 #endif, before "Default normalization"
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

echo "Copying other unmodified sources..."
for f in gwtables.c gwthread.cpp gwini.c gwbench.c gwutil.c gwdbldbl.cpp giants.c radix.c; do
    if [ -f "$GWNUM_SRC/$f" ]; then
        cp "$GWNUM_SRC/$f" "$OUT_DIR/$f"
    fi
done

echo "Done. Patched sources in $OUT_DIR"