#!/bin/bash
set -e

# Configurable paths
ARMCRUNCH_DIR="${ARMCRUNCH_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PRST_DIR="${PRST_DIR:-$HOME/code/prst}"

cd "$ARMCRUNCH_DIR"

# Step 1: Build gwnum.a for ARM64
cd gwnum_arm64
make -f makemacarm64 clean
rm -f build/macarm64/patched/.patched
make -f makemacarm64
echo "=== gwnum.a built successfully ==="

# Step 2: If PRST is available, build and test it
if [ -d "$PRST_DIR/src" ]; then
    echo "=== PRST found at $PRST_DIR, building... ==="

    # Ensure submodules are initialized (framework/arithmetic)
    if [ ! -f "$PRST_DIR/framework/md5.c" ]; then
        echo "  Framework submodule not found, initializing..."
        cd "$PRST_DIR"
        git submodule update --init --recursive 2>/dev/null || true
        if [ ! -f "$PRST_DIR/framework/md5.c" ]; then
            echo "  Submodule init failed, cloning framework directly..."
            rm -rf "$PRST_DIR/framework"
            git clone --depth 1 https://github.com/patnashev/arithmetic.git "$PRST_DIR/framework" 2>/dev/null || true
        fi
        cd "$ARMCRUNCH_DIR/gwnum_arm64"
    fi

    if [ ! -f "$PRST_DIR/framework/md5.c" ]; then
        echo "=== ERROR: PRST framework sources not found, skipping PRST build ==="
        exit 1
    fi

    # Install gwnum.a
    mkdir -p "$PRST_DIR/framework/gwnum/macarm64"
    cp gwnum.a "$PRST_DIR/framework/gwnum/macarm64/"

    # Copy gwnum headers needed by PRST
    cp "$ARMCRUNCH_DIR/reference/prime95/gwnum/"*.h "$PRST_DIR/framework/gwnum/" 2>/dev/null || true
    cp "$ARMCRUNCH_DIR/gwnum_arm64/arm64_asm_data.h" "$PRST_DIR/framework/gwnum/" 2>/dev/null || true

    # Write Makefile
    mkdir -p "$PRST_DIR/src/macarm64"
    cat << 'EOF' > "$PRST_DIR/src/macarm64/Makefile"
CC      = clang
CXX     = clang++
AR      = ar
RM      = rm -f

EXE       = prst
LIB_GWNUM = ../../framework/gwnum/macarm64/gwnum.a

COMPOBJS_COMMON = md5.o arithmetic.o group.o giant.o lucas.o config.o inputnum.o integer.o logging.o file.o container.o task.o exp.o fermat.o order.o pocklington.o lucasmul.o morrison.o proof.o testing.o support.o batch.o
COMPOBJS        = $(COMPOBJS_COMMON) prst.o

VPATH     = ..:../../framework:../../framework/arithmetic

LIBS      = -L/opt/homebrew/lib -lm -lpthread -lgmp

COMMON_CFLAGS   = -arch arm64 -O2 \
                  -I.. -I../../framework -I../../framework/arithmetic -I../../framework/gwnum \
                  -I/opt/homebrew/include \
                  -DGMP -DARM64 -DNDEBUG -Wall -Wextra

CFLAGS   = -std=c99
CXXFLAGS = -std=gnu++17

CXXFLAGS += -Wno-sign-compare -Wno-unused-parameter -Wno-unused-private-field

LDFLAGS  = -arch arm64

all: $(EXE)

%.o: %.c
	$(CC)  $(COMMON_CFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(COMMON_CFLAGS) $(CXXFLAGS) -c $< -o $@

$(EXE): $(COMPOBJS)
	$(CXX) $(LDFLAGS) -o $@ $(COMPOBJS) $(LIB_GWNUM) $(LIBS)

clean:
	$(RM) $(EXE) $(COMPOBJS)

.PHONY: all clean
EOF

    cd "$PRST_DIR/src/macarm64"
    make clean 2>/dev/null || true
    rm -f logging.h

    # Patch framework's logging.h for missing param handling
    LOGGING_H="$PRST_DIR/framework/logging.h"
    if [ -f "$LOGGING_H" ]; then
        python3 -c "
with open('$LOGGING_H') as f:
    content = f.read()
if 'std::stod(_params[name], nullptr)' in content:
    content = content.replace(
        'return std::stod(_params[name], nullptr);',
        'auto it = _params.find(name); return (it != _params.end() && !it->second.empty()) ? std::stod(it->second) : 0.0;'
    )
    content = content.replace(
        'return std::stoi(_params[name], nullptr, 10);',
        'auto it = _params.find(name); return (it != _params.end() && !it->second.empty()) ? std::stoi(it->second) : 0;'
    )
    with open('$LOGGING_H', 'w') as f:
        f.write(content)
    print('  Patched logging.h')
else:
    print('  logging.h already patched')
"
    fi

    rm -f logging.h
    make

    echo "=== PRST built successfully ==="
    ARM64_PRST="$PRST_DIR/src/macarm64/prst"

    # Validate ARM64 binary exists and is executable
    if [ ! -x "$ARM64_PRST" ]; then
        echo "ERROR: ARM64 PRST binary not found or not executable at $ARM64_PRST"
        exit 1
    fi

    # Step 3: Benchmark comparison
    BENCHMARKS=(
        "2^1279-1"
        "3*2^4000-1"
        "5*2^10000-1"
        "7*2^1000-1"
        "11*2^5000-1"
    )

    echo ""
    echo "============================================"
    echo "  BENCHMARK: ARM64 native"
    echo "============================================"
    echo ""

    echo "ARM64 binary:  $ARM64_PRST"
    echo ""

    # Create a temp directory for benchmark runs to avoid checkpoint interference
    BENCH_DIR=$(mktemp -d)

    BENCH_FAILURES=0

    for NUM in "${BENCHMARKS[@]}"; do
        echo "--- $NUM ---"

        # Clean any leftover checkpoint files
        rm -f "$BENCH_DIR"/*.txt "$BENCH_DIR"/*.param 2>/dev/null

        # ARM64 native build
        ARM64_TIME=""
        ARM64_RESULT=""
        cd "$BENCH_DIR"
        if ARM64_OUT=$("$ARM64_PRST" "$NUM" 2>&1); then
            ARM64_TIME=$(echo "$ARM64_OUT" | grep -o 'Time: [0-9.]*' | head -1 | awk '{print $2}')
            ARM64_RESULT=$(echo "$ARM64_OUT" | grep -E 'prime|composite' | head -1)
        else
            ARM64_EXIT=$?
            # PRST returns exit code 2 for "prime found" which is not an error
            if [ $ARM64_EXIT -eq 2 ]; then
                ARM64_TIME=$(echo "$ARM64_OUT" | grep -o 'Time: [0-9.]*' | head -1 | awk '{print $2}')
                ARM64_RESULT=$(echo "$ARM64_OUT" | grep -E 'prime|composite' | head -1)
            else
                echo "  ARM64:   FAILED (exit code $ARM64_EXIT)"
                echo "$ARM64_OUT" | grep '\[ARM64' || true
                BENCH_FAILURES=$((BENCH_FAILURES + 1))
            fi
        fi

        # Report results
        if [ -n "$ARM64_TIME" ]; then
            echo "  ARM64:   ${ARM64_TIME}s  $ARM64_RESULT"
        fi
        echo ""
    done

    # Cleanup temp directory
    rm -rf "$BENCH_DIR"

    echo "============================================"

    if [ "$BENCH_FAILURES" -gt 0 ]; then
        echo "WARNING: $BENCH_FAILURES benchmark(s) failed"
    fi

else
    echo "=== PRST not found at $PRST_DIR, skipping PRST build ==="
    echo "=== gwnum.a is ready at $ARMCRUNCH_DIR/gwnum_arm64/gwnum.a ==="
fi