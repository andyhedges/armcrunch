cd ~/code/armcrunch

cat << 'EOF' > ~/code/prst/src/macarm64/Makefile
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
                  -DGMP -DARM64 -Wall -Wextra

CFLAGS   = -std=c99
CXXFLAGS = -std=gnu++17

# Suppress noisy warnings from upstream code
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

cd gwnum_arm64
make -f makemacarm64 clean
make -f makemacarm64

cp gwnum.a ~/code/prst/framework/gwnum/macarm64/

cd ~/code/prst/src/macarm64
make clean

# Patch framework's logging.h in-place to handle missing params gracefully.
# Save backup first, then restore after build.
# PRST's Progress::param_double/param_int call std::stod/stoi on potentially
# empty strings when a .param file doesn't exist on first run.
LOGGING_H="../../framework/logging.h"
cp "$LOGGING_H" "${LOGGING_H}.bak"

python3 -c "
with open('${LOGGING_H}') as f:
    content = f.read()
content = content.replace(
    'return std::stod(_params[name], nullptr);',
    'auto it = _params.find(name); return (it != _params.end() && !it->second.empty()) ? std::stod(it->second) : 0.0;'
)
content = content.replace(
    'return std::stoi(_params[name], nullptr, 10);',
    'auto it = _params.find(name); return (it != _params.end() && !it->second.empty()) ? std::stoi(it->second) : 0;'
)
with open('${LOGGING_H}', 'w') as f:
    f.write(content)
"

echo "=== Verifying patched logging.h ==="
grep 'param_double\|param_int' "$LOGGING_H"
echo "=== End verify ==="

# Belt-and-suspenders: ensure no local logging.h exists right before build
rm -f logging.h

make

# Restore original logging.h
cp "${LOGGING_H}.bak" "$LOGGING_H"
rm -f "${LOGGING_H}.bak"

./prst "2^61-1"
