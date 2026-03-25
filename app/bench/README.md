# gwnum reference benchmark

Standalone C benchmark using the [gwnum](https://github.com/shafferjohn/Prime95) library (from Prime95) for modular squaring. Provides a performance reference target for the Rust implementations.

## Quick start (pre-built binaries)

Pre-built binaries are available via GitHub Releases for three platforms:

| Binary | Platform | FFT implementation |
|--------|----------|-------------------|
| `gwnum_bench-linux-amd64` | Linux x86_64 | x86 SSE2/AVX2 assembly |
| `gwnum_bench-macos-amd64` | macOS x86_64 (or Apple Silicon via Rosetta) | x86 SSE2/AVX2 assembly |
| `gwnum_bench-macos-arm64` | macOS Apple Silicon (native) | Generic C (no x86 assembly) |

### Download

```bash
cd armcrunch/app
./bench/download_gwnum.sh
```

On Apple Silicon Macs, this downloads both the native ARM64 and x86_64 (Rosetta) binaries for side-by-side comparison.

### Run with fermat_bench

Once downloaded, `fermat_bench` automatically detects and runs the gwnum binaries:

```bash
cargo run --release --bin fermat_bench -- --duration 60
```

### Run standalone

```bash
./bench/gwnum_bench-linux-amd64 --iters 1000
./bench/gwnum_bench-macos-arm64 --iters 1000
```

## Building from source

If you want to build locally instead of using pre-built binaries:

### Prerequisites

```bash
# 1. Clone Prime95 (gwnum source)
git clone --depth 1 https://github.com/shafferjohn/Prime95.git ~/Prime95

# 2. Install GMP development headers
sudo dnf install gmp-devel        # Fedora/RHEL
sudo apt-get install libgmp-dev   # Debian/Ubuntu
brew install gmp                  # macOS

# 3. Build gwnum
cd ~/Prime95/gwnum && make -f make64    # Linux
cd ~/Prime95/gwnum && make -f makemac   # macOS x86_64
```

### Build

```bash
cd armcrunch/app/bench
make -f Makefile.gwnum PRIME95_DIR=~/Prime95
```

## Creating a new release

The GitHub Actions workflow builds binaries for all three platforms:

```bash
git tag gwnum-v1
git push origin gwnum-v1
```

This triggers `.github/workflows/gwnum-bench.yml` which builds, tests, and uploads
binaries to a GitHub Release.

## Example output

On a 2-vCPU x86_64 sandbox for 1003*2^2499999-1:

| Method | Per iteration | Projected | Ratio |
|--------|-------------|-----------|-------|
| gwnum-x86 | 0.36 ms | 0.3h | 1.0× |
| fft | 10.6 ms | 7.4h | 29× |
| dwt | 11.2 ms | 7.8h | 31× |
| kbn | 43.3 ms | 30.1h | 120× |

The 30× gap on x86 is primarily due to gwnum's hand-tuned SIMD assembly FFT,
mixed-radix transform lengths (e.g. 3×2^16), and true in-domain IBDWT that
avoids all BigUint allocation in the squaring loop. On ARM64 the gap will be
much smaller since gwnum falls back to generic C code.