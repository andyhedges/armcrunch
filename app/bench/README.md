# gwnum reference benchmark

Standalone C benchmark using the [gwnum](https://github.com/shafferjohn/Prime95) library (from Prime95) for modular squaring. Provides a performance reference target for the Rust implementations.

## Prerequisites

```bash
# 1. Clone Prime95 (gwnum source)
git clone --depth 1 https://github.com/shafferjohn/Prime95.git ~/Prime95

# 2. Install GMP development headers
sudo dnf install gmp-devel        # Fedora/RHEL
# or: sudo apt-get install libgmp-dev  # Debian/Ubuntu

# 3. Build gwnum
cd ~/Prime95/gwnum && make -f make64
```

## Build

```bash
cd armcrunch/app/bench
make -f Makefile.gwnum PRIME95_DIR=~/Prime95
```

## Run

```bash
./gwnum_bench                          # defaults: k=1003, n=2499999, 1000 iters
./gwnum_bench --k 1 --n 262144         # Mersenne 2^262144-1
./gwnum_bench --iters 10000            # more iterations for stability
```

## Example output

```
gwnum benchmark: 1003*2^2499999-1  (1000 squarings)
gwnum version: 30.19
FFT length: 196608

  Iterations:  1000
  Wall time:   0.364 s
  Per iter:    0.364 ms
  Projected:   0.3 hours
```

## Comparison with armcrunch methods

Run the Rust benchmark in a separate terminal:

```bash
cd armcrunch/app
cargo run --release --bin fermat_bench -- --duration 120
```

On the same 2-vCPU x86_64 sandbox for 1003*2^2499999-1:

| Method | Per iteration | Projected | Ratio |
|--------|-------------|-----------|-------|
| gwnum  | 0.36 ms     | 0.3h      | 1.0×  |
| fft    | 10.6 ms     | 7.4h      | 29×   |
| dwt    | 11.2 ms     | 7.8h      | 31×   |
| kbn    | 43.3 ms     | 30.1h     | 120×  |

The 30× gap is primarily due to gwnum's hand-tuned x86 SIMD assembly FFT,
mixed-radix transform lengths (e.g. 3×2^16), and true in-domain IBDWT that
avoids all BigUint allocation in the squaring loop.