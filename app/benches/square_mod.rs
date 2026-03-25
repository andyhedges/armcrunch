use armcrunch::{square_mod, square_mod_kbn, square_mod_naive, DwtSquarer, FftSquarer, Kbn, NegacyclicSquarer, NttSquarer};
use criterion::{criterion_group, criterion_main, Criterion};
use num_bigint::BigUint;

fn bench_square_mod(c: &mut Criterion) {
    // Use 1005*2^33144-1 as a representative modulus (~10000 digit number)
    let kbn = Kbn::new(1005, 33144);
    let modulus = kbn.value();
    let mut x = BigUint::from(3u32).modpow(&BigUint::from(1005u32), &modulus);
    let mut squarer = FftSquarer::new(&modulus);
    let n_bytes = (modulus.bits() as usize + 7) / 8;
    let ntt_len = (2 * n_bytes).next_power_of_two();
    let mut squarer_ntt = NttSquarer::new(ntt_len);
    let mut squarer_negacyclic = NegacyclicSquarer::new(ntt_len);
    let mut squarer_dwt = DwtSquarer::new(kbn.k, kbn.n, &x);

    let mut group = c.benchmark_group("square_mod");

    group.bench_function("naive", |b| {
        b.iter(|| square_mod_naive(&mut x, &modulus));
    });

    group.bench_function("current", |b| {
        b.iter(|| square_mod(&mut x, &modulus));
    });

    group.bench_function("kbn", |b| {
        b.iter(|| square_mod_kbn(&mut x, 1005, 33144, &modulus));
    });

    group.bench_function("fft", |b| {
        b.iter(|| squarer.square_kbn(&mut x, 1005, 33144, &modulus));
    });

    group.bench_function("ntt", |b| {
        b.iter(|| squarer_ntt.square_kbn(&mut x, 1005, 33144, &modulus));
    });

    group.bench_function("negacyclic_ntt", |b| {
        b.iter(|| x = squarer_negacyclic.fft_square(&x) % &modulus);
    });

    group.bench_function("dwt", |b| {
        b.iter(|| squarer_dwt.square());
    });

    group.bench_function("dwt_extract", |b| {
        b.iter(|| squarer_dwt.to_biguint());
    });

    group.finish();
}

fn bench_fft_phases(c: &mut Criterion) {
    let kbn = Kbn::new(1005, 33144);
    let modulus = kbn.value();
    let x = BigUint::from(3u32).modpow(&BigUint::from(1005u32), &modulus);
    let mut squarer = FftSquarer::new(&modulus);

    // Warm up
    let mut xw = x.clone();
    squarer.square_kbn(&mut xw, 1005, 33144, &modulus);

    // Sample phases over many iterations and print averages
    let n = 1000;
    let mut totals = [0u64; 7];
    let mut xi = x.clone();
    for _ in 0..n {
        let t = squarer.profile_phases(&xi, 1005, 33144, &modulus);
        for (a, b) in totals.iter_mut().zip(t) { *a += b; }
        squarer.square_kbn(&mut xi, 1005, 33144, &modulus);
    }

    let labels = ["to_bytes", "fill_signal", "fft", "pointwise", "ifft", "carry", "from_bytes"];
    println!("\nFFT phase breakdown ({n} iterations):");
    for (label, total) in labels.iter().zip(totals) {
        println!("  {label:12}: {:5} µs/call", total / n);
    }

    let mut group = c.benchmark_group("fft_phases");
    group.bench_function("noop", |b| b.iter(|| {}));
    group.finish();
}

/// Benchmark at 8x the "small" size to show asymptotic separation.
///
/// At 262K bits the O(n) per-call overhead in FftSquarer (bytes conversion,
/// BigUint allocation, Kbn mod) becomes a larger fraction of total time versus
/// the O(n log n) FFT work — widening the gap over DwtSquarer, which keeps x
/// in the transform domain and does no allocations in the hot path.
///
/// naive and NTT variants are excluded: they are already too slow at this size.
fn bench_square_mod_large(c: &mut Criterion) {
    let k = 1u64;
    let exp = 262144u64; // 2^18 — clean power-of-2, b_lo is an exact integer
    let kbn = Kbn::new(k, exp);
    let modulus = kbn.value();
    let mut x = BigUint::from(3u32).modpow(&BigUint::from(1000u32), &modulus);
    let mut squarer_fft = FftSquarer::new(&modulus);
    let mut squarer_dwt = DwtSquarer::new(k, exp, &x);

    let mut group = c.benchmark_group("square_mod_large");
    group.sample_size(20);

    group.bench_function("kbn", |b| {
        b.iter(|| square_mod_kbn(&mut x, k, exp, &modulus));
    });

    group.bench_function("fft", |b| {
        b.iter(|| squarer_fft.square_kbn(&mut x, k, exp, &modulus));
    });

    group.bench_function("dwt", |b| {
        b.iter(|| squarer_dwt.square());
    });

    group.finish();
}

criterion_group!(benches, bench_square_mod, bench_fft_phases, bench_square_mod_large);
criterion_main!(benches);