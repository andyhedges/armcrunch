//! Fermat test benchmark for 1003*2^2499999-1.
//!
//! Runs a configurable number of squaring iterations with each viable method
//! (fft, dwt, kbn), verifies they produce identical results, and projects
//! total runtime for a full Fermat test.

use armcrunch::{fermat_test, fermat_test_dwt, Kbn, FftSquarer, DwtSquarer};
use num_bigint::BigUint;
use num_traits::One;
use std::time::Instant;

const K: u64 = 1003;
const N: u64 = 2_499_999;
const BASE: u64 = 3;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut iters: u64 = 100;
    let mut full_method: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                iters = args.get(i).expect("--iters requires a number")
                    .parse().expect("--iters requires a number");
            }
            "--full" => {
                i += 1;
                full_method = Some(args.get(i).expect("--full requires a method name").clone());
            }
            _ => {
                eprintln!("Usage: fermat_bench [--iters N] [--full fft|dwt]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let kbn = Kbn::new(K, N);
    println!("Target: {}*2^{}-1 (~{} decimal digits)",
        K, N, (N as f64 * 0.30103 + (K as f64).log10()).ceil() as u64);
    println!("Fermat test requires {} squarings\n", N);

    if let Some(method) = full_method {
        run_full_test(&kbn, &method);
    } else {
        run_comparison(&kbn, iters);
    }
}

struct MethodResult {
    name: &'static str,
    value: BigUint,
    elapsed_secs: f64,
    iters: u64,
}

impl MethodResult {
    fn ms_per_iter(&self) -> f64 {
        self.elapsed_secs * 1000.0 / self.iters as f64
    }

    fn projected_hours(&self) -> f64 {
        (self.ms_per_iter() / 1000.0) * N as f64 / 3600.0
    }

    fn res64(&self) -> u64 {
        self.value.iter_u64_digits().next().unwrap_or(0)
    }
}

fn run_comparison(kbn: &Kbn, iters: u64) {
    let modulus = kbn.value();
    let a_big = BigUint::from(BASE);

    println!("Computing a^k mod M (k={})...", K);
    let t0 = Instant::now();
    let x0 = a_big.modpow(&BigUint::from(K), &modulus);
    println!("  done in {:.2}s\n", t0.elapsed().as_secs_f64());

    println!("Running {} squaring iterations with each method:\n", iters);

    let mut results: Vec<MethodResult> = Vec::new();

    // --- FftSquarer (fft) ---
    {
        let mut x = x0.clone();
        let mut squarer = FftSquarer::new(&modulus);
        let t = Instant::now();
        for _ in 0..iters {
            squarer.square_kbn(&mut x, K, N, &modulus);
        }
        let elapsed = t.elapsed().as_secs_f64();
        results.push(MethodResult { name: "fft", value: x, elapsed_secs: elapsed, iters });
    }

    // --- DwtSquarer (dwt) ---
    {
        let mut squarer = DwtSquarer::new(K, N, &x0);
        let t = Instant::now();
        for _ in 0..iters {
            squarer.square();
        }
        let elapsed = t.elapsed().as_secs_f64();
        let x = squarer.to_biguint();
        results.push(MethodResult { name: "dwt", value: x, elapsed_secs: elapsed, iters });
    }

    // --- kbn (BigUint multiply + kbn reduce) ---
    {
        let mut x = x0.clone();
        let t = Instant::now();
        for _ in 0..iters {
            armcrunch::square_mod_kbn(&mut x, K, N, &modulus);
        }
        let elapsed = t.elapsed().as_secs_f64();
        results.push(MethodResult { name: "kbn", value: x, elapsed_secs: elapsed, iters });
    }

    // Print results
    for r in &results {
        println!("  {:4}: {:.3}s for {} iters ({:.3} ms/iter), projected: {:.1} hours, RES64: {:016X}",
            r.name, r.elapsed_secs, r.iters, r.ms_per_iter(), r.projected_hours(), r.res64());
    }

    // Verify all methods agree
    println!();
    let mut all_agree = true;
    for i in 1..results.len() {
        if results[i].value != results[0].value {
            eprintln!("ERROR: method '{}' disagrees with '{}' after {} iterations!",
                results[i].name, results[0].name, iters);
            eprintln!("  {} RES64: {:016X}", results[0].name, results[0].res64());
            eprintln!("  {} RES64: {:016X}", results[i].name, results[i].res64());
            all_agree = false;
        }
    }
    if all_agree {
        println!("All {} methods agree on the result after {} iterations.", results.len(), iters);
    } else {
        eprintln!("\nFATAL: Method disagreement detected!");
        std::process::exit(1);
    }
}

fn run_full_test(kbn: &Kbn, method: &str) {
    println!("Running FULL Fermat test with method '{}' (base {})...\n", method, BASE);
    let start = Instant::now();

    let residual = match method {
        "fft" => fermat_test(kbn, BASE, |pct| {
            eprint!("\r  {:.2}%", pct);
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }),
        "dwt" => fermat_test_dwt(kbn, BASE, |pct| {
            eprint!("\r  {:.2}%", pct);
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }),
        other => {
            eprintln!("Unknown method '{}'. Use 'fft' or 'dwt'.", other);
            std::process::exit(1);
        }
    };
    eprintln!("\r  done!    ");

    let elapsed = start.elapsed();
    let res64 = residual.iter_u64_digits().next().unwrap_or(0);

    println!("\nResult: {}*2^{}-1", K, N);
    if residual == BigUint::one() {
        println!("  PROBABLE PRIME (Fermat base {})", BASE);
    } else {
        println!("  COMPOSITE (Fermat base {})", BASE);
        println!("  RES64: {:016X}", res64);
    }
    println!("  Time: {:.1}s ({:.1} hours)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0);
}