//! Fermat test benchmark for 1003*2^2499999-1.
//!
//! Runs a configurable number of squaring iterations with each viable method
//! (fft, dwt, kbn), verifies they produce identical results, and projects
//! total runtime for a full Fermat test.

use armcrunch::{fermat_test, fermat_test_dwt, Kbn, FftSquarer, DwtSquarer};
use num_bigint::BigUint;
use num_traits::One;
use std::time::{Duration, Instant};

const K: u64 = 1003;
const N: u64 = 2_499_999;
const BASE: u64 = 3;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut duration_secs: u64 = 120;
    let mut full_method: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--duration" => {
                i += 1;
                duration_secs = args.get(i).expect("--duration requires seconds")
                    .parse().expect("--duration requires a number");
            }
            "--full" => {
                i += 1;
                full_method = Some(args.get(i).expect("--full requires a method name").clone());
            }
            _ => {
                eprintln!("Usage: fermat_bench [--duration SECS] [--full fft|dwt]");
                eprintln!("  --duration SECS  Time limit per method (default: 120)");
                eprintln!("  --full METHOD    Run a complete Fermat test (fft or dwt)");
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
        run_timed_comparison(&kbn, Duration::from_secs(duration_secs));
    }
}

fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1}m", secs / 60.0)
    } else if secs < 86400.0 {
        format!("{:.1}h", secs / 3600.0)
    } else {
        format!("{:.1}d", secs / 86400.0)
    }
}

fn run_timed_comparison(kbn: &Kbn, time_limit: Duration) {
    let modulus = kbn.value();
    let a_big = BigUint::from(BASE);

    println!("Computing a^k mod M (k={})...", K);
    let t0 = Instant::now();
    let x0 = a_big.modpow(&BigUint::from(K), &modulus);
    println!("  done in {:.1}s\n", t0.elapsed().as_secs_f64());

    println!("Running each method for up to {} seconds:\n", time_limit.as_secs());

    // Phase 1: timed runs to measure throughput
    struct TimingResult {
        name: &'static str,
        elapsed: Duration,
        iters: u64,
    }

    let mut timings: Vec<TimingResult> = Vec::new();

    // --- fft ---
    {
        eprint!("  Timing fft...");
        let mut x = x0.clone();
        let mut squarer = FftSquarer::new(&modulus);
        let start = Instant::now();
        let mut count: u64 = 0;
        while start.elapsed() < time_limit {
            squarer.square_kbn(&mut x, K, N, &modulus);
            count += 1;
        }
        let elapsed = start.elapsed();
        eprintln!("\r  fft:  {} iters in {:.1}s    ", count, elapsed.as_secs_f64());
        timings.push(TimingResult { name: "fft", elapsed, iters: count });
    }

    // --- dwt ---
    {
        eprint!("  Timing dwt...");
        let mut squarer = DwtSquarer::new(K, N, &x0);
        let start = Instant::now();
        let mut count: u64 = 0;
        while start.elapsed() < time_limit {
            squarer.square();
            count += 1;
        }
        let elapsed = start.elapsed();
        eprintln!("\r  dwt:  {} iters in {:.1}s    ", count, elapsed.as_secs_f64());
        timings.push(TimingResult { name: "dwt", elapsed, iters: count });
    }

    // --- kbn ---
    {
        eprint!("  Timing kbn...");
        let mut x = x0.clone();
        let start = Instant::now();
        let mut count: u64 = 0;
        while start.elapsed() < time_limit {
            armcrunch::square_mod_kbn(&mut x, K, N, &modulus);
            count += 1;
        }
        let elapsed = start.elapsed();
        eprintln!("\r  kbn:  {} iters in {:.1}s    ", count, elapsed.as_secs_f64());
        timings.push(TimingResult { name: "kbn", elapsed, iters: count });
    }

    // Print results table
    println!();
    println!("  {:<8} {:>10} {:>12} {:>12} {:>14}",
        "Method", "Iters", "Wall time", "ms/iter", "Projected");
    println!("  {:-<8} {:-<10} {:-<12} {:-<12} {:-<14}", "", "", "", "", "");
    for t in &timings {
        let ms_per = t.elapsed.as_secs_f64() * 1000.0 / t.iters as f64;
        let projected_secs = (ms_per / 1000.0) * N as f64;
        println!("  {:<8} {:>10} {:>12} {:>12.3} {:>14}",
            t.name,
            t.iters,
            format_duration(t.elapsed.as_secs_f64()),
            ms_per,
            format_duration(projected_secs));
    }

    // Phase 2: verify correctness by replaying min(iters) with each method
    // and comparing full BigUint results.
    let min_iters = timings.iter().map(|t| t.iters).min().unwrap();
    println!("\nVerifying all methods agree after {} iterations...", min_iters);

    let mut verify_values: Vec<(&str, BigUint)> = Vec::new();

    {
        let mut x = x0.clone();
        let mut squarer = FftSquarer::new(&modulus);
        for _ in 0..min_iters {
            squarer.square_kbn(&mut x, K, N, &modulus);
        }
        verify_values.push(("fft", x));
    }
    {
        let mut squarer = DwtSquarer::new(K, N, &x0);
        for _ in 0..min_iters {
            squarer.square();
        }
        verify_values.push(("dwt", squarer.to_biguint()));
    }
    {
        let mut x = x0.clone();
        for _ in 0..min_iters {
            armcrunch::square_mod_kbn(&mut x, K, N, &modulus);
        }
        verify_values.push(("kbn", x));
    }

    let mut all_agree = true;
    for i in 1..verify_values.len() {
        if verify_values[i].1 != verify_values[0].1 {
            eprintln!("ERROR: '{}' disagrees with '{}' after {} iterations!",
                verify_values[i].0, verify_values[0].0, min_iters);
            eprintln!("  {} RES64: {:016X}", verify_values[0].0,
                verify_values[0].1.iter_u64_digits().next().unwrap_or(0));
            eprintln!("  {} RES64: {:016X}", verify_values[i].0,
                verify_values[i].1.iter_u64_digits().next().unwrap_or(0));
            all_agree = false;
        }
    }

    if all_agree {
        println!("  All {} methods agree.", verify_values.len());
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