//! Fermat test benchmark for 1003*2^2499999-1.
//!
//! Benchmarks all viable squaring methods using a common fixed iteration count:
//! 1. Warmup: estimates throughput of each method
//! 2. Derives iteration count so the fastest method runs for ~duration seconds
//! 3. Runs every method (including gwnum if available) for that exact count
//! 4. Reports a table sorted by speed with ratio vs the fastest method
//! 5. Verifies all Rust methods produce identical results

use armcrunch::{fermat_test, fermat_test_dwt, Kbn, FftSquarer, DwtSquarer};
use num_bigint::BigUint;
use num_traits::One;
use std::time::Instant;

const K: u64 = 1003;
const N: u64 = 2_499_999;
const BASE: u64 = 3;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut duration_secs: u64 = 120;
    let mut full_method: Option<String> = None;
    let mut include_kbn = false;
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
            "--kbn" => {
                include_kbn = true;
            }
            _ => {
                eprintln!("Usage: fermat_bench [--duration SECS] [--full fft|dwt] [--kbn]");
                eprintln!("  --duration SECS  Target duration for the fastest method (default: 120)");
                eprintln!("                   All methods run the same number of iterations.");
                eprintln!("  --full METHOD    Run a complete Fermat test (fft or dwt)");
                eprintln!("  --kbn            Include kbn method (slow, excluded by default)");
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
        run_fixed_iteration_benchmark(&kbn, duration_secs, include_kbn);
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

// ---------------------------------------------------------------------------
// gwnum subprocess helpers
// ---------------------------------------------------------------------------

fn run_gwnum_bench(path: &std::path::Path, label: &str, iters: u64) -> Option<(String, f64)> {
    if !path.exists() {
        return None;
    }

    eprint!("  Running {} ({} iters)...", label, iters);
    let output = std::process::Command::new(path)
        .args(["--k", &K.to_string(), "--n", &N.to_string(), "--iters", &iters.to_string()])
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!("\r  {}: failed (exit code {:?})    ", label, output.status.code());
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("GWNUM_RESULT ") {
            if let Ok(ms) = rest.trim().parse::<f64>() {
                eprintln!("\r  {}: {:.3} ms/iter    ", label, ms);
                return Some((label.to_string(), ms));
            }
        }
    }

    eprintln!("\r  {}: could not parse result    ", label);
    None
}

fn find_gwnum_binary() -> Option<(std::path::PathBuf, &'static str)> {
    let search_dirs: Vec<std::path::PathBuf> = {
        let mut dirs = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.join("../../bench"));
                dirs.push(parent.join("../bench"));
            }
        }
        dirs.push(std::path::PathBuf::from("bench"));
        dirs.push(std::path::PathBuf::from("."));
        dirs
    };

    let binaries = [
        ("gwnum_bench-linux-amd64", "gwnum-x86"),
        ("gwnum_bench-macos-amd64", "gwnum-x86"),
        ("gwnum_bench-macos-arm64", "gwnum-arm"),
        ("gwnum_bench", "gwnum"),
    ];

    for (binary_name, label) in &binaries {
        for dir in &search_dirs {
            let path = dir.join(binary_name);
            if path.exists() {
                return Some((path, label));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Main benchmark flow
// ---------------------------------------------------------------------------

fn run_fixed_iteration_benchmark(kbn: &Kbn, target_duration_secs: u64, include_kbn: bool) {
    let modulus = kbn.value();
    let a_big = BigUint::from(BASE);

    println!("Computing a^k mod M (k={})...", K);
    let t0 = Instant::now();
    let x0 = a_big.modpow(&BigUint::from(K), &modulus);
    println!("  done in {:.1}s\n", t0.elapsed().as_secs_f64());

    // Phase 1: Warmup — estimate throughput of each method (50 iters each)
    println!("Warmup (estimating throughput)...\n");
    let warmup_iters: u64 = 50;

    let gwnum_binary = find_gwnum_binary();

    let gwnum_ms_per = if let Some((ref path, label)) = gwnum_binary {
        run_gwnum_bench(path, label, warmup_iters).map(|(_, ms)| ms)
    } else {
        eprintln!("  gwnum: not found (run bench/download_gwnum.sh to fetch)");
        None
    };

    let fft_ms_per = {
        let mut x = x0.clone();
        let mut squarer = FftSquarer::new(&modulus);
        let start = Instant::now();
        for _ in 0..warmup_iters { squarer.square_kbn(&mut x, K, N, &modulus); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / warmup_iters as f64;
        eprintln!("  fft:  {:.3} ms/iter (warmup)", ms);
        ms
    };

    let dwt_ms_per = {
        let mut squarer = DwtSquarer::new(K, N, &x0);
        let start = Instant::now();
        for _ in 0..warmup_iters { squarer.square(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / warmup_iters as f64;
        eprintln!("  dwt:  {:.3} ms/iter (warmup)", ms);
        ms
    };

    let kbn_ms_per = if include_kbn {
        let mut x = x0.clone();
        let start = Instant::now();
        for _ in 0..warmup_iters { armcrunch::square_mod_kbn(&mut x, K, N, &modulus); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / warmup_iters as f64;
        eprintln!("  kbn:  {:.3} ms/iter (warmup)", ms);
        Some(ms)
    } else {
        None
    };

    // Phase 2: Derive common iteration count from the fastest method
    let mut best_ms = fft_ms_per.min(dwt_ms_per);
    if let Some(gms) = gwnum_ms_per { best_ms = best_ms.min(gms); }
    if let Some(kms) = kbn_ms_per { best_ms = best_ms.min(kms); }
    let iters = ((target_duration_secs as f64 * 1000.0) / best_ms).max(100.0) as u64;

    let slowest_ms = if let Some(kms) = kbn_ms_per { kms.max(fft_ms_per).max(dwt_ms_per) } else { fft_ms_per.max(dwt_ms_per) };
    println!("\nRunning all methods for {} iterations\n  (fastest ~{}, slowest ~{})\n",
        iters,
        format_duration(best_ms * iters as f64 / 1000.0),
        format_duration(slowest_ms * iters as f64 / 1000.0));

    struct RowData {
        name: String,
        ms_per: f64,
    }

    let mut rows: Vec<RowData> = Vec::new();

    // Run gwnum
    if let Some((ref path, label)) = gwnum_binary {
        if let Some((name, ms)) = run_gwnum_bench(path, label, iters) {
            rows.push(RowData { name, ms_per: ms });
        }
    }

    // Run fft
    {
        eprint!("  Running fft ({} iters)...", iters);
        let mut x = x0.clone();
        let mut squarer = FftSquarer::new(&modulus);
        let start = Instant::now();
        for _ in 0..iters { squarer.square_kbn(&mut x, K, N, &modulus); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        eprintln!("\r  fft:  {:.3} ms/iter ({:.1}s)    ", ms, start.elapsed().as_secs_f64());
        rows.push(RowData { name: "fft".into(), ms_per: ms });
    }

    // Run dwt
    {
        eprint!("  Running dwt ({} iters)...", iters);
        let mut squarer = DwtSquarer::new(K, N, &x0);
        let start = Instant::now();
        for _ in 0..iters { squarer.square(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        eprintln!("\r  dwt:  {:.3} ms/iter ({:.1}s)    ", ms, start.elapsed().as_secs_f64());
        rows.push(RowData { name: "dwt".into(), ms_per: ms });
    }

    // Run kbn
    if include_kbn {
        eprint!("  Running kbn ({} iters)...", iters);
        let mut x = x0.clone();
        let start = Instant::now();
        for _ in 0..iters { armcrunch::square_mod_kbn(&mut x, K, N, &modulus); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        eprintln!("\r  kbn:  {:.3} ms/iter ({:.1}s)    ", ms, start.elapsed().as_secs_f64());
        rows.push(RowData { name: "kbn".into(), ms_per: ms });
    }

    // Sort by ms/iter (fastest first)
    rows.sort_by(|a, b| a.ms_per.partial_cmp(&b.ms_per).unwrap());
    let best_ms = rows.first().map(|r| r.ms_per).unwrap_or(1.0);

    // Print results table
    println!();
    println!("  {:<12} {:>10} {:>12} {:>14} {:>8}",
        "Method", "Iters", "ms/iter", "Projected", "Ratio");
    println!("  {:-<12} {:-<10} {:-<12} {:-<14} {:-<8}", "", "", "", "", "");

    for row in &rows {
        let projected_secs = (row.ms_per / 1000.0) * N as f64;
        let ratio = row.ms_per / best_ms;
        println!("  {:<12} {:>10} {:>12.3} {:>14} {:>7.1}×",
            row.name, iters, row.ms_per, format_duration(projected_secs), ratio);
    }

    // Phase 3: Verify Rust methods agree
    println!("\nVerifying all methods agree after {} iterations...", iters);

    let mut verify_values: Vec<(&str, BigUint)> = Vec::new();

    {
        let mut x = x0.clone();
        let mut squarer = FftSquarer::new(&modulus);
        for _ in 0..iters { squarer.square_kbn(&mut x, K, N, &modulus); }
        verify_values.push(("fft", x));
    }
    {
        let mut squarer = DwtSquarer::new(K, N, &x0);
        for _ in 0..iters { squarer.square(); }
        verify_values.push(("dwt", squarer.to_biguint()));
    }
    if include_kbn {
        let mut x = x0.clone();
        for _ in 0..iters { armcrunch::square_mod_kbn(&mut x, K, N, &modulus); }
        verify_values.push(("kbn", x));
    }

    let mut all_agree = true;
    for i in 1..verify_values.len() {
        if verify_values[i].1 != verify_values[0].1 {
            eprintln!("ERROR: '{}' disagrees with '{}' after {} iterations!",
                verify_values[i].0, verify_values[0].0, iters);
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