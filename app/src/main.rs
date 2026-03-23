mod cli;

use armcrunch::{fermat_test, Kbn};
use num_bigint::BigUint;
use num_traits::One;

fn main() {
    let args = cli::parse();
    let kbn = Kbn::new(args.k, args.n);
    let a = args.a;
    let (k, n) = (args.k, args.n);
    let start = std::time::Instant::now();
    let residual = fermat_test(&kbn, a, |pct| {
        eprint!("\r{pct:.2}%");
        let _ = std::io::Write::flush(&mut std::io::stderr());
    });
    eprintln!("\rdone!  ");
    let elapsed = start.elapsed();
    let res64 = residual.iter_u64_digits().next().unwrap_or(0);
    println!("{k}*2^{n}-1");
    if residual == BigUint::one() {
        println!("probable prime (Fermat, a={a}), time: {:.1}s", elapsed.as_secs_f64());
    } else {
        println!("composite (Fermat, a={a}), RES64: {res64:016X}, time: {:.1}s", elapsed.as_secs_f64());
    }
}
