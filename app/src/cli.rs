pub struct Args {
    pub k: u64,
    pub n: u64,
    pub a: u64,
}

pub fn parse() -> Args {
    let args: Vec<String> = std::env::args().collect();

    match args.len() {
        3 | 4 => {
            let k = args[1].parse::<u64>().unwrap_or_else(|_| { eprintln!("Error: k must be a positive integer"); std::process::exit(1); });
            let n = args[2].parse::<u64>().unwrap_or_else(|_| { eprintln!("Error: n must be a positive integer"); std::process::exit(1); });
            let a = if args.len() == 4 {
                args[3].parse::<u64>().unwrap_or_else(|_| { eprintln!("Error: a must be a positive integer"); std::process::exit(1); })
            } else {
                3
            };
            Args { k, n, a }
        }
        _ => {
            eprintln!("Usage: armcrunch <k> <n> [a]");
            eprintln!("  Tests whether k*2^n-1 is a probable prime using Fermat test to base a (default 3).");
            std::process::exit(1);
        }
    }
}
