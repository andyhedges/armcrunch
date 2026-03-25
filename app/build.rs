fn main() {
    let vdsp_enabled = std::env::var_os("CARGO_FEATURE_VDSP").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if vdsp_enabled && target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}