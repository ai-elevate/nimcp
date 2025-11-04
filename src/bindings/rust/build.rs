fn main() {
    // Link to nimcp_core library
    println!("cargo:rustc-link-search=native=../../../build/src/lib");
    println!("cargo:rustc-link-lib=nimcp_core");
    println!("cargo:rerun-if-changed=../../include/nimcp.h");
}
