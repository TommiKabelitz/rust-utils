# rust-utils

Small, zero-dependency utility crates for Rust. All crates are licensed in the public domain (or MIT licensed).

# Usage

These crates are intended to be vendored directly into the repositories that use them. Hence they are not listed on crates.io. The layout presented here utilises a Cargo workspace which is likely the simplest way to utilise the crates, though you are naturally free to use them in any way you see fit.

# Crates

<a name="rust-utils"></a>

crate | latest version | description
-------|----------------|-----------------------
two_dim_array | 0.1.0 | A simple struct which wraps a generic array slice allowing simple access as a two-dimensional array. Access is limited to contiguous row-major slices.