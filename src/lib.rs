//! `Lenia_ca` — GPU-accelerated Lenia cellular automaton simulation.
//!
//! All computation (FFT, complex multiply, growth function, channel update)
//! happens as GPU compute shaders via `wgpu`. Channel data is only read back
//! to the CPU when explicitly requested (for display or export).
//!
//! See [`gpu_lenia::GpuLenia`] for the main simulation entry point.

pub mod gpu_lenia;
pub mod growth_functions;
pub mod kernels;
pub mod wfft;

/// Samples the normal distribution where the peak (at `x = mu`) is 1.
/// This is not suitable for use as a gaussian probability density function!
///
/// ### Parameters
///
/// * `x` - Point of the normal distribution to sample.
///
/// * `mu` - The mean (point of the highest value/peak) of the normal distribution.
///
/// * `stddev` - Standard deviation of the normal distribution.
fn sample_normal(x: f64, mu: f64, stddev: f64) -> f64 {
    (-(((x - mu) * (x - mu)) / (2.0 * (stddev * stddev)))).exp()
}

/// Euclidean distance between points `a` and `b`.
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut out: f64 = 0.0;
    for i in 0..a.len() {
        out += (a[i] - b[i]) * (a[i] - b[i]);
    }
    out.sqrt()
}
