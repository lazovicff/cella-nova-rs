//! `Lenia_ca` — GPU-accelerated Flow Lenia cellular automaton simulation.
//!
//! All computation (FFT, complex multiply, growth function, channel aggregation,
//! Sobel gradients, flow field, reintegration tracking)
//! happens as GPU compute shaders via `wgpu`. Channel data is only read back
//! to the CPU when explicitly requested (for display or export).
//!
//! See [`gpu_flow_lenia::GpuFlowLenia`] for the main simulation entry point.

pub mod gpu_flow_lenia;
pub mod wfft;
