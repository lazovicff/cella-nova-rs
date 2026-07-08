//! GPU-resident Flow Lenia simulation.
//!
//! Extends standard Lenia with:
//! - Multi-kernel, multi-channel architecture
//! - Flow field computation via Sobel gradients
//! - Reintegration tracking (semi-Lagrangian advection)
//!
//! Reference: "Flow Lenia: Mass conservation for the simulation of
//! continuous cellular automata" (https://arxiv.org/abs/2212.07906)
//!
//! All computation runs on the GPU with zero CPU readback between frames.

use crate::wfft::{WgpuContext, WgpuFFT1D};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

const COPY_TO_CONV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv: array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&channel)) { return; }
    conv[i] = vec2<f32>(channel[i], 0.0);
}
"#;

const COMPLEX_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&conv)) { return; }
    conv[i] = complex_mul(conv[i], kernel[i]);
}
"#;

const NORMALIZE_SHADER: &str = r#"
struct Params { norm_factor: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&data)) { return; }
    data[i] = data[i] * params.norm_factor;
}
"#;

const GROWTH_FLOW_SHADER: &str = r#"
struct Params { m: f32, s: f32, h: f32 }
@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv_x: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&result)) { return; }
    let x: f32 = conv[i].x;
    let diff: f32 = x - params.m;
    let g: f32 = exp(-(diff * diff) / (2.0 * params.s * params.s));
    result[i] = (2.0 * g - 1.0) * params.h;
    conv_x[i] = x;
}
"#;

const CHANNEL_AGGREGATE_SHADER: &str = r#"
struct Params { width: u32, num_kernels: u32, num_channels: u32 }
@group(0) @binding(0) var<storage, read> u_all: array<f32>;
@group(0) @binding(1) var<storage, read_write> u_channels: array<f32>;
@group(0) @binding(2) var<storage, read> c1_flat: array<u32>;
@group(0) @binding(3) var<storage, read> c1_offsets: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = params.width * params.width * params.num_channels;
    if (i >= total) { return; }
    let c: u32 = i / (params.width * params.width);
    let pixel: u32 = i % (params.width * params.width);
    let start: u32 = c1_offsets[c];
    let end: u32 = c1_offsets[c + 1u];
    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        let k: u32 = c1_flat[j];
        sum = sum + u_all[k * params.width * params.width + pixel];
    }
    u_channels[i] = sum;
}
"#;

const SOBEL_SHADER: &str = r#"
struct Params { width: u32, height: u32, num_fields: u32 }
@group(0) @binding(0) var<storage, read> input_field: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = params.width * params.height * params.num_fields;
    if (i >= total) { return; }
    let field: u32 = i / (params.width * params.height);
    let pixel: u32 = i % (params.width * params.height);
    let x: u32 = pixel % params.width;
    let y: u32 = pixel / params.width;
    let w: u32 = params.width;
    let h: u32 = params.height;
    let base: u32 = field * w * h;
    let xl: u32 = select(x - 1u, 0u, x == 0u);
    let xr: u32 = select(x + 1u, w - 1u, x + 1u >= w);
    let yu: u32 = select(y - 1u, 0u, y == 0u);
    let yd: u32 = select(y + 1u, h - 1u, y + 1u >= h);
    let tl = input_field[base + yu * w + xl];
    let tc = input_field[base + yu * w + x];
    let tr = input_field[base + yu * w + xr];
    let ml = input_field[base + y * w + xl];
    let mr = input_field[base + y * w + xr];
    let bl = input_field[base + yd * w + xl];
    let bc = input_field[base + yd * w + x];
    let br = input_field[base + yd * w + xr];
    grad_x[i] = (-tl + tr) + 2.0 * (-ml + mr) + (-bl + br);
    grad_y[i] = (-tl - 2.0 * tc - tr) + (bl + 2.0 * bc + br);
}
"#;

const SUM_CHANNELS_SHADER: &str = r#"
struct Params { width: u32, num_channels: u32 }
@group(0) @binding(0) var<storage, read> channels: array<f32>;
@group(0) @binding(1) var<storage, read_write> sum_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= params.width * params.width) { return; }
    var sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params.num_channels; c = c + 1u) {
        sum = sum + channels[c * params.width * params.width + i];
    }
    sum_out[i] = sum;
}
"#;

const FLOW_FIELD_SHADER: &str = r#"
struct Params { width: u32, num_channels: u32, num_channels_f32: f32 }
@group(0) @binding(0) var<storage, read> channels: array<f32>;
@group(0) @binding(1) var<storage, read> nabla_u_x: array<f32>;
@group(0) @binding(2) var<storage, read> nabla_u_y: array<f32>;
@group(0) @binding(3) var<storage, read> nabla_a_x: array<f32>;
@group(0) @binding(4) var<storage, read> nabla_a_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> flow_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> flow_y: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = params.width * params.width * params.num_channels;
    if (i >= total) { return; }
    let c: u32 = i / (params.width * params.width);
    let pixel: u32 = i % (params.width * params.width);
    let a: f32 = channels[i];
    let alpha: f32 = clamp((a / params.num_channels_f32) * (a / params.num_channels_f32), 0.0, 1.0);
    let nux: f32 = nabla_u_x[i];
    let nuy: f32 = nabla_u_y[i];
    let nax: f32 = nabla_a_x[pixel];
    let nay: f32 = nabla_a_y[pixel];
    flow_x[i] = nux * (1.0 - alpha) - nax * alpha;
    flow_y[i] = nuy * (1.0 - alpha) - nay * alpha;
}
"#;

const REINTEGRATION_SHADER: &str = r#"
struct Params { width: u32, height: u32, dd: i32, sigma: f32, dt: f32, num_channels: u32, ma: f32, basal_rate: f32, kinetic_cost: f32 }
@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<storage, read> flow_x: array<f32>;
@group(0) @binding(2) var<storage, read> flow_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> new_channel: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = id.x;
    let total: u32 = params.width * params.height * params.num_channels;
    if (idx >= total) { return; }
    let c: u32 = idx / (params.width * params.height);
    let pixel: u32 = idx % (params.width * params.height);
    let x: u32 = pixel % params.width;
    let y: u32 = pixel / params.width;
    let pos_x: f32 = f32(x) + 0.5;
    let pos_y: f32 = f32(y) + 0.5;
    let dd: i32 = params.dd;
    let sigma: f32 = params.sigma;
    let dt: f32 = params.dt;
    let ma: f32 = params.ma;
    let w: u32 = params.width;
    let h: u32 = params.height;
    let w_i32: i32 = i32(w);
    let h_i32: i32 = i32(h);
    let max_sz: f32 = min(1.0, 2.0 * sigma);
    let area_norm: f32 = 4.0 * sigma * sigma;
    let c_base: u32 = c * w * h;
    var sum: f32 = 0.0;
    for (var dx: i32 = -dd; dx <= dd; dx = dx + 1) {
        for (var dy: i32 = -dd; dy <= dd; dy = dy + 1) {
            let nx: i32 = i32(x) + dx;
            let ny: i32 = i32(y) + dy;
            if (nx < 0 || nx >= w_i32 || ny < 0 || ny >= h_i32) { continue; }
            let n_idx: u32 = u32(ny) * w + u32(nx);
            let a: f32 = channel[c_base + n_idx];
            if (a <= 0.0) { continue; }
            let n_pos_x: f32 = f32(nx) + 0.5;
            let n_pos_y: f32 = f32(ny) + 0.5;
            let fx: f32 = clamp(flow_x[c_base + n_idx], -ma, ma);
            let fy: f32 = clamp(flow_y[c_base + n_idx], -ma, ma);
            let mu_x: f32 = clamp(n_pos_x + fx * dt, sigma, f32(w) - sigma);
            let mu_y: f32 = clamp(n_pos_y + fy * dt, sigma, f32(h) - sigma);
            let dpx: f32 = abs(pos_x - mu_x);
            let dpy: f32 = abs(pos_y - mu_y);
            let sz_x: f32 = clamp(0.5 - dpx + sigma, 0.0, max_sz);
            let sz_y: f32 = clamp(0.5 - dpy + sigma, 0.0, max_sz);
            let area: f32 = (sz_x * sz_y) / area_norm;
            sum = sum + a * area;
        }
    }
    // Metabolic costs: basal decay + kinetic cost proportional to local flow
    let fx_self: f32 = clamp(flow_x[idx], -ma, ma);
    let fy_self: f32 = clamp(flow_y[idx], -ma, ma);
    let flow_mag: f32 = sqrt(fx_self * fx_self + fy_self * fy_self);
    sum = sum * (1.0 - params.basal_rate * dt) - params.kinetic_cost * flow_mag * dt;
    new_channel[idx] = max(sum, 0.0);
}
"#;

const WALL_INTERACTION_SHADER: &str = r#"
// Wall kernel from Sensorimotor Lenia:
//   K_wall(r) = exp(-(r/2)^2/2) * sigmoid(-10*(r/2 - 1))
//   G_wall(x) = -10 * max(0, x - 0.001)
// Small-radius spatial convolution (7x7 neighborhood).
struct Params { width: u32, height: u32, num_channels: u32 }
@group(0) @binding(0) var<storage, read> obstacle: array<f32>;
@group(0) @binding(1) var<storage, read_write> u_channel: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn wall_kernel(dist: f32) -> f32 {
    let r = dist / 2.0;
    let gaussian = exp(-r * r / 2.0);
    let cutoff = 1.0 / (1.0 + exp(10.0 * (r - 1.0)));
    return gaussian * cutoff;
}

fn wall_growth(x: f32) -> f32 {
    return -10.0 * max(0.0, x - 0.001);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = id.x;
    let total: u32 = params.width * params.height * params.num_channels;
    if (idx >= total) { return; }
    let c: u32 = idx / (params.width * params.height);
    let pixel: u32 = idx % (params.width * params.height);
    let px: u32 = pixel % params.width;
    let py: u32 = pixel / params.width;
    let w: u32 = params.width;
    let h: u32 = params.height;
    let w_i32: i32 = i32(w);
    let h_i32: i32 = i32(h);
    // 7x7 neighborhood convolution
    var conv: f32 = 0.0;
    for (var dy: i32 = -3; dy <= 3; dy = dy + 1) {
        for (var dx: i32 = -3; dx <= 3; dx = dx + 1) {
            let nx: i32 = i32(px) + dx;
            let ny: i32 = i32(py) + dy;
            if (nx < 0 || nx >= w_i32 || ny < 0 || ny >= h_i32) { continue; }
            let n_idx: u32 = u32(ny) * w + u32(nx);
            let obs: f32 = obstacle[n_idx];
            if (obs <= 0.0) { continue; }
            let dist: f32 = sqrt(f32(dx * dx + dy * dy));
            conv = conv + obs * wall_kernel(dist);
        }
    }
    let growth: f32 = wall_growth(conv);
    u_channel[idx] = u_channel[idx] + growth;
}
"#;

// ---------------------------------------------------------------------------
// CPU-side kernel generation
// ---------------------------------------------------------------------------

/// Generates Flow Lenia kernels matching the reference implementation.
///
/// Each kernel k uses parameters R (global scale), r[k], a[k][3], w[k][3], b[k][3].
/// Returns pre-FFT'd complex kernels ready for GPU upload.
pub fn generate_flow_kernels(
    size: usize,
    global_r: f32,
    r: &[f32],
    a: &[[f32; 3]],
    w: &[[f32; 3]],
    b: &[[f32; 3]],
) -> Vec<Vec<num_complex::Complex32>> {
    use rustfft::{FftDirection, FftPlanner};

    let k = r.len();
    let padded = size.next_power_of_two();
    let mid = padded as i32 / 2;

    let mut kernels_fft = Vec::with_capacity(k);

    for ki in 0..k {
        let mut kernel_real = ndarray::Array2::<f64>::zeros([padded, padded]);

        for i in 0..padded {
            for j in 0..padded {
                let dx = i as i32 - mid;
                let dy = j as i32 - mid;
                let dist = ((dx * dx + dy * dy) as f64).sqrt();
                let d_scaled = dist / ((global_r as f64 + 15.0) * r[ki] as f64);
                let sig = 0.5 * (((-d_scaled + 1.0) * 5.0).tanh() + 1.0);
                let mut ker_val = 0.0f64;
                for p in 0..3 {
                    let diff = d_scaled - a[ki][p] as f64;
                    ker_val += b[ki][p] as f64 * (-(diff * diff) / w[ki][p] as f64).exp();
                }
                kernel_real[[i, j]] = sig * ker_val;
            }
        }

        let sum: f64 = kernel_real.iter().sum();
        if sum > 0.0 {
            kernel_real.mapv_inplace(|v| v / sum);
        }

        // FFT shift
        let mut shifted = ndarray::Array2::<f64>::zeros([padded, padded]);
        for i in 0..padded {
            for j in 0..padded {
                let si = (i + padded / 2) % padded;
                let sj = (j + padded / 2) % padded;
                shifted[[i, j]] = kernel_real[[si, sj]];
            }
        }

        // 2D FFT
        let mut data: Vec<num_complex::Complex<f64>> = shifted
            .iter()
            .map(|&v| num_complex::Complex::new(v, 0.0))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(padded, FftDirection::Forward);
        let mut scratch = vec![num_complex::Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        for i in 0..padded {
            let mut row: Vec<_> = (0..padded).map(|j| data[i * padded + j]).collect();
            fft.process_with_scratch(&mut row, &mut scratch);
            for j in 0..padded {
                data[i * padded + j] = row[j];
            }
        }
        for j in 0..padded {
            let mut col: Vec<_> = (0..padded).map(|i| data[i * padded + j]).collect();
            fft.process_with_scratch(&mut col, &mut scratch);
            for i in 0..padded {
                data[i * padded + j] = col[i];
            }
        }

        kernels_fft.push(
            data.iter()
                .map(|&c| num_complex::Complex32::new(c.re as f32, c.im as f32))
                .collect(),
        );
    }

    kernels_fft
}

// ---------------------------------------------------------------------------
// GpuFlowLenia
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct GpuFlowLenia {
    context: Arc<WgpuContext>,
    shape: Vec<usize>,
    total_elements: usize,
    dt: f32,
    num_channels: usize,
    num_kernels: usize,
    dd: i32,
    sigma: f32,
    basal_metabolic_rate: f32,
    kinetic_cost: f32,

    // Channel mapping
    c0: Vec<u32>,
    c1_flat: Vec<u32>,
    c1_offsets: Vec<u32>,

    // Per-kernel growth params
    kernel_m: Vec<f32>,
    kernel_s: Vec<f32>,
    kernel_h: Vec<f32>,

    // --- Packed GPU buffers ---
    /// All channels packed: [X*Y*C] f32
    channel_buffer: wgpu::Buffer,
    /// Output of reintegration: [X*Y*C] f32
    new_channel_buffer: wgpu::Buffer,
    /// Working buffer for FFT: [X*Y] vec2<f32>
    conv_buffer: wgpu::Buffer,
    /// Convolution real part per kernel: [K * X * Y] f32
    conv_x_buffer: wgpu::Buffer,
    /// All kernels packed: [X*Y*k] vec2<f32>
    kernel_buffer: wgpu::Buffer,
    /// All growth results: [X*Y*k] f32
    u_buffer: wgpu::Buffer,
    /// Aggregated per channel: [X*Y*C] f32
    u_channel_buffer: wgpu::Buffer,
    /// Gradient of U: [X*Y*C] f32 each
    nabla_u_x_buffer: wgpu::Buffer,
    nabla_u_y_buffer: wgpu::Buffer,
    /// Gradient of sum(A): [X*Y] f32 each
    nabla_a_x_buffer: wgpu::Buffer,
    nabla_a_y_buffer: wgpu::Buffer,
    /// Sum of all channels: [X*Y] f32
    sum_a_buffer: wgpu::Buffer,
    /// Flow field: [X*Y*C] f32 each
    flow_x_buffer: wgpu::Buffer,
    flow_y_buffer: wgpu::Buffer,

    // Obstacle channel
    obstacle_buffer: wgpu::Buffer,

    // FFT
    forward_fft_1d: Vec<WgpuFFT1D>,
    inverse_fft_1d: Vec<WgpuFFT1D>,

    // Pipelines
    copy_to_conv_pipeline: wgpu::ComputePipeline,
    complex_mul_pipeline: wgpu::ComputePipeline,
    normalize_pipeline: wgpu::ComputePipeline,
    growth_pipeline: wgpu::ComputePipeline,
    channel_aggregate_pipeline: wgpu::ComputePipeline,
    sobel_pipeline: wgpu::ComputePipeline,
    sum_channels_pipeline: wgpu::ComputePipeline,
    flow_field_pipeline: wgpu::ComputePipeline,
    reintegration_pipeline: wgpu::ComputePipeline,
    wall_interaction_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    copy_to_conv_bgl: wgpu::BindGroupLayout,
    complex_mul_bgl: wgpu::BindGroupLayout,
    normalize_bgl: wgpu::BindGroupLayout,
    growth_bgl: wgpu::BindGroupLayout,
    channel_aggregate_bgl: wgpu::BindGroupLayout,
    sobel_bgl: wgpu::BindGroupLayout,
    sum_channels_bgl: wgpu::BindGroupLayout,
    flow_field_bgl: wgpu::BindGroupLayout,
    reintegration_bgl: wgpu::BindGroupLayout,
    wall_interaction_bgl: wgpu::BindGroupLayout,

    // Cached bind groups (static bindings)
    normalize_bg: wgpu::BindGroup,
    channel_aggregate_bg: wgpu::BindGroup,
    sobel_u_bg: wgpu::BindGroup,
    sobel_a_bg: wgpu::BindGroup,
    sum_channels_bg: wgpu::BindGroup,
    flow_field_bg: wgpu::BindGroup,
    reintegration_bg: wgpu::BindGroup,
    wall_interaction_bg: wgpu::BindGroup,

    // Uniform buffers
    growth_params_buffer: wgpu::Buffer,
    normalize_params_buffer: wgpu::Buffer,
    channel_aggregate_params_buffer: wgpu::Buffer,
    sobel_params_u_buffer: wgpu::Buffer,
    sobel_params_a_buffer: wgpu::Buffer,
    sum_channels_params_buffer: wgpu::Buffer,
    flow_field_params_buffer: wgpu::Buffer,
    reintegration_params_buffer: wgpu::Buffer,
    wall_params_buffer: wgpu::Buffer,

    // Mapping buffers
    c1_flat_buffer: wgpu::Buffer,
    c1_offsets_buffer: wgpu::Buffer,

    // Readback
    readback_buffer: wgpu::Buffer,
}

impl GpuFlowLenia {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: Arc<WgpuContext>,
        shape: &[usize],
        num_channels: usize,
        num_kernels: usize,
        c0: &[u32],
        c1: &[Vec<u32>],
        kernel_m: &[f32],
        kernel_s: &[f32],
        kernel_h: &[f32],
        dt: f32,
        dd: i32,
        sigma: f32,
        basal_metabolic_rate: f32,
        kinetic_cost: f32,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;

        assert_eq!(shape.len(), 2, "Flow Lenia requires 2D grids");
        let total_elements: usize = shape.iter().product();
        let buf_size = (total_elements * 4) as u64;
        let conv_buf_size = (total_elements * 8) as u64;

        // Flatten c1
        let mut c1_flat = Vec::new();
        let mut c1_offsets = vec![0u32]; // start with 0
        for c in 0..num_channels {
            c1_flat.extend(c1[c].iter().cloned());
            c1_offsets.push(c1_flat.len() as u32);
        }

        // FFT instances
        let mut forward_fft_1d = Vec::with_capacity(shape.len());
        let mut inverse_fft_1d = Vec::with_capacity(shape.len());
        for &dim in shape {
            forward_fft_1d.push(WgpuFFT1D::new(Arc::clone(&context), dim, false));
            inverse_fft_1d.push(WgpuFFT1D::new(Arc::clone(&context), dim, true));
        }

        // --- Buffer helpers ---
        let make_storage = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let make_uniform = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // --- Create buffers ---
        let ch_size = (total_elements * num_channels * 4) as u64;
        let channel_buffer = make_storage("fl::channel", ch_size);
        let new_channel_buffer = make_storage("fl::new_channel", ch_size);
        let conv_buffer = make_storage("fl::conv", conv_buf_size);

        let k_size = (total_elements * num_kernels * 8) as u64;
        let kernel_buffer = make_storage("fl::kernel", k_size);

        let u_size = (total_elements * num_kernels * 4) as u64;
        let u_buffer = make_storage("fl::u", u_size);
        let conv_x_buffer = make_storage("fl::conv_x", u_size);

        let uc_size = (total_elements * num_channels * 4) as u64;
        let u_channel_buffer = make_storage("fl::u_channel", uc_size);
        let nabla_u_x_buffer = make_storage("fl::nabla_u_x", uc_size);
        let nabla_u_y_buffer = make_storage("fl::nabla_u_y", uc_size);

        let nabla_a_x_buffer = make_storage("fl::nabla_a_x", buf_size);
        let nabla_a_y_buffer = make_storage("fl::nabla_a_y", buf_size);
        let sum_a_buffer = make_storage("fl::sum_a", buf_size);

        let flow_x_buffer = make_storage("fl::flow_x", uc_size);
        let flow_y_buffer = make_storage("fl::flow_y", uc_size);

        // Obstacle channel (single channel, X*Y f32)
        let obstacle_buffer = make_storage("fl::obstacle", buf_size);

        // Uniform buffers
        let growth_params_buffer = make_uniform("fl::growth_params", 12);
        let normalize_params_buffer = make_uniform("fl::normalize_params", 4);
        let channel_aggregate_params_buffer = make_uniform("fl::ca_params", 12);
        let sobel_params_u_buffer = make_uniform("fl::sobel_u_params", 12);
        let sobel_params_a_buffer = make_uniform("fl::sobel_a_params", 12);
        let sum_channels_params_buffer = make_uniform("fl::sc_params", 8);
        let flow_field_params_buffer = make_uniform("fl::ff_params", 12);
        let reintegration_params_buffer = make_uniform("fl::ri_params", 36);
        let wall_params_buffer = make_uniform("fl::wall_params", 12);

        // Mapping buffers
        let c1_flat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::c1_flat"),
            size: (c1_flat.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&c1_flat_buffer, 0, bytemuck::cast_slice(&c1_flat));

        let c1_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::c1_offsets"),
            size: (c1_offsets.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&c1_offsets_buffer, 0, bytemuck::cast_slice(&c1_offsets));

        // Readback
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Shader modules ---
        let sm = |label: &str, src: &str| -> wgpu::ShaderModule {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let copy_to_conv_sm = sm("fl::copy_to_conv", COPY_TO_CONV_SHADER);
        let complex_mul_sm = sm("fl::complex_mul", COMPLEX_MUL_SHADER);
        let normalize_sm = sm("fl::normalize", NORMALIZE_SHADER);
        let growth_sm = sm("fl::growth", GROWTH_FLOW_SHADER);
        let channel_aggregate_sm = sm("fl::channel_aggregate", CHANNEL_AGGREGATE_SHADER);
        let sobel_sm = sm("fl::sobel", SOBEL_SHADER);
        let sum_channels_sm = sm("fl::sum_channels", SUM_CHANNELS_SHADER);
        let flow_field_sm = sm("fl::flow_field", FLOW_FIELD_SHADER);
        let reintegration_sm = sm("fl::reintegration", REINTEGRATION_SHADER);
        let wall_interaction_sm = sm("fl::wall_interaction", WALL_INTERACTION_SHADER);

        // --- Bind group layout helpers ---
        let sro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let srw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let unif = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let copy_to_conv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::copy_to_conv bgl"),
            entries: &[sro(0), srw(1)],
        });
        let complex_mul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::complex_mul bgl"),
            entries: &[srw(0), sro(1)],
        });
        let normalize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::normalize bgl"),
            entries: &[srw(0), unif(1)],
        });
        let growth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::growth bgl"),
            entries: &[sro(0), srw(1), srw(2), unif(3)],
        });
        let channel_aggregate_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fl::ca bgl"),
                entries: &[sro(0), srw(1), sro(2), sro(3), unif(4)],
            });
        let sobel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::sobel bgl"),
            entries: &[sro(0), srw(1), srw(2), unif(3)],
        });
        let sum_channels_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::sc bgl"),
            entries: &[sro(0), srw(1), unif(2)],
        });
        let flow_field_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ff bgl"),
            entries: &[
                sro(0),
                sro(1),
                sro(2),
                sro(3),
                sro(4),
                srw(5),
                srw(6),
                unif(7),
            ],
        });
        let reintegration_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ri bgl"),
            entries: &[sro(0), sro(1), sro(2), srw(3), unif(4)],
        });
        let wall_interaction_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fl::wall bgl"),
                entries: &[sro(0), srw(1), unif(2)],
            });

        // --- Pipeline layouts & pipelines ---
        let pl = |label: &str, bgl: &wgpu::BindGroupLayout| -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
        };
        let cp = |label: &str,
                  layout: &wgpu::PipelineLayout,
                  module: &wgpu::ShaderModule|
         -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: "main",
            })
        };

        let copy_to_conv_pipeline = cp(
            "fl::copy_to_conv",
            &pl("fl::copy_to_conv pl", &copy_to_conv_bgl),
            &copy_to_conv_sm,
        );
        let complex_mul_pipeline = cp(
            "fl::complex_mul",
            &pl("fl::complex_mul pl", &complex_mul_bgl),
            &complex_mul_sm,
        );
        let normalize_pipeline = cp(
            "fl::normalize",
            &pl("fl::normalize pl", &normalize_bgl),
            &normalize_sm,
        );
        let growth_pipeline = cp("fl::growth", &pl("fl::growth pl", &growth_bgl), &growth_sm);
        let channel_aggregate_pipeline = cp(
            "fl::ca",
            &pl("fl::ca pl", &channel_aggregate_bgl),
            &channel_aggregate_sm,
        );
        let sobel_pipeline = cp("fl::sobel", &pl("fl::sobel pl", &sobel_bgl), &sobel_sm);
        let sum_channels_pipeline = cp(
            "fl::sc",
            &pl("fl::sc pl", &sum_channels_bgl),
            &sum_channels_sm,
        );
        let flow_field_pipeline = cp("fl::ff", &pl("fl::ff pl", &flow_field_bgl), &flow_field_sm);
        let reintegration_pipeline = cp(
            "fl::ri",
            &pl("fl::ri pl", &reintegration_bgl),
            &reintegration_sm,
        );
        let wall_interaction_pipeline = cp(
            "fl::wall",
            &pl("fl::wall pl", &wall_interaction_bgl),
            &wall_interaction_sm,
        );

        // --- Write uniform params ---
        let padded_n = forward_fft_1d[0].padded_len() as f32;
        let norm_factor = 1.0 / padded_n.powi(shape.len() as i32);
        queue.write_buffer(
            &normalize_params_buffer,
            0,
            bytemuck::cast_slice(&[norm_factor]),
        );

        let ca: [u32; 3] = [shape[0] as u32, num_kernels as u32, num_channels as u32];
        queue.write_buffer(
            &channel_aggregate_params_buffer,
            0,
            bytemuck::cast_slice(&ca),
        );

        let su: [u32; 3] = [shape[0] as u32, shape[1] as u32, num_channels as u32];
        queue.write_buffer(&sobel_params_u_buffer, 0, bytemuck::cast_slice(&su));

        let sa: [u32; 3] = [shape[0] as u32, shape[1] as u32, 1];
        queue.write_buffer(&sobel_params_a_buffer, 0, bytemuck::cast_slice(&sa));

        let sc: [u32; 2] = [shape[0] as u32, num_channels as u32];
        queue.write_buffer(&sum_channels_params_buffer, 0, bytemuck::cast_slice(&sc));

        // flow_field_params: width(u32), num_channels(u32), num_channels_f32(f32)
        {
            let mut data = Vec::with_capacity(12);
            data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
            data.extend_from_slice(&(num_channels as u32).to_le_bytes());
            data.extend_from_slice(&(num_channels as f32).to_le_bytes());
            queue.write_buffer(&flow_field_params_buffer, 0, &data);
        }

        // reintegration_params: width(u32), height(u32), dd(i32), sigma(f32), dt(f32), num_channels(u32), ma(f32), basal_rate(f32), kinetic_cost(f32)
        {
            let ma = dd as f32 - sigma;
            let mut data = Vec::with_capacity(36);
            data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
            data.extend_from_slice(&(shape[1] as u32).to_le_bytes());
            data.extend_from_slice(&(dd as i32).to_le_bytes());
            data.extend_from_slice(&sigma.to_le_bytes());
            data.extend_from_slice(&dt.to_le_bytes());
            data.extend_from_slice(&(num_channels as u32).to_le_bytes());
            data.extend_from_slice(&ma.to_le_bytes());
            data.extend_from_slice(&basal_metabolic_rate.to_le_bytes());
            data.extend_from_slice(&kinetic_cost.to_le_bytes());
            queue.write_buffer(&reintegration_params_buffer, 0, &data);
        }

        // wall_params: width(u32), height(u32), num_channels(u32)
        {
            let wp: [u32; 3] = [shape[0] as u32, shape[1] as u32, num_channels as u32];
            queue.write_buffer(&wall_params_buffer, 0, bytemuck::cast_slice(&wp));
        }

        // --- Cached bind groups ---
        let normalize_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::normalize bg"),
            layout: &normalize_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: normalize_params_buffer.as_entire_binding(),
                },
            ],
        });

        let channel_aggregate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ca bg"),
            layout: &channel_aggregate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c1_flat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c1_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: channel_aggregate_params_buffer.as_entire_binding(),
                },
            ],
        });

        let sobel_u_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::sobel_u bg"),
            layout: &sobel_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nabla_u_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nabla_u_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sobel_params_u_buffer.as_entire_binding(),
                },
            ],
        });

        let sobel_a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::sobel_a bg"),
            layout: &sobel_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sum_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nabla_a_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nabla_a_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sobel_params_a_buffer.as_entire_binding(),
                },
            ],
        });

        let sum_channels_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::sc bg"),
            layout: &sum_channels_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sum_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sum_channels_params_buffer.as_entire_binding(),
                },
            ],
        });

        let flow_field_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ff bg"),
            layout: &flow_field_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nabla_u_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nabla_u_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: nabla_a_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: nabla_a_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flow_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: flow_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: flow_field_params_buffer.as_entire_binding(),
                },
            ],
        });

        let reintegration_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ri bg"),
            layout: &reintegration_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flow_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flow_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: new_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: reintegration_params_buffer.as_entire_binding(),
                },
            ],
        });

        let wall_interaction_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::wall bg"),
            layout: &wall_interaction_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: obstacle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wall_params_buffer.as_entire_binding(),
                },
            ],
        });

        GpuFlowLenia {
            context,
            shape: shape.to_vec(),
            total_elements,
            dt,
            num_channels,
            num_kernels,
            dd,
            sigma,
            basal_metabolic_rate,
            kinetic_cost,
            c0: c0.to_vec(),
            c1_flat,
            c1_offsets,
            kernel_m: kernel_m.to_vec(),
            kernel_s: kernel_s.to_vec(),
            kernel_h: kernel_h.to_vec(),
            channel_buffer,
            new_channel_buffer,
            conv_buffer,
            kernel_buffer,
            u_buffer,
            conv_x_buffer,
            u_channel_buffer,
            nabla_u_x_buffer,
            nabla_u_y_buffer,
            nabla_a_x_buffer,
            nabla_a_y_buffer,
            sum_a_buffer,
            flow_x_buffer,
            flow_y_buffer,
            obstacle_buffer,
            forward_fft_1d,
            inverse_fft_1d,
            copy_to_conv_pipeline,
            complex_mul_pipeline,
            normalize_pipeline,
            growth_pipeline,
            channel_aggregate_pipeline,
            sobel_pipeline,
            sum_channels_pipeline,
            flow_field_pipeline,
            reintegration_pipeline,
            wall_interaction_pipeline,
            copy_to_conv_bgl,
            complex_mul_bgl,
            normalize_bgl,
            growth_bgl,
            channel_aggregate_bgl,
            sobel_bgl,
            sum_channels_bgl,
            flow_field_bgl,
            reintegration_bgl,
            wall_interaction_bgl,
            normalize_bg,
            channel_aggregate_bg,
            sobel_u_bg,
            sobel_a_bg,
            sum_channels_bg,
            flow_field_bg,
            reintegration_bg,
            wall_interaction_bg,
            growth_params_buffer,
            normalize_params_buffer,
            channel_aggregate_params_buffer,
            sobel_params_u_buffer,
            sobel_params_a_buffer,
            sum_channels_params_buffer,
            flow_field_params_buffer,
            reintegration_params_buffer,
            wall_params_buffer,
            c1_flat_buffer,
            c1_offsets_buffer,
            readback_buffer,
        }
    }

    // --- Accessors ---

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn channel_buffer(&self) -> &wgpu::Buffer {
        &self.channel_buffer
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn num_kernels(&self) -> usize {
        self.num_kernels
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
        let ma = self.dd as f32 - self.sigma;
        let mut data = Vec::with_capacity(36);
        data.extend_from_slice(&(self.shape[0] as u32).to_le_bytes());
        data.extend_from_slice(&(self.shape[1] as u32).to_le_bytes());
        data.extend_from_slice(&(self.dd as i32).to_le_bytes());
        data.extend_from_slice(&self.sigma.to_le_bytes());
        data.extend_from_slice(&dt.to_le_bytes());
        data.extend_from_slice(&(self.num_channels as u32).to_le_bytes());
        data.extend_from_slice(&ma.to_le_bytes());
        data.extend_from_slice(&self.basal_metabolic_rate.to_le_bytes());
        data.extend_from_slice(&self.kinetic_cost.to_le_bytes());
        self.context
            .queue
            .write_buffer(&self.reintegration_params_buffer, 0, &data);
    }

    /// Uploads a pre-FFT'd kernel at the given index.
    pub fn set_kernel(&self, kernel_fft: &[num_complex::Complex32], kernel_idx: usize) {
        assert_eq!(kernel_fft.len(), self.total_elements);
        let offset = (kernel_idx * self.total_elements * 8) as u64;
        let data: Vec<[f32; 2]> = kernel_fft.iter().map(|c| [c.re, c.im]).collect();
        self.context
            .queue
            .write_buffer(&self.kernel_buffer, offset, bytemuck::cast_slice(&data));
    }

    /// Uploads channel data at the given index.
    pub fn upload_channel(&self, data: &[f64], channel: usize) {
        assert_eq!(data.len(), self.total_elements);
        let offset = (channel * self.total_elements * 4) as u64;
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        self.context.queue.write_buffer(
            &self.channel_buffer,
            offset,
            bytemuck::cast_slice(&f32_data),
        );
    }

    /// Uploads obstacle data (single channel, values 0.0 or 1.0).
    pub fn upload_obstacles(&self, data: &[f32]) {
        assert_eq!(data.len(), self.total_elements);
        self.context
            .queue
            .write_buffer(&self.obstacle_buffer, 0, bytemuck::cast_slice(data));
    }

    /// Returns a reference to the obstacle buffer for rendering.
    pub fn obstacle_buffer(&self) -> &wgpu::Buffer {
        &self.obstacle_buffer
    }

    /// Update growth parameters for a single kernel.
    pub fn set_growth_param(&mut self, k: usize, m: f32, s: f32, h: f32) {
        self.kernel_m[k] = m;
        self.kernel_s[k] = s;
        self.kernel_h[k] = h;
    }

    /// Update all growth parameters at once.
    pub fn set_all_growth_params(&mut self, m: &[f32], s: &[f32], h: &[f32]) {
        self.kernel_m.copy_from_slice(m);
        self.kernel_s.copy_from_slice(s);
        self.kernel_h.copy_from_slice(h);
    }

    /// Run N iterations (forward pass).
    pub fn run_steps(&self, n: usize) {
        for _ in 0..n {
            self.iterate();
        }
    }

    /// Download all channel data concatenated.
    pub fn download_all_channels(&self) -> Vec<f32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total_floats = self.total_elements * self.num_channels;
        let total_bytes = (total_floats * 4) as u64;

        // Need a bigger readback buffer
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::readback_all"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::download_all"),
        });
        encoder.copy_buffer_to_buffer(&self.channel_buffer, 0, &readback, 0, total_bytes);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback.unmap();
        result
    }

    /// Download kernel FFT weights for a given kernel.
    pub fn download_kernel(&self, k: usize) -> Vec<num_complex::Complex32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements;
        let total_bytes = (total * 8) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::readback_k"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::download_k"),
        });
        let offset = (k * total * 8) as u64;
        encoder.copy_buffer_to_buffer(&self.kernel_buffer, offset, &readback, 0, total_bytes);
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        let raw: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback.unmap();
        raw.chunks(2)
            .map(|c| num_complex::Complex32::new(c[0], c[1]))
            .collect()
    }

    /// Re-initialize all channels from flat f64 data.
    pub fn reinit_channels(&self, data: &[Vec<f64>]) {
        for (c, ch_data) in data.iter().enumerate() {
            self.upload_channel(ch_data, c);
        }
    }

    /// Get total elements (width * height).
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Get kernel_m slice.
    pub fn kernel_m_slice(&self) -> &[f32] {
        &self.kernel_m
    }

    /// Get kernel_s slice.
    pub fn kernel_s_slice(&self) -> &[f32] {
        &self.kernel_s
    }

    /// Get kernel_h slice.
    pub fn kernel_h_slice(&self) -> &[f32] {
        &self.kernel_h
    }

    pub fn download_channel(&self, channel: usize) -> Vec<f32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let buf_size = (self.total_elements * 4) as u64;
        let offset = (channel * self.total_elements * 4) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::download"),
        });
        encoder.copy_buffer_to_buffer(
            &self.channel_buffer,
            offset,
            &self.readback_buffer,
            0,
            buf_size,
        );
        queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.readback_buffer.unmap();
        result
    }

    // --- Main iteration ---

    /// Performs a single Flow Lenia iteration entirely on the GPU.
    pub fn iterate(&self) {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements as u32;
        let wg = (total + 255) / 256;
        let _wg_k = ((total * self.num_kernels as u32) + 255) / 256;
        let wg_c = ((total * self.num_channels as u32) + 255) / 256;

        let axis0 = self.shape[0] as u32;
        let axis1 = self.shape[1] as u32;
        let lanes0 = self.total_elements / axis0 as usize;
        let lanes1 = self.total_elements / axis1 as usize;

        // Cache FFT bind groups
        let fft_bgs: Vec<_> = self
            .forward_fft_1d
            .iter()
            .chain(self.inverse_fft_1d.iter())
            .map(|fft| fft.ensure_bind_groups(&self.conv_buffer))
            .collect();
        let (fw_bgs, inv_bgs) = fft_bgs.split_at(self.forward_fft_1d.len());

        // ================================================================
        // Phase 1: Per-kernel convolution + growth
        // ================================================================
        for k in 0..self.num_kernels {
            let src_c = self.c0[k] as usize;
            let ch_offset = (src_c * self.total_elements * 4) as u64;
            let k_offset = (k * self.total_elements * 8) as u64;
            let u_offset = (k * self.total_elements * 4) as u64;
            let ch_size = std::num::NonZeroU64::new((self.total_elements * 4) as u64);
            let k_size = std::num::NonZeroU64::new((self.total_elements * 8) as u64);
            let u_size = std::num::NonZeroU64::new((self.total_elements * 4) as u64);

            // Create per-kernel bind groups
            let copy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fl::copy_bg k={k}")),
                layout: &self.copy_to_conv_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.channel_buffer,
                            offset: ch_offset,
                            size: ch_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.conv_buffer.as_entire_binding(),
                    },
                ],
            });

            let cmul_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fl::cmul_bg k={k}")),
                layout: &self.complex_mul_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.conv_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.kernel_buffer,
                            offset: k_offset,
                            size: k_size,
                        }),
                    },
                ],
            });

            let grow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fl::grow_bg k={k}")),
                layout: &self.growth_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.conv_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.u_buffer,
                            offset: u_offset,
                            size: u_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.conv_x_buffer,
                            offset: u_offset,
                            size: u_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.growth_params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Write growth params
            let gp: [f32; 3] = [self.kernel_m[k], self.kernel_s[k], self.kernel_h[k]];
            queue.write_buffer(&self.growth_params_buffer, 0, bytemuck::cast_slice(&gp));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("fl::conv_k{k}")),
            });

            // Copy channel → conv
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("copy"),
                });
                p.set_pipeline(&self.copy_to_conv_pipeline);
                p.set_bind_group(0, &copy_bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            // Forward FFT axis 0
            {
                let (br, fft) = fw_bgs[0];
                self.forward_fft_1d[0].record_transform(&mut encoder, lanes0, axis0, 1, br, fft);
            }

            // Forward FFT axis 1
            {
                let (br, fft) = fw_bgs[1];
                self.forward_fft_1d[1].record_transform(&mut encoder, lanes1, 1, axis0, br, fft);
            }

            // Complex multiply
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cmul"),
                });
                p.set_pipeline(&self.complex_mul_pipeline);
                p.set_bind_group(0, &cmul_bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            // Inverse FFT axis 1
            {
                let (br, fft) = inv_bgs[1];
                self.inverse_fft_1d[1].record_transform(&mut encoder, lanes1, 1, axis0, br, fft);
            }

            // Inverse FFT axis 0
            {
                let (br, fft) = inv_bgs[0];
                self.inverse_fft_1d[0].record_transform(&mut encoder, lanes0, axis0, 1, br, fft);
            }

            // Normalize
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("norm"),
                });
                p.set_pipeline(&self.normalize_pipeline);
                p.set_bind_group(0, &self.normalize_bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            // Growth
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grow"),
                });
                p.set_pipeline(&self.growth_pipeline);
                p.set_bind_group(0, &grow_bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
        }

        // ================================================================
        // Phase 2: Channel aggregation, gradients, flow, reintegration
        // ================================================================
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fl::phase2"),
            });

            // Channel aggregate: u_buffer → u_channel_buffer
            {
                let mut p =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ca") });
                p.set_pipeline(&self.channel_aggregate_pipeline);
                p.set_bind_group(0, &self.channel_aggregate_bg, &[]);
                p.dispatch_workgroups(wg_c, 1, 1);
            }

            // Wall interaction: obstacle → negative growth on u_channel_buffer
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("wall"),
                });
                p.set_pipeline(&self.wall_interaction_pipeline);
                p.set_bind_group(0, &self.wall_interaction_bg, &[]);
                p.dispatch_workgroups(wg_c, 1, 1);
            }

            // Sum channels: channel_buffer → sum_a_buffer
            {
                let mut p =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sc") });
                p.set_pipeline(&self.sum_channels_pipeline);
                p.set_bind_group(0, &self.sum_channels_bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            // Sobel U: u_channel_buffer → nabla_u_x, nabla_u_y
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sobel_u"),
                });
                p.set_pipeline(&self.sobel_pipeline);
                p.set_bind_group(0, &self.sobel_u_bg, &[]);
                p.dispatch_workgroups(wg_c, 1, 1);
            }

            // Sobel A: sum_a_buffer → nabla_a_x, nabla_a_y
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sobel_a"),
                });
                p.set_pipeline(&self.sobel_pipeline);
                p.set_bind_group(0, &self.sobel_a_bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            // Flow field
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("flow"),
                });
                p.set_pipeline(&self.flow_field_pipeline);
                p.set_bind_group(0, &self.flow_field_bg, &[]);
                p.dispatch_workgroups(wg_c, 1, 1);
            }

            // Reintegration tracking
            {
                let mut p =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ri") });
                p.set_pipeline(&self.reintegration_pipeline);
                p.set_bind_group(0, &self.reintegration_bg, &[]);
                p.dispatch_workgroups(wg_c, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
        }

        // ================================================================
        // Phase 3: Swap new_channel → channel
        // ================================================================
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fl::swap"),
            });
            encoder.copy_buffer_to_buffer(
                &self.new_channel_buffer,
                0,
                &self.channel_buffer,
                0,
                (self.total_elements * self.num_channels * 4) as u64,
            );
            queue.submit(Some(encoder.finish()));
        }
    }
}
