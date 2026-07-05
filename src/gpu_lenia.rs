//! GPU-resident Lenia simulation.
//!
//! Runs the entire Lenia iteration on the GPU with zero CPU readback between frames.
//! Only reads back channel data when explicitly requested (for display).
//!
//! # Performance
//!
//! For large grids (512×512+), this is significantly faster than the CPU-based
//! Lenia implementations because:
//! - All data stays in GPU memory between iterations
//! - FFT, complex multiply, growth function, and channel update all run as GPU compute shaders
//! - No CPU↔GPU data transfer per iteration

use crate::wfft::{WgpuContext, WgpuFFT1D};
use num_complex::Complex32;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WGSL shaders for the Lenia pipeline
// ---------------------------------------------------------------------------

/// Copies channel field data into the convolution buffer (real = field, imag = 0).
const COPY_TO_CONV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = arrayLength(&channel);
    if (i >= total) { return; }
    conv[i] = vec2<f32>(channel[i], 0.0);
}
"#;

/// Complex multiplication: conv[i] *= kernel[i]
const COMPLEX_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = arrayLength(&conv);
    if (i >= total) { return; }
    conv[i] = complex_mul(conv[i], kernel[i]);
}
"#;

/// Standard Lenia growth function: 2 * exp(-(x-mu)^2 / (2*sigma^2)) - 1
const GROWTH_STANDARD_SHADER: &str = r#"
struct Params {
    mu: f32,
    sigma: f32,
}
@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = arrayLength(&result);
    if (i >= total) { return; }
    let x: f32 = conv[i].x;
    let diff: f32 = x - params.mu;
    let g: f32 = exp(-(diff * diff) / (2.0 * params.sigma * params.sigma));
    result[i] = 2.0 * g - 1.0;
}
"#;

/// Pass-through growth function: result[i] = conv[i].x * multiplier
const GROWTH_PASS_SHADER: &str = r#"
struct Params {
    multiplier: f32,
}
@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = arrayLength(&result);
    if (i >= total) { return; }
    result[i] = conv[i].x * params.multiplier;
}
"#;

/// Normalize inverse FFT result: divide by padded_n.
const NORMALIZE_SHADER: &str = r#"
struct Params {
    norm_factor: f32,
}
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = arrayLength(&data);
    if (i >= total) { return; }
    data[i] = data[i] * params.norm_factor;
}
"#;

/// Weighted sum of growth results and channel update:
/// delta[i] = sum(weight[j] * growth_result_j[i])
/// channel[i] = clamp(channel[i] + delta[i] * dt, 0, 1)
const UPDATE_SHADER: &str = r#"
struct Params {
    total: f32,
    dt: f32,
    weight: f32,
    weight_sum_reciprocal: f32,
}
@group(0) @binding(0) var<storage, read_write> channel: array<f32>;
@group(0) @binding(1) var<storage, read_write> delta: array<f32>;
@group(0) @binding(2) var<storage, read> growth_result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= u32(params.total)) { return; }

    let g: f32 = growth_result[i];
    let sum: f32 = g * params.weight * params.weight_sum_reciprocal;

    delta[i] = sum;
    channel[i] = clamp(channel[i] + sum * params.dt, 0.0, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// GpuLenia
// ---------------------------------------------------------------------------

/// Growth function types supported on the GPU.
#[derive(Clone, Copy, Debug)]
pub enum GpuGrowthFn {
    /// Standard Lenia gaussian bump: 2 * exp(-(x-mu)^2 / (2*sigma^2)) - 1
    StandardLenia { mu: f32, sigma: f32 },
    /// Pass-through: result = x * multiplier
    Pass { multiplier: f32 },
}

/// A Lenia simulation that runs entirely on the GPU.
///
/// All computation (FFT, complex multiply, growth function, channel update)
/// happens as GPU compute shaders. Channel data is only read back to the CPU
/// when explicitly requested (for display or export).
pub struct GpuLenia {
    context: Arc<WgpuContext>,
    shape: Vec<usize>,
    total_elements: usize,
    dt: f32,
    _num_channels: usize,
    num_conv_channels: usize,

    // GPU buffers
    channel_buffers: Vec<wgpu::Buffer>,
    delta_buffers: Vec<wgpu::Buffer>,
    conv_buffers: Vec<wgpu::Buffer>,
    kernel_buffers: Vec<wgpu::Buffer>,
    growth_result_buffers: Vec<wgpu::Buffer>,

    // FFT instances
    forward_fft_1d: Vec<WgpuFFT1D>,
    inverse_fft_1d: Vec<WgpuFFT1D>,

    // Compute pipelines
    copy_to_conv_pipeline: wgpu::ComputePipeline,
    complex_mul_pipeline: wgpu::ComputePipeline,
    growth_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    normalize_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    copy_to_conv_bgl: wgpu::BindGroupLayout,
    complex_mul_bgl: wgpu::BindGroupLayout,
    growth_bgl: wgpu::BindGroupLayout,
    update_bgl: wgpu::BindGroupLayout,
    normalize_bgl: wgpu::BindGroupLayout,

    // Cached bind groups (buffer bindings never change, only contents)
    copy_to_conv_bg: wgpu::BindGroup,
    complex_mul_bg: wgpu::BindGroup,
    normalize_bg: wgpu::BindGroup,
    growth_bg: wgpu::BindGroup,
    update_bg: wgpu::BindGroup,

    // Growth function configuration
    _growth_fn: GpuGrowthFn,
    growth_params_buffer: wgpu::Buffer,

    // Weights
    _weights: Vec<f32>,
    weight_sum_reciprocal: f32,

    // Uniform buffers
    update_params_buffer: wgpu::Buffer,
    normalize_params_buffer: wgpu::Buffer,

    // Readback buffer for display
    readback_buffer: wgpu::Buffer,
    // CPU-side staging buffer for uploads
    _staging_buffer: wgpu::Buffer,
}

impl GpuLenia {
    /// Creates a new GPU-resident Lenia simulation.
    ///
    /// * `shape` - Shape of the simulation grid (each dim must be a power of two).
    /// * `growth_fn` - Growth function to use.
    /// * `dt` - Time step.
    pub fn new(
        context: Arc<WgpuContext>,
        shape: &[usize],
        growth_fn: GpuGrowthFn,
        dt: f32,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;

        let total_elements: usize = shape.iter().product();
        let buf_size = (total_elements * 4) as u64; // f32 = 4 bytes
        let conv_buf_size = (total_elements * 8) as u64; // vec2<f32> = 8 bytes

        // --- Create FFT instances ---
        let mut forward_fft_1d = Vec::with_capacity(shape.len());
        let mut inverse_fft_1d = Vec::with_capacity(shape.len());
        for &dim in shape {
            forward_fft_1d.push(WgpuFFT1D::new(Arc::clone(&context), dim, false));
            inverse_fft_1d.push(WgpuFFT1D::new(Arc::clone(&context), dim, true));
        }

        // --- Create GPU buffers ---
        let make_storage_buffer = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let channel_buf = make_storage_buffer("gpu_lenia::channel", buf_size);
        let delta_buf = make_storage_buffer("gpu_lenia::delta", buf_size);
        let conv_buf = make_storage_buffer("gpu_lenia::conv", conv_buf_size);
        let kernel_buf = make_storage_buffer("gpu_lenia::kernel", conv_buf_size);
        let growth_buf = make_storage_buffer("gpu_lenia::growth_result", buf_size);

        // Uniform buffers
        let growth_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_lenia::growth_params"),
            size: 8, // 2 x f32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_lenia::update_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let normalize_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_lenia::normalize_params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging and readback buffers
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_lenia::staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_lenia::readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Create shader modules ---
        let copy_to_conv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_lenia::copy_to_conv shader"),
            source: wgpu::ShaderSource::Wgsl(COPY_TO_CONV_SHADER.into()),
        });
        let complex_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_lenia::complex_mul shader"),
            source: wgpu::ShaderSource::Wgsl(COMPLEX_MUL_SHADER.into()),
        });
        let growth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_lenia::growth shader"),
            source: wgpu::ShaderSource::Wgsl(
                match growth_fn {
                    GpuGrowthFn::StandardLenia { .. } => GROWTH_STANDARD_SHADER,
                    GpuGrowthFn::Pass { .. } => GROWTH_PASS_SHADER,
                }
                .into(),
            ),
        });
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_lenia::update shader"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let normalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_lenia::normalize shader"),
            source: wgpu::ShaderSource::Wgsl(NORMALIZE_SHADER.into()),
        });

        // --- Bind group layouts ---
        let copy_to_conv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_lenia::copy_to_conv bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let complex_mul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_lenia::complex_mul bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let growth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_lenia::growth bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_lenia::update bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let normalize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_lenia::normalize bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // --- Pipeline layouts ---
        let make_pipeline_layout =
            |label: &str, bgl: &wgpu::BindGroupLayout| -> wgpu::PipelineLayout {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[bgl],
                    push_constant_ranges: &[],
                })
            };

        let copy_to_conv_pl = make_pipeline_layout("gpu_lenia::copy_to_conv pl", &copy_to_conv_bgl);
        let complex_mul_pl = make_pipeline_layout("gpu_lenia::complex_mul pl", &complex_mul_bgl);
        let growth_pl = make_pipeline_layout("gpu_lenia::growth pl", &growth_bgl);
        let update_pl = make_pipeline_layout("gpu_lenia::update pl", &update_bgl);
        let normalize_pl = make_pipeline_layout("gpu_lenia::normalize pl", &normalize_bgl);

        // --- Compute pipelines ---
        let make_pipeline = |label: &str,
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

        let copy_to_conv_pipeline = make_pipeline(
            "gpu_lenia::copy_to_conv",
            &copy_to_conv_pl,
            &copy_to_conv_shader,
        );
        let complex_mul_pipeline = make_pipeline(
            "gpu_lenia::complex_mul",
            &complex_mul_pl,
            &complex_mul_shader,
        );
        let growth_pipeline = make_pipeline("gpu_lenia::growth", &growth_pl, &growth_shader);
        let update_pipeline = make_pipeline("gpu_lenia::update", &update_pl, &update_shader);
        let normalize_pipeline =
            make_pipeline("gpu_lenia::normalize", &normalize_pl, &normalize_shader);

        // Initialize channel buffer with zeros
        let zero_data: Vec<u8> = vec![0u8; buf_size as usize];
        queue.write_buffer(&channel_buf, 0, &zero_data);
        queue.write_buffer(&delta_buf, 0, &zero_data);
        queue.write_buffer(&growth_buf, 0, &zero_data);

        // Initialize weights
        let weight_sum_reciprocal = 1.0;

        // Initialize update params
        let update_params: [f32; 4] = [total_elements as f32, dt, 1.0, weight_sum_reciprocal];
        queue.write_buffer(&update_params_buf, 0, bytemuck::cast_slice(&update_params));

        // Initialize growth params
        match growth_fn {
            GpuGrowthFn::StandardLenia { mu, sigma } => {
                let params: [f32; 2] = [mu, sigma];
                queue.write_buffer(&growth_params_buf, 0, bytemuck::cast_slice(&params));
            }
            GpuGrowthFn::Pass { multiplier } => {
                let params: [f32; 1] = [multiplier];
                queue.write_buffer(&growth_params_buf, 0, bytemuck::cast_slice(&params));
            }
        }

        // --- Create cached bind groups (buffer bindings are static) ---
        let copy_to_conv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_lenia::copy_to_conv bg"),
            layout: &copy_to_conv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: channel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: conv_buf.as_entire_binding(),
                },
            ],
        });

        let complex_mul_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_lenia::complex_mul bg"),
            layout: &complex_mul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel_buf.as_entire_binding(),
                },
            ],
        });

        let normalize_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_lenia::normalize bg"),
            layout: &normalize_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: normalize_params_buf.as_entire_binding(),
                },
            ],
        });

        let growth_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_lenia::growth bg"),
            layout: &growth_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: growth_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: growth_params_buf.as_entire_binding(),
                },
            ],
        });

        let update_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_lenia::update bg"),
            layout: &update_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: channel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: delta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: growth_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: update_params_buf.as_entire_binding(),
                },
            ],
        });

        GpuLenia {
            context,
            shape: shape.to_vec(),
            total_elements,
            dt,
            _num_channels: 1,
            num_conv_channels: 1,
            channel_buffers: vec![channel_buf],
            delta_buffers: vec![delta_buf],
            conv_buffers: vec![conv_buf],
            kernel_buffers: vec![kernel_buf],
            growth_result_buffers: vec![growth_buf],
            forward_fft_1d,
            inverse_fft_1d,
            copy_to_conv_pipeline,
            complex_mul_pipeline,
            growth_pipeline,
            update_pipeline,
            normalize_pipeline,
            copy_to_conv_bgl,
            complex_mul_bgl,
            growth_bgl,
            update_bgl,
            normalize_bgl,
            copy_to_conv_bg,
            complex_mul_bg,
            normalize_bg,
            growth_bg,
            update_bg,
            _growth_fn: growth_fn,
            growth_params_buffer: growth_params_buf,
            _weights: Vec::new(),
            weight_sum_reciprocal,
            update_params_buffer: update_params_buf,
            normalize_params_buffer: normalize_params_buf,
            readback_buffer: readback_buf,
            _staging_buffer: staging_buf,
        }
    }

    /// Returns the shape of the simulation grid.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns a reference to the channel buffer (for binding in render pipelines).
    pub fn channel_buffer(&self) -> &wgpu::Buffer {
        &self.channel_buffers[0]
    }

    /// Returns the current dt value.
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Sets the dt value.
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
        let params: [f32; 3] = [
            self.num_conv_channels as f32,
            dt,
            self.weight_sum_reciprocal,
        ];
        self.context.queue.write_buffer(
            &self.update_params_buffer,
            0,
            bytemuck::cast_slice(&params),
        );
    }

    /// Sets the kernel for a convolution channel.
    ///
    /// `kernel_fft` is the pre-FFT'd kernel data (complex f32 values).
    pub fn set_kernel(&self, kernel_fft: &[Complex32], conv_channel: usize) {
        assert_eq!(kernel_fft.len(), self.total_elements);
        let buf = &self.kernel_buffers[conv_channel];
        let data: Vec<[f32; 2]> = kernel_fft.iter().map(|c| [c.re, c.im]).collect();
        self.context
            .queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&data));
    }

    /// Uploads initial channel data from the CPU to the GPU.
    pub fn upload_channel(&self, data: &[f64], channel: usize) {
        assert_eq!(data.len(), self.total_elements);
        let buf = &self.channel_buffers[channel];
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        self.context
            .queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&f32_data));
    }

    /// Reads channel data from the GPU to the CPU (for display).
    pub fn download_channel(&self, channel: usize) -> Vec<f32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let buf_size = (self.total_elements * 4) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_lenia::download encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.channel_buffers[channel],
            0,
            &self.readback_buffer,
            0,
            buf_size,
        );
        queue.submit(Some(encoder.finish()));

        let readback_slice = self.readback_buffer.slice(..);
        readback_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let view = readback_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.readback_buffer.unmap();
        result
    }

    /// Performs a single iteration of the Lenia simulation entirely on the GPU.
    /// All dispatches are recorded into a single command encoder and submitted once.
    pub fn iterate(&self) {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total_elements = self.total_elements as u32;
        let wg_count = (total_elements + 255) / 256;

        for conv_idx in 0..self.num_conv_channels {
            // Ensure FFT bind groups are cached (created once, reused thereafter)
            let conv_buf = &self.conv_buffers[conv_idx];
            let fft_bgs: Vec<_> = self
                .forward_fft_1d
                .iter()
                .chain(self.inverse_fft_1d.iter())
                .map(|fft| fft.ensure_bind_groups(conv_buf))
                .collect();
            let (fw_bgs, inv_bgs) = fft_bgs.split_at(self.forward_fft_1d.len());

            // Pre-compute FFT lane parameters for both axes
            let axis0_len = self.shape[0] as u32;
            let axis1_len = self.shape[1] as u32;
            let num_lanes0 = self.total_elements / axis0_len as usize;
            let num_lanes1 = self.total_elements / axis1_len as usize;

            // Pre-write normalize params (immediate, visible to subsequent dispatches)
            let padded_n = self.forward_fft_1d[0].padded_len() as f32;
            let norm_factor = 1.0 / padded_n.powi(self.shape.len() as i32);
            queue.write_buffer(
                &self.normalize_params_buffer,
                0,
                bytemuck::cast_slice(&[norm_factor]),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_lenia::iterate encoder"),
            });

            // 1. Copy channel to convolution buffer
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_lenia::copy_to_conv pass"),
                });
                cpass.set_pipeline(&self.copy_to_conv_pipeline);
                cpass.set_bind_group(0, &self.copy_to_conv_bg, &[]);
                cpass.dispatch_workgroups(wg_count, 1, 1);
            }

            // 2. Forward FFT axis 0 (rows: contiguous)
            {
                let (bit_rev_bg, fft_bg) = fw_bgs[0];
                self.forward_fft_1d[0].record_transform(
                    &mut encoder,
                    num_lanes0,
                    axis0_len,
                    1,
                    bit_rev_bg,
                    fft_bg,
                );
            }

            // 3. Forward FFT axis 1 (columns: strided)
            {
                let (bit_rev_bg, fft_bg) = fw_bgs[1];
                self.forward_fft_1d[1].record_transform(
                    &mut encoder,
                    num_lanes1,
                    1,
                    axis0_len,
                    bit_rev_bg,
                    fft_bg,
                );
            }

            // 4. Complex multiply with kernel
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_lenia::complex_mul pass"),
                });
                cpass.set_pipeline(&self.complex_mul_pipeline);
                cpass.set_bind_group(0, &self.complex_mul_bg, &[]);
                cpass.dispatch_workgroups(wg_count, 1, 1);
            }

            // 5. Inverse FFT axis 1 (columns: strided, reverse order)
            {
                let (bit_rev_bg, fft_bg) = inv_bgs[1];
                self.inverse_fft_1d[1].record_transform(
                    &mut encoder,
                    num_lanes1,
                    1,
                    axis0_len,
                    bit_rev_bg,
                    fft_bg,
                );
            }

            // 6. Inverse FFT axis 0 (rows: contiguous)
            {
                let (bit_rev_bg, fft_bg) = inv_bgs[0];
                self.inverse_fft_1d[0].record_transform(
                    &mut encoder,
                    num_lanes0,
                    axis0_len,
                    1,
                    bit_rev_bg,
                    fft_bg,
                );
            }

            // 7. Normalize inverse FFT result
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_lenia::normalize pass"),
                });
                cpass.set_pipeline(&self.normalize_pipeline);
                cpass.set_bind_group(0, &self.normalize_bg, &[]);
                cpass.dispatch_workgroups(wg_count, 1, 1);
            }

            // 8. Apply growth function
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_lenia::growth pass"),
                });
                cpass.set_pipeline(&self.growth_pipeline);
                cpass.set_bind_group(0, &self.growth_bg, &[]);
                cpass.dispatch_workgroups(wg_count, 1, 1);
            }

            // 9. Weighted sum and channel update
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_lenia::update pass"),
                });
                cpass.set_pipeline(&self.update_pipeline);
                cpass.set_bind_group(0, &self.update_bg, &[]);
                cpass.dispatch_workgroups(wg_count, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
        }
    }
}
