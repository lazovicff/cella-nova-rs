//! GPU-accelerated Fast Fourier Transform using wgpu compute shaders.
//!
//! This module provides an alternative to the CPU-based FFT in [`super::fft`],
//! offloading the computation to the GPU (Metal on Apple Silicon, Vulkan/DX12 elsewhere).
//!
//! # Precision
//!
//! Uses `f32` (single-precision) instead of the `f64` used by the CPU FFT. This is
//! because:
//! - `f32` is universally supported in WebGPU/WGSL
//! - `f32` is significantly faster on GPUs (2-4x throughput vs f64)
//! - For visual simulations like Lenia, `f32` precision is sufficient
//!
//! # Algorithm
//!
//! Implements the iterative Cooley-Tukey radix-2 FFT using compute shaders:
//! 1. Bit-reversal permutation of the input
//! 2. log2(N) stages of butterfly operations
//!
//! All lanes along an axis are processed in a single GPU pass (batched).

use num_complex::Complex32;
use rustfft::{Fft, FftDirection, FftPlanner};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// WGSL shader sources (embedded as constants)
// ---------------------------------------------------------------------------

/// WGSL compute shader for bit-reversal permutation (batched, multi-lane).
const BIT_REVERSE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;

struct Params {
    n: u32,
    num_lanes: u32,
    lane_stride: u32,
    element_stride: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var result: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | ((x >> i) & 1u);
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total: u32 = params.n * params.num_lanes;
    let i: u32 = id.x;
    if (i >= total) {
        return;
    }
    let lane: u32 = i / params.n;
    let offset_in_lane: u32 = i % params.n;
    let bits: u32 = u32(log2(f32(params.n)));
    let j: u32 = bit_reverse(offset_in_lane, bits);
    if (offset_in_lane < j) {
        let base: u32 = lane * params.lane_stride;
        let a: u32 = base + offset_in_lane * params.element_stride;
        let b: u32 = base + j * params.element_stride;
        let tmp: vec2<f32> = data[a];
        data[a] = data[b];
        data[b] = tmp;
    }
}
"#;

/// WGSL compute shader for a single FFT butterfly stage (batched, multi-lane).
const FFT_STAGE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<f32>>;

struct Params {
    n: u32,
    stage: u32,
    inverse: u32,
    num_lanes: u32,
    lane_stride: u32,
    element_stride: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_n: u32 = params.n / 2u;
    let butterflies_per_lane: u32 = half_n;
    let total_butterflies: u32 = butterflies_per_lane * params.num_lanes;
    let i: u32 = id.x;
    if (i >= total_butterflies) {
        return;
    }
    let lane: u32 = i / butterflies_per_lane;
    let butterfly: u32 = i % butterflies_per_lane;
    let base: u32 = lane * params.lane_stride;
    let es: u32 = params.element_stride;

    let stride: u32 = 1u << params.stage;
    let block_size: u32 = stride * 2u;
    let block: u32 = butterfly / stride;
    let offset: u32 = butterfly % stride;
    let j: u32 = base + (block * block_size + offset) * es;
    let k: u32 = j + stride * es;

    let stage_offset: u32 = (1u << params.stage) - 1u;
    let w: vec2<f32> = twiddles[stage_offset + offset];

    let even: vec2<f32> = data[j];
    let odd: vec2<f32> = complex_mul(w, data[k]);

    data[j] = even + odd;
    data[k] = even - odd;
}
"#;

// ---------------------------------------------------------------------------
// WgpuContext
// ---------------------------------------------------------------------------

/// Manages the wgpu device and queue for GPU compute operations.
///
/// Create one instance and share it via `Arc` across all FFT instances.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    /// Initializes a wgpu device and queue.
    ///
    /// Uses the high-performance GPU adapter (discrete GPU if available).
    /// Panics if no suitable adapter is found.
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("wfft::WgpuContext::new() - No suitable GPU adapter found!");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wfft::WgpuContext Device"),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ))
        .expect("wfft::WgpuContext::new() - Failed to request GPU device!");

        WgpuContext { device, queue }
    }

    /// Creates a context from an existing device and queue (for sharing with a renderer).
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        WgpuContext { device, queue }
    }
}

/// Returns a global shared `WgpuContext`, creating it on first call.
pub fn global_context() -> Arc<WgpuContext> {
    static GLOBAL_CONTEXT: std::sync::OnceLock<Arc<WgpuContext>> = std::sync::OnceLock::new();
    GLOBAL_CONTEXT
        .get_or_init(|| Arc::new(WgpuContext::new()))
        .clone()
}

// ---------------------------------------------------------------------------
// WgpuFFT1D
// ---------------------------------------------------------------------------

/// A pre-planned 1D FFT using wgpu compute shaders.
///
/// Supports batched processing of multiple independent lanes in a single GPU pass.
pub struct WgpuFFT1D {
    context: Arc<WgpuContext>,
    n: usize,        // original size
    padded_n: usize, // next power of two (used for GPU FFT; equals n if n is a power of two)
    inverse: bool,
    num_stages: u32,
    // CPU fallback for non-power-of-two sizes
    cpu_fft: Option<Arc<dyn Fft<f32>>>,
    // GPU resources
    twiddle_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bit_rev_pipeline: wgpu::ComputePipeline,
    fft_stage_pipeline: wgpu::ComputePipeline,
    bit_rev_bind_group_layout: wgpu::BindGroupLayout,
    fft_bind_group_layout: wgpu::BindGroupLayout,
    // Cached bind groups (created lazily when data buffer is known)
    cached_bit_rev_bg: OnceLock<wgpu::BindGroup>,
    cached_fft_bg: OnceLock<wgpu::BindGroup>,
}

impl WgpuFFT1D {
    /// Creates a new 1D FFT plan.
    ///
    /// * `context` - Shared wgpu context.
    /// * `n` - Length of the FFT. If not a power of two, the input will be
    ///   zero-padded to the next power of two internally.
    /// * `inverse` - If true, performs the inverse FFT.
    pub fn new(context: Arc<WgpuContext>, n: usize, inverse: bool) -> Self {
        assert!(n > 0, "WgpuFFT1D::new() - n must be > 0");

        // Round up to the next power of two for the GPU FFT
        let padded_n = n.next_power_of_two();
        let use_gpu = n == padded_n;

        // CPU fallback for non-power-of-two sizes
        let cpu_fft = if use_gpu {
            None
        } else {
            let mut planner = FftPlanner::new();
            let direction = match inverse {
                true => FftDirection::Inverse,
                false => FftDirection::Forward,
            };
            Some(planner.plan_fft(n, direction))
        };

        let num_stages = if use_gpu {
            (padded_n as f64).log2() as u32
        } else {
            0
        };
        let device = &context.device;

        // --- Twiddle factors (precomputed on CPU) ---
        let twiddles: Vec<[f32; 2]> = Self::compute_twiddle_factors(padded_n, inverse);
        let twiddle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D twiddle buffer"),
            size: (twiddles.len() * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context
            .queue
            .write_buffer(&twiddle_buffer, 0, bytemuck::cast_slice(&twiddles));

        // --- Parameters uniform buffer ---
        // Sized for the FFT stage params struct (6 x u32 = 24 bytes)
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D params buffer"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Shader modules ---
        let bit_rev_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wfft::bit_reverse shader"),
            source: wgpu::ShaderSource::Wgsl(BIT_REVERSE_SHADER.into()),
        });
        let fft_stage_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wfft::fft_stage shader"),
            source: wgpu::ShaderSource::Wgsl(FFT_STAGE_SHADER.into()),
        });

        // --- Bind group layouts ---
        let bit_rev_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wfft::bit_rev bind group layout"),
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

        let fft_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wfft::fft_stage bind group layout"),
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

        // --- Pipeline layouts ---
        let bit_rev_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wfft::bit_rev pipeline layout"),
                bind_group_layouts: &[&bit_rev_bind_group_layout],
                push_constant_ranges: &[],
            });
        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wfft::fft_stage pipeline layout"),
            bind_group_layouts: &[&fft_bind_group_layout],
            push_constant_ranges: &[],
        });

        // --- Compute pipelines ---
        let bit_rev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wfft::bit_reverse pipeline"),
            layout: Some(&bit_rev_pipeline_layout),
            module: &bit_rev_shader,
            entry_point: "main",
        });

        let fft_stage_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wfft::fft_stage pipeline"),
            layout: Some(&fft_pipeline_layout),
            module: &fft_stage_shader,
            entry_point: "main",
        });

        // --- Compute pipelines ---
        WgpuFFT1D {
            context,
            n,
            padded_n,
            inverse,
            num_stages,
            cpu_fft,
            twiddle_buffer,
            params_buffer,
            bit_rev_pipeline,
            fft_stage_pipeline,
            bit_rev_bind_group_layout,
            fft_bind_group_layout,
            cached_bit_rev_bg: OnceLock::new(),
            cached_fft_bg: OnceLock::new(),
        }
    }

    /// Returns the original length of this FFT.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns the padded length (next power of two) used for GPU FFT.
    pub fn padded_len(&self) -> usize {
        self.padded_n
    }

    /// Returns whether this is an inverse FFT.
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Creates (or returns cached) bind groups for GPU-resident transforms.
    /// `data_buffer` is the storage buffer that will be transformed.
    pub fn ensure_bind_groups(
        &self,
        data_buffer: &wgpu::Buffer,
    ) -> (&wgpu::BindGroup, &wgpu::BindGroup) {
        let device = &self.context.device;
        let bit_rev = self.cached_bit_rev_bg.get_or_init(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wfft::bit_rev bg (cached)"),
                layout: &self.bit_rev_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        let fft = self.cached_fft_bg.get_or_init(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wfft::fft_stage bg (cached)"),
                layout: &self.fft_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        (bit_rev, fft)
    }

    /// Records all FFT dispatches into an external command encoder (no internal submits).
    /// Uses `copy_buffer_to_buffer` (ordered within the encoder) instead of
    /// `queue.write_buffer` (immediate) so that each stage sees its own params.
    pub fn record_transform(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        num_lanes: usize,
        lane_stride: u32,
        element_stride: u32,
        bit_rev_bg: &wgpu::BindGroup,
        fft_bg: &wgpu::BindGroup,
    ) {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let num_stages = self.num_stages as usize;

        // Build all params for all passes into one staging buffer.
        // Layout: [bit_rev: 4 u32s] [stage_0: 6 u32s] ... [stage_N: 6 u32s]
        let params_size = 4 * 4 + num_stages * 6 * 4; // u32 = 4 bytes each
        let mut params_data: Vec<u8> = Vec::with_capacity(params_size);

        // Bit-reverse params (4 u32s)
        params_data.extend_from_slice(bytemuck::cast_slice(&[
            self.padded_n as u32,
            num_lanes as u32,
            lane_stride,
            element_stride,
        ]));

        // Stage params (6 u32s each)
        for stage in 0..self.num_stages {
            params_data.extend_from_slice(bytemuck::cast_slice(&[
                self.padded_n as u32,
                stage,
                if self.inverse { 1 } else { 0 },
                num_lanes as u32,
                lane_stride,
                element_stride,
            ]));
        }

        // Upload all params to a staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::record_transform staging"),
            size: params_size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&staging, 0, &params_data);

        // Bit-reverse pass: copy params[0..16] to uniform buffer, then dispatch
        encoder.copy_buffer_to_buffer(&staging, 0, &self.params_buffer, 0, 16);
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wfft::bit_rev gpu"),
            });
            cp.set_pipeline(&self.bit_rev_pipeline);
            cp.set_bind_group(0, bit_rev_bg, &[]);
            cp.dispatch_workgroups((self.padded_n as u32 * num_lanes as u32 + 255) / 256, 1, 1);
        }

        // Butterfly stages: copy each stage's params, then dispatch
        for stage in 0..num_stages {
            let offset = (16 + stage * 24) as u64; // 16 = bit-rev size, 24 = 6 u32s
            encoder.copy_buffer_to_buffer(&staging, offset, &self.params_buffer, 0, 24);
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("wfft::stage {} gpu", stage)),
            });
            cp.set_pipeline(&self.fft_stage_pipeline);
            cp.set_bind_group(0, fft_bg, &[]);
            cp.dispatch_workgroups(
                ((self.padded_n as u32 / 2) * num_lanes as u32 + 255) / 256,
                1,
                1,
            );
        }
    }

    /// Transforms a single lane of data in-place using the GPU.
    pub fn transform(&self, data: &mut [Complex32]) {
        self.transform_batch(data, 1);
    }

    /// Transforms `num_lanes` independent lanes in a single GPU pass.
    ///
    /// `data` must have exactly `num_lanes * self.n` elements, laid out as
    /// `[lane_0, lane_1, ..., lane_{num_lanes-1}]` where each lane has `self.n` elements.
    /// If `self.n` is not a power of two, each lane is zero-padded to `self.padded_n`.
    pub fn transform_batch(&self, data: &mut [Complex32], num_lanes: usize) {
        assert_eq!(
            data.len(),
            num_lanes * self.n,
            "WgpuFFT1D::transform_batch() - data length {} does not match {} * {}",
            data.len(),
            num_lanes,
            self.n
        );

        // CPU fallback for non-power-of-two sizes
        if let Some(ref fft) = self.cpu_fft {
            let mut scratch: Vec<Complex32> =
                vec![Complex32::new(0.0, 0.0); fft.get_inplace_scratch_len()];
            for lane in 0..num_lanes {
                let start = lane * self.n;
                let mut lane_data: Vec<Complex32> = data[start..start + self.n].to_vec();
                fft.process_with_scratch(&mut lane_data, &mut scratch);
                if self.inverse {
                    let inv = 1.0 / self.n as f32;
                    for v in lane_data.iter_mut() {
                        v.re *= inv;
                        v.im *= inv;
                    }
                }
                data[start..start + self.n].copy_from_slice(&lane_data);
            }
            return;
        }

        let device = &self.context.device;
        let queue = &self.context.queue;

        // Total padded buffer size
        let total_padded = num_lanes * self.padded_n;
        let padded_buf_size = (total_padded * 8) as u64;

        // --- Create a GPU storage buffer ---
        let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D data buffer"),
            size: padded_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data: pad each lane with zeros if needed
        {
            let mut flat: Vec<[f32; 2]> = Vec::with_capacity(total_padded);
            for lane in 0..num_lanes {
                let start = lane * self.n;
                for i in 0..self.n {
                    let c = data[start + i];
                    flat.push([c.re, c.im]);
                }
                for _ in self.n..self.padded_n {
                    flat.push([0.0, 0.0]);
                }
            }
            queue.write_buffer(&data_buffer, 0, bytemuck::cast_slice(&flat));
        }

        // --- Create bind groups ---
        let bit_rev_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wfft::bit_rev bind group"),
            layout: &self.bit_rev_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let fft_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wfft::fft_stage bind group"),
            layout: &self.fft_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Bit-reversal pass ---
        {
            let params_data: [u32; 4] = [
                self.padded_n as u32,
                num_lanes as u32,
                self.padded_n as u32,
                1,
            ];
            queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params_data));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wfft::bit_rev encoder"),
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wfft::bit_reverse pass"),
            });
            cpass.set_pipeline(&self.bit_rev_pipeline);
            cpass.set_bind_group(0, &bit_rev_bind_group, &[]);
            let total_elements = self.padded_n as u32 * num_lanes as u32;
            let wg_count = (total_elements + 255) / 256;
            cpass.dispatch_workgroups(wg_count, 1, 1);
            drop(cpass);
            queue.submit(Some(encoder.finish()));
        }

        // --- Butterfly stages ---
        for stage in 0..self.num_stages {
            let params_data: [u32; 6] = [
                self.padded_n as u32,
                stage,
                if self.inverse { 1u32 } else { 0u32 },
                num_lanes as u32,
                self.padded_n as u32,
                1,
            ];
            queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params_data));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("wfft::fft_stage {} encoder", stage)),
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("wfft::fft_stage {} pass", stage)),
            });
            cpass.set_pipeline(&self.fft_stage_pipeline);
            cpass.set_bind_group(0, &fft_bind_group, &[]);
            let total_butterflies = (self.padded_n as u32 / 2) * num_lanes as u32;
            let wg_count = (total_butterflies + 255) / 256;
            cpass.dispatch_workgroups(wg_count, 1, 1);
            drop(cpass);
            queue.submit(Some(encoder.finish()));
        }

        // --- Create readback buffer (per-call, sized for the batch) ---
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::readback buffer"),
            size: padded_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Copy result back to readback buffer ---
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wfft::readback encoder"),
            });
            encoder.copy_buffer_to_buffer(&data_buffer, 0, &readback_buffer, 0, padded_buf_size);
            queue.submit(Some(encoder.finish()));
        }

        // --- Read back results (only first n elements per lane) ---
        {
            let readback_slice = readback_buffer.slice(..);
            readback_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let readback_view = readback_slice.get_mapped_range();
            let result_bytes: &[[f32; 2]] = bytemuck::cast_slice(&readback_view);
            for lane in 0..num_lanes {
                let dst_start = lane * self.n;
                let src_start = lane * self.padded_n;
                for i in 0..self.n {
                    let val = result_bytes[src_start + i];
                    data[dst_start + i] = Complex32::new(val[0], val[1]);
                }
            }
            drop(readback_view);
            readback_buffer.unmap();
        }

        // Normalize inverse FFT: divide by padded_n
        if self.inverse {
            let inv_n = 1.0 / self.padded_n as f32;
            for v in data.iter_mut() {
                v.re *= inv_n;
                v.im *= inv_n;
            }
        }
    }

    /// Precomputes all twiddle factors for all stages of the FFT.
    fn compute_twiddle_factors(n: usize, inverse: bool) -> Vec<[f32; 2]> {
        let num_stages = (n as f64).log2() as u32;
        let total_twiddles = n;
        let mut twiddles = Vec::with_capacity(total_twiddles);

        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };

        for stage in 0..num_stages {
            let block_size = 1u64 << (stage + 1);
            let _num_blocks = n as u64 / block_size;
            for k in 0..(block_size / 2) {
                let angle = sign * 2.0 * std::f64::consts::PI * (k as f64) / (block_size as f64);
                twiddles.push([angle.cos() as f32, angle.sin() as f32]);
            }
        }

        while twiddles.len() < n {
            twiddles.push([0.0, 0.0]);
        }

        twiddles
    }
}

// ---------------------------------------------------------------------------
// WgpuFFTND
// ---------------------------------------------------------------------------

/// A pre-planned N-dimensional FFT using wgpu compute shaders.
///
/// Performs a separable N-dimensional FFT by applying 1D FFTs along each axis.
/// All lanes along an axis are processed in a single GPU pass.
pub struct WgpuFFTND {
    _context: Arc<WgpuContext>,
    shape: Vec<usize>,
    fft_1d_instances: Vec<WgpuFFT1D>,
    inverse: bool,
}

impl std::fmt::Debug for WgpuFFTND {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuFFTND")
            .field("shape", &self.shape)
            .field("inverse", &self.inverse)
            .field("num_1d_ffts", &self.fft_1d_instances.len())
            .finish()
    }
}

impl WgpuFFTND {
    /// Creates a new N-dimensional FFT plan.
    pub fn new(context: Arc<WgpuContext>, shape: &[usize], inverse: bool) -> Self {
        assert!(
            !shape.is_empty(),
            "WgpuFFTND::new() - shape must not be empty!"
        );
        let mut fft_1d_instances = Vec::with_capacity(shape.len());
        for &dim in shape {
            fft_1d_instances.push(WgpuFFT1D::new(Arc::clone(&context), dim, inverse));
        }
        WgpuFFTND {
            _context: context,
            shape: shape.to_vec(),
            fft_1d_instances,
            inverse,
        }
    }

    /// Creates a new N-dimensional FFT plan using the global wgpu context.
    pub fn new_global(shape: &[usize], inverse: bool) -> Self {
        let context = global_context();
        Self::new(context, shape, inverse)
    }

    /// Returns the shape of the arrays this FFT can transform.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns whether this is an inverse FFT.
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Transforms `Complex<f32>` data in-place using the GPU.
    ///
    /// All lanes along each axis are processed in a single batched GPU pass.
    pub fn transform(&self, data: &mut ndarray::ArrayD<Complex32>) {
        assert_eq!(
            data.shape(),
            &self.shape[..],
            "WgpuFFTND::transform() - data shape mismatch"
        );

        let axis_order: Vec<usize> = if self.inverse {
            (0..self.shape.len()).rev().collect()
        } else {
            (0..self.shape.len()).collect()
        };

        for &axis in &axis_order {
            let fft_1d = &self.fft_1d_instances[axis];
            let axis_len = self.shape[axis];
            let num_lanes = data.len() / axis_len;

            let mut flat: Vec<Complex32> = Vec::with_capacity(data.len());
            for lane in data.lanes(ndarray::Axis(axis)) {
                for c in lane.iter() {
                    flat.push(*c);
                }
            }

            fft_1d.transform_batch(&mut flat, num_lanes);

            for (lane_idx, mut lane) in data.lanes_mut(ndarray::Axis(axis)).into_iter().enumerate()
            {
                let start = lane_idx * axis_len;
                for (i, c) in lane.iter_mut().enumerate() {
                    *c = flat[start + i];
                }
            }
        }
    }

    /// Transforms `Complex<f64>` data in-place, converting to/from `f32` for the GPU.
    ///
    /// All lanes along each axis are processed in a single batched GPU pass.
    pub fn transform_f64(&self, data: &mut ndarray::ArrayD<num_complex::Complex<f64>>) {
        assert_eq!(
            data.shape(),
            &self.shape[..],
            "WgpuFFTND::transform_f64() - data shape mismatch"
        );

        let axis_order: Vec<usize> = if self.inverse {
            (0..self.shape.len()).rev().collect()
        } else {
            (0..self.shape.len()).collect()
        };

        for &axis in &axis_order {
            let fft_1d = &self.fft_1d_instances[axis];
            let axis_len = self.shape[axis];

            // Number of lanes along this axis = total_elements / axis_len
            let num_lanes = data.len() / axis_len;

            // Collect all lanes into a single flat buffer
            // Each lane is a 1D slice along `axis`
            let mut flat: Vec<Complex32> = Vec::with_capacity(data.len());
            for lane in data.lanes(ndarray::Axis(axis)) {
                for c in lane.iter() {
                    flat.push(Complex32::new(c.re as f32, c.im as f32));
                }
            }

            // Transform all lanes in a single batched GPU pass
            fft_1d.transform_batch(&mut flat, num_lanes);

            // Scatter back
            for (lane_idx, mut lane) in data.lanes_mut(ndarray::Axis(axis)).into_iter().enumerate()
            {
                let start = lane_idx * axis_len;
                for (i, c) in lane.iter_mut().enumerate() {
                    let val = flat[start + i];
                    *c = num_complex::Complex::new(val.re as f64, val.im as f64);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    fn test_context() -> Arc<WgpuContext> {
        static INIT: std::sync::OnceLock<Arc<WgpuContext>> = std::sync::OnceLock::new();
        INIT.get_or_init(|| Arc::new(WgpuContext::new())).clone()
    }

    #[test]
    fn test_fft_1d_forward_inverse_roundtrip() {
        let ctx = test_context();
        let n = 256;

        let original: Vec<Complex32> = (0..n)
            .map(|i| {
                let val = (2.0 * std::f32::consts::PI * i as f32 / 16.0).cos();
                Complex32::new(val, 0.0)
            })
            .collect();

        let mut data = original.clone();

        let fft = WgpuFFT1D::new(Arc::clone(&ctx), n, false);
        fft.transform(&mut data);

        let ifft = WgpuFFT1D::new(Arc::clone(&ctx), n, true);
        ifft.transform(&mut data);

        for (a, b) in data.iter().zip(original.iter()) {
            let diff = (a.re - b.re).abs();
            assert!(
                diff < 1e-3,
                "Roundtrip error too large: |{} - {}| = {}",
                a.re,
                b.re,
                diff
            );
        }
    }

    #[test]
    fn test_fft_1d_impulse() {
        let ctx = test_context();
        let n = 128;

        let mut data: Vec<Complex32> = (0..n).map(|_| Complex32::new(0.0, 0.0)).collect();
        data[0] = Complex32::new(1.0, 0.0);

        let fft = WgpuFFT1D::new(Arc::clone(&ctx), n, false);
        fft.transform(&mut data);

        for (i, val) in data.iter().enumerate() {
            let mag = (val.re * val.re + val.im * val.im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-3,
                "Impulse response at {}: magnitude {} != 1.0",
                i,
                mag
            );
        }
    }

    #[test]
    fn test_fft_nd_2d_roundtrip() {
        let ctx = test_context();
        let shape = vec![32, 32];

        let original = ndarray::ArrayD::from_shape_fn(shape.clone(), |idx| {
            let val = ((idx[0] as f32).sin() * (idx[1] as f32).cos()) * 0.5;
            Complex32::new(val, 0.0)
        });

        let mut data = original.clone();

        let fft = WgpuFFTND::new(Arc::clone(&ctx), &shape, false);
        fft.transform(&mut data);

        let ifft = WgpuFFTND::new(Arc::clone(&ctx), &shape, true);
        ifft.transform(&mut data);

        for (a, b) in data.iter().zip(original.iter()) {
            let diff = (a.re - b.re).abs();
            assert!(
                diff < 1e-2,
                "2D roundtrip error too large: |{} - {}| = {}",
                a.re,
                b.re,
                diff
            );
        }
    }

    #[test]
    fn test_fft_1d_batch() {
        let ctx = test_context();
        let n = 64;
        let num_lanes = 16;

        // Create independent lanes: each lane is an impulse at a different position
        let mut data: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_lanes * n];
        for lane in 0..num_lanes {
            data[lane * n + lane] = Complex32::new(1.0, 0.0);
        }

        let fft = WgpuFFT1D::new(Arc::clone(&ctx), n, false);
        fft.transform_batch(&mut data, num_lanes);

        // Each lane's FFT should have magnitude 1.0 everywhere
        for lane in 0..num_lanes {
            let start = lane * n;
            for i in 0..n {
                let val = data[start + i];
                let mag = (val.re * val.re + val.im * val.im).sqrt();
                assert!(
                    (mag - 1.0).abs() < 1e-3,
                    "Batch lane {} element {}: magnitude {} != 1.0",
                    lane,
                    i,
                    mag
                );
            }
        }
    }

    #[test]
    fn test_fft_non_power_of_two() {
        let ctx = test_context();
        let n = 300; // not a power of two

        let original: Vec<Complex32> = (0..n)
            .map(|i| {
                let val = (2.0 * std::f32::consts::PI * i as f32 / 16.0).cos();
                Complex32::new(val, 0.0)
            })
            .collect();

        let mut data = original.clone();

        let fft = WgpuFFT1D::new(Arc::clone(&ctx), n, false);
        fft.transform(&mut data);

        let ifft = WgpuFFT1D::new(Arc::clone(&ctx), n, true);
        ifft.transform(&mut data);

        for (a, b) in data.iter().zip(original.iter()) {
            let diff = (a.re - b.re).abs();
            assert!(
                diff < 1e-2,
                "Non-power-of-two roundtrip error too large: |{} - {}| = {}",
                a.re,
                b.re,
                diff
            );
        }
    }
}

// ----
