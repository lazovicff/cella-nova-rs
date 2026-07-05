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
//! Each stage dispatches one compute shader invocation per N/2 butterfly pairs.

use num_complex::Complex32;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WGSL shader sources (embedded as constants)
// ---------------------------------------------------------------------------

/// WGSL compute shader for bit-reversal permutation.
const BIT_REVERSE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;

struct Params {
    n: u32,
    _padding: u32,
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
    let i: u32 = id.x;
    if (i >= params.n) {
        return;
    }
    // Number of bits needed to represent indices 0..n-1
    let bits: u32 = u32(log2(f32(params.n)));
    let j: u32 = bit_reverse(i, bits);
    if (i < j) {
        let tmp: vec2<f32> = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}
"#;

/// WGSL compute shader for a single FFT butterfly stage.
const FFT_STAGE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<f32>>;

struct Params {
    n: u32,
    stage: u32,
    inverse: u32,
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
    let i: u32 = id.x;
    let half_n: u32 = params.n / 2u;
    if (i >= half_n) {
        return;
    }

    let stride: u32 = 1u << params.stage;
    let block_size: u32 = stride * 2u;
    let block: u32 = i / stride;
    let offset: u32 = i % stride;
    let j: u32 = block * block_size + offset;
    let k: u32 = j + stride;

    // Twiddle factor: exp(-2*pi*i*offset / block_size) for forward,
    // or exp(2*pi*i*offset / block_size) for inverse.
    // Precomputed on CPU and stored sequentially per stage in the twiddle buffer.
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

        // Block on the async adapter request
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
}

// ---------------------------------------------------------------------------
// WgpuFFT1D
// ---------------------------------------------------------------------------

/// A pre-planned 1D FFT using wgpu compute shaders.
///
/// Once created for a given length and direction, it can efficiently
/// transform many arrays of that same length.
///
/// # Panics
///
/// - If `n` is not a power of two.
/// - If `n` is 0.
pub struct WgpuFFT1D {
    context: Arc<WgpuContext>,
    n: usize,
    inverse: bool,
    num_stages: u32,
    // GPU resources
    twiddle_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bit_rev_pipeline: wgpu::ComputePipeline,
    fft_stage_pipeline: wgpu::ComputePipeline,
    bit_rev_bind_group_layout: wgpu::BindGroupLayout,
    fft_bind_group_layout: wgpu::BindGroupLayout,
    // Readback buffer — sized to hold `n` complex f32 values
    readback_buffer: wgpu::Buffer,
}

impl WgpuFFT1D {
    /// Creates a new 1D FFT plan.
    ///
    /// * `context` - Shared wgpu context.
    /// * `n` - Length of the FFT (must be a power of two).
    /// * `inverse` - If true, performs the inverse FFT.
    pub fn new(context: Arc<WgpuContext>, n: usize, inverse: bool) -> Self {
        assert!(n > 0, "WgpuFFT1D::new() - n must be > 0");
        assert!(
            n.is_power_of_two(),
            "WgpuFFT1D::new() - n must be a power of two, got {}",
            n
        );

        let num_stages = (n as f64).log2() as u32;
        let device = &context.device;

        // --- Twiddle factors (precomputed on CPU) ---
        let twiddles: Vec<[f32; 2]> = Self::compute_twiddle_factors(n, inverse);
        let twiddle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D twiddle buffer"),
            size: (twiddles.len() * 8) as u64, // 2 f32 per complex = 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context
            .queue
            .write_buffer(&twiddle_buffer, 0, bytemuck::cast_slice(&twiddles));

        // --- Parameters uniform buffer ---
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D params buffer"),
            size: 12, // 3 x u32
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
        // Bit-reversal: binding 0 = data (storage), binding 1 = params (uniform)
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

        // FFT stage: binding 0 = data (storage), binding 1 = twiddles (storage), binding 2 = params (uniform)
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

        // --- Readback buffer ---
        let buf_size = (n * 8) as u64; // n complex f32 = n * 8 bytes
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D readback buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        WgpuFFT1D {
            context,
            n,
            inverse,
            num_stages,
            twiddle_buffer,
            params_buffer,
            bit_rev_pipeline,
            fft_stage_pipeline,
            bit_rev_bind_group_layout,
            fft_bind_group_layout,
            readback_buffer,
        }
    }

    /// Returns the length of this FFT.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns whether this is an inverse FFT.
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Transforms `data` in-place using the GPU.
    ///
    /// `data` must have exactly `self.n` elements.
    pub fn transform(&self, data: &mut [Complex32]) {
        assert_eq!(
            data.len(),
            self.n,
            "WgpuFFT1D::transform() - data length {} does not match FFT length {}",
            data.len(),
            self.n
        );

        let device = &self.context.device;
        let queue = &self.context.queue;

        // --- Create a GPU storage buffer for the data ---
        let buf_size = (self.n * 8) as u64;
        let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wfft::WgpuFFT1D data buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data directly via queue.write_buffer
        let complex_reinterpreted: Vec<[f32; 2]> = data.iter().map(|c| [c.re, c.im]).collect();
        queue.write_buffer(
            &data_buffer,
            0,
            bytemuck::cast_slice(&complex_reinterpreted),
        );

        // --- Create bind groups ---
        // Bit-reversal: binding 0 = data (storage), binding 1 = params (uniform)
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

        // FFT stage: binding 0 = data (storage), binding 1 = twiddles (storage), binding 2 = params (uniform)
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
        // Note: each pass uses a separate submit() so that queue.write_buffer
        // calls for params are visible to the subsequent compute pass.
        {
            let params_data: [u32; 2] = [self.n as u32, 0];
            queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params_data));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wfft::bit_rev encoder"),
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wfft::bit_reverse pass"),
            });
            cpass.set_pipeline(&self.bit_rev_pipeline);
            cpass.set_bind_group(0, &bit_rev_bind_group, &[]);
            let bit_rev_wg_count = (self.n as u32 + 255) / 256;
            cpass.dispatch_workgroups(bit_rev_wg_count, 1, 1);
            drop(cpass);
            queue.submit(Some(encoder.finish()));
        }

        // --- Butterfly stages ---
        for stage in 0..self.num_stages {
            let params_data: [u32; 3] =
                [self.n as u32, stage, if self.inverse { 1u32 } else { 0u32 }];
            queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params_data));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("wfft::fft_stage {} encoder", stage)),
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("wfft::fft_stage {} pass", stage)),
            });
            cpass.set_pipeline(&self.fft_stage_pipeline);
            cpass.set_bind_group(0, &fft_bind_group, &[]);
            let fft_wg_count = ((self.n as u32 / 2) + 255) / 256;
            cpass.dispatch_workgroups(fft_wg_count, 1, 1);
            drop(cpass);
            queue.submit(Some(encoder.finish()));
        }

        // --- Copy result back to readback buffer ---
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wfft::readback encoder"),
            });
            encoder.copy_buffer_to_buffer(&data_buffer, 0, &self.readback_buffer, 0, buf_size);
            queue.submit(Some(encoder.finish()));
        }

        // --- Read back results ---
        {
            let readback_slice = self.readback_buffer.slice(..);
            readback_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let readback_view = readback_slice.get_mapped_range();
            let result_bytes: &[[f32; 2]] = bytemuck::cast_slice(&readback_view);
            for (i, val) in result_bytes.iter().enumerate() {
                data[i] = Complex32::new(val[0], val[1]);
            }
            drop(readback_view);
            self.readback_buffer.unmap();
        }

        // Normalize inverse FFT: divide by N
        if self.inverse {
            let inv_n = 1.0 / self.n as f32;
            for v in data.iter_mut() {
                v.re *= inv_n;
                v.im *= inv_n;
            }
        }
    }

    /// Precomputes all twiddle factors for all stages of the FFT.
    ///
    /// Returns a flat array indexed by `stage_offset + k` where
    /// `stage_offset = 2^stage - 1` and `k` ranges from 0 to `2^stage - 1`.
    fn compute_twiddle_factors(n: usize, inverse: bool) -> Vec<[f32; 2]> {
        let num_stages = (n as f64).log2() as u32;
        let total_twiddles = n; // sum_{s=0}^{m-1} 2^s = 2^m - 1 = n - 1, plus one extra for alignment
        let mut twiddles = Vec::with_capacity(total_twiddles);

        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };

        for stage in 0..num_stages {
            let block_size = 1u64 << (stage + 1); // 2, 4, 8, ..., n
            let _num_blocks = n as u64 / block_size;
            for k in 0..(block_size / 2) {
                let angle = sign * 2.0 * std::f64::consts::PI * (k as f64) / (block_size as f64);
                twiddles.push([angle.cos() as f32, angle.sin() as f32]);
            }
        }

        // Pad to exactly n entries (should already be n-1, pad to n)
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
///
/// # Panics
///
/// - If any dimension is not a power of two.
/// - If any dimension is 0.
pub struct WgpuFFTND {
    _context: Arc<WgpuContext>,
    shape: Vec<usize>,
    fft_1d_instances: Vec<WgpuFFT1D>,
    inverse: bool,
}

impl WgpuFFTND {
    /// Creates a new N-dimensional FFT plan.
    ///
    /// * `context` - Shared wgpu context.
    /// * `shape` - Shape of the N-dimensional array (each dim must be a power of two).
    /// * `inverse` - If true, performs the inverse FFT.
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

    /// Returns the shape of the arrays this FFT can transform.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns whether this is an inverse FFT.
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Transforms `data` in-place using the GPU.
    ///
    /// `data` must have a shape matching `self.shape`.
    ///
    /// This processes one axis at a time. For each axis, it extracts each 1D lane
    /// (row/column/etc.), sends it to the GPU for the 1D FFT, and writes the result back.
    ///
    /// For large N-dimensional data, this involves multiple GPU round-trips per axis.
    /// A more optimized implementation would process all lanes of an axis in a single
    /// GPU pass, but this simpler approach is correct and functional.
    pub fn transform(&self, data: &mut ndarray::ArrayD<Complex32>) {
        assert_eq!(
            data.shape(),
            &self.shape[..],
            "WgpuFFTND::transform() - data shape {:?} does not match expected shape {:?}",
            data.shape(),
            &self.shape
        );

        // Process axes in order (forward) or reverse order (inverse),
        // matching the convention used in the CPU FFT module.
        let axis_order: Vec<usize> = if self.inverse {
            (0..self.shape.len()).rev().collect()
        } else {
            (0..self.shape.len()).collect()
        };

        for &axis in &axis_order {
            let fft_1d = &self.fft_1d_instances[axis];
            let _axis_len = self.shape[axis];

            // Process each lane along this axis
            for mut lane in data.lanes_mut(ndarray::Axis(axis)) {
                // Collect lane data into a contiguous Vec<Complex32>
                let mut buf: Vec<Complex32> = lane
                    .iter()
                    .map(|&c| Complex32::new(c.re as f32, c.im as f32))
                    .collect();

                // Transform on GPU
                fft_1d.transform(&mut buf);

                // Write back
                for (i, val) in buf.iter().enumerate() {
                    lane[i] = Complex32::new(val.re, val.im).into();
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

    /// Helper: create a shared wgpu context for tests.
    fn test_context() -> Arc<WgpuContext> {
        static INIT: std::sync::OnceLock<Arc<WgpuContext>> = std::sync::OnceLock::new();
        INIT.get_or_init(|| Arc::new(WgpuContext::new())).clone()
    }

    #[test]
    fn test_fft_1d_forward_inverse_roundtrip() {
        let ctx = test_context();
        let n = 256;

        // Create a simple signal: a cosine wave
        let original: Vec<Complex32> = (0..n)
            .map(|i| {
                let val = (2.0 * std::f32::consts::PI * i as f32 / 16.0).cos();
                Complex32::new(val, 0.0)
            })
            .collect();

        let mut data = original.clone();

        // Forward FFT
        let fft = WgpuFFT1D::new(Arc::clone(&ctx), n, false);
        fft.transform(&mut data);

        // Inverse FFT
        let ifft = WgpuFFT1D::new(Arc::clone(&ctx), n, true);
        ifft.transform(&mut data);

        // Check roundtrip: result should be close to original
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

        // Impulse at index 0 -> all frequencies should be 1.0
        let mut data: Vec<Complex32> = (0..n).map(|_| Complex32::new(0.0, 0.0)).collect();
        data[0] = Complex32::new(1.0, 0.0);

        let fft = WgpuFFT1D::new(Arc::clone(&ctx), n, false);
        fft.transform(&mut data);

        // All outputs should have magnitude 1.0
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

        // Create a 2D pattern
        let original = ndarray::ArrayD::from_shape_fn(shape.clone(), |idx| {
            let val = ((idx[0] as f32).sin() * (idx[1] as f32).cos()) * 0.5;
            Complex32::new(val, 0.0)
        });

        let mut data = original.clone();

        // Forward FFT
        let fft = WgpuFFTND::new(Arc::clone(&ctx), &shape, false);
        fft.transform(&mut data);

        // Inverse FFT
        let ifft = WgpuFFTND::new(Arc::clone(&ctx), &shape, true);
        ifft.transform(&mut data);

        // Check roundtrip
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
}
