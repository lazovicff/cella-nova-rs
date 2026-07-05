use lenia_ca::gpu_lenia::{GpuGrowthFn, GpuLenia};
use lenia_ca::kernels;
use lenia_ca::wfft::WgpuContext;
use ndarray::Array2;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const GRID_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------

/// Full-screen triangle vertex shader (no vertex buffer needed).
const RENDER_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Single triangle covering the entire clip space
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}
"#;

/// Fragment shader: samples the channel buffer directly, no CPU readback.
const RENDER_FRAGMENT_SHADER: &str = r#"
struct Params {
    grid_size: u32,
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let cell_w = params.screen_width / f32(params.grid_size);
    let cell_h = params.screen_height / f32(params.grid_size);
    let col = u32(pos.x / cell_w);
    let row = u32(pos.y / cell_h);

    if (col >= params.grid_size || row >= params.grid_size) {
        return vec4<f32>(0.01, 0.01, 0.02, 1.0);
    }

    let idx = row * params.grid_size + col;
    let val = channel[idx];
    if (val <= 0.015) {
        return vec4<f32>(0.01, 0.01, 0.02, 1.0);
    }

    let intensity = pow(clamp(val, 0.0, 1.0), 0.65);
    return vec4<f32>(
        0.0,
        intensity * 0.8,
        intensity * 0.4,
        pow(intensity, 0.5) * 0.85,
    );
}
"#;

// ---------------------------------------------------------------------------
// Classic Orbium pattern
// ---------------------------------------------------------------------------

fn add_orbium(array: &mut Array2<f64>, cx: usize, cy: usize) {
    let shape = [array.shape()[0], array.shape()[1]];
    let radius = 20;

    let i_min = (cx as i32 - radius).max(0) as usize;
    let i_max = (cx as usize + radius as usize).min(shape[0]);
    let j_min = (cy as i32 - radius).max(0) as usize;
    let j_max = (cy as usize + radius as usize).min(shape[1]);

    for i in i_min..i_max {
        for j in j_min..j_max {
            let dx = i as f64 - cx as f64;
            let dy = j as f64 - cy as f64;
            let r = (dx * dx + dy * dy).sqrt() / 13.0;

            if r < 1.0 {
                let val = (-((r - 0.5) * (r - 0.5)) / (2.0 * 0.15 * 0.15)).exp();
                array[[i, j]] = (array[[i, j]] + val * 0.5).min(1.0);
            }
        }
    }
}

fn precompute_kernel_fft(
    kernel: &ndarray::ArrayD<f64>,
    size: usize,
) -> Vec<num_complex::Complex32> {
    use rustfft::{FftDirection, FftPlanner};

    let padded_size = size.next_power_of_two();
    let mut kernel_padded = ndarray::Array2::<f64>::zeros([padded_size; 2]);

    let k_shape = kernel.shape();
    let offset_i = (padded_size - k_shape[0]) / 2;
    let offset_j = (padded_size - k_shape[1]) / 2;
    for i in 0..k_shape[0] {
        for j in 0..k_shape[1] {
            kernel_padded[[offset_i + i, offset_j + j]] = kernel[[i, j]];
        }
    }

    let mut shifted = kernel_padded.clone();
    for i in 0..padded_size {
        for j in 0..padded_size {
            let si = (i + padded_size / 2) % padded_size;
            let sj = (j + padded_size / 2) % padded_size;
            shifted[[i, j]] = kernel_padded[[si, sj]];
        }
    }

    let sum: f64 = shifted.iter().sum();
    if sum > 0.0 {
        shifted.mapv_inplace(|v| v / sum);
    }

    let mut data: Vec<num_complex::Complex<f64>> = shifted
        .iter()
        .map(|&v| num_complex::Complex::new(v, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(padded_size, FftDirection::Forward);
    let mut scratch = vec![num_complex::Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

    for i in 0..padded_size {
        let mut row: Vec<num_complex::Complex<f64>> = (0..padded_size)
            .map(|j| data[i * padded_size + j])
            .collect();
        fft.process_with_scratch(&mut row, &mut scratch);
        for j in 0..padded_size {
            data[i * padded_size + j] = row[j];
        }
    }

    for j in 0..padded_size {
        let mut col: Vec<num_complex::Complex<f64>> = (0..padded_size)
            .map(|i| data[i * padded_size + j])
            .collect();
        fft.process_with_scratch(&mut col, &mut scratch);
        for i in 0..padded_size {
            data[i * padded_size + j] = col[i];
        }
    }

    data.iter()
        .map(|&c| num_complex::Complex32::new(c.re as f32, c.im as f32))
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Lenia: GPU-Accelerated Self-Organizing Life")
        .with_inner_size(winit::dpi::LogicalSize::new(
            GRID_SIZE as f64,
            GRID_SIZE as f64,
        ))
        .build(&event_loop)
        .unwrap();

    // --- wgpu setup ---
    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found!");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Lenia Device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to request GPU device!");

    let window_size = window.inner_size();
    let mut surface_config = surface
        .get_default_config(&adapter, window_size.width, window_size.height)
        .unwrap();
    surface_config.present_mode = wgpu::PresentMode::AutoVsync;
    surface.configure(&device, &surface_config);

    // --- Shared context for compute and rendering ---
    let context = Arc::new(WgpuContext::from_device(device, queue));

    // --- Render pipeline ---
    let vertex_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vertex shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_VERTEX_SHADER.into()),
        });

    let fragment_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fragment shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_FRAGMENT_SHADER.into()),
        });

    let render_bgl = context
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let render_pl = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pl"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

    let render_pipeline = context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    // --- Render params uniform buffer ---
    let render_params_buf = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render params"),
        size: 12,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    {
        let mut data = Vec::with_capacity(12);
        data.extend_from_slice(&(GRID_SIZE as u32).to_le_bytes());
        data.extend_from_slice(&(surface_config.width as f32).to_le_bytes());
        data.extend_from_slice(&(surface_config.height as f32).to_le_bytes());
        context.queue.write_buffer(&render_params_buf, 0, &data);
    }

    // --- Lenia simulation ---
    let shape = GRID_SIZE.next_power_of_two();
    let game = GpuLenia::new(
        Arc::clone(&context),
        &[shape, shape],
        GpuGrowthFn::StandardLenia {
            mu: 0.15,
            sigma: 0.017,
        },
        0.1,
    );

    let kernel = kernels::gaussian_donut_2d(13, 1.0 / 6.7);
    let kernel_fft = precompute_kernel_fft(&kernel, shape);
    game.set_kernel(&kernel_fft, 0);

    // Initialize with random orbium gliders
    {
        let mut rng = rand::thread_rng();
        let mut ch0 = Array2::<f64>::zeros([shape; 2]);
        for _ in 0..12 {
            let x = rng.gen_range(50..shape - 50);
            let y = rng.gen_range(50..shape - 50);
            add_orbium(&mut ch0, x, y);
        }
        let flat: Vec<f64> = ch0.iter().copied().collect();
        game.upload_channel(&flat, 0);
    }

    // --- Render bind group (binds channel buffer for fragment shader) ---
    let render_bg = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bg"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: game.channel_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_params_buf.as_entire_binding(),
                },
            ],
        });

    // --- State ---
    let mut paused = false;
    let mut last_fps_time = Instant::now();
    let mut frame_count: u32 = 0;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   🌊 Lenia: GPU-Accelerated Orbium Simulator 🌊       ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Running entirely on GPU ({}x{} grid)              ║",
        GRID_SIZE, GRID_SIZE
    );
    println!("║  Zero CPU readback — render samples GPU buffer       ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Controls:                                              ║");
    println!("║    [Space]     Pause/Resume simulation                 ║");
    println!("║    [R]         Reset with new random creatures         ║");
    println!("║    [Q/Esc]     Quit                                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,

            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(key),
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    },
                ..
            } => match key {
                VirtualKeyCode::Space => paused = !paused,
                VirtualKeyCode::R => {
                    let mut rng = rand::thread_rng();
                    let mut ch0 = Array2::<f64>::zeros([shape; 2]);
                    for _ in 0..12 {
                        let x = rng.gen_range(50..shape - 50);
                        let y = rng.gen_range(50..shape - 50);
                        add_orbium(&mut ch0, x, y);
                    }
                    let flat: Vec<f64> = ch0.iter().copied().collect();
                    game.upload_channel(&flat, 0);
                }
                VirtualKeyCode::Q | VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                _ => {}
            },

            Event::MainEventsCleared => {
                if !paused {
                    game.iterate();
                }
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                frame_count += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(last_fps_time).as_secs_f64();
                if elapsed >= 1.0 {
                    let fps = frame_count as f64 / elapsed;
                    window.set_title(&format!(
                        "Lenia GPU — {:.0} FPS ({}×{} grid)",
                        fps, GRID_SIZE, GRID_SIZE
                    ));
                    last_fps_time = now;
                    frame_count = 0;
                }

                let frame = surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("render encoder"),
                        });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.01,
                                    g: 0.01,
                                    b: 0.02,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });

                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &render_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }

                context.queue.submit(Some(encoder.finish()));
                frame.present();
            }

            _ => {}
        }
    });
}
