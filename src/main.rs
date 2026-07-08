use lenia_ca::gpu_flow_lenia::{generate_flow_kernels, GpuFlowLenia};
use lenia_ca::wfft::WgpuContext;
use ndarray::Array2;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const GRID_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------

const RENDER_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}
"#;

const RENDER_FRAGMENT_SHADER: &str = r#"
struct Params {
    grid_size: u32,
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> obstacle: array<f32>;

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

    // Obstacles render as dark red
    if (obstacle[idx] > 0.5) {
        return vec4<f32>(0.25, 0.05, 0.05, 1.0);
    }

    // Read all 3 channels from packed buffer
    let total_pixels = params.grid_size * params.grid_size;
    let c0 = channel[idx];
    let c1 = channel[total_pixels + idx];
    let c2 = channel[2u * total_pixels + idx];

    // RGB mapping with boosted contrast
    let r = clamp(c0 * 1.5, 0.0, 1.0);
    let g = clamp(c1 * 1.5, 0.0, 1.0);
    let b = clamp(c2 * 1.5, 0.0, 1.0);
    let intensity = r + g + b;

    if (intensity < 0.005) {
        return vec4<f32>(0.01, 0.01, 0.02, 1.0);
    }

    return vec4<f32>(
        pow(r, 0.5),
        pow(g, 0.5),
        pow(b, 0.5),
        1.0,
    );
}
"#;

// ---------------------------------------------------------------------------
// Load trained kernels
// ---------------------------------------------------------------------------
/// Load trained kernel FFT weights from a binary file and upload to the simulation.
fn load_trained_kernels(game: &GpuFlowLenia, path: &str) {
    let data = std::fs::read(path).expect("Failed to load kernels file");
    let header_bytes = 3 * 4; // 3 u32s
    let header: &[u32] = bytemuck::cast_slice(&data[..header_bytes]);
    let num_kernels = header[0] as usize;
    let _grid_size = header[1] as usize;
    let total = header[2] as usize;
    let kernel_floats: &[f32] = bytemuck::cast_slice(&data[header_bytes..]);

    println!("Loading {} kernels from '{}'...", num_kernels, path);
    for k in 0..num_kernels {
        let offset = k * total * 2;
        let mut kernel_data: Vec<num_complex::Complex32> = Vec::with_capacity(total);
        for i in 0..total {
            let re = kernel_floats[offset + i * 2];
            let im = kernel_floats[offset + i * 2 + 1];
            kernel_data.push(num_complex::Complex32::new(re, im));
        }
        game.set_kernel(&kernel_data, k);
    }
    println!("Loaded {} kernels.", num_kernels);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --- Interactive mode ---
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Flow Lenia: GPU-Accelerated Mass-Conserving Life")
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
            label: Some("Flow Lenia Device"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

    // --- Render params ---
    let render_params_buf = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render params"),
        size: 12,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Flow Lenia setup ---
    let shape = GRID_SIZE.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 9;

    // Channel mapping: M = [[2,1,0],[0,2,1],[1,0,2]]
    //   c0: source channel for each kernel
    //   c1: which kernels contribute to each target channel
    let c0: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let c1: Vec<Vec<u32>> = vec![
        vec![0, 1, 6], // ch0: 2 self + 1 from ch2
        vec![2, 3, 4], // ch1: 1 from ch0 + 2 self
        vec![5, 7, 8], // ch2: 1 from ch1 + 2 self
    ];

    // Per-kernel growth parameters — tuned for glider formation.
    // Classic Lenia uses growth around μ≈0.15–0.3 with small σ.
    // Mix of positive (growth) and negative (inhibition) heights.
    let mut rng = rand::thread_rng();
    let kernel_m: Vec<f32> = vec![0.15, 0.12, 0.28, 0.18, 0.10, 0.30, 0.22, 0.14, 0.25];
    let kernel_s: Vec<f32> = vec![0.02, 0.03, 0.015, 0.025, 0.02, 0.018, 0.022, 0.028, 0.02];
    let kernel_h: Vec<f32> = vec![0.8, -0.3, 0.6, 0.5, -0.4, 0.7, 0.4, -0.25, 0.55];

    let dt: f32 = 0.2;
    let dd: i32 = 5;
    let sigma: f32 = 0.65;
    let basal_metabolic_rate: f32 = 0.001;
    let kinetic_cost: f32 = 0.0005;

    let game = GpuFlowLenia::new(
        Arc::clone(&context),
        &[shape, shape],
        num_channels,
        num_kernels,
        &c0,
        &c1,
        &kernel_m,
        &kernel_s,
        &kernel_h,
        dt,
        dd,
        sigma,
        basal_metabolic_rate,
        kinetic_cost,
    );

    // Generate Flow Lenia kernels — Mexican-hat style for glider formation.
    // Each kernel: positive center bump + negative ring = local excitation, global inhibition.
    let global_r: f32 = 42.0; // characteristic scale for 512 grid
    let radii: Vec<f32> = vec![0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85];
    // a: bump positions (0=center, 1=edge). Center bump + ring bump.
    let a: Vec<[f32; 3]> = vec![
        [0.0, 0.6, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, 0.55, 0.0],
        [0.0, 0.45, 0.0],
        [0.0, 0.65, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.55, 0.0],
    ];
    let w: Vec<[f32; 3]> = vec![
        [0.08, 0.06, 0.01],
        [0.07, 0.05, 0.01],
        [0.09, 0.07, 0.01],
        [0.08, 0.06, 0.01],
        [0.07, 0.05, 0.01],
        [0.09, 0.07, 0.01],
        [0.08, 0.06, 0.01],
        [0.07, 0.05, 0.01],
        [0.08, 0.06, 0.01],
    ];
    let b: Vec<[f32; 3]> = vec![
        [0.8, -0.3, 0.0],
        [0.7, -0.25, 0.0],
        [0.9, -0.35, 0.0],
        [0.75, -0.3, 0.0],
        [0.65, -0.2, 0.0],
        [0.85, -0.35, 0.0],
        [0.7, -0.25, 0.0],
        [0.6, -0.2, 0.0],
        [0.8, -0.3, 0.0],
    ];

    let kernels_fft = generate_flow_kernels(shape, global_r, &radii, &a, &w, &b);
    for (k, kfft) in kernels_fft.iter().enumerate() {
        game.set_kernel(kfft, k);
    }

    // Load trained kernels if requested
    if let Some(pos) = args.iter().position(|a| a == "--load-kernels") {
        if let Some(path) = args.get(pos + 1) {
            load_trained_kernels(&game, path);
        } else {
            load_trained_kernels(&game, "trained_kernels.bin");
        }
    }

    // Initialize all channels with random noise patches.
    // All channels need initial activation for cross-talk to develop.
    let patch_size = (global_r as f64 * 1.5) as usize;
    let patch_half = patch_size / 2;
    let glider_positions: [(usize, usize); 3] = [
        (shape / 2, shape / 2),
        (shape / 3, shape / 2),
        (2 * shape / 3, shape / 2),
    ];
    for c in 0..num_channels {
        let mut ch = Array2::<f64>::zeros([shape; 2]);
        let (cx, cy) = glider_positions[c];
        let x0 = cx.saturating_sub(patch_half);
        let y0 = cy.saturating_sub(patch_half);
        for dy in 0..patch_size {
            for dx in 0..patch_size {
                let px = x0 + dx;
                let py = y0 + dy;
                if px < shape && py < shape {
                    ch[[py, px]] = rng.gen_range(0.0..1.0);
                }
            }
        }
        let flat: Vec<f64> = ch.iter().copied().collect();
        let max_val = flat.iter().cloned().fold(0.0f64, f64::max);
        println!("Channel {c}: max initial value = {max_val:.4}");
        game.upload_channel(&flat, c);
    }

    // Initialize obstacles: vertical walls and scattered blocks (scaled for 256 grid)
    {
        let mut obstacles = vec![0.0f32; shape * shape];
        // Vertical wall at x = shape/3
        let wall_x = shape / 3;
        for y in (shape / 8)..(7 * shape / 8) {
            if y % 8 > 5 {
                continue;
            }
            for dx in 0..2 {
                let idx = y * shape + wall_x + dx;
                if idx < obstacles.len() {
                    obstacles[idx] = 1.0;
                }
            }
        }
        // Vertical wall at x = 2*shape/3
        let wall_x2 = 2 * shape / 3;
        for y in (shape / 8)..(7 * shape / 8) {
            if y % 10 > 7 {
                continue;
            }
            for dx in 0..2 {
                let idx = y * shape + wall_x2 + dx;
                if idx < obstacles.len() {
                    obstacles[idx] = 1.0;
                }
            }
        }
        // Scattered obstacle disks
        for _ in 0..8 {
            let cx = rng.gen_range(shape / 6..5 * shape / 6);
            let cy = rng.gen_range(shape / 6..5 * shape / 6);
            let radius: i32 = rng.gen_range(2..6);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy > radius * radius {
                        continue;
                    }
                    let px = (cx as i32 + dx) as usize;
                    let py = (cy as i32 + dy) as usize;
                    if px < shape && py < shape {
                        obstacles[py * shape + px] = 1.0;
                    }
                }
            }
        }
        game.upload_obstacles(&obstacles);
    }

    // --- Render bind group ---
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: game.obstacle_buffer().as_entire_binding(),
                },
            ],
        });

    // --- State ---
    let mut paused = false;
    let mut last_fps_time = Instant::now();
    let mut frame_count: u32 = 0;
    let mut total_frames: u32 = 0;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   🌊 Flow Lenia: GPU-Accelerated Mass-Conserving CA  🌊 ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  {} channels, {} kernels, {}×{} grid                 ║",
        num_channels, num_kernels, GRID_SIZE, GRID_SIZE
    );
    println!(
        "║  Reintegration tracking: dd={}, σ={}                ║",
        dd, sigma
    );
    println!(
        "║  Metabolism: basal={}, kinetic={}            ║",
        basal_metabolic_rate, kinetic_cost
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Controls:                                              ║");
    println!("║    [Space]     Pause/Resume simulation                 ║");
    println!("║    [R]         Reset with new random state             ║");
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
                    let patch_size = (global_r as f64 * 1.5) as usize;
                    let patch_half = patch_size / 2;
                    let glider_positions: [(usize, usize); 3] = [
                        (shape / 2, shape / 2),
                        (shape / 3, shape / 2),
                        (2 * shape / 3, shape / 2),
                    ];
                    for c in 0..num_channels {
                        let mut ch = Array2::<f64>::zeros([shape; 2]);
                        let (cx, cy) = glider_positions[c];
                        let x0 = cx.saturating_sub(patch_half);
                        let y0 = cy.saturating_sub(patch_half);
                        for dy in 0..patch_size {
                            for dx in 0..patch_size {
                                let px = x0 + dx;
                                let py = y0 + dy;
                                if px < shape && py < shape {
                                    ch[[py, px]] = rng.gen_range(0.0..1.0);
                                }
                            }
                        }
                        let flat: Vec<f64> = ch.iter().copied().collect();
                        game.upload_channel(&flat, c);
                    }
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
                // Debug: print channel stats + center of mass at key frames
                let debug_frames = [10, 30, 60, 120];
                if debug_frames.contains(&total_frames) {
                    for c in 0..num_channels {
                        let data = game.download_channel(c);
                        let max_val = data.iter().cloned().fold(0.0f32, f32::max);
                        let sum: f32 = data.iter().sum();
                        let non_zero = data.iter().filter(|&&v| v > 0.001).count();
                        // Center of mass
                        let mut cx: f32 = 0.0;
                        let mut cy: f32 = 0.0;
                        for (i, &v) in data.iter().enumerate() {
                            if v > 0.001 {
                                let x = (i % shape) as f32;
                                let y = (i / shape) as f32;
                                cx += x * v;
                                cy += y * v;
                            }
                        }
                        if sum > 0.0 {
                            cx /= sum;
                            cy /= sum;
                        }
                        println!(
                            "Frame {frame_count} ch{c}: max={max_val:.4}, sum={sum:.2}, non_zero={non_zero}/{}, com=({cx:.0},{cy:.0})",
                            data.len()
                        );
                    }
                }
                frame_count += 1;
                total_frames += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(last_fps_time).as_secs_f64();
                if elapsed >= 1.0 {
                    let fps = frame_count as f64 / elapsed;
                    window.set_title(&format!(
                        "Flow Lenia GPU — {:.0} FPS ({}×{} grid, {}ch, {}kernels)",
                        fps, GRID_SIZE, GRID_SIZE, num_channels, num_kernels
                    ));
                    last_fps_time = now;
                    frame_count = 0;
                }

                // Update render params
                {
                    let mut data = Vec::with_capacity(12);
                    data.extend_from_slice(&(GRID_SIZE as u32).to_le_bytes());
                    data.extend_from_slice(&(surface_config.width as f32).to_le_bytes());
                    data.extend_from_slice(&(surface_config.height as f32).to_le_bytes());
                    context.queue.write_buffer(&render_params_buf, 0, &data);
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
