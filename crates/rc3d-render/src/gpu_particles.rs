//! Persistent GPU particle systems updated by compute shader.

use std::collections::HashMap;
use std::time::Instant;

use wgpu::util::DeviceExt;

use crate::render_passes::pass_effects::GpuEmitterParams;

const PARTICLE_STRIDE: u64 = 64;
const WORKGROUP: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmitterUniforms {
    origin: [f32; 4],
    origin_jitter: [f32; 4],
    velocity: [f32; 4],
    velocity_jitter: [f32; 4],
    acceleration: [f32; 4],
    start_color: [f32; 4],
    end_color: [f32; 4],
    sizes_time: [f32; 4],
}

struct GpuParticleSystem {
    particle_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    compute_bg: wgpu::BindGroup,
    count: u32,
}

/// Compute pipeline + per-node particle buffers.
pub struct GpuParticleSim {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    systems: HashMap<u64, GpuParticleSystem>,
    last_tick: Instant,
    elapsed: f32,
}

impl GpuParticleSim {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Particles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle_update.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Particle BGL"),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU Particle PLL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU Particle"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            bgl,
            systems: HashMap::new(),
            last_tick: Instant::now(),
            elapsed: 0.0,
        }
    }

    fn ensure_system(
        &mut self,
        device: &wgpu::Device,
        key: u64,
        count: u32,
    ) -> &GpuParticleSystem {
        let recreate = self
            .systems
            .get(&key)
            .map(|s| s.count != count)
            .unwrap_or(true);
        if recreate {
            let nbytes = (count as u64).max(1) * PARTICLE_STRIDE;
            let zeros = vec![0u8; nbytes as usize];
            let particle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU particle buffer"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU particle params"),
                size: std::mem::size_of::<GpuEmitterUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GPU particle compute BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            self.systems.insert(
                key,
                GpuParticleSystem {
                    particle_buf,
                    params_buf,
                    compute_bg,
                    count,
                },
            );
        }
        self.systems.get(&key).expect("particle system just inserted")
    }

    /// Advance simulation for one emitter.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        key: u64,
        emitter: &GpuEmitterParams,
        dt: f32,
        time: f32,
    ) {
        let count = emitter.max_particles.max(1);
        let uniforms = GpuEmitterUniforms {
            origin: [
                emitter.origin[0],
                emitter.origin[1],
                emitter.origin[2],
                emitter.spawn_rate,
            ],
            origin_jitter: [
                emitter.origin_jitter[0],
                emitter.origin_jitter[1],
                emitter.origin_jitter[2],
                count as f32,
            ],
            velocity: [
                emitter.velocity[0],
                emitter.velocity[1],
                emitter.velocity[2],
                emitter.lifetime,
            ],
            velocity_jitter: [
                emitter.velocity_jitter[0],
                emitter.velocity_jitter[1],
                emitter.velocity_jitter[2],
                emitter.lifetime_jitter,
            ],
            acceleration: [
                emitter.acceleration[0],
                emitter.acceleration[1],
                emitter.acceleration[2],
                dt,
            ],
            start_color: emitter.start_color,
            end_color: emitter.end_color,
            sizes_time: [emitter.start_size, emitter.end_size, time, key as f32],
        };
        let _ = self.ensure_system(device, key, count);
        let sys = self.systems.get(&key).expect("particle system");
        queue.write_buffer(&sys.params_buf, 0, bytemuck::bytes_of(&uniforms));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Particles"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &sys.compute_bg, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP), 1, 1);
    }

    pub fn buffer(&self, key: u64) -> Option<(&wgpu::Buffer, u32)> {
        self.systems.get(&key).map(|s| (&s.particle_buf, s.count))
    }

    pub fn tick_dt(&mut self) -> (f32, f32) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().clamp(0.0, 0.1);
        self.last_tick = now;
        self.elapsed += dt;
        (dt, self.elapsed)
    }
}
