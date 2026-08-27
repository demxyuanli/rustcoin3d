//! GPU particle emitter (Three.js GPUParticleSystem analogue).
//!
//! Emitter parameters live on [`PointCloudNode`](crate::PointCloudNode).
//! By default simulation runs on a compute shader; set
//! [`ParticleEmitter::simulate_on_cpu`] to keep the CPU integrator.

use rc3d_core::math::Vec3;
use serde::{Deserialize, Serialize};

use crate::node_data::NodeData;
use crate::SceneGraph;

/// One simulated particle.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            age: 0.0,
            lifetime: 1.0,
            size: 1.0,
            color: [1.0; 4],
        }
    }
}

/// Continuous emitter that fills a [`PointCloudNode`](crate::PointCloudNode) particle list.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ParticleEmitter {
    pub origin: Vec3,
    pub origin_jitter: Vec3,
    pub velocity: Vec3,
    pub velocity_jitter: Vec3,
    pub acceleration: Vec3,
    /// Particles spawned per second.
    pub spawn_rate: f32,
    pub max_particles: u32,
    pub lifetime: f32,
    pub lifetime_jitter: f32,
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub start_size: f32,
    pub end_size: f32,
    /// When true, CPU `tick()` owns simulation. Default false = GPU compute path.
    #[serde(default)]
    pub simulate_on_cpu: bool,
    #[serde(default, skip)]
    spawn_accum: f32,
    #[serde(default, skip)]
    last_time: f64,
    #[serde(default, skip)]
    rng: u32,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            origin_jitter: Vec3::new(0.1, 0.1, 0.1),
            velocity: Vec3::new(0.0, 1.5, 0.0),
            velocity_jitter: Vec3::new(0.4, 0.4, 0.4),
            acceleration: Vec3::new(0.0, -0.4, 0.0),
            spawn_rate: 80.0,
            max_particles: 2048,
            lifetime: 2.0,
            lifetime_jitter: 0.5,
            start_color: [1.0, 0.85, 0.4, 1.0],
            end_color: [1.0, 0.2, 0.05, 0.0],
            start_size: 8.0,
            end_size: 2.0,
            simulate_on_cpu: false,
            spawn_accum: 0.0,
            last_time: 0.0,
            rng: 0x9E37_79B9,
        }
    }
}

impl ParticleEmitter {
    pub fn fountain() -> Self {
        Self::default()
    }

    fn rand01(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }

    fn rand_signed(&mut self) -> f32 {
        self.rand01() * 2.0 - 1.0
    }

    fn jittered(&mut self, base: Vec3, jitter: Vec3) -> Vec3 {
        Vec3::new(
            base.x + jitter.x * self.rand_signed(),
            base.y + jitter.y * self.rand_signed(),
            base.z + jitter.z * self.rand_signed(),
        )
    }

    fn spawn_one(&mut self) -> Particle {
        let lifetime = (self.lifetime + self.lifetime_jitter * self.rand_signed()).max(0.05);
        Particle {
            position: self.jittered(self.origin, self.origin_jitter),
            velocity: self.jittered(self.velocity, self.velocity_jitter),
            age: 0.0,
            lifetime,
            size: self.start_size,
            color: self.start_color,
        }
    }

    /// Advance simulation. `time` is absolute seconds from the world clock.
    pub fn tick(&mut self, particles: &mut Vec<Particle>, time: f64) {
        let dt = if self.last_time <= 0.0 {
            0.0
        } else {
            ((time - self.last_time) as f32).clamp(0.0, 0.1)
        };
        self.last_time = time;
        if dt <= 0.0 {
            return;
        }

        let cap = self.max_particles as usize;
        self.spawn_accum += self.spawn_rate * dt;
        while self.spawn_accum >= 1.0 && particles.len() < cap {
            self.spawn_accum -= 1.0;
            particles.push(self.spawn_one());
        }
        if particles.len() >= cap {
            self.spawn_accum = 0.0;
        }

        let start_c = self.start_color;
        let end_c = self.end_color;
        let start_s = self.start_size;
        let end_s = self.end_size;
        let acc = self.acceleration;

        let mut write = 0usize;
        for i in 0..particles.len() {
            let mut p = particles[i];
            p.age += dt;
            if p.age >= p.lifetime {
                continue;
            }
            p.velocity += acc * dt;
            p.position += p.velocity * dt;
            let t = (p.age / p.lifetime).clamp(0.0, 1.0);
            p.size = start_s + (end_s - start_s) * t;
            p.color = [
                start_c[0] + (end_c[0] - start_c[0]) * t,
                start_c[1] + (end_c[1] - start_c[1]) * t,
                start_c[2] + (end_c[2] - start_c[2]) * t,
                start_c[3] + (end_c[3] - start_c[3]) * t,
            ];
            particles[write] = p;
            write += 1;
        }
        particles.truncate(write);
    }
}

/// Tick every `PointCloudNode` that has an emitter. Called from the world clock.
pub fn tick_particle_emitters(graph: &mut SceneGraph, time: f64) {
    let mut stack: Vec<rc3d_core::NodeId> = graph.roots().to_vec();
    let mut ids = Vec::new();
    while let Some(id) = stack.pop() {
        ids.push(id);
        if let Some(entry) = graph.get(id) {
            stack.extend_from_slice(&entry.children);
        }
    }
    for id in ids {
        let Some(entry) = graph.get_mut(id) else { continue };
        let NodeData::PointCloud(pc) = &mut entry.data else { continue };
        let Some(ref mut emitter) = pc.emitter else { continue };
        if !emitter.simulate_on_cpu {
            continue;
        }
        emitter.tick(&mut pc.particles, time);
        entry.dirty_flags |= crate::node_entry::dirty_flags::GEOMETRY;
    }
}
