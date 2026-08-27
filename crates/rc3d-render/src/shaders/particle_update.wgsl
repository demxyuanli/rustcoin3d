// GPU particle integrate + recycle (Three.js GPUParticleSystem analogue).
// Layout matches point_cloud.wgsl Point (64B): position/color/velocity/extra.

struct Particle {
    position: vec4<f32>, // xyz, size
    color: vec4<f32>,
    velocity: vec4<f32>, // xyz, age
    extra: vec4<f32>,    // lifetime, rng seed, pad
}

struct EmitterParams {
    origin: vec4<f32>,          // xyz, spawn_rate
    origin_jitter: vec4<f32>,   // xyz, max_particles
    velocity: vec4<f32>,        // xyz, lifetime
    velocity_jitter: vec4<f32>, // xyz, lifetime_jitter
    acceleration: vec4<f32>,    // xyz, dt
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    sizes_time: vec4<f32>,      // start_size, end_size, time, frame_seed
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: EmitterParams;

fn hash_u32(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn rand01(seed: ptr<function, u32>) -> f32 {
    *seed = hash_u32(*seed);
    return f32(*seed) * (1.0 / 4294967296.0);
}

fn rand_signed(seed: ptr<function, u32>) -> f32 {
    return rand01(seed) * 2.0 - 1.0;
}

fn jitter(base: vec3<f32>, j: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(
        base.x + j.x * rand_signed(seed),
        base.y + j.y * rand_signed(seed),
        base.z + j.z * rand_signed(seed),
    );
}

fn spawn_particle(seed: ptr<function, u32>) -> Particle {
    let lifetime = max(params.velocity.w + params.velocity_jitter.w * rand_signed(seed), 0.05);
    var p: Particle;
    let pos = jitter(params.origin.xyz, params.origin_jitter.xyz, seed);
    let vel = jitter(params.velocity.xyz, params.velocity_jitter.xyz, seed);
    p.position = vec4<f32>(pos, params.sizes_time.x);
    p.color = params.start_color;
    p.velocity = vec4<f32>(vel, 0.0);
    p.extra = vec4<f32>(lifetime, f32(*seed), 0.0, 0.0);
    return p;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let max_p = u32(params.origin_jitter.w + 0.5);
    if (idx >= max_p) { return; }

    let dt = params.acceleration.w;
    let spawn_rate = params.origin.w;
    let avg_life = max(params.velocity.w, 0.05);
    let alive_count = min(u32(spawn_rate * avg_life + 0.5), max_p);
    if (alive_count == 0u || idx >= alive_count) {
        particles[idx].color.a = 0.0;
        particles[idx].position.w = 0.0;
        particles[idx].velocity.w = params.velocity.w + 1.0;
        return;
    }

    var seed = hash_u32(idx ^ bitcast<u32>(params.sizes_time.w) ^ bitcast<u32>(params.sizes_time.z));
    var p = particles[idx];
    if (p.extra.y != 0.0) {
        seed = bitcast<u32>(p.extra.y);
    }

    var age = p.velocity.w;
    var life = p.extra.x;
    if (life <= 0.001 || age >= life) {
        p = spawn_particle(&seed);
        particles[idx] = p;
        return;
    }

    age = age + dt;
    if (age >= life) {
        p.color.a = 0.0;
        p.position.w = 0.0;
        p.velocity.w = age;
        p.extra.x = life;
        p.extra.y = f32(seed);
        particles[idx] = p;
        return;
    }

    var vel = p.velocity.xyz + params.acceleration.xyz * dt;
    var pos = p.position.xyz + vel * dt;
    let t = clamp(age / max(life, 0.001), 0.0, 1.0);
    let size = mix(params.sizes_time.x, params.sizes_time.y, t);
    let col = mix(params.start_color, params.end_color, t);

    p.position = vec4<f32>(pos, size);
    p.color = col;
    p.velocity = vec4<f32>(vel, age);
    p.extra = vec4<f32>(life, f32(seed), 0.0, 0.0);
    particles[idx] = p;
}
