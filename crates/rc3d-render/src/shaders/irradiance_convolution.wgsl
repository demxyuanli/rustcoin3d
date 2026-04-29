// Cosine-weighted hemisphere integration for diffuse IBL irradiance map.
// Convolves the environment cubemap with the Lambertian BRDF: ∫ L(ω) * max(0, n·ω) * dω / π
// Result is a single cubemap face at the output resolution.

@group(0) @binding(0) var env_map: texture_cube<f32>;
@group(0) @binding(1) var result_tex: texture_storage_2d<rgba16float, write>;

struct PushConstants {
    face: u32,
    sample_count: u32,
}

var<push_constant> pc: PushConstants;

const PI: f32 = 3.141592653589793;

fn cubemap_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let uc = 2.0 * uv - 1.0;
    switch face {
        case 0u: { return vec3<f32>( 1.0, -uc.y, -uc.x); }  // +X
        case 1u: { return vec3<f32>(-1.0, -uc.y,  uc.x); }  // -X
        case 2u: { return vec3<f32>( uc.x,  1.0,  uc.y); }  // +Y
        case 3u: { return vec3<f32>( uc.x, -1.0, -uc.y); }  // -Y
        case 4u: { return vec3<f32>( uc.x, -uc.y,  1.0); }  // +Z
        default: { return vec3<f32>(-uc.x, -uc.y, -1.0); }  // -Z
    }
}

fn tangent_space(n: vec3<f32>) -> (vec3<f32>, vec3<f32>) {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return (t, b);
}

fn hammersley(i: u32, N: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(N), f32(bit_reverse(i)) * 2.3283064365386963e-10);
}

fn bit_reverse(v: u32) -> u32 {
    var r = v;
    r = ((r & 0xaaaaaaaa) >> 1u) | ((r & 0x55555555) << 1u);
    r = ((r & 0xcccccccc) >> 2u) | ((r & 0x33333333) << 2u);
    r = ((r & 0xf0f0f0f0) >> 4u) | ((r & 0x0f0f0f0f) << 4u);
    r = ((r & 0xff00ff00) >> 8u) | ((r & 0x00ff00ff) << 8u);
    return (r >> 16u) | (r << 16u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(result_tex);
    if gid.x >= dims || gid.y >= dims {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let n = normalize(cubemap_direction(pc.face, uv));
    let (t, b) = tangent_space(n);

    var irradiance = vec3<f32>(0.0);
    let N = max(pc.sample_count, 1u);
    let Nf = f32(N);

    for (var i = 0u; i < N; i = i + 1u) {
        let xi = hammersley(i, N);
        let phi = 2.0 * PI * xi.x;
        let cos_theta = sqrt(xi.y);
        let sin_theta = sqrt(1.0 - xi.y);

        let h = t * (cos(phi) * sin_theta) + b * (sin(phi) * sin_theta) + n * cos_theta;
        let sample_dir = normalize(h);

        irradiance = irradiance + textureSample(env_map, env_map, sample_dir).rgb * cos_theta;
    }

    irradiance = irradiance * (PI / Nf);
    textureStore(result_tex, gid.xy, vec4<f32>(irradiance, 1.0));
}
