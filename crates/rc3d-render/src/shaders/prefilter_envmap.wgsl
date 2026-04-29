// GGX importance-sampled prefiltered environment map for specular IBL.
// Generates a mip chain where mip 0 = highest roughness, high mip = sharpest reflection.
// Each dispatch handles one cubemap face at one mip roughness level.

@group(0) @binding(0) var env_map: texture_cube<f32>;
@group(0) @binding(1) var result_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const SAMPLE_COUNT: u32 = 128u;

struct PushConstants {
    face: u32,
    roughness: f32,
    resolution: u32,
}

var<push_constant> pc: PushConstants;

fn cubemap_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let uc = 2.0 * uv - 1.0;
    switch face {
        case 0u: { return vec3<f32>( 1.0, -uc.y, -uc.x); }
        case 1u: { return vec3<f32>(-1.0, -uc.y,  uc.x); }
        case 2u: { return vec3<f32>( uc.x,  1.0,  uc.y); }
        case 3u: { return vec3<f32>( uc.x, -1.0, -uc.y); }
        case 4u: { return vec3<f32>( uc.x, -uc.y,  1.0); }
        default: { return vec3<f32>(-uc.x, -uc.y, -1.0); }
    }
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

fn ggx_importance_sample(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= pc.resolution || gid.y >= pc.resolution {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / f32(pc.resolution);
    let n = normalize(cubemap_direction(pc.face, uv));
    let v = n;

    var prefiltered = vec3<f32>(0.0);
    var total_weight = 0.0;

    let roughness = clamp(pc.roughness, 0.001, 1.0);

    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h_local = ggx_importance_sample(xi, roughness);
        // Build tangent space from n
        let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) < 0.999);
        let t = normalize(cross(up, n));
        let b = cross(n, t);
        let h = t * h_local.x + b * h_local.y + n * h_local.z;
        let h = normalize(h);

        let l = 2.0 * dot(v, h) * h - v;
        let n_dot_l = max(dot(n, l), 0.0);

        if n_dot_l > 0.0 {
            prefiltered = prefiltered + textureSample(env_map, env_map, l).rgb * n_dot_l;
            total_weight = total_weight + n_dot_l;
        }
    }

    prefiltered = prefiltered / max(total_weight, 0.0001);
    textureStore(result_tex, gid.xy, vec4<f32>(prefiltered, 1.0));
}
