// BRDF integration LUT for split-sum approximation.
// Output: RG = (scale, bias) for the specular IBL term.
// x-axis = NdotV, y-axis = roughness

@group(0) @binding(0) var lut_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const SAMPLE_COUNT: u32 = 1024u;

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

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn integrate_brdf(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    var scale = 0.0;
    var bias = 0.0;

    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = ggx_importance_sample(xi, roughness);
        let l = 2.0 * dot(v, h) * h - v;

        let n_dot_l = l.z;
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = pow(1.0 - v_dot_h, 5.0);
            scale = scale + (1.0 - fc) * g_vis;
            bias  = bias  + fc * g_vis;
        }
    }

    return vec2<f32>(scale, bias) / f32(SAMPLE_COUNT);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(lut_out);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let n_dot_v = uv.x;
    let roughness = uv.y;
    let result = integrate_brdf(max(n_dot_v, 0.001), max(roughness, 0.001));
    textureStore(lut_out, gid.xy, vec4<f32>(result.x, result.y, 0.0, 1.0));
}
