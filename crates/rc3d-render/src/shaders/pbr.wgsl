const PI: f32 = 3.141592653589793;

struct SceneUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dirs: array<vec4<f32>, 4>,
    light_colors: array<vec4<f32>, 4>,
    light_types: array<vec4<f32>, 4>,
    light_positions: array<vec4<f32>, 4>,
    spot_params: array<vec4<f32>, 4>,
    light_count: vec4<f32>,
    diffuse_color: vec4<f32>,
    ambient_color: vec4<f32>,
    specular_color: vec4<f32>,
    shininess: vec4<f32>,
    clip_planes: array<vec4<f32>, 6>,
    clip_count: vec4<f32>,
    pbr_base_color: vec4<f32>,
    pbr_metallic_roughness: vec4<f32>,
    ibl_diffuse: vec4<f32>,
    ibl_specular: vec4<f32>,
    light_view_proj: mat4x4<f32>,
    shadow_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(1) @binding(0) var t_albedo: texture_2d<f32>;
@group(1) @binding(1) var s_albedo: sampler;
@group(2) @binding(0) var t_shadow: texture_depth_2d;
@group(2) @binding(1) var s_shadow: sampler_comparison;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) light_clip: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    let wp = (u.model * vec4<f32>(in.position, 1.0)).xyz;
    out.world_pos = wp;
    out.world_normal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.texcoord;
    out.light_clip = u.light_view_proj * vec4<f32>(wp, 1.0);
    return out;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn shadow_factor_dir(light_clip: vec4<f32>, world_normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    if (u.shadow_params.w < 0.5) {
        return 1.0;
    }
    let ndc = light_clip.xyz / max(light_clip.w, 1e-6);
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let bias = u.shadow_params.y + (1.0 - dot(normalize(world_normal), normalize(light_dir))) * 0.01;
    let z_ref = ndc.z - bias;
    let hw = i32(floor(u.shadow_params.z + 0.5));
    var sum = 0.0;
    var cnt = 0.0;
    let inv = u.shadow_params.x;
    if (hw <= 0) {
        sum = textureSampleCompare(t_shadow, s_shadow, uv, z_ref);
        return sum;
    }
    for (var y = -hw; y <= hw; y = y + 1) {
        for (var x = -hw; x <= hw; x = x + 1) {
            let off = vec2<f32>(f32(x), f32(y)) * inv;
            sum += textureSampleCompare(t_shadow, s_shadow, uv + off, z_ref);
            cnt += 1.0;
        }
    }
    return sum / max(cnt, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let clip_count = i32(u.clip_count.x);
    for (var i = 0; i < clip_count; i = i + 1) {
        let plane = u.clip_planes[i];
        let dist = dot(plane.xyz, in.world_pos) + plane.w;
        if (dist < 0.0) {
            discard;
        }
    }

    let n = normalize(in.world_normal);
    let v = normalize(u.camera_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 0.0001);

    let albedo_sample = textureSample(t_albedo, s_albedo, in.uv).rgb;
    var albedo = albedo_sample * u.pbr_base_color.xyz;
    let metallic = clamp(u.pbr_metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(u.pbr_metallic_roughness.y, 0.04, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let light_count = min(i32(u.light_count.x), 4);
    for (var i = 0; i < light_count; i = i + 1) {
        let light_type = i32(u.light_types[i].x + 0.5);
        let raw_dir = normalize(u.light_dirs[i].xyz);
        let point_to_light = u.light_positions[i].xyz - in.world_pos;
        let dist = max(length(point_to_light), 0.0001);
        let to_light = point_to_light / dist;
        let attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        var light_dir = -raw_dir;
        var intensity_scale = 1.0;
        if (light_type == 1) {
            light_dir = to_light;
            intensity_scale = attenuation;
        } else if (light_type == 2) {
            light_dir = to_light;
            let cos_cutoff = u.spot_params[i].x;
            let drop_off = u.spot_params[i].y;
            let spot_cos = dot(normalize(-raw_dir), light_dir);
            if (spot_cos < cos_cutoff) {
                continue;
            }
            let spot_factor = pow(spot_cos, max(drop_off, 0.0));
            intensity_scale = attenuation * spot_factor;
        }
        var sh = 1.0;
        if (light_type == 0) {
            sh = shadow_factor_dir(in.light_clip, in.world_normal, -light_dir);
        }
        let light_color = u.light_colors[i].xyz * intensity_scale;
        let l = normalize(light_dir);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_h = max(dot(n, h), 0.0);
        let h_dot_v = max(dot(h, v), 0.0);

        let ndf = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let dfg = ndf * g;
        let spec = (dfg / max(4.0 * n_dot_v * n_dot_l, 0.001)) * fresnel_schlick(h_dot_v, f0);
        let kd = (1.0 - fresnel_schlick(n_dot_l, f0)) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;
        lo += (diffuse + spec) * light_color * n_dot_l * sh;
    }

    let ibl_kd = (1.0 - fresnel_schlick_roughness(n_dot_v, f0, roughness)) * (1.0 - metallic);
    let ambient = u.ibl_diffuse.xyz * albedo * ibl_kd;
    let spec_amb = u.ibl_specular.xyz * fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let color = lo + ambient + spec_amb * 0.2;
    return vec4<f32>(color, 1.0);
}
