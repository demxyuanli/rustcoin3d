const PI: f32 = 3.141592653589793;
const MAX_LIGHTS: u32 = 16u;
const CSM_CASCADE_COUNT: u32 = 4u;

struct SceneUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dirs: array<vec4<f32>, MAX_LIGHTS>,
    light_colors: array<vec4<f32>, MAX_LIGHTS>,
    light_types: array<vec4<f32>, MAX_LIGHTS>,
    light_positions: array<vec4<f32>, MAX_LIGHTS>,
    spot_params: array<vec4<f32>, MAX_LIGHTS>,
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
    csm_view_proj: array<mat4x4<f32>, CSM_CASCADE_COUNT>,
    csm_split_depths: vec4<f32>,
    shadow_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(1) @binding(0) var t_albedo: texture_2d<f32>;
@group(1) @binding(1) var s_mat: sampler;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(2) @binding(0) var t_shadow: texture_depth_2d_array;
@group(2) @binding(1) var s_shadow: sampler_comparison;
@group(3) @binding(0) var t_envmap: texture_2d<f32>;
@group(3) @binding(1) var t_brdf_lut: texture_2d<f32>;
@group(3) @binding(2) var s_ibl: sampler;

struct InstanceData {
    model: mat4x4<f32>,
    mvp: mat4x4<f32>,
    diffuse_color: vec4<f32>,
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
}

@group(3) @binding(3) var<storage, read> instances: array<InstanceData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_pos: vec4<f32>,
    @location(4) flat_instance_idx: u32,
    @location(5) world_tangent: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let inst = instances[instance_idx];
    let world_pos4 = inst.model * vec4<f32>(in.position, 1.0);
    out.clip_position = inst.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos4.xyz;
    out.world_normal = normalize((inst.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.texcoord;
    // Transform tangent (xyz) to world space; w carries handedness
    out.world_tangent = vec4<f32>(
        normalize((inst.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w,
    );
    let view_pos4 = u.csm_view_proj[0] * world_pos4;
    out.view_pos = view_pos4;
    out.flat_instance_idx = instance_idx;
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

// Select CSM cascade based on view-space depth
fn select_cascade(view_depth: f32) -> u32 {
    for (var i = 0u; i < CSM_CASCADE_COUNT - 1u; i = i + 1u) {
        if view_depth < u.csm_split_depths[i] {
            return i;
        }
    }
    return CSM_CASCADE_COUNT - 1u;
}

fn shadow_factor_csm(world_pos: vec3<f32>, world_normal: vec3<f32>, light_dir: vec3<f32>, cascade_idx: u32) -> f32 {
    if (u.shadow_params.w < 0.5) {
        return 1.0;
    }
    let light_clip = u.csm_view_proj[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / max(light_clip.w, 1e-6);
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let bias = u.shadow_params.y + (1.0 - dot(normalize(world_normal), normalize(light_dir))) * 0.01;
    let z_ref = ndc.z - bias;
    let hw = i32(floor(u.shadow_params.z + 0.5));
    var inv = u.shadow_params.x;
    if (hw <= 0) {
        return textureSampleCompareLevel(t_shadow, s_shadow, uv, i32(cascade_idx), z_ref);
    }
    var sum = 0.0;
    var cnt = 0.0;
    for (var y = -hw; y <= hw; y = y + 1) {
        for (var x = -hw; x <= hw; x = x + 1) {
            let off = vec2<f32>(f32(x), f32(y)) * inv;
            sum += textureSampleCompareLevel(t_shadow, s_shadow, uv + off, i32(cascade_idx), z_ref);
            cnt += 1.0;
        }
    }
    return sum / max(cnt, 1.0);
}

fn direction_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let u = atan2(d.x, -d.z) / (2.0 * PI) + 0.5;
    let v = asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5;
    return vec2<f32>(u, v);
}

fn sample_prefiltered_envmap(reflection: vec3<f32>, roughness: f32) -> vec3<f32> {
    let uv = direction_to_uv(reflection);
    let r = max(roughness, 0.001);
    return textureSampleLevel(t_envmap, s_ibl, uv, r * 3.0).rgb;
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

    let v = normalize(u.camera_pos.xyz - in.world_pos);

    let inst = instances[in.flat_instance_idx];
    // Robust normal mapping: if tangent basis is invalid, fall back to geometry normal.
    let N = normalize(in.world_normal);
    var n = N;
    let t_len2 = dot(in.world_tangent.xyz, in.world_tangent.xyz);
    if (t_len2 > 1e-8) {
        let T = normalize(in.world_tangent.xyz - dot(in.world_tangent.xyz, N) * N);
        let B = cross(N, T) * in.world_tangent.w;
        let TBN = mat3x3<f32>(T, B, N);
        // Sample normal map and decode from [0,1] to [-1,1]
        let tangent_normal_tex = textureSample(t_normal, s_mat, in.uv).rgb;
        let tangent_normal = normalize(tangent_normal_tex * 2.0 - 1.0);
        n = normalize(TBN * tangent_normal);
    }
    let n_dot_v = max(dot(n, v), 0.0001);

    let albedo_sample = textureSample(t_albedo, s_mat, in.uv).rgb;
    var albedo = albedo_sample * inst.base_color.xyz;
    let metallic = clamp(inst.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(inst.metallic_roughness.y, 0.04, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Compute view-space depth for CSM cascade selection
    let view_depth = abs(in.view_pos.z);

    var lo = vec3<f32>(0.0);
    let light_count = min(u32(u.light_count.x), MAX_LIGHTS);
    for (var i = 0u; i < light_count; i = i + 1u) {
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
            let cascade_idx = select_cascade(view_depth);
            sh = shadow_factor_csm(in.world_pos, in.world_normal, -light_dir, cascade_idx);
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

    let r = reflect(-v, n);
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    let ibl_kd = (1.0 - f) * (1.0 - metallic);
    let diffuse_ibl = u.ibl_diffuse.xyz * albedo * ibl_kd;

    let prefiltered_color = sample_prefiltered_envmap(r, roughness);
    let env_brdf = textureSample(t_brdf_lut, s_ibl, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered_color * (f * env_brdf.x + env_brdf.y);

    let color = lo + diffuse_ibl + specular_ibl;
    return vec4<f32>(color, 1.0);
}
