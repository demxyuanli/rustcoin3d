const PI: f32 = 3.141592653589793;
const MAX_LIGHTS: u32 = 16u;
const CSM_CASCADE_COUNT: u32 = 4u;

struct PerDrawUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    diffuse_color: vec4<f32>,
    ambient_color: vec4<f32>,
    specular_color: vec4<f32>,
    shininess: vec4<f32>,
    clip_planes: array<vec4<f32>, 6>,
    clip_count: vec4<f32>,
    pbr_base_color: vec4<f32>,
    pbr_metallic_roughness: vec4<f32>,
    pbr_emissive_alpha: vec4<f32>,
    pbr_alpha_flags: vec4<f32>,
    pbr_clearcoat: vec4<f32>,
    pbr_sheen: vec4<f32>,
    pbr_specular: vec4<f32>,
    pbr_transmission: vec4<f32>,
    light_set_index: vec4<f32>,
}

struct GlobalFrameUniforms {
    light_dirs: array<vec4<f32>, MAX_LIGHTS>,
    light_colors: array<vec4<f32>, MAX_LIGHTS>,
    light_types: array<vec4<f32>, MAX_LIGHTS>,
    light_positions: array<vec4<f32>, MAX_LIGHTS>,
    spot_params: array<vec4<f32>, MAX_LIGHTS>,
    light_count: vec4<f32>,
    ibl_diffuse: vec4<f32>,
    ibl_specular: vec4<f32>,
    sh_l2: array<vec4<f32>, 9>,
    sh_intensity: vec4<f32>,
    csm_view_proj: array<mat4x4<f32>, CSM_CASCADE_COUNT>,
    csm_split_depths: vec4<f32>,
    shadow_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: PerDrawUniforms;
@group(2) @binding(2) var<uniform> g: GlobalFrameUniforms;
@group(1) @binding(0) var t_albedo: texture_2d<f32>;
@group(1) @binding(1) var s_mat: sampler;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var t_metallic_roughness: texture_2d<f32>;
@group(1) @binding(4) var t_emissive: texture_2d<f32>;
@group(1) @binding(5) var t_occlusion: texture_2d<f32>;
@group(2) @binding(0) var t_shadow: texture_depth_2d_array;
@group(2) @binding(1) var s_shadow: sampler_comparison;
@group(2) @binding(3) var t_omni_shadow: texture_depth_cube_array;
@group(2) @binding(4) var s_omni_shadow: sampler_comparison;
@group(3) @binding(0) var t_envmap: texture_2d<f32>;
@group(3) @binding(1) var t_brdf_lut: texture_2d<f32>;
@group(3) @binding(2) var s_ibl: sampler;

struct InstanceData {
    model: mat4x4<f32>,
    mvp: mat4x4<f32>,
    diffuse_color: vec4<f32>,
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    emissive_alpha: vec4<f32>,
    morph_weights: array<f32, 8>,
    morph_count: vec4<f32>,
}

@group(3) @binding(3) var<storage, read> instances: array<InstanceData>;

// Morph target position deltas: flat vec3 array indexed as
//   target_stride * target_index + vertex_index
// where target_stride = array_length / morph_target_count.
@group(3) @binding(4) var<storage, read> morph_position_deltas: array<vec4<f32>>;
@group(3) @binding(5) var<storage, read> morph_normal_deltas: array<vec4<f32>>;
// x = vertex_count (stride per morph target)
@group(3) @binding(6) var<uniform> morph_params: vec4<f32>;

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
    @location(4) world_tangent: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_idx: u32, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let inst = instances[instance_idx];

    // Apply morph target blending (weights still per-instance in SSBO)
    var pos = in.position;
    var nrm = in.normal;
    let mt_count = i32(inst.morph_count.x + 0.5);
    if (mt_count > 0) {
        let stride = i32(morph_params.x + 0.5);
        for (var t = 0; t < mt_count; t = t + 1) {
            let w = inst.morph_weights[t];
            if (abs(w) > 0.0001) {
                let base_idx = t * stride + i32(vertex_idx);
                pos = pos + morph_position_deltas[base_idx].xyz * w;
                if (base_idx < i32(arrayLength(&morph_normal_deltas))) {
                    nrm = nrm + morph_normal_deltas[base_idx].xyz * w;
                }
            }
        }
    }

    // World / clip from per-instance data so instanced draws get correct transform.
    let world_pos4 = inst.model * vec4<f32>(pos, 1.0);
    out.clip_position = inst.mvp * vec4<f32>(pos, 1.0);
    out.world_pos = world_pos4.xyz;
    let world_normal = normalize((inst.model * vec4<f32>(nrm, 0.0)).xyz);
    out.world_normal = world_normal;
    out.uv = in.texcoord;
    var tangent_w = in.tangent.w;
    // Flip tangent-space handedness when geometry normal faces away from camera
    // so that the TBN basis and normal mapping stay consistent for back-facing surfaces.
    let view_dir = normalize(u.camera_pos.xyz - world_pos4.xyz);
    if (dot(world_normal, view_dir) < 0.0) {
        tangent_w = -tangent_w;
    }
    out.world_tangent = vec4<f32>(
        normalize((inst.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        tangent_w,
    );
    // Camera-relative offset; FS uses length() as view-space cascade depth.
    out.view_pos = vec4<f32>(world_pos4.xyz - u.camera_pos.xyz, 1.0);
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

struct AnisoFrame {
    t: vec3<f32>,
    b: vec3<f32>,
}

fn anisotropy_frame(n: vec3<f32>, world_tangent: vec4<f32>, rotation: f32) -> AnisoFrame {
    var t: vec3<f32>;
    let t_len2 = dot(world_tangent.xyz, world_tangent.xyz);
    if (t_len2 > 1e-8) {
        t = normalize(world_tangent.xyz - n * dot(world_tangent.xyz, n));
    } else {
        let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.z) > 0.999);
        t = normalize(cross(axis, n));
    }
    var b = cross(n, t);
    if (t_len2 > 1e-8 && abs(world_tangent.w) > 0.001) {
        b = b * world_tangent.w;
    }
    b = normalize(b);
    if (abs(rotation) > 0.0001) {
        let c = cos(rotation);
        let s = sin(rotation);
        let rotated = normalize(c * t + s * b);
        b = normalize(cross(n, rotated));
        t = rotated;
    }
    return AnisoFrame(t, b);
}

/// Anisotropic GGX NDF (Burley / KHR_materials_anisotropy).
fn distribution_ggx_aniso(
    t: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
    h: vec3<f32>,
    at: f32,
    ab: f32,
) -> f32 {
    let to_h = dot(t, h);
    let bo_h = dot(b, h);
    let no_h = max(dot(n, h), 0.0001);
    let a2 = at * ab;
    let d = vec3<f32>(ab * to_h, at * bo_h, a2 * no_h);
    let d2 = max(dot(d, d), 1e-7);
    let inv = a2 / d2;
    return a2 * inv * inv / PI;
}

/// Height-correlated Smith visibility; includes the 1/(4 nDotV nDotL) factor.
fn visibility_ggx_aniso(
    t: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    at: f32,
    ab: f32,
    n_dot_v: f32,
    n_dot_l: f32,
) -> f32 {
    let lambda_v = n_dot_l * length(vec3<f32>(at * dot(t, v), ab * dot(b, v), n_dot_v));
    let lambda_l = n_dot_v * length(vec3<f32>(at * dot(t, l), ab * dot(b, l), n_dot_l));
    return 0.5 / max(lambda_v + lambda_l, 0.001);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn eval_iridescence(eta2: f32, n_dot_v: f32, thickness_nm: f32, base_f0: vec3<f32>) -> vec3<f32> {
    let eta = max(eta2, 1.0001);
    let sin_t2 = (1.0 - n_dot_v * n_dot_v) / (eta * eta);
    if (sin_t2 >= 1.0) {
        return vec3<f32>(1.0);
    }
    let cos_t2 = sqrt(1.0 - sin_t2);
    let opd = 2.0 * eta * thickness_nm * cos_t2;
    let lambda = vec3<f32>(650.0, 510.0, 475.0);
    let interf = 0.5 + 0.5 * cos(2.0 * PI * opd / lambda);
    let f_film = pow((eta - 1.0) / (eta + 1.0), 2.0);
    let f_schlick = f_film + (1.0 - f_film) * pow(1.0 - n_dot_v, 5.0);
    return mix(base_f0, vec3<f32>(f_schlick) * interf + base_f0 * (1.0 - interf), 1.0);
}

// Select CSM cascade with blend zone for smooth transitions.
const CSM_BLEND_ZONE: f32 = 0.08; // ±8% blend zone around each split

struct CascadeSelection {
    cascade_a: u32,
    cascade_b: u32,
    blend: f32, // 0.0 = fully cascade_a, 1.0 = fully cascade_b
}

fn select_cascade_blended(view_depth: f32) -> CascadeSelection {
    for (var i = 0u; i < CSM_CASCADE_COUNT - 1u; i = i + 1u) {
        let split = g.csm_split_depths[i];
        if split <= 0.0 { continue; }
        let blend_half = split * CSM_BLEND_ZONE;
        let blend_start = split - blend_half;
        let blend_end = split + blend_half;
        if view_depth < blend_start {
            return CascadeSelection(i, i, 0.0);
        }
        if view_depth < blend_end {
            let t = clamp((view_depth - blend_start) / (blend_end - blend_start), 0.0, 1.0);
            return CascadeSelection(i, i + 1u, t);
        }
    }
    return CascadeSelection(CSM_CASCADE_COUNT - 1u, CSM_CASCADE_COUNT - 1u, 0.0);
}

fn shadow_factor_csm_blended(world_pos: vec3<f32>, world_normal: vec3<f32>, light_dir: vec3<f32>, sel: CascadeSelection) -> f32 {
    let s0 = shadow_factor_csm(world_pos, world_normal, light_dir, sel.cascade_a);
    if sel.blend <= 0.001 {
        return s0;
    }
    let s1 = shadow_factor_csm(world_pos, world_normal, light_dir, sel.cascade_b);
    return mix(s0, s1, sel.blend);
}

fn shadow_factor_omni(world_pos: vec3<f32>, light_pos: vec3<f32>, far_plane: f32, cube_idx: u32) -> f32 {
    let to_light = world_pos - light_pos;
    let dist = length(to_light);
    let light_dir = to_light / max(dist, 1e-6);
    let z_ref = dist / far_plane;
    let bias = 0.002;
    let z_biased = z_ref - bias;
    return textureSampleCompare(t_omni_shadow, s_omni_shadow, -light_dir, cube_idx, z_biased);
}

fn shadow_factor_csm(world_pos: vec3<f32>, world_normal: vec3<f32>, light_dir: vec3<f32>, cascade_idx: u32) -> f32 {
    if (g.shadow_params.w < 0.5) {
        return 1.0;
    }
    let light_clip = g.csm_view_proj[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / max(light_clip.w, 1e-6);
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    // light_dir is the ray travel direction (from light toward the scene).
    // Slope-scale bias needs NdotL (toward the light), so invert before the dot.
    let n_dot_l = max(dot(normalize(world_normal), normalize(-light_dir)), 0.0);
    let bias = g.shadow_params.y + (1.0 - n_dot_l) * 0.002;
    let z_ref = ndc.z - bias;
    let hw = i32(floor(g.shadow_params.z + 0.5));
    var inv = g.shadow_params.x;
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

/// L2 SH irradiance (three.js `shGetIrradianceAt`) times Lambert `1/PI`.
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let x = n.x;
    let y = n.y;
    let z = n.z;
    var r = g.sh_l2[0].xyz * 0.886227;
    r += g.sh_l2[1].xyz * (2.0 * 0.511664 * y);
    r += g.sh_l2[2].xyz * (2.0 * 0.511664 * z);
    r += g.sh_l2[3].xyz * (2.0 * 0.511664 * x);
    r += g.sh_l2[4].xyz * (2.0 * 0.429043 * x * y);
    r += g.sh_l2[5].xyz * (2.0 * 0.429043 * y * z);
    r += g.sh_l2[6].xyz * (0.743125 * z * z - 0.247708);
    r += g.sh_l2[7].xyz * (2.0 * 0.429043 * x * z);
    r += g.sh_l2[8].xyz * (0.429043 * (x * x - y * y));
    return r * g.sh_intensity.x / PI;
}

/// Core PBR shading logic. Shared by `fs_main` and `fs_main_wboit`.
/// Returns (color, alpha) as a vec4.
fn pbr_shade(in: VertexOutput) -> vec4<f32> {
    let clip_count = i32(u.clip_count.x);
    for (var i = 0; i < clip_count; i = i + 1) {
        let plane = u.clip_planes[i];
        let dist = dot(plane.xyz, in.world_pos) + plane.w;
        if (dist < 0.0) {
            discard;
        }
    }

    let v = normalize(u.camera_pos.xyz - in.world_pos);

    // Two-sided lighting: flip geometry normal when facing away from camera.
    let N_raw = normalize(in.world_normal);
    let front_facing = dot(N_raw, v) >= 0.0;
    let N = select(-N_raw, N_raw, front_facing);
    var n = N;
#ifdef HAS_NORMAL_MAP
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
#endif
    let n_dot_v = max(dot(n, v), 0.0001);
    if (u.pbr_alpha_flags.w > 199.0) {
        let d = saturate(length(in.view_pos.xyz) / 40.0);
        return vec4<f32>(vec3<f32>(d), 1.0);
    }
    if (u.pbr_alpha_flags.w > 99.0) {
        return vec4<f32>(n * 0.5 + 0.5, 1.0);
    }

#ifdef HAS_ALBEDO_TEX
    let albedo_sample = textureSample(t_albedo, s_mat, in.uv).rgba;
    var albedo = albedo_sample.rgb * u.pbr_base_color.xyz;
    let alpha_sample = albedo_sample.a * u.pbr_alpha_flags.y;
#else
    var albedo = u.pbr_base_color.xyz;
    let alpha_sample = u.pbr_alpha_flags.y;
#endif

    // Alpha mode handling
    let alpha_mode = i32(u.pbr_alpha_flags.x + 0.5);
    if (alpha_mode == 1) {
        // Mask mode: discard fragments below cutoff
        if (alpha_sample < u.pbr_emissive_alpha.w) {
            discard;
        }
    }

    // Sample ORM textures (Occlusion/Roughness/Metallic)
#ifdef HAS_MR_TEX
    let mr_sample = textureSample(t_metallic_roughness, s_mat, in.uv);
    // glTF convention: B=metallic, G=roughness
    let metallic = clamp(mr_sample.b * u.pbr_metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(mr_sample.g * u.pbr_metallic_roughness.y, 0.04, 1.0);
#else
    let metallic = clamp(u.pbr_metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(u.pbr_metallic_roughness.y, 0.04, 1.0);
#endif
    let anisotropy = clamp(u.pbr_metallic_roughness.z, 0.0, 1.0);
    let use_aniso = anisotropy > 0.001;
    var aniso_t = vec3<f32>(1.0, 0.0, 0.0);
    var aniso_b = vec3<f32>(0.0, 1.0, 0.0);
    if (use_aniso) {
        let frame = anisotropy_frame(n, in.world_tangent, u.pbr_metallic_roughness.w);
        aniso_t = frame.t;
        aniso_b = frame.b;
    }
    let alpha_r = roughness * roughness;
    let at = mix(alpha_r, 1.0, anisotropy * anisotropy);
    let ab = alpha_r * alpha_r / max(at, 0.001);
    // KHR_materials_specular: tint dielectric F0 by specular_color * specular_factor
    let dielectric_f0 = u.pbr_specular.rgb * u.pbr_specular.a * 0.08;
    var f0 = mix(dielectric_f0, albedo, metallic);
    let iri_factor = u.pbr_clearcoat.z;
    if (iri_factor > 0.001) {
        let iri_ior = select(1.3, u.pbr_clearcoat.w, u.pbr_clearcoat.w > 1.001);
        let tmin = u.pbr_transmission.z;
        let tmax = select(400.0, u.pbr_transmission.w, u.pbr_transmission.w > 1.0);
        let thick = mix(tmin, tmax, 1.0 - n_dot_v);
        f0 = mix(f0, eval_iridescence(iri_ior, n_dot_v, thick, f0), iri_factor);
    }

    // Occlusion
#ifdef HAS_OCCLUSION_TEX
    let ao = textureSample(t_occlusion, s_mat, in.uv).r;
#else
    let ao = 1.0;
#endif

    // Distance to camera (view-space units) for CSM cascade selection.
    let view_depth = length(in.view_pos.xyz);

    var lo = vec3<f32>(0.0);
    let light_count = min(u32(g.light_count.x), MAX_LIGHTS);
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light_type = i32(g.light_types[i].x + 0.5);
        if (light_type == 4) {
            let up_sq = dot(g.light_dirs[i].xyz, g.light_dirs[i].xyz);
            if (up_sq < 1e-8) { continue; }
            let up = g.light_dirs[i].xyz * inverseSqrt(up_sq);
            let w = clamp(dot(n, up) * 0.5 + 0.5, 0.0, 1.0);
            let hemi = mix(g.light_positions[i].xyz, g.light_colors[i].xyz, w);
            lo += (albedo * (1.0 - metallic) / PI) * hemi;
            continue;
        }
        let raw_dir = normalize(g.light_dirs[i].xyz);
        let point_to_light = g.light_positions[i].xyz - in.world_pos;
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
            let cos_cutoff = g.spot_params[i].x;
            let drop_off = g.spot_params[i].y;
            let spot_cos = dot(normalize(-raw_dir), light_dir);
            if (spot_cos < cos_cutoff) {
                continue;
            }
            let spot_factor = pow(spot_cos, max(drop_off, 0.0));
            intensity_scale = attenuation * spot_factor;
        }
#ifdef HAS_SHADOW
        var sh = 1.0;
        if (light_type == 0) {
            let cascade_sel = select_cascade_blended(view_depth);
            sh = shadow_factor_csm_blended(in.world_pos, in.world_normal, -light_dir, cascade_sel);
            sh = max(sh, 0.12);
        } else if (light_type == 1) {
            let slot = i32(g.light_types[i].y + 0.5);
            if (slot > 0) {
                let far_plane = max(g.light_positions[i].w, 1.0);
                sh = shadow_factor_omni(in.world_pos, g.light_positions[i].xyz, far_plane, u32(slot - 1));
                sh = max(sh, 0.2);
            }
        }
#else
        let sh = 1.0;
#endif
        let light_color = g.light_colors[i].xyz * intensity_scale;
        let l = normalize(light_dir);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_h = max(dot(n, h), 0.0);
        let h_dot_v = max(dot(h, v), 0.0);

        var spec: vec3<f32>;
        if (use_aniso) {
            let ndf = distribution_ggx_aniso(aniso_t, aniso_b, n, h, at, ab);
            let vis = visibility_ggx_aniso(aniso_t, aniso_b, n, v, l, at, ab, n_dot_v, n_dot_l);
            spec = (ndf * vis) * fresnel_schlick(h_dot_v, f0);
        } else {
            let ndf = distribution_ggx(n_dot_h, roughness);
            let geo = geometry_smith(n_dot_v, n_dot_l, roughness);
            spec = (ndf * geo / max(4.0 * n_dot_v * n_dot_l, 0.001)) * fresnel_schlick(h_dot_v, f0);
        }
        let kd = (1.0 - fresnel_schlick(n_dot_l, f0)) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;
        lo += (diffuse + spec) * light_color * n_dot_l * sh;
    }

    // Ambient contribution from scene ambient_color uniform (always applied).
    // This guarantees surfaces facing away from all lights remain visible,
    // especially for complex concave models (engines, interiors).
    let scene_ambient = u.ambient_color.xyz * albedo * (1.0 - metallic);

#ifdef HAS_IBL
    var r = reflect(-v, n);
    if (use_aniso) {
        let aniso_tangent = cross(aniso_b, v);
        let aniso_normal = cross(aniso_tangent, aniso_b);
        if (dot(aniso_tangent, aniso_tangent) > 1e-8 && dot(aniso_normal, aniso_normal) > 1e-8) {
            let bent = normalize(mix(n, normalize(aniso_normal), anisotropy));
            r = reflect(-v, bent);
        }
    }
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    let ibl_kd = (1.0 - f) * (1.0 - metallic);
    let diffuse_ibl = (g.ibl_diffuse.xyz + sh_irradiance(n)) * albedo * ibl_kd;

    let prefiltered_color = sample_prefiltered_envmap(r, roughness);
    let env_brdf = textureSample(t_brdf_lut, s_ibl, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered_color * (f * env_brdf.x + env_brdf.y);

    var color = lo + diffuse_ibl + specular_ibl + scene_ambient;
#else
    var color = lo + scene_ambient + sh_irradiance(n) * albedo * (1.0 - metallic);
#endif

    // Emissive contribution
#ifdef HAS_EMISSIVE_TEX
    let emissive = textureSample(t_emissive, s_mat, in.uv).rgb * u.pbr_emissive_alpha.xyz;
#else
    let emissive = u.pbr_emissive_alpha.xyz;
#endif
    color = color + emissive;

    // Apply ambient occlusion
    color = color * ao;

#ifdef HAS_SHEEN
    // Sheen layer: fabric/velvet microfiber BRDF (KHR_materials_sheen).
    let sheen_color = u.pbr_sheen.xyz;
    let sheen_roughness = max(u.pbr_sheen.w, 0.01);
    if (any(sheen_color > vec3<f32>(0.001))) {
        let inv_r = 1.0 / max(sheen_roughness * sheen_roughness, 0.001);
        let sheen_brdf = sheen_color * inv_r * (2.0 + inv_r) / (2.0 * PI * 4.0);
        color = color + sheen_brdf * n_dot_v * 0.25;
    }
#endif

#ifdef HAS_CLEARCOAT
    // Clearcoat layer: second specular lobe with fixed IOR=1.5 (F0=0.04)
    // Energy-conserving per KHR_materials_clearcoat extension
    let cc_factor = u.pbr_clearcoat.x;
    let cc_roughness = clamp(u.pbr_clearcoat.y, 0.01, 1.0);
    if (cc_factor > 0.01) {
        let cc_n = N;
        let cc_n_dot_v = max(dot(cc_n, v), 0.0001);
        let cc_f0 = 0.04;  // Fixed index ~1.5
        // Two-layer Fresnel: attenuate base specular by clearcoat absorption
        let cc_fresnel = fresnel_schlick(cc_n_dot_v, vec3<f32>(cc_f0));
        // Base layer loses energy to clearcoat
        color = color * (1.0 - cc_fresnel * cc_factor);

        // Recompute specular lighting for clearcoat layer
        var cc_lo = vec3<f32>(0.0);
        for (var j = 0u; j < light_count; j = j + 1u) {
            let lt = i32(g.light_types[j].x + 0.5);
            let raw_ldir = normalize(g.light_dirs[j].xyz);
            let ptl = g.light_positions[j].xyz - in.world_pos;
            let dst = max(length(ptl), 0.0001);
            let tl = ptl / dst;
            let att = 1.0 / (1.0 + 0.09 * dst + 0.032 * dst * dst);
            var ldir = -raw_ldir;
            var l_scale = 1.0;
            if (lt == 1) { ldir = tl; l_scale = att; }
            else if (lt == 2) {
                ldir = tl;
                let cos_cut = g.spot_params[j].x;
                if (dot(normalize(-raw_ldir), ldir) < cos_cut) { continue; }
                l_scale = att * pow(dot(normalize(-raw_ldir), ldir), max(g.spot_params[j].y, 0.0));
            }
            let cl = normalize(ldir);
            let ch = normalize(v + cl);
            let cl_n_dot_l = max(dot(cc_n, cl), 0.0);
            let cl_n_dot_h = max(dot(cc_n, ch), 0.0);
            let cl_h_dot_v = max(dot(ch, v), 0.0);
            let cl_ndf = distribution_ggx(cl_n_dot_h, cc_roughness);
            let cl_g = geometry_smith(cc_n_dot_v, cl_n_dot_l, cc_roughness);
            let cl_spec = (cl_ndf * cl_g / max(4.0 * cc_n_dot_v * cl_n_dot_l, 0.001))
                * fresnel_schlick(cl_h_dot_v, vec3<f32>(cc_f0));
            cc_lo += cl_spec * g.light_colors[j].xyz * l_scale * cl_n_dot_l;
        }
        // IBL for clearcoat layer (simplified: sample envmap at fixed roughness)
        let cc_r = reflect(-v, cc_n);
        let cc_env = textureSampleLevel(t_envmap, s_ibl, direction_to_uv(cc_r), cc_roughness * 2.0).rgb * 0.5;
        color = color + (cc_lo + cc_env) * cc_factor;
    }
#endif

#ifdef HAS_TRANSMISSION
    // Screen-space-free refraction: sample IBL along the refracted view ray.
    let trans = clamp(u.pbr_transmission.x, 0.0, 1.0);
    let ior = max(u.pbr_transmission.y, 1.0);
    if (trans > 0.001) {
        let eta = 1.0 / ior;
        let refr_dir = refract(-v, n, eta);
        var trans_color = albedo;
        if (dot(refr_dir, refr_dir) > 0.001) {
#ifdef HAS_IBL
            trans_color = textureSampleLevel(t_envmap, s_ibl, direction_to_uv(refr_dir), roughness * 4.0).rgb;
#endif
        }
        color = mix(color, trans_color * albedo, trans);
    }
#endif

    let final_alpha = select(alpha_sample, 1.0, alpha_mode == 0);
    let toon_steps = u.pbr_alpha_flags.w;
    if (toon_steps > 1.5 && toon_steps < 99.0) {
        let lum = max(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4);
        let q = floor(lum * toon_steps + 0.5) / toon_steps;
        color = color * (q / lum);
    }
#ifdef HAS_TRANSMISSION
    let trans_a = clamp(u.pbr_transmission.x, 0.0, 1.0);
    let out_alpha = mix(final_alpha, max(0.12, 1.0 - trans_a * 0.85), step(0.001, trans_a));
    return vec4<f32>(color, out_alpha);
#else
    return vec4<f32>(color, final_alpha);
#endif
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return pbr_shade(in);
}

// WBOIT (Weighted Blended OIT) output: two render targets.
// RT0 (accum):     premultiplied color * alpha * weight, alpha * weight  (additive blend)
// RT1 (revealage): 1 - alpha  (multiplicative blend: dst *= src)
// Weight follows McGuire & Bavoil 2012: w(z) = clamp(alpha / (1e-5 + z^3), 1e-3, 300)
struct WboitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: vec4<f32>,
}

@fragment
fn fs_main_wboit(in: VertexOutput) -> WboitOutput {
    let result = pbr_shade(in);
    let color = result.rgb;
    let alpha = result.a;

    // McGuire 2013: closer fragments get higher weight. clip.w after raster is 1/view_z.
    let z = 1.0 / max(in.clip_position.w, 1e-5);
    let luma_a = max(max(color.r, color.g), color.b) * alpha;
    let weight = max(min(1.0, luma_a), alpha)
        * clamp(0.03 / (1e-5 + pow(z / 200.0, 4.0)), 1e-2, 3e3);

    var out: WboitOutput;
    out.accum = vec4<f32>(color * alpha * weight, alpha * weight);
    out.revealage = vec4<f32>(1.0 - alpha, 1.0 - alpha, 1.0 - alpha, 1.0 - alpha);
    return out;
}
