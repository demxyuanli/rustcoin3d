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
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = (u.model * vec4<f32>(in.position, 1.0)).xyz;
    out.world_normal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clip planes
    let clip_count = i32(u.clip_count.x);
    for (var i = 0; i < clip_count; i = i + 1) {
        let plane = u.clip_planes[i];
        let dist = dot(plane.xyz, in.world_pos) + plane.w;
        if (dist < 0.0) {
            discard;
        }
    }

    let n = normalize(in.world_normal);
    let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
    let light_count = min(i32(u.light_count.x), 4);
    let shininess = max(u.shininess.x, 1.0);
    let ambient_strength = 0.15;

    // Fast path: single directional light (most common case)
    if (light_count == 1 && i32(u.light_types[0].x + 0.5) == 0) {
        let light_dir = -normalize(u.light_dirs[0].xyz);
        let light_color = u.light_colors[0].xyz;
        let diff = max(dot(n, light_dir), 0.0);
        let half_dir = normalize(light_dir + view_dir);
        let spec = pow(max(dot(n, half_dir), 0.0), shininess);
        let ambient = ambient_strength * light_color * u.ambient_color.xyz;
        let diffuse = diff * light_color * u.diffuse_color.xyz;
        let specular = spec * light_color * u.specular_color.xyz;
        return vec4<f32>(ambient + diffuse + specular, 1.0);
    }

    // General path: multiple / typed lights
    var result = vec3<f32>(0.0, 0.0, 0.0);
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
        let light_color = u.light_colors[i].xyz;

        let ambient = ambient_strength * light_color * u.ambient_color.xyz * intensity_scale;
        let diff = max(dot(n, light_dir), 0.0);
        let diffuse = diff * light_color * u.diffuse_color.xyz * intensity_scale;
        let half_dir = normalize(light_dir + view_dir);
        let spec = pow(max(dot(n, half_dir), 0.0), shininess);
        let specular = spec * light_color * u.specular_color.xyz * intensity_scale;
        result += ambient + diffuse + specular;
    }
    return vec4<f32>(result, 1.0);
}
