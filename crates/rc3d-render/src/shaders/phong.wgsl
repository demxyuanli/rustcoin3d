struct SceneUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    object_color: vec4<f32>,
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
    let n = normalize(in.world_normal);
    let light_dir = normalize(-u.light_dir.xyz);
    let view_dir = normalize(u.camera_pos.xyz - in.world_pos);

    // Ambient
    let ambient_strength = 0.15;
    let ambient = ambient_strength * u.light_color.xyz * u.object_color.xyz;

    // Diffuse
    let diff = max(dot(n, light_dir), 0.0);
    let diffuse = diff * u.light_color.xyz * u.object_color.xyz;

    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_dir), 0.0), 32.0);
    let specular = spec * u.light_color.xyz * vec3<f32>(0.5, 0.5, 0.5);

    let result = ambient + diffuse + specular;
    return vec4<f32>(result, 1.0);
}
