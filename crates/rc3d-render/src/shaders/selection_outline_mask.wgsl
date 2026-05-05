struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FlatUniforms;

struct MaskParams {
    reversed_z: f32,
    depth_bias: f32,
    far_ndc_z: f32,
    _pad0: f32,
};

@group(1) @binding(0) var prepass_depth: texture_2d<f32>;
@group(1) @binding(1) var<uniform> mask_params: MaskParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) clip_z: f32,
    @location(1) clip_w: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
    var o: VsOut;
    o.clip_pos = u.mvp * vec4<f32>(in.position, 1.0);
    o.clip_z = o.clip_pos.z;
    o.clip_w = o.clip_pos.w;
    return o;
}

@fragment
fn fs_main(
    @location(0) clip_z: f32,
    @location(1) clip_w: f32,
    @builtin(position) frag_coord: vec4<f32>,
) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(prepass_depth));
    let p = vec2<i32>(i32(frag_coord.x), i32(frag_coord.y));
    let pclamp = vec2<i32>(
        clamp(p.x, 0, i32(dims.x) - 1),
        clamp(p.y, 0, i32(dims.y) - 1),
    );
    let ref_z = textureLoad(prepass_depth, pclamp, 0).x;
    let cur_z = clip_z / clip_w;

    var visible = 1.0;
    if mask_params.reversed_z > 0.5 {
        visible = select(0.0, 1.0, cur_z > ref_z + mask_params.depth_bias);
    } else {
        visible = select(0.0, 1.0, cur_z < ref_z - mask_params.depth_bias);
    }

    let no_occluder = abs(ref_z - mask_params.far_ndc_z) < 1e-5;
    if no_occluder {
        visible = 1.0;
    }

    return vec4<f32>(0.0, visible, 1.0, 1.0);
}
