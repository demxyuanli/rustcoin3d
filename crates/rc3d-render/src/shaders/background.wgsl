// Fullscreen background pass — gradient, image, or solid color.

struct BgParams {
    mode: u32,             // 0=gradient, 1=image, 2=solid
    top_color: vec4<f32>,  // gradient top / solid color
    bot_color: vec4<f32>,  // gradient bottom
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> bg: BgParams;
@group(0) @binding(1) var t_image: texture_2d<f32>;
@group(0) @binding(2) var s_image: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    var o: VsOut;
    o.clip_pos = vec4<f32>(positions[vi], 1.0, 1.0); // z=1.0 = near plane (reverse Z)
    o.uv = vec2<f32>((positions[vi].x + 1.0) * 0.5, (positions[vi].y + 1.0) * 0.5);
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    if bg.mode == 0u {
        // Gradient: lerp top to bottom using uv.y
        return mix(bg.bot_color, bg.top_color, vec4<f32>(i.uv.y));
    } else if bg.mode == 1u {
        // Image: sample texture
        let tex = textureSample(t_image, s_image, i.uv);
        return tex;
    }
    // Solid color (mode 2 or default): top_color
    return bg.top_color;
}
