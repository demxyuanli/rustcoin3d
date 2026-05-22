// Fullscreen background pass — gradients, 2D image (stretch/tile/original), solid.

struct BgParams {
    mode: u32,             // 0=vertical, 1=horizontal, 2=center, 3=diagonal, 4=image, 5=solid
    image_fit: u32,        // 0=stretch, 1=tile, 2=original
    _pad0: vec2<u32>,
    top_color: vec4<f32>,
    bot_color: vec4<f32>,
    image_size: vec2<f32>,
    screen_size: vec2<f32>,
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

fn sample_image(uv: vec2<f32>) -> vec4<f32> {
    if bg.image_fit == 0u {
        // Stretch: direct UV mapping
        return textureSample(t_image, s_image, uv);
    } else if bg.image_fit == 1u {
        // Tile: repeat UV by image-to-screen ratio
        let tile = bg.screen_size / bg.image_size;
        return textureSample(t_image, s_image, fract(uv * tile));
    }
    // Original size, centered; bot_color fills borders
    let scale = bg.image_size / bg.screen_size;
    let img_uv = (uv - 0.5) / scale + 0.5;
    if img_uv.x < 0.0 || img_uv.x > 1.0 || img_uv.y < 0.0 || img_uv.y > 1.0 {
        return bg.bot_color;
    }
    return textureSample(t_image, s_image, img_uv);
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    if bg.mode == 0u {
        // Vertical gradient: top → bottom
        return mix(bg.bot_color, bg.top_color, vec4<f32>(i.uv.y));
    } else if bg.mode == 1u {
        // Horizontal gradient: left → right
        return mix(bg.bot_color, bg.top_color, vec4<f32>(i.uv.x));
    } else if bg.mode == 2u {
        // Center (radial) gradient: center → edges
        let dist = distance(i.uv, vec2<f32>(0.5, 0.5)) * 1.414;
        return mix(bg.top_color, bg.bot_color, vec4<f32>(saturate(dist)));
    } else if bg.mode == 3u {
        // Diagonal gradient: bottom-left → top-right
        let t = (i.uv.x + i.uv.y) * 0.5;
        return mix(bg.bot_color, bg.top_color, vec4<f32>(t));
    } else if bg.mode == 4u {
        // Image with fit mode
        return sample_image(i.uv);
    } else if bg.mode == 6u {
        // Sky-ground: sky above horizon (y>0.5), ground below
        // uv.y: 0=bottom, 1=top
        let horizon = 0.5;
        let sky_t = smoothstep(horizon - 0.1, horizon, i.uv.y);
        // Sky (top) fades toward horizon, ground (bottom) fades toward horizon
        let sky = mix(bg.top_color, bg.bot_color, vec4<f32>(1.0 - i.uv.y));
        let ground = mix(bg.bot_color, bg.top_color, vec4<f32>(i.uv.y * 2.0));
        return mix(ground, sky, vec4<f32>(sky_t));
    }
    // Solid (mode 5 or default): top_color
    return bg.top_color;
}
