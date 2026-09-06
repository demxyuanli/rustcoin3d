struct CompParams {
    op: u32,
    mix_mode: u32,
    _p0: u32,
    _p1: u32,
    fac: f32,
    radius: f32,
    brightness: f32,
    contrast: f32,
    dir: vec2<f32>,
    stop_count: u32,
    _p2: u32,
    stop0: vec4<f32>,
    stop1: vec4<f32>,
    stop2: vec4<f32>,
    stop3: vec4<f32>,
    const_color: vec4<f32>,
    scalars0: vec4<f32>, // [value, value_a, tx, ty]
    scalars1: vec4<f32>, // [rot(rad), scale, gamma, exposure]
    scalars2: vec4<f32>, // [hue, sat, val, invert_fac]
    crop: vec4<f32>,     // (x0, y0, x1, y1) normalized UV
    morph: f32,          // signed pixel distance
}

@group(0) @binding(0) var src_a: texture_2d<f32>;
@group(0) @binding(1) var src_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: CompParams;
@group(0) @binding(4) var dst_hdr: texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sample_uv(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(tex, samp, uv, 0.0);
}

// ---------------------------------------------------------------- mix modes

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let r = c.r;
    let g = c.g;
    let b = c.b;
    let mx = max(r, max(g, b));
    let mn = min(r, min(g, b));
    let d = mx - mn;
    var h = 0.0;
    if d > 1.0e-6 {
        if mx == r {
            h = fract((g - b) / d);
        } else if mx == g {
            h = fract((b - r) / d + 2.0);
        } else {
            h = fract((r - g) / d + 4.0);
        }
        h = h / 6.0;
    }
    let s = select(0.0, d / max(mx, 1.0e-6), mx > 1.0e-6);
    return vec3<f32>(h, s, mx);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let h = fract(c.x);
    let s = clamp(c.y, 0.0, 1.0);
    let v = max(c.z, 0.0);
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let idx = u32(i) % 6u;
    var rgb = vec3<f32>(v, t, p);
    if idx == 1u {
        rgb = vec3<f32>(q, v, p);
    } else if idx == 2u {
        rgb = vec3<f32>(p, v, t);
    } else if idx == 3u {
        rgb = vec3<f32>(p, q, v);
    } else if idx == 4u {
        rgb = vec3<f32>(t, p, v);
    } else if idx == 5u {
        rgb = vec3<f32>(v, p, q);
    }
    return rgb;
}

fn set_hsv(a: vec3<f32>, h: f32, s: f32, v: f32) -> vec3<f32> {
    let ah = rgb_to_hsv(a);
    return hsv_to_rgb(vec3<f32>(fract(ah.x + h), ah.y * s, ah.z * v));
}

// HLS-style hue/sat on channels, matching Blender's legacy hue mix modes.
fn mix_hue(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let bh = rgb_to_hsv(b);
    return set_hsv(a, bh.x, 1.0, 1.0);
}

fn mix_saturation(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ah = rgb_to_hsv(a);
    let bh = rgb_to_hsv(b);
    return set_hsv(a, ah.x, bh.y, 1.0);
}

fn mix_value(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let bh = rgb_to_hsv(b);
    return set_hsv(a, 0.0, 0.0, bh.z);
}

fn mix_color_mode(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let bh = rgb_to_hsv(b);
    return set_hsv(a, bh.x, bh.y, 1.0);
}

fn blend_pair(a: vec3<f32>, b: vec3<f32>, mode: u32) -> vec3<f32> {
    var w = b;
    if mode == 3u {
        // screen: 1 - (1-a)(1-b)
        w = vec3<f32>(1.0) - (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b);
    } else if mode == 4u {
        w = select(vec3<f32>(0.0), a / max(b, vec3<f32>(1.0e-4)), b > vec3<f32>(0.0));
    } else if mode == 5u {
        w = abs(a - b);
    } else if mode == 6u {
        w = min(a, b);
    } else if mode == 7u {
        w = max(a, b);
    } else if mode == 8u {
        // overlay
        let test = step(vec3<f32>(0.5), a);
        let lo = 2.0 * a * b;
        let hi = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b);
        w = mix(lo, hi, test);
    } else if mode == 9u {
        // dodge
        w = select(vec3<f32>(1.0), a / max(vec3<f32>(1.0) - b, vec3<f32>(1.0e-4)), b < vec3<f32>(1.0));
    } else if mode == 10u {
        // burn
        w = select(vec3<f32>(0.0), vec3<f32>(1.0) - (vec3<f32>(1.0) - a) / max(b, vec3<f32>(1.0e-4)), b > vec3<f32>(0.0));
    } else if mode == 11u {
        w = mix_hue(a, b);
    } else if mode == 12u {
        w = mix_saturation(a, b);
    } else if mode == 13u {
        w = mix_value(a, b);
    } else if mode == 14u {
        w = mix_color_mode(a, b);
    } else if mode == 15u {
        w = a - b;
    }
    return w;
}

fn mix_color(a: vec3<f32>, b: vec3<f32>, fac: f32, mode: u32) -> vec3<f32> {
    let f = clamp(fac, 0.0, 1.0);
    return mix(a, blend_pair(a, b, mode), f);
}

fn gaussian_blur(uv: vec2<f32>, dst_dims: vec2<f32>) -> vec4<f32> {
    let dir = params.dir;
    let r = max(params.radius, 0.0);
    if r < 0.01 {
        return sample_uv(src_a, uv);
    }
    let px = dir / dst_dims;
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    let taps = 8;
    for (var i = -taps; i <= taps; i = i + 1) {
        let t = f32(i);
        let w = exp(-0.5 * (t / max(r, 0.5)) * (t / max(r, 0.5)));
        acc += sample_uv(src_a, uv + px * t * r) * w;
        wsum += w;
    }
    return acc / max(wsum, 1.0e-4);
}

fn bright_contrast(c: vec3<f32>) -> vec3<f32> {
    let b = params.brightness;
    let k = params.contrast;
    let mid = vec3<f32>(0.5);
    return clamp((c - mid) * (1.0 + k) + mid + vec3<f32>(b), vec3<f32>(0.0), vec3<f32>(16.0));
}

fn ramp_eval(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let n = max(params.stop_count, 2u);
    var stops: array<vec4<f32>, 4>;
    stops[0] = params.stop0;
    stops[1] = params.stop1;
    stops[2] = params.stop2;
    stops[3] = params.stop3;
    if t <= stops[0].w {
        return stops[0].xyz;
    }
    for (var i = 1u; i < n; i = i + 1u) {
        let a = stops[i - 1u];
        let b = stops[i];
        if t <= b.w {
            let span = max(b.w - a.w, 1.0e-4);
            let u = (t - a.w) / span;
            return mix(a.xyz, b.xyz, u);
        }
    }
    return stops[n - 1u].xyz;
}

fn math_eval(x: f32, y: f32, mode: u32) -> f32 {
    if mode == 1u {
        return x - y;
    }
    if mode == 2u {
        return x * y;
    }
    if mode == 3u {
        return select(x, x / max(y, 1.0e-6), abs(y) > 1.0e-6);
    }
    if mode == 4u {
        return pow(max(x, 0.0), y);
    }
    if mode == 5u {
        return min(x, y);
    }
    if mode == 6u {
        return max(x, y);
    }
    if mode == 7u {
        return select(0.0, 1.0, x < y);
    }
    if mode == 8u {
        return select(0.0, 1.0, x > y);
    }
    if mode == 9u {
        return abs(x);
    }
    if mode == 10u {
        return floor(x);
    }
    if mode == 11u {
        return ceil(x);
    }
    if mode == 12u {
        return sin(x);
    }
    if mode == 13u {
        return cos(x);
    }
    return x + y;
}

fn transform_uv(uv: vec2<f32>) -> vec2<f32> {
    // inverse mapping so dst pixel samples the correct src texel
    let tx = params.scalars0.z;
    let ty = params.scalars0.w;
    let rot = params.scalars1.x;
    let scale = max(params.scalars1.y, 1.0e-4);
    let c = cos(rot);
    let s = sin(rot);
    let centered = uv - vec2<f32>(0.5);
    let sc = centered / vec2<f32>(scale);
    let rot_back = vec2<f32>(c * sc.x + s * sc.y, -s * sc.x + c * sc.y);
    return rot_back + vec2<f32>(tx, ty) + vec2<f32>(0.5);
}

fn morph_sample(uv: vec2<f32>, dims: vec2<f32>) -> vec4<f32> {
    let amt = params.morph;
    if abs(amt) < 0.5 {
        return sample_uv(src_a, uv);
    }
    let step_px = vec2<f32>(amt, 0.0) / dims;
    let n = i32(abs(amt));
    var best = sample_uv(src_a, uv);
    for (var i = 1; i <= n; i = i + 1) {
        let t = f32(i);
        let offs = step_px * t;
        if amt > 0 {
            let a = sample_uv(src_a, uv + offs);
            let b = sample_uv(src_a, uv - offs);
            best = vec4<f32>(max(best.rgb, max(a.rgb, b.rgb)), max(best.a, max(a.a, b.a)));
        } else {
            let a = sample_uv(src_a, uv + offs);
            let b = sample_uv(src_a, uv - offs);
            best = vec4<f32>(min(best.rgb, min(a.rgb, b.rgb)), min(best.a, min(a.a, b.a)));
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn hdr_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_hdr);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if params.op == 0u {
        color = sample_uv(src_a, uv);
    } else if params.op == 1u {
        let a = sample_uv(src_a, uv).xyz;
        let b = sample_uv(src_b, uv).xyz;
        color = vec4<f32>(mix_color(a, b, params.fac, params.mix_mode), 1.0);
    } else if params.op == 2u {
        color = gaussian_blur(uv, vec2<f32>(dims));
    } else if params.op == 3u {
        color = vec4<f32>(bright_contrast(sample_uv(src_a, uv).xyz), 1.0);
    } else if params.op == 4u {
        let src = sample_uv(src_a, uv);
        color = vec4<f32>(ramp_eval(luma(src.xyz)), src.a);
    } else if params.op == 6u {
        color = params.const_color;
    } else if params.op == 7u {
        let v = params.scalars0.x;
        color = vec4<f32>(vec3<f32>(v), 1.0);
    } else if params.op == 8u {
        // Math: luminance of A (or value_a), scalar B
        let a = sample_uv(src_a, uv).x;
        let b = sample_uv(src_b, uv).x;
        let x = select(params.scalars0.y, a, a > 0.0);
        let y = select(params.scalars0.x, b, b > 0.0);
        color = vec4<f32>(vec3<f32>(math_eval(x, y, params.mix_mode)), 1.0);
    } else if params.op == 9u {
        // Alpha Over: b on top of a weighted by fac * b.a
        let a = sample_uv(src_a, uv);
        let b = sample_uv(src_b, uv);
        let w = clamp(params.fac * b.a, 0.0, 1.0);
        let over = mix(a.rgb, b.rgb, w);
        color = vec4<f32>(over, clamp(a.a + b.a * (1.0 - a.a), 0.0, 1.0));
    } else if params.op == 10u {
        let src = sample_uv(src_a, uv);
        color = vec4<f32>(src.rgb * exp2(params.scalars1.w), src.a);
    } else if params.op == 11u {
        let src = sample_uv(src_a, uv);
        color = vec4<f32>(pow(max(src.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / params.scalars1.z)), src.a);
    } else if params.op == 12u {
        let src = sample_uv(src_a, uv);
        color = vec4<f32>(set_hsv(src.rgb, params.scalars2.x - 0.5, params.scalars2.y, params.scalars2.z), src.a);
    } else if params.op == 13u {
        let src = sample_uv(src_a, uv);
        let f = params.scalars2.w;
        let inv = vec3<f32>(1.0) - src.rgb;
        color = vec4<f32>(mix(src.rgb, inv, f), src.a);
    } else if params.op == 14u {
        color = sample_uv(src_a, transform_uv(uv));
    } else if params.op == 15u {
        color = sample_uv(src_a, transform_uv(uv));
    } else if params.op == 16u {
        color = sample_uv(src_a, transform_uv(uv));
    } else if params.op == 17u {
        // Crop: keep rect, black outside
        let c = params.crop;
        let inside = uv.x >= c.x && uv.x <= c.z && uv.y >= c.y && uv.y <= c.w;
        if inside {
            color = sample_uv(src_a, uv);
        } else {
            color = vec4<f32>(0.0);
        }
    } else if params.op == 18u {
        color = morph_sample(uv, vec2<f32>(dims));
    } else {
        color = sample_uv(src_a, uv);
    }
    textureStore(dst_hdr, vec2<i32>(gid.xy), color);
}

@group(0) @binding(0) var prev_src: texture_2d<f32>;
@group(0) @binding(1) var prev_samp: sampler;
@group(0) @binding(2) var dst_ldr: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn preview_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_ldr);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let c = textureSampleLevel(prev_src, prev_samp, uv, 0.0);
    textureStore(dst_ldr, vec2<i32>(gid.xy), vec4<f32>(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
