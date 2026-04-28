struct CullUniforms {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    meshlet_count: u32,
    lod_stride: u32,
    meshlet_phase: u32,
    meshlet_stride_spatial: u32,
    hzb_dims: vec4<u32>,
    hzb_mip_max: u32,
    hzb_enabled: u32,
    depth_reversed_z: u32,
    orthographic_projection: u32,
};

struct GpuMeshlet {
    index_offset: u32,
    index_count: u32,
    vertex_offset: u32,
    vertex_count: u32,
};

struct GpuMeshletBounds {
    center: vec3<f32>,
    radius: f32,
    cone_axis_cutoff: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: CullUniforms;
@group(0) @binding(1) var<storage, read> meshlets: array<GpuMeshlet>;
@group(0) @binding(2) var<storage, read> bounds: array<GpuMeshletBounds>;
@group(0) @binding(3) var<storage, read_write> visible: array<atomic<u32>>;
@group(0) @binding(4) var hzb_max_tex: texture_2d<f32>;
@group(0) @binding(5) var hzb_min_tex: texture_2d<f32>;

// WebGPU NDC: x,y in [-1,1] with +y up; UV [0,1] with v=0 at top texel row.
// Matches depth attachment / HZB mip0 addressing used by textureLoad (origin top-left).
fn ndc_to_uv(ndc: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn spatial_stride_bucket(center: vec3<f32>) -> u32 {
    let x = bitcast<u32>(center.x);
    let y = bitcast<u32>(center.y);
    let z = bitcast<u32>(center.z);
    return x ^ (y << 5u) ^ (z << 10u);
}

@compute @workgroup_size(64)
fn cull_meshlets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= u.meshlet_count {
        return;
    }

    let b = bounds[idx];

    let s = max(u.lod_stride, 1u);
    if s > 1u {
        let phase = u.meshlet_phase % s;
        if u.meshlet_stride_spatial != 0u {
            if (spatial_stride_bucket(b.center) % s) != phase {
                return;
            }
        } else {
            if (idx % s) != phase {
                return;
            }
        }
    }

    let clip = u.view_proj * vec4<f32>(b.center, 1.0);
    var visible_flag = true;

    if clip.w > 0.0 {
        let r = b.radius;
        if (clip.x < -clip.w - r) || (clip.x > clip.w + r) ||
           (clip.y < -clip.w - r) || (clip.y > clip.w + r) {
            visible_flag = false;
        }
        // WebGPU clip depth: visible hull uses 0 <= clip.z <= clip.w (ZO depth after divide is [0,1]).
        if (clip.z < -r) || (clip.z > clip.w + r) {
            visible_flag = false;
        }
    } else {
        visible_flag = false;
    }

    if visible_flag {
        let axis = b.cone_axis_cutoff.xyz;
        let cutoff = b.cone_axis_cutoff.w;
        if u.orthographic_projection != 0u {
            let view_dir = normalize(u.view_pos.xyz - b.center);
            if dot(axis, view_dir) >= cutoff {
                visible_flag = false;
            }
        } else {
            let to_c = b.center - u.view_pos.xyz;
            let dtc = length(to_c);
            if dtc > 1e-6 {
                if dot(to_c, axis) >= cutoff * dtc + b.radius {
                    visible_flag = false;
                }
            }
        }
    }

    // max pyramid: reverse-Z (larger depth = closer). min pyramid: forward-Z (smaller = closer).
    if visible_flag && (u.hzb_enabled != 0u) {
        let c = b.center;
        let r = b.radius;
        var u_min = 1.0e10;
        var u_max = -1.0e10;
        var v_min = 1.0e10;
        var v_max = -1.0e10;
        var z_max = -1.0e10;
        var z_min = 1.0e10;
        var had_proj = false;

        for (var i = 0u; i < 8u; i++) {
            let ox = select(-r, r, (i & 1u) != 0u);
            let oy = select(-r, r, (i & 2u) != 0u);
            let oz = select(-r, r, (i & 4u) != 0u);
            let clip_c = u.view_proj * vec4<f32>(c + vec3<f32>(ox, oy, oz), 1.0);
            if clip_c.w > 1e-6 {
                let ndc = clip_c.xyz / clip_c.w;
                let uv = ndc_to_uv(ndc);
                u_min = min(u_min, uv.x);
                u_max = max(u_max, uv.x);
                v_min = min(v_min, uv.y);
                v_max = max(v_max, uv.y);
                z_max = max(z_max, ndc.z);
                z_min = min(z_min, ndc.z);
                had_proj = true;
            }
        }

        let to_c = u.view_pos.xyz - c;
        let dtc = length(to_c);
        if dtc > 1e-6 {
            let nearest = c + normalize(to_c) * r;
            let clip_n = u.view_proj * vec4<f32>(nearest, 1.0);
            if clip_n.w > 1e-6 {
                let ndc_n = clip_n.xyz / clip_n.w;
                let uv_n = ndc_to_uv(ndc_n);
                u_min = min(u_min, uv_n.x);
                u_max = max(u_max, uv_n.x);
                v_min = min(v_min, uv_n.y);
                v_max = max(v_max, uv_n.y);
                z_max = max(z_max, ndc_n.z);
                z_min = min(z_min, ndc_n.z);
                had_proj = true;
            }
        }

        if had_proj {
            let fw = f32(u.hzb_dims.x);
            let fh = f32(u.hzb_dims.y);
            let max_x = i32(max(u.hzb_dims.x, 1u)) - 1;
            let max_y = i32(max(u.hzb_dims.y, 1u)) - 1;
            let xi0 = i32(clamp(floor(u_min * fw), 0.0, f32(max_x)));
            let xi1 = i32(clamp(ceil(u_max * fw) - 1.0, f32(xi0), f32(max_x)));
            let yi0 = i32(clamp(floor(v_min * fh), 0.0, f32(max_y)));
            let yi1 = i32(clamp(ceil(v_max * fh) - 1.0, f32(yi0), f32(max_y)));
            let rw = xi1 - xi0 + 1;
            let rh = yi1 - yi0 + 1;
            let mip_level = i32(clamp(
                i32(floor(log2(f32(max(max(rw, rh), 1))))),
                0,
                i32(u.hzb_mip_max),
            ));
            let mip = u32(mip_level);

            let dw = max(i32(u.hzb_dims.x >> mip), 1);
            let dh = max(i32(u.hzb_dims.y >> mip), 1);
            let mx0 = i32(clamp(xi0 >> mip, 0, dw - 1));
            let mx1 = i32(clamp(xi1 >> mip, mx0, dw - 1));
            let my0 = i32(clamp(yi0 >> mip, 0, dh - 1));
            let my1 = i32(clamp(yi1 >> mip, my0, dh - 1));

            let mxc = (mx0 + mx1) / 2;
            let myc = (my0 + my1) / 2;

            if u.depth_reversed_z != 0u {
                let z00 = textureLoad(hzb_max_tex, vec2<i32>(mx0, my0), mip_level).x;
                let z10 = textureLoad(hzb_max_tex, vec2<i32>(mx1, my0), mip_level).x;
                let z01 = textureLoad(hzb_max_tex, vec2<i32>(mx0, my1), mip_level).x;
                let z11 = textureLoad(hzb_max_tex, vec2<i32>(mx1, my1), mip_level).x;
                var hzb_cover = max(max(z00, z10), max(z01, z11));
                hzb_cover = max(hzb_cover, textureLoad(hzb_max_tex, vec2<i32>(mxc, myc), mip_level).x);
                let ref_z = z_max;
                if ref_z < hzb_cover - 0.0001 {
                    visible_flag = false;
                }
            } else {
                let z00 = textureLoad(hzb_min_tex, vec2<i32>(mx0, my0), mip_level).x;
                let z10 = textureLoad(hzb_min_tex, vec2<i32>(mx1, my0), mip_level).x;
                let z01 = textureLoad(hzb_min_tex, vec2<i32>(mx0, my1), mip_level).x;
                let z11 = textureLoad(hzb_min_tex, vec2<i32>(mx1, my1), mip_level).x;
                var hzb_cover = min(min(z00, z10), min(z01, z11));
                hzb_cover = min(hzb_cover, textureLoad(hzb_min_tex, vec2<i32>(mxc, myc), mip_level).x);
                let ref_z = z_min;
                if ref_z > hzb_cover + 0.0001 {
                    visible_flag = false;
                }
            }
        }
    }

    if visible_flag {
        let slot = atomicAdd(&visible[0], 1u);
        if (slot + 1u) < arrayLength(&visible) {
            atomicStore(&visible[slot + 1u], idx);
        }
    }
}
