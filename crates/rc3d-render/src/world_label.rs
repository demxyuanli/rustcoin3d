//! World-space annotation labels: textured quads on the annotation plane.

use glam::{Mat4, Vec3};
use glyphon::{Buffer, FontSystem, Metrics, Shaping, SwashCache, SwashContent};
use wgpu::util::DeviceExt;

use crate::render_passes::pass_markup::projection::{ndc_to_screen, project_point_ndc};
use crate::gpu_resource::GpuUniformPool;
use crate::pipelines::PipelineSet;
use crate::sdf;
use crate::vertex::{FlatUniforms, WorldLabelVertex};
use rc3d_scene::node_data::FontStyle;

/// Label drawn as a 3D quad on the annotation plane (model-local space).
#[derive(Clone, Debug)]
pub struct WorldLabelCommand {
    pub string: String,
    pub model_matrix: Mat4,
    pub at: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    /// Seed world height; draw pass rescales to `screen_height_px`.
    pub height_world: f32,
    /// Target full label height on screen (pixels), from `AnnotationStyle::font_size`.
    pub screen_height_px: f32,
    pub color: [f32; 4],
    /// Empty uses [`FontStyle`] generic family (Coin3D `SoFont::name`).
    pub font_name: String,
    pub font_style: FontStyle,
}

/// Camera-facing basis for world-space text (Text3 / billboards).
pub fn camera_billboard_basis(world_pos: Vec3, camera_pos: Vec3) -> ([f32; 3], [f32; 3]) {
    let mut to_cam = camera_pos - world_pos;
    if to_cam.length_squared() < 1e-12 {
        to_cam = Vec3::Z;
    } else {
        to_cam = to_cam.normalize();
    }
    let up = Vec3::Y;
    let mut tangent = up.cross(to_cam);
    if tangent.length_squared() < 1e-12 {
        tangent = Vec3::X;
    } else {
        tangent = tangent.normalize();
    }
    let bitangent = to_cam.cross(tangent).normalize();
    (tangent.into(), bitangent.into())
}

pub fn normalize3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > 1e-12 {
        let inv = len_sq.sqrt().recip();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

pub fn bitangent_from(normal: [f32; 3], tangent: [f32; 3]) -> [f32; 3] {
    let n = Vec3::from(normal);
    let t = Vec3::from(tangent);
    let b = n.cross(t);
    if b.length_squared() > 1e-12 {
        b.normalize().into()
    } else {
        normalize3([normal[1], -normal[0], 0.0], [0.0, 1.0, 0.0])
    }
}

pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    Vec3::from(a).cross(Vec3::from(b)).into()
}

/// Linear dimension: tangent along the dimension line, bitangent along `offset_dir`.
pub fn linear_dimension_label_basis(
    start: [f32; 3],
    end: [f32; 3],
    offset_dir: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let tangent = normalize3(
        [end[0] - start[0], end[1] - start[1], end[2] - start[2]],
        [1.0, 0.0, 0.0],
    );
    let mut bitangent = normalize3(offset_dir, [0.0, 1.0, 0.0]);
    if Vec3::from(tangent)
        .cross(Vec3::from(bitangent))
        .length_squared()
        < 1e-12
    {
        bitangent = bitangent_from([0.0, 0.0, 1.0], tangent);
    }
    (tangent, bitangent)
}

/// Angular dimension: same plane as the arc (normal = arm1 x arm2).
pub fn angle_dimension_label_basis(
    center: [f32; 3],
    arm1: [f32; 3],
    arm2: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let v1 = [
        arm1[0] - center[0],
        arm1[1] - center[1],
        arm1[2] - center[2],
    ];
    let v2 = [
        arm2[0] - center[0],
        arm2[1] - center[1],
        arm2[2] - center[2],
    ];
    let plane_normal = normalize3(cross3(v1, v2), [0.0, 0.0, 1.0]);
    let tangent = normalize3(
        [arm2[0] - arm1[0], arm2[1] - arm1[1], arm2[2] - arm1[2]],
        [1.0, 0.0, 0.0],
    );
    let bitangent = bitangent_from(plane_normal, tangent);
    (tangent, bitangent)
}

/// Radial/diameter: in-plane bitangent matches arrow geometry (`dir x world Y`).
pub fn extent_label_basis(from: [f32; 3], to: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let tangent = normalize3(
        [to[0] - from[0], to[1] - from[1], to[2] - from[2]],
        [1.0, 0.0, 0.0],
    );
    let perp = Vec3::from(tangent).cross(Vec3::Y);
    let bitangent = if perp.length_squared() > 1e-12 {
        perp.normalize().into()
    } else {
        [0.0, 0.0, 1.0]
    };
    (tangent, bitangent)
}

/// Leader/callout: label offset is in the anchor XY plane (normal +Z).
pub fn leader_label_basis(tangent: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let tangent = normalize3(tangent, [1.0, 0.0, 0.0]);
    let bitangent = bitangent_from([0.0, 0.0, 1.0], tangent);
    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn linear_label_quad_normal_matches_annotation_plane() {
        let start = [0.0, -1.0, 0.0];
        let end = [0.0, 1.0, 0.0];
        let offset_dir = [-1.2, 0.0, 0.0];
        let (tangent, bitangent) = linear_dimension_label_basis(start, end, offset_dir);
        let quad_normal = Vec3::from(tangent).cross(Vec3::from(bitangent)).normalize();
        // Dimension + offset lie in z=0 (XY); collinear start/end must not pick YZ via AABB.
        assert!(quad_normal.dot(Vec3::Z).abs() > 0.99);
    }
}

/// Annotation labels: lie on the annotation plane, readable from most viewing angles.
/// Falls back to camera-facing billboard only when the plane is nearly edge-on
/// (within ~8° of camera direction, where text would be unreadably thin).
pub fn readable_label_basis(
    at: Vec3,
    camera_pos: Vec3,
    preferred_tangent: Vec3,
    preferred_bitangent: Vec3,
) -> ([f32; 3], [f32; 3]) {
    let mut to_cam = camera_pos - at;
    if to_cam.length_squared() < 1e-12 {
        return (
            normalize3(preferred_tangent.into(), [1.0, 0.0, 0.0]),
            normalize3(preferred_bitangent.into(), [0.0, 1.0, 0.0]),
        );
    }
    to_cam = to_cam.normalize();

    // Prefer annotation-plane alignment. Only fall back to billboard when
    // the plane normal is nearly parallel to the camera direction (edge-on).
    let plane_n = preferred_tangent.cross(preferred_bitangent);
    let edge_on = plane_n.length_squared() > 1e-8
        && plane_n.normalize().dot(to_cam).abs() > 0.99;
    if !edge_on {
        let tangent = normalize3(preferred_tangent.into(), [1.0, 0.0, 0.0]);
        let bitangent = normalize3(preferred_bitangent.into(), [0.0, 1.0, 0.0]);
        return (tangent, bitangent);
    }

    // Edge-on fallback: camera-facing billboard
    let mut display_b =
        preferred_bitangent - to_cam * preferred_bitangent.dot(to_cam);
    if display_b.length_squared() < 1e-8 {
        display_b = preferred_tangent - to_cam * preferred_tangent.dot(to_cam);
    }
    if display_b.length_squared() < 1e-8 {
        let mut up = Vec3::Y;
        if up.dot(to_cam).abs() > 0.95 {
            up = Vec3::Z;
        }
        display_b = up - to_cam * up.dot(to_cam);
    }
    display_b = display_b.normalize();
    let display_t = to_cam.cross(display_b).normalize();
    (display_t.into(), display_b.into())
}

pub fn readable_label_basis_local(
    at: Vec3,
    camera_world: Vec3,
    model: Mat4,
    preferred_tangent: [f32; 3],
    preferred_bitangent: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let cam_local = model.inverse().transform_point3(camera_world);
    readable_label_basis(
        at,
        cam_local,
        Vec3::from(preferred_tangent),
        Vec3::from(preferred_bitangent),
    )
}

fn screen_half_extent_pixels(
    at: Vec3,
    axis: Vec3,
    half_extent: f32,
    model: Mat4,
    scene_vp: Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> f32 {
    if half_extent < 1e-8 {
        return 0.0;
    }
    let tip = at + axis * half_extent;
    let Some(ndc0) = project_point_ndc(at, model, scene_vp, depth_reversed_z) else {
        return 0.0;
    };
    let Some(ndc1) = project_point_ndc(tip, model, scene_vp, depth_reversed_z) else {
        return 0.0;
    };
    let p0 = ndc_to_screen(ndc0, screen_w, screen_h);
    let p1 = ndc_to_screen(ndc1, screen_w, screen_h);
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    (dx * dx + dy * dy).sqrt()
}

/// World half-extents so the label projects to a fixed screen height and texture aspect.
pub fn resolve_screen_stable_quad_extents(
    at: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    height_world: f32,
    screen_height_px: f32,
    texture_aspect: f32,
    model: Mat4,
    scene_vp: Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> (f32, f32, [f32; 3], [f32; 3]) {
    let at_v = Vec3::from(at);
    let t = Vec3::from(tangent);
    let b = Vec3::from(bitangent);
    let half_h0 = height_world.max(1e-6) * 0.5;
    let half_w0 = half_h0 * texture_aspect.max(1e-3);

    let px_b = screen_half_extent_pixels(
        at_v, b, half_h0, model, scene_vp, screen_w, screen_h, depth_reversed_z,
    );
    let px_t = screen_half_extent_pixels(
        at_v, t, half_w0, model, scene_vp, screen_w, screen_h, depth_reversed_z,
    );

    let target_b = screen_height_px.max(8.0) * 0.5;
    let half_h = half_h0 * target_b / px_b.max(0.5);
    let half_w = half_w0 * (target_b * texture_aspect.max(1e-3)) / px_t.max(0.5);

    (half_w, half_h, tangent, bitangent)
}

/// Base world height from extension length and style factor, then screen-pixel floor.
pub fn resolve_label_height_world(
    extension_len: f32,
    label_height_factor: f32,
    min_screen_px: f32,
    at: [f32; 3],
    bitangent: [f32; 3],
    model: Mat4,
    scene_vp: Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> f32 {
    let mut h = extension_len.max(1e-4) * label_height_factor.max(1e-4);
    let at_v = Vec3::from(at);
    let bit = Vec3::from(bitangent);
    let Some(ndc0) = project_point_ndc(at_v, model, scene_vp, depth_reversed_z) else {
        return h;
    };
    let tip = at_v + bit * h;
    let Some(ndc1) = project_point_ndc(tip, model, scene_vp, depth_reversed_z) else {
        return h;
    };
    let p0 = ndc_to_screen(ndc0, screen_w, screen_h);
    let p1 = ndc_to_screen(ndc1, screen_w, screen_h);
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let pixel_h = (dx * dx + dy * dy).sqrt();
    if pixel_h < min_screen_px && pixel_h > 0.5 {
        h *= min_screen_px / pixel_h;
    }
    h
}

struct LabelRaster {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

fn raster_label_mask(
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    font_name: &str,
    font_style: FontStyle,
    text: &str,
    raster_px: f32,
) -> Option<LabelRaster> {
    crate::font_loader::ensure_named_font(font_system, font_name);
    let attrs = crate::font_loader::attrs_from_font(font_name, font_style);
    let mut buffer = Buffer::new(font_system, Metrics::new(raster_px, raster_px * 1.2));
    buffer.set_size(font_system, Some(4096.0), Some(4096.0));
    buffer.set_text(font_system, text, attrs, Shaping::Advanced);
    buffer.shape_until_scroll(font_system, false);

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for run in buffer.layout_runs() {
        for glyph in run.glyphs.iter() {
            let pg = glyph.physical((0.0, 0.0), 1.0);
            let Some(image) = swash_cache.get_image(font_system, pg.cache_key).as_ref() else {
                continue;
            };
            if image.content != SwashContent::Mask {
                continue;
            }
            let left = pg.x + image.placement.left;
            let top = pg.y + image.placement.top;
            let right = left + image.placement.width as i32;
            let bottom = top + image.placement.height as i32;
            min_x = min_x.min(left);
            min_y = min_y.min(top);
            max_x = max_x.max(right);
            max_y = max_y.max(bottom);
        }
    }
    if min_x == i32::MAX {
        return None;
    }

    let w = (max_x - min_x).max(1) as u32;
    let h = (max_y - min_y).max(1) as u32;
    let mut mask = vec![0u8; (w * h) as usize];
    let offset = (-min_x as f32, -min_y as f32);

    for run in buffer.layout_runs() {
        for glyph in run.glyphs.iter() {
            let pg = glyph.physical(offset, 1.0);
            let Some(image) = swash_cache.get_image(font_system, pg.cache_key).as_ref() else {
                continue;
            };
            if image.content != SwashContent::Mask {
                continue;
            }
            blit_mask(&mut mask, w, h, pg.x as f32, pg.y as f32, image);
        }
    }
    let sdf = sdf::coverage_to_sdf(&mask, w, h, 8.0);
    Some(LabelRaster {
        data: sdf,
        width: w,
        height: h,
    })
}

fn blit_mask(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    x: f32,
    y: f32,
    image: &glyphon::SwashImage,
) {
    let gw = image.placement.width as i32;
    let gh = image.placement.height as i32;
    for row in 0..gh {
        for col in 0..gw {
            let src_i = (row * gw + col) as usize;
            if src_i >= image.data.len() {
                continue;
            }
            let alpha = image.data[src_i];
            if alpha == 0 {
                continue;
            }
            let dx = x as i32 + col + image.placement.left;
            let dy = y as i32 + row + image.placement.top;
            if dx < 0 || dy < 0 {
                continue;
            }
            let (dx, dy) = (dx as u32, dy as u32);
            if dx >= dst_w || dy >= dst_h {
                continue;
            }
            let di = (dy * dst_w + dx) as usize;
            dst[di] = dst[di].saturating_add(alpha);
        }
    }
}

fn build_quad_vertices(
    at: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    half_w: f32,
    half_h: f32,
) -> [WorldLabelVertex; 6] {
    let corner = |sx: f32, sy: f32, u: f32, v: f32| WorldLabelVertex {
        position: [
            at[0] + tangent[0] * sx + bitangent[0] * sy,
            at[1] + tangent[1] * sx + bitangent[1] * sy,
            at[2] + tangent[2] * sx + bitangent[2] * sy,
        ],
        uv: [u, v],
    };
    [
        corner(-half_w, -half_h, 0.0, 1.0),
        corner(half_w, -half_h, 1.0, 1.0),
        corner(half_w, half_h, 1.0, 0.0),
        corner(-half_w, -half_h, 0.0, 1.0),
        corner(half_w, half_h, 1.0, 0.0),
        corner(-half_w, half_h, 0.0, 0.0),
    ]
}

/// Draw world-space annotation labels (depth-tested, same pass target as markup lines).
pub fn draw_world_labels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    flat_pool: &mut GpuUniformPool,
    pipelines: &PipelineSet,
    pass: &mut wgpu::RenderPass<'_>,
    labels: &[WorldLabelCommand],
    scene_vp: Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    camera_world: Vec3,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    _label_attrs: glyphon::Attrs<'static>,
) {
    if labels.is_empty() {
        return;
    }
    let pipeline = if depth_reversed_z {
        &pipelines.world_label_reverse
    } else {
        &pipelines.world_label_forward
    };
    pass.set_pipeline(pipeline);

    let raster_px = 64.0_f32;
    for cmd in labels {
        let Some(raster) = raster_label_mask(
            font_system,
            swash_cache,
            &cmd.font_name,
            cmd.font_style,
            &cmd.string,
            raster_px,
        )
        else {
            continue;
        };
        if raster.width == 0 || raster.height == 0 {
            continue;
        }

        let aspect = raster.width as f32 / raster.height as f32;
        let at_v = Vec3::from(cmd.at);
        let (tangent, bitangent) = readable_label_basis_local(
            at_v,
            camera_world,
            cmd.model_matrix,
            cmd.tangent,
            cmd.bitangent,
        );
        let (half_w, half_h, tangent, bitangent) = resolve_screen_stable_quad_extents(
            cmd.at,
            tangent,
            bitangent,
            cmd.height_world,
            cmd.screen_height_px,
            aspect,
            cmd.model_matrix,
            scene_vp,
            screen_w,
            screen_h,
            depth_reversed_z,
        );
        let verts = build_quad_vertices(cmd.at, tangent, bitangent, half_w, half_h);
        let mvp = scene_vp * cmd.model_matrix;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("world label"),
            size: wgpu::Extent3d {
                width: raster.width,
                height: raster.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &raster.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(raster.width),
                rows_per_image: Some(raster.height),
            },
            wgpu::Extent3d {
                width: raster.width,
                height: raster.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniforms = FlatUniforms {
            mvp: mvp.to_cols_array_2d(),
            color: cmd.color,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let Some(offset) = flat_pool.push_flat(&uniforms) else {
            continue;
        };
        let stride =
            std::num::NonZero::new(flat_pool.stride()).expect("flat uniform stride");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world label bg"),
            layout: &pipelines.world_label_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &flat_pool.buffer,
                        offset: 0,
                        size: Some(stride),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&pipelines.world_label_sampler),
                },
            ],
        });

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("world label vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        pass.set_bind_group(0, &bind_group, &[offset]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..6, 0..1);
    }
}
