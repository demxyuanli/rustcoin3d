//! Incremental constrained Delaunay adapter for UV face fill.

use std::collections::HashMap;

use super::triangulation::Delaunay2d;
use super::half_edge::VertIdx;
use super::DelaunayConfig;

/// Opaque vertex handle (internal Delaunay vertex index).
pub type CdtVertHandle = u32;

pub struct NativeCdt {
    delaunay: Delaunay2d,
    handles: Vec<VertIdx>,
    handle_gi: Vec<usize>,
    constraints: usize,
}

impl NativeCdt {
    pub fn from_uv_bbox(u_min: f32, v_min: f32, u_max: f32, v_max: f32) -> Self {
        let mut delaunay = Delaunay2d::new(DelaunayConfig {
            sort_by_diagonal: false,
            optimize: true,
            max_optimize_passes: 1,
        });
        delaunay.begin_live(
            u_min as f64,
            v_min as f64,
            u_max as f64,
            v_max as f64,
        );
        Self {
            delaunay,
            handles: Vec::new(),
            handle_gi: Vec::new(),
            constraints: 0,
        }
    }

    pub fn insert(&mut self, u: f64, v: f64, gi: usize) -> Option<CdtVertHandle> {
        let vidx = self.delaunay.insert_live(u, v, gi as u32);
        let hi = self.handles.len() as CdtVertHandle;
        self.handles.push(vidx);
        self.handle_gi.push(gi);
        Some(hi)
    }

    pub fn try_add_constraint(&mut self, ha: CdtVertHandle, hb: CdtVertHandle) -> bool {
        let Some(a) = self.handles.get(ha as usize).copied() else {
            return false;
        };
        let Some(b) = self.handles.get(hb as usize).copied() else {
            return false;
        };
        let ok = self.delaunay.constrain_live(a, b);
        if ok {
            self.constraints += 1;
        }
        ok
    }

    pub fn exists_constraint(&self, ha: CdtVertHandle, hb: CdtVertHandle) -> bool {
        let a = self.handles.get(ha as usize).copied();
        let b = self.handles.get(hb as usize).copied();
        match (a, b) {
            (Some(a), Some(b)) => self.delaunay.has_edge(a, b),
            _ => false,
        }
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints
    }

    pub fn vertex_uv(&self, ha: CdtVertHandle) -> (f32, f32) {
        let v = self.handles[ha as usize];
        let p = self.delaunay.vertex_point(v);
        (p.x as f32, p.y as f32)
    }

    pub fn handle_global_index(&self, ha: CdtVertHandle) -> usize {
        self.handle_gi[ha as usize]
    }

    pub fn extract_triangles(
        &self,
        filter: impl Fn(f32, f32) -> bool,
    ) -> Vec<usize> {
        let mut tris = Vec::new();
        for (gids, uvs) in self.inner_faces_detail() {
            let cu = (uvs[0].0 + uvs[1].0 + uvs[2].0) / 3.0;
            let cv = (uvs[0].1 + uvs[1].1 + uvs[2].1) / 3.0;
            if !filter(cu, cv) {
                continue;
            }
            tris.push(gids[0]);
            tris.push(gids[1]);
            tris.push(gids[2]);
        }
        tris
    }

    /// Global indices and UV coords per inner triangle.
    pub fn inner_faces_detail(&self) -> Vec<([usize; 3], [(f32, f32); 3])> {
        let mut out = Vec::new();
        for face in self.delaunay.inner_faces() {
            let gids = [
                self.delaunay.vertex_data(face[0]) as usize,
                self.delaunay.vertex_data(face[1]) as usize,
                self.delaunay.vertex_data(face[2]) as usize,
            ];
            let uvs = [
                self.delaunay.vertex_point(face[0]).to_f32(),
                self.delaunay.vertex_point(face[1]).to_f32(),
                self.delaunay.vertex_point(face[2]).to_f32(),
            ];
            out.push((gids, uvs));
        }
        out
    }

    pub fn vertex_count(&self) -> usize {
        self.handle_gi.len()
    }

    pub fn finalize(&mut self) {
        self.delaunay.finalize_constraints();
    }
}

/// UV bbox (min_u, min_v, max_u, max_v) from trim loops.
pub fn uv_bbox_from_loops(
    outer: &[(f32, f32)],
    inners: &[Vec<(f32, f32)>],
) -> (f32, f32, f32, f32) {
    let mut u_min = f32::MAX;
    let mut u_max = f32::MIN;
    let mut v_min = f32::MAX;
    let mut v_max = f32::MIN;
    for &(u, v) in outer {
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    for inner in inners {
        for &(u, v) in inner {
            u_min = u_min.min(u);
            u_max = u_max.max(u);
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        }
    }
    if u_min > u_max {
        return (0.0, 0.0, 1.0, 1.0);
    }
    (u_min, v_min, u_max, v_max)
}

/// Insert UV with quantization / bump logic for coincident UV keys.
pub fn insert_uv_native(
    cdt: &mut NativeCdt,
    mut uv: (f32, f32),
    gi: usize,
    uv_to_handle: &mut HashMap<(u64, u64), CdtVertHandle>,
    uv_span: Option<(f32, f32)>,
    quant_key: fn((f32, f32), Option<(f32, f32)>) -> (u64, u64),
) -> Option<CdtVertHandle> {
    for bump in 0..32usize {
        let key = quant_key(uv, uv_span);
        if let Some(&hi) = uv_to_handle.get(&key) {
            if cdt.handle_global_index(hi) == gi {
                return Some(hi);
            }
            let bump_scale = 1e-5 * (bump as f32 + 1.0);
            if let Some(span) = uv_span {
                uv.0 += bump_scale * span.0;
                uv.1 += bump_scale * span.1 * ((bump % 2) as f32 * 2.0 - 1.0);
            } else {
                uv.0 += bump_scale;
                uv.1 += bump_scale * ((bump % 2) as f32 * 2.0 - 1.0);
            }
            continue;
        }
        let hi = cdt.insert(uv.0 as f64, uv.1 as f64, gi)?;
        uv_to_handle.insert(key, hi);
        return Some(hi);
    }
    None
}
