use rc3d_core::math::Vec3;
use rc3d_scene::node_entry::dirty_flags::GEOMETRY;
use rc3d_scene::{NodeData, SceneGraph};

use crate::dirty_flags::mark_node_dirty;

type MeshLodStage = (Vec<Vec3>, Option<Vec<[f32; 2]>>, Vec<i32>);

const PREVIEW_TRIANGLE_THRESHOLD: usize = 1_000_000;

/// Gaussian-CDF streaming: approx erf function (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

const STREAM_SIGMA: f64 = 0.22;
const STREAM_DURATION_S: f64 = 2.2;
const STREAM_LOD_TARGETS: &[usize] = &[
    12_000, 30_000, 60_000, 110_000, 180_000, 280_000, 430_000, 660_000, 1_000_000,
];
const STREAM_STEP_MS: u64 = 180;

pub struct FullResPatch {
    pub coord_node: rc3d_core::NodeId,
    pub ifs_node: rc3d_core::NodeId,
    pub tex_node: Option<rc3d_core::NodeId>,
    pub full_points: Vec<Vec3>,
    pub full_tex: Option<Vec<[f32; 2]>>,
    pub full_coord_index: Vec<i32>,
    pub total_full_tris: usize,
    pub stream_stages: Vec<MeshLodStage>,
    pub stage_tri_counts: Vec<usize>,
    pub current_stage: usize,
    pub stream_start: Option<std::time::Instant>,
}

pub fn apply_lod_stage_to_graph(
    graph: &mut SceneGraph,
    coord_id: rc3d_core::NodeId,
    ifs_id: rc3d_core::NodeId,
    tex_node: Option<rc3d_core::NodeId>,
    stage: &MeshLodStage,
) {
    let (points, tex, coord_index) = stage;
    {
        if let Some(coord_mut) = graph.get_mut(coord_id) {
            if let NodeData::Coordinate3(c) = &mut coord_mut.data {
                c.point = points.clone();
            }
        }
        if let (Some(tid), Some(tex_pts)) = (tex_node, tex.as_ref()) {
            if let Some(tex_mut) = graph.get_mut(tid) {
                if let NodeData::TextureCoordinate2(t) = &mut tex_mut.data {
                    t.point = tex_pts.clone();
                }
            }
        }
        if let Some(ifs_mut) = graph.get_mut(ifs_id) {
            if let NodeData::IndexedFaceSet(ifs) = &mut ifs_mut.data {
                ifs.coord_index = coord_index.clone();
            }
        }
    }
    mark_node_dirty(graph, coord_id, GEOMETRY);
    mark_node_dirty(graph, ifs_id, GEOMETRY);
    if let Some(tid) = tex_node {
        mark_node_dirty(graph, tid, GEOMETRY);
    }
}

impl FullResPatch {
    pub fn apply_stage_to_graph(&self, graph: &mut SceneGraph, stage_i: usize) {
        apply_lod_stage_to_graph(
            graph,
            self.coord_node,
            self.ifs_node,
            self.tex_node,
            &self.stream_stages[stage_i],
        );
    }

    pub fn apply_full_to_graph(self, graph: &mut SceneGraph) {
        {
            if let Some(coord_mut) = graph.get_mut(self.coord_node) {
                if let NodeData::Coordinate3(c) = &mut coord_mut.data {
                    c.point = self.full_points;
                }
            }
            if let (Some(tex_id), Some(full_tex)) = (self.tex_node, self.full_tex) {
                if let Some(tex_mut) = graph.get_mut(tex_id) {
                    if let NodeData::TextureCoordinate2(t) = &mut tex_mut.data {
                        t.point = full_tex;
                    }
                }
            }
            if let Some(ifs_mut) = graph.get_mut(self.ifs_node) {
                if let NodeData::IndexedFaceSet(ifs) = &mut ifs_mut.data {
                    ifs.coord_index = self.full_coord_index;
                }
            }
        }
        mark_node_dirty(graph, self.coord_node, GEOMETRY);
        mark_node_dirty(graph, self.ifs_node, GEOMETRY);
        if let Some(tid) = self.tex_node {
            mark_node_dirty(graph, tid, GEOMETRY);
        }
    }

    pub fn stage_for_budget(&self, budget: usize) -> Option<usize> {
        let n = self.stage_tri_counts.len();
        if n == 0 || budget == 0 {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.stage_tri_counts[mid] <= budget {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            None
        } else {
            Some(lo - 1)
        }
    }

    pub fn final_stage_index(&self) -> Option<usize> {
        self.stream_stages.len().checked_sub(1)
    }
}

pub fn gaussian_triangle_budget(t_s: f64, total_tris: usize) -> usize {
    if t_s <= 0.0 {
        return 0;
    }
    if t_s >= STREAM_DURATION_S {
        return total_tris;
    }
    let x = (t_s / STREAM_DURATION_S - 0.5) / STREAM_SIGMA;
    let fraction = normal_cdf(x).clamp(0.005, 0.999);
    ((total_tris as f64) * fraction) as usize
}

pub fn apply_decimated_preview(graph: &mut SceneGraph) -> Vec<FullResPatch> {
    let mut patches = Vec::new();
    for &root in graph.roots().to_vec().iter() {
        collect_preview_patches(graph, root, &mut patches);
    }
    patches
}

fn collect_preview_patches(
    graph: &mut SceneGraph,
    node: rc3d_core::NodeId,
    patches: &mut Vec<FullResPatch>,
) {
    let children: Vec<rc3d_core::NodeId> = graph.children(node).unwrap_or(&[]).to_vec();
    if children.len() >= 3 {
        for trip in children.windows(3) {
            let coord_id = trip[0];
            let tex_id = trip[1];
            let ifs_id = trip[2];
            let (Some(coord_entry), Some(tex_entry), Some(ifs_entry)) =
                (graph.get(coord_id), graph.get(tex_id), graph.get(ifs_id))
            else {
                continue;
            };
            let (
                NodeData::Coordinate3(coord),
                NodeData::TextureCoordinate2(tex),
                NodeData::IndexedFaceSet(ifs),
            ) = (&coord_entry.data, &tex_entry.data, &ifs_entry.data)
            else {
                continue;
            };
            if tex.point.len() != coord.point.len() {
                continue;
            }
            let tri_count = ifs.coord_index.iter().filter(|&&v| v == -1).count();
            if tri_count <= PREVIEW_TRIANGLE_THRESHOLD {
                continue;
            }
            let full_points = coord.point.clone();
            let full_tex = tex.point.clone();
            let full_coord_index = ifs.coord_index.clone();
            let stream_stages =
                build_stream_stages(&full_points, Some(&full_tex), &full_coord_index, tri_count);
            let stage_tri_counts: Vec<usize> = stream_stages
                .iter()
                .map(|s| s.2.iter().filter(|&&v| v == -1).count())
                .collect();
            apply_lod_stage_to_graph(graph, coord_id, ifs_id, Some(tex_id), &stream_stages[0]);
            patches.push(FullResPatch {
                coord_node: coord_id,
                ifs_node: ifs_id,
                tex_node: Some(tex_id),
                full_points,
                full_tex: Some(full_tex),
                full_coord_index,
                total_full_tris: tri_count,
                stream_stages,
                stage_tri_counts,
                current_stage: 0,
                stream_start: None,
            });
        }
    }
    if children.len() >= 2 {
        for pair in children.windows(2) {
            let coord_id = pair[0];
            let ifs_id = pair[1];
            let (Some(coord_entry), Some(ifs_entry)) = (graph.get(coord_id), graph.get(ifs_id))
            else {
                continue;
            };
            let (NodeData::Coordinate3(coord), NodeData::IndexedFaceSet(ifs)) =
                (&coord_entry.data, &ifs_entry.data)
            else {
                continue;
            };
            let tri_count = ifs.coord_index.iter().filter(|&&v| v == -1).count();
            if tri_count <= PREVIEW_TRIANGLE_THRESHOLD {
                continue;
            }
            let full_points = coord.point.clone();
            let full_coord_index = ifs.coord_index.clone();
            let stream_stages =
                build_stream_stages(&full_points, None, &full_coord_index, tri_count);
            let stage_tri_counts: Vec<usize> = stream_stages
                .iter()
                .map(|s| s.2.iter().filter(|&&v| v == -1).count())
                .collect();
            apply_lod_stage_to_graph(graph, coord_id, ifs_id, None, &stream_stages[0]);
            patches.push(FullResPatch {
                coord_node: coord_id,
                ifs_node: ifs_id,
                tex_node: None,
                full_points,
                full_tex: None,
                full_coord_index,
                total_full_tris: tri_count,
                stream_stages,
                stage_tri_counts,
                current_stage: 0,
                stream_start: None,
            });
        }
    }
    for child in children {
        collect_preview_patches(graph, child, patches);
    }
}

fn build_stream_stages(
    full_points: &[Vec3],
    tex: Option<&[[f32; 2]]>,
    full_coord_index: &[i32],
    tri_count: usize,
) -> Vec<MeshLodStage> {
    let mut out = Vec::new();
    let n = STREAM_LOD_TARGETS.len();
    let mut last_tris: usize = 0;
    for i in 0..n {
        let t = STREAM_LOD_TARGETS[i];
        if t >= tri_count {
            break;
        }
        let stage = decimate_indexed_face_set(full_points, tex, full_coord_index, t);
        let tris = stage.2.iter().filter(|&&v| v == -1).count();
        if tris > last_tris {
            last_tris = tris;
            out.push(stage);
        }
    }
    if out.is_empty() {
        let fallback = STREAM_LOD_TARGETS
            .iter()
            .copied()
            .find(|&x| x < tri_count)
            .unwrap_or(250_000);
        out.push(decimate_indexed_face_set(
            full_points,
            tex,
            full_coord_index,
            fallback,
        ));
    }
    out
}

fn decimate_indexed_face_set(
    points: &[Vec3],
    tex: Option<&[[f32; 2]]>,
    coord_index: &[i32],
    target_triangles: usize,
) -> (Vec<Vec3>, Option<Vec<[f32; 2]>>, Vec<i32>) {
    if let Some(t) = tex {
        assert_eq!(t.len(), points.len(), "texture coords must match positions");
    }
    let tri_count = coord_index.iter().filter(|&&v| v == -1).count();
    if tri_count <= target_triangles || tri_count == 0 {
        return (
            points.to_vec(),
            tex.map(|t| t.to_vec()),
            coord_index.to_vec(),
        );
    }
    let stride = (tri_count / target_triangles).max(2);
    let mut new_points = Vec::new();
    let mut new_tex: Option<Vec<[f32; 2]>> = tex.map(|_| Vec::new());
    let mut new_index = Vec::new();
    let mut remap: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();

    let mut face = Vec::with_capacity(4);
    let mut face_id = 0usize;
    for &idx in coord_index {
        if idx < 0 {
            if face.len() == 3 && face_id % stride == 0 {
                for &src in &face {
                    let mapped = if let Some(&m) = remap.get(&src) {
                        m
                    } else {
                        let m = new_points.len() as i32;
                        remap.insert(src, m);
                        new_points.push(points[src as usize]);
                        if let (Some(t_in), Some(ref mut t_out)) = (tex, new_tex.as_mut()) {
                            t_out.push(t_in[src as usize]);
                        }
                        m
                    };
                    new_index.push(mapped);
                }
                new_index.push(-1);
            }
            face.clear();
            face_id += 1;
        } else {
            face.push(idx);
        }
    }
    if new_points.is_empty() || new_index.is_empty() {
        return (
            points.to_vec(),
            tex.map(|t| t.to_vec()),
            coord_index.to_vec(),
        );
    }
    (new_points, new_tex, new_index)
}

pub const fn stream_step_ms() -> u64 {
    STREAM_STEP_MS
}
