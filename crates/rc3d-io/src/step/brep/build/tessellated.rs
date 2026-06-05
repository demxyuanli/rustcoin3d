//! Tessellated geometry builder for STEP entities.
//!
//! Handles TESSELLATED_SHELL, TESSELLATED_FACE, TRIANGULATED_FACE,
//! COMPLEX_TRIANGULATED_FACE, and COORDINATES_LIST entities from AP242.
//!
//! These provide pre-triangulated geometry (common in CATIA, NX exports)
//! as a fallback when precise B-Rep surfaces are unavailable. We convert
//! them directly to TriangleMesh data.

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use crate::step::parser::EntityIndex;
use crate::step::value::StepValue;

/// Raw triangle mesh extracted from tessellated STEP geometry.
#[derive(Debug, Clone)]
pub struct TessellatedMesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub normals: Option<Vec<Vec3>>,
    pub color: Option<[f32; 3]>,
}

/// Read a COORDINATES_LIST entity into a flat Vec<Vec3>.
///
/// STEP structure: COORDINATES_LIST('name', (x1,y1,z1, x2,y2,z2, ...))
/// Each coordinate is 3 consecutive real values; total count is n/3.
fn read_coordinates_list(coords_id: u64, entities: &EntityIndex) -> Option<Vec<Vec3>> {
    let record = entities.get(&coords_id)?;
    let list = record.params.nth_param(1)?.as_list()?;
    if list.len() < 3 || list.len() % 3 != 0 {
        log::warn!(
            "[STEP] COORDINATES_LIST #{} has {} values (expected multiple of 3)",
            coords_id, list.len()
        );
        return None;
    }
    let n = list.len() / 3;
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        let x = list[i * 3].as_real()? as f32;
        let y = list[i * 3 + 1].as_real()? as f32;
        let z = list[i * 3 + 2].as_real()? as f32;
        positions.push(Vec3::new(x, y, z));
    }
    Some(positions)
}

/// Read a TRIANGULATED_FACE or COMPLEX_TRIANGULATED_FACE entity into index triples.
///
/// STEP structure (TRIANGULATED_FACE):
///   TRIANGULATED_FACE('name', #pn_max, ((i1,i2,i3), (i4,i5,i6), ...))
///   where pn_max is the number of triangles, followed by triples of 1-based indices.
///
/// COMPLEX_TRIANGULATED_FACE is similar but may contain per-triangle PnIndex,
/// normal indices, and color indices. We extract only the position indices.
fn read_triangulated_face(tri_id: u64, entities: &EntityIndex) -> Option<Vec<u32>> {
    let record = entities.get(&tri_id)?;
    let name = record.name.as_str();

    // Param 0: pn_max (integer, number of triangles) — may be Omitted
    // Remaining params: alternating (index_triples, optional pn_index)
    // For COMPLEX_TRIANGULATED_FACE: ((i1,i2,i3), pn1, n1, c1, (i4,i5,i6), pn2, n2, c2, ...)

    let triples = if name == "COMPLEX_TRIANGULATED_FACE" {
        // Each triangle occupies 4 entries: (i1,i2,i3), pn_idx, normal_idx, color_idx
        read_complex_triangulated_face_indices(record)
    } else {
        // TRIANGULATED_FACE: positions are at params[1], params[3], params[5], ...
        // Each even-indexed param is a list of 3 integers
        // Actually: params list alternates between index triples and optional pn_index ints
        // So params[0] may be pn_max, then (i1,i2,i3) lists at odd indices
        read_simple_triangulated_face_indices(record)
    }?;

    if triples.is_empty() {
        return None;
    }
    Some(triples)
}

fn read_simple_triangulated_face_indices(record: &crate::step::parser::EntityRecord) -> Option<Vec<u32>> {
    let params = record.params.as_list()?;
    let mut indices = Vec::new();
    for (i, val) in params.iter().enumerate() {
        // Skip pn_max (first param, integer)
        if i == 0 && val.as_int().is_some() {
            continue;
        }
        // Try to read as a list of 3 integers (index triple)
        if let Some(tri_list) = val.as_list() {
            if tri_list.len() == 3 {
                for v in tri_list {
                    if let Some(idx) = v.as_int() {
                        indices.push((idx as u32).saturating_sub(1)); // 1-based → 0-based
                    }
                }
            }
        }
    }
    Some(indices)
}

fn read_complex_triangulated_face_indices(record: &crate::step::parser::EntityRecord) -> Option<Vec<u32>> {
    let params = record.params.as_list()?;
    let mut indices = Vec::new();
    // COMPLEX_TRIANGULATED_FACE: groups of 4 values:
    // (i1,i2,i3) list, pn_index int, normal_index int, color_index int
    let mut i = 0;
    while i + 3 < params.len() {
        // Skip pn_max if first param is integer
        if i == 0 {
            if let Some(tri_list) = params[i].as_list() {
                if tri_list.len() == 3 {
                    for v in tri_list {
                        if let Some(idx) = v.as_int() {
                            indices.push((idx as u32).saturating_sub(1));
                        }
                    }
                }
                i += 4; // skip the 3 following params
                continue;
            } else if params[i].as_int().is_some() {
                i += 1;
                continue;
            }
        }
        // Index triple
        if let Some(tri_list) = params[i].as_list() {
            if tri_list.len() == 3 {
                for v in tri_list {
                    if let Some(idx) = v.as_int() {
                        indices.push((idx as u32).saturating_sub(1));
                    }
                }
            }
        }
        i += 4; // skip pn_index, normal_index, color_index
    }
    Some(indices)
}

/// Build a TessellatedMesh from a TESSELLATED_SHELL entity.
///
/// STEP structure:
///   TESSELLATED_SHELL('name', #coords, #normals?, (#face1, #face2, ...))
///
/// Each face reference points to a TESSELLATED_FACE or TRIANGULATED_FACE.
/// The shell shares a single COORDINATES_LIST and optional normals list.
pub fn build_tessellated_shell(shell_id: u64, entities: &EntityIndex) -> Option<Vec<TessellatedMesh>> {
    let record = entities.get(&shell_id)?;
    if record.name != "TESSELLATED_SHELL" {
        return None;
    }

    let params = record.params.as_list()?;
    if params.len() < 3 {
        return None;
    }

    // Param 1: COORDINATES_LIST reference
    let coords_id = params[1].as_ref_id()?;
    let positions = read_coordinates_list(coords_id, entities)?;

    // Param 2: optional normals COORDINATES_LIST reference (may be omitted)
    let normals = if params.len() > 2 {
        params[2].as_ref_id().and_then(|nid| read_coordinates_list(nid, entities))
    } else {
        None
    };

    // Param 3: face list
    let face_list = if params.len() > 3 {
        params[3].as_list()
    } else {
        params.get(2).and_then(|v| v.as_list())
    }?;

    let mut meshes = Vec::new();
    for face_val in face_list {
        if let Some(face_id) = face_val.as_ref_id() {
            if let Some(indices) = read_triangulated_face(face_id, entities) {
                meshes.push(TessellatedMesh {
                    positions: positions.clone(),
                    indices,
                    normals: normals.clone(),
                    color: None,
                });
            }
        }
    }

    if meshes.is_empty() { None } else { Some(meshes) }
}

/// Detect tessellated geometry shells in the entity index and return them
/// as a map of shell_id → Vec<TessellatedMesh>.
///
/// This is a fallback for STEP files that use tessellated geometry instead of
/// precise B-Rep surfaces. Call after the main B-Rep shell collection to fill
/// any missing geometry.
pub fn collect_tessellated_shells(entities: &EntityIndex) -> HashMap<u64, Vec<TessellatedMesh>> {
    let mut result = HashMap::new();
    for (&id, record) in entities.iter() {
        if record.name == "TESSELLATED_SHELL" {
            match build_tessellated_shell(id, entities) {
                Some(meshes) => {
                    log::info!(
                        "[STEP] Tessellated shell #{}: {} faces, {} total vertices",
                        id,
                        meshes.len(),
                        meshes.first().map(|m| m.positions.len()).unwrap_or(0)
                    );
                    result.insert(id, meshes);
                }
                None => {
                    log::warn!("[STEP] Failed to build tessellated shell #{}", id);
                }
            }
        }
    }
    result
}
