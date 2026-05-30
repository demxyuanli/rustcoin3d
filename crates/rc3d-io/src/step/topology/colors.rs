//! Face color extraction from STYLED_ITEM.
use std::collections::HashMap;
use super::super::parser::EntityIndex;
use super::super::value::StepValue;
use super::helpers::*;

/// Extract per-face colors from STYLED_ITEM → SURFACE_STYLE_FILL_AREA → COLOUR_RGB chain.
/// Returns a map from face entity ID to (R, G, B) color.
pub fn collect_face_colors(entities: &EntityIndex) -> HashMap<u64, [f32; 3]> {
    let mut colors: HashMap<u64, [f32; 3]> = HashMap::new();

    for (_, record) in entities.iter() {
        if record.name != "STYLED_ITEM" {
            continue;
        }
        // STYLED_ITEM(name, (#psa1, ...), #item)
        let style_list = record.params.nth_param(1).and_then(|v| v.as_list());
        let item_ref = record.params.nth_param(2).and_then(|v| v.as_ref_id());

        let color = extract_color_from_styles(style_list, entities);
        if let (Some(item_id), Some(rgb)) = (item_ref, color) {
            // The item may be a face or a shape representation containing faces
            for face_id in resolve_item_to_faces(item_id, entities) {
                colors.insert(face_id, rgb);
            }
        }
    }

    colors
}

/// Walk PRESENTATION_STYLE_ASSIGNMENT → SURFACE_STYLE_USAGE → SURFACE_STYLE_FILL_AREA
/// → FILL_AREA_STYLE → COLOUR_RGB chain to extract RGB triplet.
fn extract_color_from_styles(style_list: Option<&[StepValue]>, entities: &EntityIndex) -> Option<[f32; 3]> {
    let list = style_list?;
    for psa_val in list {
        let psa_id = psa_val.as_ref_id()?;
        let psa_record = entities.get(&psa_id)?;
        if psa_record.name != "PRESENTATION_STYLE_ASSIGNMENT" {
            continue;
        }
        let inner_styles = psa_record.params.nth_param(1).and_then(|v| v.as_list())?;
        for style_val in inner_styles {
            let style_id = style_val.as_ref_id()?;
            let style_record = entities.get(&style_id)?;
            match style_record.name.as_str() {
                "SURFACE_STYLE_USAGE" => {
                    // SURFACE_STYLE_USAGE(usage_type, #side_style)
                    let side_id = nth_ref(&style_record.params, 1)?;
                    if let Some(rgb) = resolve_surface_style_to_rgb(side_id, entities) {
                        return Some(rgb);
                    }
                }
                "SURFACE_SIDE_STYLE" => {
                    if let Some(rgb) = resolve_surface_style_to_rgb(style_id, entities) {
                        return Some(rgb);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn resolve_surface_style_to_rgb(side_style_id: u64, entities: &EntityIndex) -> Option<[f32; 3]> {
    let record = entities.get(&side_style_id)?;
    match record.name.as_str() {
        "SURFACE_SIDE_STYLE" => {
            // SURFACE_SIDE_STYLE(name, #fill_area_style)
            let fill_id = nth_ref(&record.params, 1)?;
            resolve_surface_style_to_rgb(fill_id, entities)
        }
        "SURFACE_STYLE_FILL_AREA" => {
            // SURFACE_STYLE_FILL_AREA(#fill_area)
            let fill_id = nth_ref(&record.params, 0)?;
            resolve_surface_style_to_rgb(fill_id, entities)
        }
        "FILL_AREA_STYLE" => {
            // FILL_AREA_STYLE(name, #fill_colour)
            let colour_id = nth_ref(&record.params, 1)?;
            resolve_colour_rgb(colour_id, entities)
        }
        _ => None,
    }
}

fn resolve_colour_rgb(colour_id: u64, entities: &EntityIndex) -> Option<[f32; 3]> {
    let record = entities.get(&colour_id)?;
    if record.name != "COLOUR_RGB" && record.name != "COLOUR" {
        return None;
    }
    let r = record.params.nth_param(1).and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
    let g = record.params.nth_param(2).and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
    let b = record.params.nth_param(3).and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
    Some([r, g, b])
}

/// Resolve a STYLED_ITEM target to a set of face entity IDs.
fn resolve_item_to_faces(item_id: u64, entities: &EntityIndex) -> Vec<u64> {
    // Direct: the item itself is a face
    if let Some(record) = entities.get(&item_id) {
        match record.name.as_str() {
            "ADVANCED_FACE" | "FACE" | "FACE_SURFACE" => {
                return vec![item_id];
            }
            "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL" => {
                // Shell: return its face IDs
                let face_ids = nth_list_refs(&record.params, 1).unwrap_or_default();
                return face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)).collect();
            }
            _ => {}
        }
    }
    // Indirect: look for this item as a member of a shape representation
    for (&eid, record) in entities.iter() {
        if record.name == "SHAPE_REPRESENTATION" || record.name == "ADVANCED_BREP_SHAPE_REPRESENTATION" {
            let items = record.params.nth_param(1).and_then(|v| v.as_list());
            if let Some(item_list) = items {
                for item in item_list {
                    if let Some(iid) = item.as_ref_id() {
                        if iid == item_id {
                            // Found the representation containing this item
                            return find_all_faces_in_representation(eid, entities);
                        }
                    }
                }
            }
        }
    }
    vec![]
}

fn resolve_face_recursive(face_id: u64, entities: &EntityIndex) -> Vec<u64> {
    let record = match entities.get(&face_id) {
        Some(r) => r,
        None => return vec![],
    };
    match record.name.as_str() {
        "ADVANCED_FACE" | "FACE" | "FACE_SURFACE" => vec![face_id],
        "ORIENTED_FACE" => {
            let inner_id = nth_ref(&record.params, 3).or_else(|| nth_ref(&record.params, 1));
            inner_id.map(|id| resolve_face_recursive(id, entities)).unwrap_or_default()
        }
        _ => vec![],
    }
}

fn find_all_faces_in_representation(rep_id: u64, entities: &EntityIndex) -> Vec<u64> {
    let record = match entities.get(&rep_id) {
        Some(r) => r,
        None => return vec![],
    };
    let items: Vec<StepValue> = record.params.nth_param(1).and_then(|v| v.as_list()).map(|s| s.to_vec()).unwrap_or_default();
    let mut faces = Vec::new();
    for item in &items {
        let id: u64 = match item.as_ref_id() {
            Some(x) => x,
            None => continue,
        };
        if let Some(r) = entities.get(&id) {
            match r.name.as_str() {
                "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" => {
                    if let Some(outer_shell_id) = nth_ref(&r.params, 1) {
                        if let Some(shell) = entities.get(&outer_shell_id) {
                            let face_ids = nth_list_refs(&shell.params, 1).unwrap_or_default();
                            faces.extend(face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)));
                        }
                    }
                }
                "SHELL_BASED_SURFACE_MODEL" => {
                    let shell_ids: Vec<u64> = r.params.nth_param(1)
                        .and_then(|v| v.as_list())
                        .map(|l| l.iter().filter_map(|x| x.as_ref_id()).collect())
                        .unwrap_or_default();
                    for sid in shell_ids {
                        if let Some(shell) = entities.get(&sid) {
                            let face_ids = nth_list_refs(&shell.params, 1).unwrap_or_default();
                            faces.extend(face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)));
                        }
                    }
                }
                "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL" => {
                    let face_ids = nth_list_refs(&r.params, 1).unwrap_or_default();
                    faces.extend(face_ids.iter().flat_map(|&fid| resolve_face_recursive(fid, entities)));
                }
                _ => {}
            }
        }
    }
    faces
}
