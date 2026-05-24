//! Path B: Cleaned pass-through from EntityIndex → ISO 10303-21 text.
//!
//! Filters out presentation/metadata entities, renumbers IDs, updates references,
//! and outputs standardized STEP text.

use std::collections::HashMap;
use super::super::parser::EntityIndex;
use super::super::value::StepValue;
use super::format::{format_param, format_header, classify_group};

/// Whitelisted entity names for Path B pass-through.
const KEEP_TYPES: &[&str] = &[
    // Geometry
    "CARTESIAN_POINT", "DIRECTION", "VECTOR", "AXIS2_PLACEMENT_3D", "AXIS2_PLACEMENT_2D",
    "AXIS1_PLACEMENT",
    // Curves
    "LINE", "CIRCLE", "ELLIPSE", "POLYLINE",
    "B_SPLINE_CURVE", "B_SPLINE_CURVE_WITH_KNOTS", "RATIONAL_B_SPLINE_CURVE",
    "PCURVE", "SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE",
    "TRIMMED_CURVE", "COMPOSITE_CURVE", "COMPOSITE_CURVE_SEGMENT", "OFFSET_CURVE_3D",
    // Surfaces
    "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS", "RATIONAL_B_SPLINE_SURFACE",
    "SURFACE_OF_REVOLUTION", "SURFACE_OF_LINEAR_EXTRUSION",
    "OFFSET_SURFACE", "RECTANGULAR_TRIMMED_SURFACE",
    "CURVE_BOUNDED_SURFACE", "BOUNDED_SURFACE",
    // Topology
    "VERTEX_POINT", "EDGE_CURVE", "ORIENTED_EDGE", "EDGE_LOOP", "POLY_LOOP",
    "FACE_OUTER_BOUND", "FACE_BOUND",
    "ADVANCED_FACE", "FACE_SURFACE", "FACE",
    "CLOSED_SHELL", "OPEN_SHELL", "SHELL",
    "ORIENTED_CLOSED_SHELL", "ORIENTED_OPEN_SHELL",
    "MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "SHELL_BASED_SURFACE_MODEL",
    // Assembly
    "PRODUCT", "PRODUCT_DEFINITION", "PRODUCT_DEFINITION_FORMATION",
    "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
    "PRODUCT_DEFINITION_SHAPE", "SHAPE_DEFINITION_REPRESENTATION",
    "SHAPE_REPRESENTATION", "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "ITEM_DEFINED_TRANSFORMATION",
];

pub fn write_entities(entities: &EntityIndex) -> String {
    let mut out = String::with_capacity(entities.len() * 256);

    // Phase 1a: initial filter
    let mut keep_ids: std::collections::HashSet<u64> = entities.iter()
        .filter(|(_, r)| KEEP_TYPES.contains(&r.name.as_str()))
        .map(|(&id, _)| id)
        .collect();

    // Phase 1b: transitive closure — include entities referenced by kept entities
    loop {
        let prev_len = keep_ids.len();
        for &id in &keep_ids.clone() {
            if let Some(record) = entities.get(&id) {
                let refs = collect_refs(&record.params);
                for ref_id in refs {
                    if entities.contains_key(&ref_id) {
                        keep_ids.insert(ref_id);
                    }
                }
            }
        }
        if keep_ids.len() == prev_len { break; }
    }

    // Phase 1c: output all transitively-needed entities (includes keep_ids closure)
    let mut kept: Vec<(u64, &str, &StepValue)> = entities.iter()
        .filter(|(id, _)| keep_ids.contains(id))
        .map(|(&id, r)| (id, r.name.as_str(), &r.params))
        .collect();

    // Phase 2: sort by entity group, then by original ID
    kept.sort_by_key(|(id, name, _)| {
        (classify_group(name), *id)
    });

    // Phase 3: build renumbering map (new IDs start from 1)
    let renumber: HashMap<u64, u64> = kept.iter()
        .enumerate()
        .map(|(new_id, (old_id, _, _))| (*old_id, new_id as u64 + 1))
        .collect();

    // Phase 4: HEADER
    out.push_str(&format_header("AP203_CONFIGURATION_CONTROLLED_3D_DESIGN_OF_MECHANICAL_PARTS_AND_ASSEMBLIES_MIM_LF"));

    // Phase 5: DATA section
    out.push_str("DATA;\n");

    for (new_idx, (_old_id, name, params)) in kept.iter().enumerate() {
        let new_id = new_idx as u64 + 1;
        let formatted = format_param(params);
        // format_param wraps List in (...), so strip outer parens for entity output
        let inner = if formatted.starts_with('(') && formatted.ends_with(')') {
            &formatted[1..formatted.len()-1]
        } else {
            &formatted
        };
        let params_text = rewrite_refs(inner.to_string(), &renumber);
        out.push_str(&format!("#{} = {}({});\n", new_id, name, params_text));
    }

    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    out
}

/// Recursively collect all entity reference IDs from a StepValue tree.
fn collect_refs(value: &StepValue) -> Vec<u64> {
    let mut refs = Vec::new();
    match value {
        StepValue::Ref(id) => refs.push(*id),
        StepValue::List(items) => {
            for item in items {
                refs.extend(collect_refs(item));
            }
        }
        StepValue::Typed(_, inner) => refs.extend(collect_refs(inner)),
        _ => {}
    }
    refs
}

/// Rewrite entity references in formatted param text from old IDs to new IDs.
fn rewrite_refs(text: String, renumber: &HashMap<u64, u64>) -> String {
    if renumber.is_empty() {
        return text;
    }
    // Parse and rebuild: find #NNN patterns and replace
    let mut result = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' && (i == 0 || !bytes[i-1].is_ascii_alphanumeric()) {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(old_id) = std::str::from_utf8(&bytes[start..end])
                    .unwrap_or("0").parse::<u64>()
                {
                    if let Some(&new_id) = renumber.get(&old_id) {
                        result.push_str(&format!("#{}", new_id));
                    } else {
                        // Entity was filtered out → replace with $
                        result.push('$');
                    }
                    i = end;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Format a float for STEP output: avoid trailing zeros, use "0." instead of "0"
pub fn format_real(v: f32) -> String {
    if v == 0.0 {
        "0.".to_string()
    } else if (v.round() - v).abs() < 1e-7 {
        format!("{}.", v.round() as i64)
    } else {
        format!("{:.6}", v)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Write a CARTESIAN_POINT entity.
pub fn write_cartesian_point(id: u64, pt: &[f32; 3]) -> String {
    format!("#{} = CARTESIAN_POINT('', ({}, {}, {}));\n",
        id, format_real(pt[0]), format_real(pt[1]), format_real(pt[2]))
}

/// Write a DIRECTION entity.
pub fn write_direction(id: u64, dir: &[f32; 3]) -> String {
    format!("#{} = DIRECTION('', ({}, {}, {}));\n",
        id, format_real(dir[0]), format_real(dir[1]), format_real(dir[2]))
}

/// Write an AXIS2_PLACEMENT_3D entity.
pub fn write_placement(id: u64, origin_id: u64, axis_id: u64, refdir_id: u64) -> String {
    format!("#{} = AXIS2_PLACEMENT_3D('', #{}, #{}, #{});\n",
        id, origin_id, axis_id, refdir_id)
}

/// Write a PLANE entity.
pub fn write_plane(id: u64, placement_id: u64) -> String {
    format!("#{} = PLANE('', #{});\n", id, placement_id)
}

/// Write a LINE entity.
pub fn write_line(id: u64, point_id: u64, dir_id: u64) -> String {
    format!("#{} = LINE('', #{}, #{});\n", id, point_id, dir_id)
}

/// Write an EDGE_CURVE entity.
pub fn write_edge_curve(id: u64, start_id: u64, end_id: u64,
                         curve_id: u64, same_sense: bool) -> String {
    let ss = if same_sense { ".T." } else { ".F." };
    format!("#{} = EDGE_CURVE('', #{}, #{}, #{}, {});\n",
        id, start_id, end_id, curve_id, ss)
}

/// Write an EDGE_LOOP entity.
pub fn write_edge_loop(id: u64, edge_ids: &[u64]) -> String {
    let refs: Vec<String> = edge_ids.iter().map(|e| format!("#{}", e)).collect();
    format!("#{} = EDGE_LOOP('', ({}));\n", id, refs.join(", "))
}

/// Write a FACE_OUTER_BOUND entity.
pub fn write_face_outer_bound(id: u64, loop_id: u64, orient: bool) -> String {
    let o = if orient { ".T." } else { ".F." };
    format!("#{} = FACE_OUTER_BOUND('', #{}, {});\n", id, loop_id, o)
}

/// Write an ADVANCED_FACE entity.
pub fn write_advanced_face(id: u64, bound_ids: &[u64],
                            surface_id: u64, same_sense: bool) -> String {
    let refs: Vec<String> = bound_ids.iter().map(|b| format!("#{}", b)).collect();
    let ss = if same_sense { ".T." } else { ".F." };
    format!("#{} = ADVANCED_FACE('', ({}), #{}, {});\n",
        id, refs.join(", "), surface_id, ss)
}

/// Write a CLOSED_SHELL entity.
pub fn write_closed_shell(id: u64, face_ids: &[u64]) -> String {
    let refs: Vec<String> = face_ids.iter().map(|f| format!("#{}", f)).collect();
    format!("#{} = CLOSED_SHELL('', ({}));\n", id, refs.join(", "))
}

/// Write a MANIFOLD_SOLID_BREP entity.
pub fn write_manifold_solid_brep(id: u64, shell_id: u64) -> String {
    format!("#{} = MANIFOLD_SOLID_BREP('', #{});\n", id, shell_id)
}

/// Write a CYLINDRICAL_SURFACE entity.
pub fn write_cylindrical_surface(id: u64, placement_id: u64, radius: f32) -> String {
    format!("#{} = CYLINDRICAL_SURFACE('', #{}, {});\n", id, placement_id, format_real(radius))
}

/// Write an ELLIPSE entity.
pub fn write_ellipse(id: u64, placement_id: u64, semi_a: f32, semi_b: f32) -> String {
    format!("#{} = ELLIPSE('', #{}, {}, {});\n", id, placement_id, format_real(semi_a), format_real(semi_b))
}

/// Write a CIRCLE entity.
pub fn write_circle(id: u64, placement_id: u64, radius: f32) -> String {
    format!("#{} = CIRCLE('', #{}, {});\n", id, placement_id, format_real(radius))
}

/// Write a complete STEP file with HEADER and DATA sections.
pub fn write_step_document(header_sections: &[&str], data_entities: &[String]) -> String {
    let mut out = String::new();
    out.push_str("ISO-10303-21;\n");
    out.push_str("HEADER;\n");
    for section in header_sections {
        out.push_str(section);
        out.push('\n');
    }
    out.push_str("ENDSEC;\n");
    out.push_str("DATA;\n");
    for entity in data_entities {
        out.push_str(entity);
    }
    out.push_str("ENDSEC;\n");
    out.push_str("END-ISO-10303-21;\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::parser;

    fn parse_text(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_write_entities_simple() {
        let entities = parse_text(
            "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#2 = CARTESIAN_POINT('', (1., 0., 0.));
#3 = DIRECTION('', (0., 0., 1.));
#10 = LINE('', #1, #3);\
",
        );
        let output = write_entities(&entities);
        assert!(output.contains("CARTESIAN_POINT"));
        assert!(output.contains("LINE"));
        assert!(output.contains("DATA;"));
        assert!(output.contains("ENDSEC;"));
        // Check that points come before lines (sorted by group)
        let pt_pos = output.find("CARTESIAN_POINT").unwrap();
        let line_pos = output.find("LINE").unwrap();
        assert!(pt_pos < line_pos, "points should be output before lines");
    }

    #[test]
    fn test_write_filters_styled_items() {
        let entities = parse_text(
            "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#2 = STYLED_ITEM('', (#3), #4);
#3 = PRESENTATION_STYLE_ASSIGNMENT('', (#5));
#4 = CLOSED_SHELL('', (#6));
#5 = SURFACE_STYLE_USAGE(.BOTH., #31);
#6 = ADVANCED_FACE('', (#8), #9, .T.);
#8 = FACE_OUTER_BOUND('', #11, .T.);
#9 = PLANE('', #12);
#12 = AXIS2_PLACEMENT_3D('', #1, #1, #1);
#20 = LINE('', #1, #1);
#21 = EDGE_CURVE('', #1, #1, #20, .T.);
#11 = EDGE_LOOP('', (#21));
#31 = SURFACE_SIDE_STYLE('', (#32));
#32 = SURFACE_STYLE_FILL_AREA(#33);
#33 = FILL_AREA_STYLE('', #34);
#34 = FILL_AREA_STYLE_COLOUR('', #15);
#15 = COLOUR_RGB('', 0.8, 0.8, 0.8);\
",
        );
        let output = write_entities(&entities);
        // Style entities should be filtered out (no geometry entity references them)
        assert!(!output.contains("STYLED_ITEM"));
        assert!(!output.contains("PRESENTATION_STYLE"));
        assert!(!output.contains("COLOUR_RGB"));
        assert!(!output.contains("FILL_AREA_STYLE"));
        // Geometry entities should be present
        assert!(output.contains("CARTESIAN_POINT"));
        assert!(output.contains("CLOSED_SHELL"));
        assert!(output.contains("PLANE"));
    }

    #[test]
    fn test_rewrite_refs() {
        let mut map = HashMap::new();
        map.insert(10, 1);
        map.insert(20, 2);
        assert_eq!(rewrite_refs("#10,#20".into(), &map), "#1,#2");
        assert_eq!(rewrite_refs("#10=LINE('',#10,#20)".into(), &map), "#1=LINE('',#1,#2)");
        // Unknown refs become $
        assert_eq!(rewrite_refs("#99".into(), &map), "$");
    }
}

#[cfg(test)]
mod entity_writer_tests {
    use super::*;

    #[test]
    fn test_write_cartesian_point() {
        let s = write_cartesian_point(1, &[0.0, 10.0, -5.5]);
        assert!(s.starts_with("#1 = CARTESIAN_POINT('', (0., 10., -5.5)"));
    }

    #[test]
    fn test_write_closed_shell() {
        let s = write_closed_shell(100, &[10, 11, 12]);
        assert!(s.contains("#100 = CLOSED_SHELL('', (#10, #11, #12))"));
    }

    #[test]
    fn test_write_manifold_solid_brep() {
        let s = write_manifold_solid_brep(200, 100);
        assert!(s.contains("#200 = MANIFOLD_SOLID_BREP('', #100)"));
    }

    #[test]
    fn test_write_step_document() {
        let entities = vec![
            write_cartesian_point(1, &[0.0, 0.0, 0.0]),
            write_cartesian_point(2, &[1.0, 0.0, 0.0]),
        ];
        let doc = write_step_document(
            &["FILE_DESCRIPTION(('test'),'2;1');",
              "FILE_NAME('t','t',(''),(''),'','','');",
              "FILE_SCHEMA(('TEST'));"],
            &entities,
        );
        assert!(doc.starts_with("ISO-10303-21;"));
        assert!(doc.contains("HEADER;"));
        assert!(doc.contains("DATA;"));
        assert!(doc.contains("ENDSEC;"));
        assert!(doc.contains("END-ISO-10303-21;"));
        assert!(doc.contains("CARTESIAN_POINT"));
    }
}
