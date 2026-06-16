//! Shared formatting helpers for ISO 10303-21 output.

use rc3d_core::math::Real;
use super::super::value::StepValue;

/// Format a StepValue as ISO 10303-21 parameter text.
pub fn format_param(value: &StepValue) -> String {
    match value {
        StepValue::Omitted => "$".to_string(),
        StepValue::Integer(v) => v.to_string(),
        StepValue::Real(v) => format_real(*v),
        StepValue::String(s) => format_string(s),
        StepValue::Enum(s) => s.clone(),
        StepValue::Ref(id) => format!("#{}", id),
        StepValue::Typed(tag, inner) => format!("{}({})", tag, format_param(inner)),
        StepValue::List(items) => {
            let inner: Vec<String> = items.iter().map(format_param).collect();
            format!("({})", inner.join(","))
        }
    }
}

/// Format f64 with up to 6 significant digits, stripping trailing zeros.
fn format_real(v: f64) -> String {
    if v == 0.0 {
        return "0.".to_string();
    }
    let abs = v.abs();
    if abs < 1e-7 {
        return format!("{:.6E}", v);
    }
    let mut s = format!("{:.6}", v);
    // Strip trailing zeros
    while s.ends_with('0') && s.contains('.') {
        s.pop();
    }
    if s.ends_with('.') {
        s.push('0'); // Keep one zero after decimal per STEP convention: 1.0
    }
    s
}

fn format_string(s: &str) -> String {
    if s.is_empty() {
        "''".to_string()
    } else {
        // Escape single quotes per ISO 10303-21: '' → '
        let escaped = s.replace('\'', "''");
        format!("'{}'", escaped)
    }
}

/// Generate a standard ISO 10303-21 HEADER section.
pub fn format_header(schema: &str) -> String {
    format!(
        "ISO-10303-21;\n\
         HEADER;\n\
         FILE_DESCRIPTION(('rustcoin3d export'),'2;1');\n\
         FILE_NAME('export','{}',('rustcoin3d'),(''),'','');\n\
         FILE_SCHEMA(('{}'));\n\
         ENDSEC;\n",
        chrono_like_date(),
        schema,
    )
}

fn chrono_like_date() -> String {
    // ISO 8601 date without external crate
    use std::time::SystemTime;
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            let days = secs / 86400;
            // Rough: count years from 1970
            let mut y = 1970i64;
            let mut remaining = days as i64;
            loop {
                let year_days = if is_leap(y) { 366 } else { 365 };
                if remaining < year_days { break; }
                remaining -= year_days;
                y += 1;
            }
            let months_days = if is_leap(y) {
                [31,29,31,30,31,30,31,31,30,31,30,31]
            } else {
                [31,28,31,30,31,30,31,31,30,31,30,31]
            };
            let mut m = 1usize;
            for &md in &months_days {
                if remaining < md as i64 { break; }
                remaining -= md as i64;
                m += 1;
            }
            let d = remaining + 1;
            format!("{:04}-{:02}-{:02}T00:00:00", y, m, d)
        }
        Err(_) => "2024-01-01T00:00:00".to_string(),
    }
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Entity type groups for sorting output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntityGroup {
    Points = 0,
    Directions = 1,
    Placements = 2,
    Curves = 3,
    Surfaces = 4,
    Topology = 5,
    Assembly = 6,
    Other = 7,
}

/// Classify an entity name into an output group for ordering.
pub fn classify_group(name: &str) -> EntityGroup {
    match name {
        "CARTESIAN_POINT" | "VERTEX_POINT" => EntityGroup::Points,
        "DIRECTION" | "VECTOR" => EntityGroup::Directions,
        "AXIS2_PLACEMENT_3D" | "AXIS2_PLACEMENT_2D" | "AXIS1_PLACEMENT" => EntityGroup::Placements,
        "LINE" | "CIRCLE" | "ELLIPSE" | "POLYLINE"
        | "B_SPLINE_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE"
        | "PCURVE" | "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE"
        | "TRIMMED_CURVE" | "COMPOSITE_CURVE" | "COMPOSITE_CURVE_SEGMENT"
        | "OFFSET_CURVE_3D" => EntityGroup::Curves,
        "PLANE" | "CYLINDRICAL_SURFACE" | "CONICAL_SURFACE" | "SPHERICAL_SURFACE"
        | "TOROIDAL_SURFACE"
        | "B_SPLINE_SURFACE" | "B_SPLINE_SURFACE_WITH_KNOTS" | "RATIONAL_B_SPLINE_SURFACE"
        | "SURFACE_OF_REVOLUTION" | "SURFACE_OF_LINEAR_EXTRUSION"
        | "OFFSET_SURFACE" | "RECTANGULAR_TRIMMED_SURFACE"
        | "CURVE_BOUNDED_SURFACE" | "BOUNDED_SURFACE" => EntityGroup::Surfaces,
        "EDGE_CURVE" | "ORIENTED_EDGE" | "EDGE_LOOP" | "POLY_LOOP"
        | "FACE_OUTER_BOUND" | "FACE_BOUND"
        | "ADVANCED_FACE" | "FACE_SURFACE" | "FACE"
        | "CLOSED_SHELL" | "OPEN_SHELL" | "SHELL"
        | "ORIENTED_CLOSED_SHELL" | "ORIENTED_OPEN_SHELL"
        | "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS"
        | "SHELL_BASED_SURFACE_MODEL" => EntityGroup::Topology,
        "PRODUCT" | "PRODUCT_DEFINITION" | "PRODUCT_DEFINITION_FORMATION"
        | "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE"
        | "PRODUCT_DEFINITION_SHAPE" | "SHAPE_DEFINITION_REPRESENTATION"
        | "SHAPE_REPRESENTATION" | "SHAPE_REPRESENTATION_RELATIONSHIP"
        | "ADVANCED_BREP_SHAPE_REPRESENTATION"
        | "NEXT_ASSEMBLY_USAGE_OCCURRENCE" | "ITEM_DEFINED_TRANSFORMATION"
        | "PRODUCT_DEFINITION_CONTEXT" | "PRODUCT_CONTEXT"
        | "APPLICATION_CONTEXT" | "APPLICATION_PROTOCOL_DEFINITION" => EntityGroup::Assembly,
        _ => EntityGroup::Other,
    }
}

/// Entity name whitelist — only these types pass through Path B.
pub fn is_geometry_entity(name: &str) -> bool {
    matches!(classify_group(name), EntityGroup::Points | EntityGroup::Directions
        | EntityGroup::Placements | EntityGroup::Curves | EntityGroup::Surfaces
        | EntityGroup::Topology | EntityGroup::Assembly)
        && !matches!(name,
            "PRODUCT_CONTEXT" | "APPLICATION_CONTEXT" | "APPLICATION_PROTOCOL_DEFINITION"
            | "PRODUCT_RELATED_PRODUCT_CATEGORY" | "PRODUCT_DEFINITION_CONTEXT"
            | "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
            | "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"
            | "SHAPE_REPRESENTATION_RELATIONSHIP"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_real_integer() {
        assert_eq!(format_real(0.0), "0.");
        assert_eq!(format_real(1.0), "1.0");
        assert_eq!(format_real(1.5), "1.5");
        assert_eq!(format_real(3.1415926535).starts_with("3.14"), true);
        assert_eq!(format_real(2.0), "2.0");
    }

    #[test]
    fn test_format_param_primitives() {
        assert_eq!(format_param(&StepValue::Integer(42)), "42");
        assert_eq!(format_param(&StepValue::Real(3.14)), "3.14");
        assert_eq!(format_param(&StepValue::String("".into())), "''");
        assert_eq!(format_param(&StepValue::String("hello".into())), "'hello'");
        assert_eq!(format_param(&StepValue::Enum(".T.".into())), ".T.");
        assert_eq!(format_param(&StepValue::Omitted), "$");
        assert_eq!(format_param(&StepValue::Ref(10)), "#10");
    }

    #[test]
    fn test_format_param_list() {
        let list = StepValue::List(vec![
            StepValue::Integer(1),
            StepValue::Real(2.0),
        ]);
        assert_eq!(format_param(&list), "(1,2.0)");
    }

    #[test]
    fn test_format_param_typed() {
        let typed = StepValue::Typed(
            "LENGTH_MEASURE".into(),
            Box::new(StepValue::Real(0.01)),
        );
        assert_eq!(format_param(&typed), "LENGTH_MEASURE(0.01)");
    }

    #[test]
    fn test_classify_groups() {
        assert_eq!(classify_group("CARTESIAN_POINT"), EntityGroup::Points);
        assert_eq!(classify_group("CLOSED_SHELL"), EntityGroup::Topology);
        assert_eq!(classify_group("PRODUCT"), EntityGroup::Assembly);
        assert_eq!(classify_group("UNKNOWN_TYPE"), EntityGroup::Other);
    }
}
