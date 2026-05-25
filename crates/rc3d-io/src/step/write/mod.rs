//! STEP writer: ISO 10303-21 serialization.
//!
//! Two paths:
//! - `write_step_from_graph`: SceneGraph → STEP (mesh export with coplanar merging)
//! - `write_step_from_entities`: EntityIndex → STEP (cleaned pass-through)

pub mod format;
pub mod entity_to_step;
pub mod scene_to_step;

use super::parser::EntityIndex;
use rc3d_scene::SceneGraph;

/// Write a STEP file from a SceneGraph (Path A: export mode).
/// Preserves assembly hierarchy, merges coplanar triangles into faces.
#[allow(dead_code)] // Will be implemented in scene_to_step.rs
pub fn write_step_from_graph(_graph: &SceneGraph) -> Result<String, String> {
    scene_to_step::write_scene(_graph)
}

/// Write a STEP file from an EntityIndex (Path B: cleaned pass-through).
/// Filters presentation/metadata entities, renumbers IDs, canonicalizes output.
pub fn write_step_from_entities(entities: &EntityIndex) -> String {
    entity_to_step::write_entities(entities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;

    #[test]
    fn test_write_step_from_entities_roundtrip() {
        let input = "\
#1 = CARTESIAN_POINT('', (0., 0., 0.));
#2 = CARTESIAN_POINT('', (10., 0., 0.));
#3 = CARTESIAN_POINT('', (10., 10., 0.));
#4 = CARTESIAN_POINT('', (0., 10., 0.));
#5 = DIRECTION('', (0., 0., 1.));
#10 = LINE('', #1, #2);
#11 = LINE('', #2, #3);
#12 = LINE('', #3, #4);
#13 = LINE('', #4, #1);
#14 = EDGE_CURVE('', #1, #2, #10, .T.);
#15 = EDGE_CURVE('', #2, #3, #11, .T.);
#16 = EDGE_CURVE('', #3, #4, #12, .T.);
#17 = EDGE_CURVE('', #4, #1, #13, .T.);
#18 = EDGE_LOOP('', (#14, #15, #16, #17));
#19 = FACE_OUTER_BOUND('', #18, .T.);
#20 = AXIS2_PLACEMENT_3D('', #1, #5, #2);
#21 = PLANE('', #20);
#22 = ADVANCED_FACE('', (#19), #21, .T.);
#23 = CLOSED_SHELL('', (#22));
#24 = MANIFOLD_SOLID_BREP('', #23);\
";
        let full_input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            input
        );
        let entities = parser::parse_exchange(&full_input).unwrap().entities;
        let output = write_step_from_entities(&entities);

        // Must be valid STEP text
        assert!(output.contains("ISO-10303-21;"));
        assert!(output.contains("HEADER;"));
        assert!(output.contains("DATA;"));
        assert!(output.contains("ENDSEC;"));
        assert!(output.contains("END-ISO-10303-21;"));

        // All geometry entities should be present
        for name in &["CARTESIAN_POINT", "CLOSED_SHELL", "PLANE", "ADVANCED_FACE"] {
            assert!(output.contains(name), "output should contain {}", name);
        }

        // Output should be parseable (round-trip)
        let parsed = parser::parse_exchange(&output);
        assert!(parsed.is_ok(), "round-trip parse should succeed: {:?}", parsed.err());
        let roundtrip = parsed.unwrap();
        assert!(roundtrip.entities.len() >= 15, "round-trip should have most entities");
    }
}
