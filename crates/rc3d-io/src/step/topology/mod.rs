//! B-rep topology traversal: Shell -> Face -> Loop -> Edge.

mod collect;
mod placement;
mod colors;
mod helpers;

pub use collect::{
    collect_solid_models, collect_shells, collect_shell_faces, global_tolerance,
    length_unit_scale,
};
pub use placement::{
    resolve_point, resolve_axis1_placement, resolve_sweep_axis, resolve_placement,
    resolve_direction, resolve_direction_public,
};
pub use colors::collect_face_colors;

pub use collect::{StepFace, StepLoop, StepEdge, StepShell, StepSolidModel};

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;

    fn make_exchange(data_section: &str) -> parser::EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data_section
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_advanced_face_resolution() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = DIRECTION('', (0.0, 0.0, 1.0));
#6 = AXIS2_PLACEMENT_3D('', #1, #5, #2);
#7 = PLANE('', #6);
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #7, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        assert!(!shells.is_empty());
        let face = &shells[0].faces[0];
        assert!(face.surface_id.is_some());
        assert!(face.same_sense);
    }

    #[test]
    fn test_open_shell_collection() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = OPEN_SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        assert_eq!(shells.len(), 1);
        assert_eq!(shells[0].faces.len(), 1);
    }

    #[test]
    fn test_oriented_face_reversed_flag() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = DIRECTION('', (0.0, 0.0, 1.0));
#6 = AXIS2_PLACEMENT_3D('', #1, #5, #2);
#7 = PLANE('', #6);
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #7, .T.);
#17 = ORIENTED_FACE('', *, *, #16, .F.);
#18 = CLOSED_SHELL('', (#17));
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        assert_eq!(shells.len(), 1);
        assert_eq!(shells[0].faces.len(), 1);
        assert!(!shells[0].faces[0].oriented_forward);
    }

    #[test]
    fn test_brep_with_voids_extracts_outer() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#18 = BREP_WITH_VOIDS('', #17, ());
#20 = LINE('', #1, #2);\
",
        );

        let shells = collect_shells(&entities);
        // Should extract outer shell #17 from BREP_WITH_VOIDS #18
        assert!(!shells.is_empty());
        assert!(shells.iter().any(|s| s.id == 17));
    }

    #[test]
    fn test_brep_with_voids_extracts_void_shells() {
        let entities = make_exchange(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = CARTESIAN_POINT('', (2.0, 2.0, 0.0));
#6 = CARTESIAN_POINT('', (8.0, 2.0, 0.0));
#7 = CARTESIAN_POINT('', (8.0, 8.0, 0.0));
#8 = CARTESIAN_POINT('', (2.0, 8.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#30 = EDGE_CURVE('', #5, #6, #20, .T.);
#31 = EDGE_CURVE('', #6, #7, #20, .T.);
#32 = EDGE_CURVE('', #7, #8, #20, .T.);
#33 = EDGE_CURVE('', #8, #5, #20, .T.);
#34 = EDGE_LOOP('', (#30, #31, #32, #33));
#35 = FACE_OUTER_BOUND('', #34, .T.);
#36 = FACE_SURFACE('', (#35));
#37 = CLOSED_SHELL('', (#36));
#18 = BREP_WITH_VOIDS('', #17, (#37));
#20 = LINE('', #1, #2);\
",
        );

        let models = collect_solid_models(&entities);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].outer.id, 17);
        assert_eq!(models[0].voids.len(), 1);
        assert_eq!(models[0].voids[0].id, 37);
    }
}
