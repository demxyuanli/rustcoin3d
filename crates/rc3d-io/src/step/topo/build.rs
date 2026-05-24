use super::vertex::VertexRegistry;
use super::edge::{EdgeId, EdgeRegistry};
use super::shape::{TopoFace, TopoLoop, TopoShell, ShapeId};
use crate::step::parser::EntityIndex;
use crate::step::topology::StepShell;

/// Result of building shared topology from STEP entities.
pub struct TopoBuildResult {
    pub shells: Vec<TopoShell>,
    pub vertices: VertexRegistry,
    pub edges: EdgeRegistry,
}

/// Build shared-topology shells from the flat STEP topology.
/// Each unique vertex position maps to one VertexId.
/// Each unique (start,end) vertex pair maps to one EdgeId.
pub fn build_shared_topology(
    shells: &[StepShell],
    _entities: &EntityIndex,
) -> TopoBuildResult {
    let mut vertices = VertexRegistry::new();
    let mut edges = EdgeRegistry::new();

    let topo_shells: Vec<TopoShell> = shells.iter().enumerate().map(|(si, shell)| {
        let topo_faces: Vec<TopoFace> = shell.faces.iter().enumerate().map(|(fi, face)| {
            let mut topo_loops = Vec::new();

            for bloop in &face.bounds {
                let loop_edges: Vec<(EdgeId, bool)> = bloop.edges.iter().map(|edge| {
                    let v_start = vertices.insert(edge.start);
                    let v_end = vertices.insert(edge.end);
                    let eid = edges.insert(v_start, v_end, edge.curve_id, edge.tolerance);
                    (eid, edge.reversed)
                }).collect();

                if !loop_edges.is_empty() {
                    topo_loops.push(TopoLoop { edges: loop_edges });
                }
            }

            let outer_loop = topo_loops.first().cloned()
                .unwrap_or(TopoLoop { edges: Vec::new() });
            let inner_loops = if topo_loops.len() > 1 {
                topo_loops[1..].to_vec()
            } else {
                Vec::new()
            };

            TopoFace {
                id: ShapeId::Face((si * 1000 + fi) as u32),
                outer_loop,
                inner_loops,
                surface_entity_id: face.surface_id,
                same_sense: face.same_sense,
            }
        }).collect();

        TopoShell {
            id: ShapeId::Shell(si as u32),
            faces: topo_faces,
            is_closed: true, // conservative default
        }
    }).collect();

    TopoBuildResult { shells: topo_shells, vertices, edges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::parser;
    use crate::step::topology;

    fn make_entities(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_shared_vertices_across_adjacent_faces() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = CARTESIAN_POINT('', (0.0, 0.0, 10.0));
#10 = EDGE_CURVE('', #1, #2, #30, .T.);
#11 = EDGE_CURVE('', #2, #3, #30, .T.);
#12 = EDGE_CURVE('', #3, #4, #30, .T.);
#13 = EDGE_CURVE('', #4, #1, #30, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #40, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = EDGE_CURVE('', #1, #2, #31, .T.);
#21 = EDGE_CURVE('', #2, #5, #31, .T.);
#22 = EDGE_CURVE('', #5, #1, #31, .T.);
#23 = EDGE_LOOP('', (#20, #21, #22));
#24 = FACE_OUTER_BOUND('', #23, .T.);
#25 = ADVANCED_FACE('', (#24), #41, .T.);
#26 = CLOSED_SHELL('', (#25));
#30 = LINE('', #1, #2);
#31 = LINE('', #1, #2);
#40 = PLANE('', #50);
#41 = PLANE('', #51);
#50 = AXIS2_PLACEMENT_3D('', #1, #60, #2);
#51 = AXIS2_PLACEMENT_3D('', #1, #61, #2);
#60 = DIRECTION('', (0.0, 0.0, 1.0));
#61 = DIRECTION('', (1.0, 0.0, 0.0));
",
        );
        let shells = topology::collect_shells(&entities);
        assert!(shells.len() >= 2, "should have at least 2 shells");
        let result = build_shared_topology(&shells, &entities);
        // Vertices #1 and #2 are shared between the two shells' faces
        // They should be deduplicated in the registry
        let total_verts: usize = shells.iter()
            .flat_map(|s| s.faces.iter())
            .flat_map(|f| f.bounds.iter())
            .flat_map(|l| l.edges.iter())
            .count() * 2; // each edge has start+end
        // Actual unique vertices should be less (shared #1 and #2 across shells)
        assert!(result.vertices.len() <= 5,
            "vertices should be deduplicated, got {} unique from {} total refs",
            result.vertices.len(), total_verts);
    }
}
