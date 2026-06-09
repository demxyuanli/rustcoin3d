use std::collections::HashMap;

use crate::store::BRepStore;
use crate::topo::{EdgeKey, ShellKey};
use crate::mesh::config::BRepMeshConfig;
use crate::mesh::edge_disc::{discretize_edge, EdgePolygon};
use crate::mesh::report::{apply_relative_deflection, shell_bbox_diagonal, ShellMeshReport};
use crate::mesh::same_param::apply_same_parameter;

/// Returns `None` when the shell key does not exist in the store.
pub(crate) fn init_shell_mesh(
    shell_key: ShellKey,
    reg: &BRepStore,
    config: &BRepMeshConfig,
) -> Option<(BRepMeshConfig, f32, ShellMeshReport, HashMap<EdgeKey, EdgePolygon>)> {
    let shell_diag = shell_bbox_diagonal(shell_key, reg);
    let mut scaled_config = config.clone();
    apply_relative_deflection(&mut scaled_config, shell_diag);
    if scaled_config.relative_deflection > 0.0 && shell_diag > 0.0 {
        scaled_config.face.shell_min_size =
            shell_diag * scaled_config.face.min_size_relative;
    }

    let shell = reg.shells.get(shell_key)?;

    let report = ShellMeshReport {
        face_count: shell.faces.len(),
        shell_diag,
        ..Default::default()
    };

    let _te = std::time::Instant::now();
    let shell_edges = crate::topo_iter::iter_edges_of_shell(shell_key, reg);
    log::debug!("[mesh] shell {:?}: {} edges to discretize", shell_key, shell_edges.len());
    let mut edge_polygons: HashMap<EdgeKey, EdgePolygon> =
        HashMap::with_capacity(shell_edges.len());
    let edge_cfg = &scaled_config.edge;
    for ek in &shell_edges {
        edge_polygons.insert(*ek, discretize_edge(*ek, reg, edge_cfg));
    }
    log::debug!(
        "[mesh] edge discretization: {:.1}s for {} edges",
        _te.elapsed().as_secs_f32(),
        shell_edges.len()
    );

    if scaled_config.same_parameter_tol > 0.0 {
        apply_same_parameter(
            &mut edge_polygons,
            reg,
            scaled_config.same_parameter_tol,
        );
    }

    Some((scaled_config, shell_diag, report, edge_polygons))
}
