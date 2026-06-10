//! STEP meshing and ASCII STL export (shared by `export_mesh` example and diagnostics).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rc3d_core::math::{Mat4, Vec3};
use rc3d_shape::{EmitPlanOptions, SceneEmitPlan, ShapeDocument};

use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::mesh::face_uv::UvSource;
use crate::step::brep::mesh::report::ShellMeshReport;
use crate::step::brep::mesh::{BRepMeshConfig, ShellMeshOutput};
use rc3d_shape::topo::{BRepFace, FaceKey};
use crate::step::brep::BRepStore;
use crate::step::brep::heal::HealLevel;
use crate::step::import_options::StepImportOptions;
use crate::step::import_step_file_with_options;
use crate::step::mesh_result::MeshResult;
use crate::{parse_stl_triangles, write_ascii_stl, StlError};

#[derive(Debug, thiserror::Error)]
pub enum MeshExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("STL error: {0}")]
    Stl(#[from] StlError),
    #[error("STEP error: {0}")]
    Step(String),
    #[error("Unsupported input format: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct MeshExportOptions {
    pub import_options: StepImportOptions,
    pub mesh_config: BRepMeshConfig,
    pub heal: bool,
    pub heal_level: HealLevel,
    pub heal_passes: usize,
}

impl Default for MeshExportOptions {
    fn default() -> Self {
        Self {
            import_options: StepImportOptions::default(),
            mesh_config: BRepMeshConfig::default(),
            heal: true,
            heal_level: HealLevel::Standard,
            heal_passes: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepMeshResult {
    pub mesh: MeshResult,
    pub report: ShellMeshReport,
    pub solid_count: usize,
    pub instance_count: usize,
}

#[derive(Debug, Clone)]
pub struct ExportSummary {
    pub input: PathBuf,
    pub output: PathBuf,
    pub tri_count: usize,
    pub vert_count: usize,
    pub meshed_faces: usize,
    pub total_faces: usize,
    pub grid_fallback_rate: f32,
    pub per_face_files: usize,
}

pub fn default_stl_output(input: &Path) -> PathBuf {
    input.with_extension("stl")
}

/// Append a tessellated solid mesh under an assembly instance transform.
pub fn append_mesh_with_transform(dst: &mut MeshResult, src: &MeshResult, world: Mat4) {
    let offset = dst.vertices.len() as i32;
    for &v in &src.vertices {
        let p = world * v.extend(1.0);
        dst.vertices.push(Vec3::new(p.x, p.y, p.z));
    }
    // Normal transform: use inverse-transpose for correct handling of non-uniform scaling.
    // OCC: gp_Trsf — applies rotation + scale correctly via inv-transpose 3x3.
    let normal_mat = world.inverse().transpose();
    for &n in &src.normals {
        let t = (normal_mat * n.extend(0.0)).truncate();
        dst.normals.push(t.normalize_or_zero());
    }
    for &idx in &src.indices {
        if idx == -1 {
            dst.indices.push(-1);
        } else {
            dst.indices.push(idx + offset);
        }
    }
}

pub fn mesh_step_file(path: &Path, options: &MeshExportOptions) -> Result<StepMeshResult, MeshExportError> {
    let (document, plan) = build_export_emit_plan(path, options)?;
    let (combined, report) = merge_emit_plan_meshes(&plan);
    let solid_count = plan.mesh_table.len();
    let instance_count = plan.instances.len();
    let _ = document;

    Ok(StepMeshResult {
        mesh: combined,
        report,
        solid_count,
        instance_count,
    })
}

pub fn export_step_to_ascii_stl(
    input: &Path,
    output: &Path,
    options: &MeshExportOptions,
) -> Result<ExportSummary, MeshExportError> {
    let meshed = mesh_step_file(input, options)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_ascii_stl(output, &meshed.mesh.vertices, &meshed.mesh.indices)?;

    Ok(summary_from_mesh(input, output, &meshed.mesh, &meshed.report, 0))
}

pub fn export_step_per_face_ascii_stl(
    input: &Path,
    output_dir: &Path,
    options: &MeshExportOptions,
) -> Result<ExportSummary, MeshExportError> {
    let (document, plan) = build_export_emit_plan(input, options)?;

    std::fs::create_dir_all(output_dir)?;

    let mut file_count = 0usize;

    for cached in plan.mesh_table.values() {
        let out = ShellMeshOutput {
            mesh: cached.mesh.clone(),
            report: cached.report.clone(),
        };
        write_per_face_stl(&out, &document.store, output_dir, &mut file_count)?;
    }

    let (combined, report) = merge_emit_plan_meshes(&plan);

    Ok(summary_from_mesh(
        input,
        output_dir,
        &combined,
        &report,
        file_count,
    ))
}

pub fn convert_stl_to_ascii(input: &Path, output: &Path) -> Result<ExportSummary, MeshExportError> {
    let data = std::fs::read(input)?;
    let tris = parse_stl_triangles(&data)?;
    let mut mesh = MeshResult::default();
    for tri in tris {
        let base = mesh.vertices.len() as i32;
        mesh.vertices.push(Vec3::from(tri.vertices[0]));
        mesh.vertices.push(Vec3::from(tri.vertices[1]));
        mesh.vertices.push(Vec3::from(tri.vertices[2]));
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, -1]);
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_ascii_stl(output, &mesh.vertices, &mesh.indices)?;
    Ok(ExportSummary {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        tri_count: mesh.indices.len() / 4,
        vert_count: mesh.vertices.len(),
        meshed_faces: 0,
        total_faces: 0,
        grid_fallback_rate: 0.0,
        per_face_files: 0,
    })
}

pub fn export_file_to_ascii_stl(
    input: &Path,
    output: &Path,
    options: &MeshExportOptions,
    per_face_dir: Option<&Path>,
) -> Result<ExportSummary, MeshExportError> {
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "step" | "stp" => {
            let meshed = mesh_step_file(input, options)?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_ascii_stl(output, &meshed.mesh.vertices, &meshed.mesh.indices)?;

            let mut summary = summary_from_mesh(
                input, output, &meshed.mesh, &meshed.report, 0,
            );

            if let Some(dir) = per_face_dir {
                let face_summary = export_step_per_face_ascii_stl(input, dir, options)?;
                summary.per_face_files = face_summary.per_face_files;
            }

            Ok(summary)
        }
        "stl" => convert_stl_to_ascii(input, output),
        other => Err(MeshExportError::Unsupported(other.to_string())),
    }
}

fn import_options_for_export(options: &MeshExportOptions) -> StepImportOptions {
    let mut io = options.import_options.clone();
    io.skip_visualization = true;
    if options.heal {
        io.heal_level = options.heal_level;
    }
    io
}

fn emit_plan_options_for_export(
    options: &MeshExportOptions,
    heal_skip_faces: &[FaceKey],
) -> EmitPlanOptions {
    EmitPlanOptions {
        mesh_config: options.mesh_config.clone(),
        heal_skip_faces: heal_skip_faces.to_vec(),
        default_material: Default::default(),
        explode_offsets: Default::default(),
    }
}

fn build_export_emit_plan(
    path: &Path,
    options: &MeshExportOptions,
) -> Result<(ShapeDocument, SceneEmitPlan), MeshExportError> {
    let import_options = import_options_for_export(options);
    let import_result = import_step_file_with_options(path, &import_options)
        .map_err(|e| MeshExportError::Step(format!("{e}")))?;
    let mut document = import_result.document;
    let mut plan_options = emit_plan_options_for_export(
        options,
        &import_result.report.heal_skip_face_keys,
    );
    document
        .store
        .tolerance
        .apply_to_mesh_config(&mut plan_options.mesh_config);
    let plan = document
        .build_emit_plan(&plan_options)
        .map_err(|e| MeshExportError::Step(format!("{e}")))?;
    Ok((document, plan))
}

fn merge_emit_plan_meshes(plan: &SceneEmitPlan) -> (MeshResult, ShellMeshReport) {
    let mut combined = MeshResult::default();
    let mut report = ShellMeshReport::default();
    let mut reported_slots = HashSet::new();
    for inst in &plan.instances {
        let Some(cached) = plan.mesh_table.get(&inst.mesh_slot) else {
            continue;
        };
        append_mesh_with_transform(&mut combined, &cached.mesh, inst.world_transform);
        if reported_slots.insert(inst.mesh_slot) {
            report = merge_reports(&report, &cached.report);
        }
    }
    (combined, report)
}

fn summary_from_mesh(
    input: &Path,
    output: &Path,
    mesh: &MeshResult,
    report: &ShellMeshReport,
    per_face_files: usize,
) -> ExportSummary {
    ExportSummary {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        tri_count: mesh.indices.len() / 4,
        vert_count: mesh.vertices.len(),
        meshed_faces: report.meshed_faces,
        total_faces: report.face_count,
        grid_fallback_rate: report.grid_fallback_rate(),
        per_face_files,
    }
}

fn face_triangle_indices(mesh: &MeshResult, fs: &crate::step::brep::mesh::report::FaceMeshStats) -> Vec<i32> {
    let start = fs.first_tri * 4;
    let end = (start + fs.tri_count * 4).min(mesh.indices.len());
    if start >= mesh.indices.len() {
        Vec::new()
    } else {
        mesh.indices[start..end].to_vec()
    }
}

fn face_stl_name(
    face_key: FaceKey,
    face: &BRepFace,
    fs: &crate::step::brep::mesh::report::FaceMeshStats,
) -> String {
    let kind = surface_kind_label(&face.surface);
    let uv = match fs.uv_source {
        UvSource::Pcurve => "pc",
        _ => "syn",
    };
    let fk_str = format!("{face_key:?}");
    let fk_id = fk_str
        .trim_start_matches("FaceKey(")
        .trim_end_matches(')');
    format!("face_{fk_id}_{kind}{uv}_{}tris.stl", fs.tri_count)
}

fn merge_reports(acc: &ShellMeshReport, shell: &ShellMeshReport) -> ShellMeshReport {
    let mut merged = acc.clone();
    merged.face_count += shell.face_count;
    merged.meshed_faces += shell.meshed_faces;
    merged.grid_fallback_count += shell.grid_fallback_count;
    merged.cdt_constraint_failure_count += shell.cdt_constraint_failure_count;
    merged.total_tris += shell.total_tris;
    merged.shell_diag = merged.shell_diag.max(shell.shell_diag);
    merged.max_equiv_edge_weld_gap = merged
        .max_equiv_edge_weld_gap
        .max(shell.max_equiv_edge_weld_gap);
    merged.faces.extend_from_slice(&shell.faces);
    merged
}

/// Write per-face STL files from a shell mesh output.
fn write_per_face_stl(
    out: &ShellMeshOutput,
    reg: &BRepStore,
    output_dir: &Path,
    file_count: &mut usize,
) -> Result<(), MeshExportError> {
    for fs in &out.report.faces {
        if fs.tri_count == 0 {
            continue;
        }
        let face_idx = face_triangle_indices(&out.mesh, fs);
        if face_idx.is_empty() {
            continue;
        }
        let face = reg.faces.get(fs.face_key).expect("face");
        let name = face_stl_name(fs.face_key, face, fs);
        let path = output_dir.join(name);
        write_ascii_stl(&path, &out.mesh.vertices, &face_idx)?;
        *file_count += 1;
    }
    Ok(())
}

fn surface_kind_label(surface: &SurfaceGeom) -> &'static str {
    match surface {
        SurfaceGeom::Revolution { .. } => "Rev",
        SurfaceGeom::BSpline(_) => "BSpl",
        SurfaceGeom::Offset { .. } => "Offs",
        SurfaceGeom::Plane { .. } => "Plane",
        SurfaceGeom::Cylinder { .. } => "Cyl",
        SurfaceGeom::Cone { .. } => "Cone",
        SurfaceGeom::Sphere { .. } => "Sph",
        SurfaceGeom::Torus { .. } => "Tor",
        SurfaceGeom::Extrusion { .. } => "Extr",
    }
}
