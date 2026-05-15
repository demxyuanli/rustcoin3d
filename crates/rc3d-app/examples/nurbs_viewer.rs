//! NURBS viewer — demonstrates 3D NURBS curves and surfaces.
//!
//! Shows:
//! - Helix curve (degree 3, 16 control points)
//! - Wavy space curve (degree 3, 12 control points)
//! - Wavy NURBS surface (degree 3×3, 6×6 control grid)
//!
//! Curves use adaptive tessellation with angle deflection (tolerance 0.005, angle_tol 0.02).
//! Surface uses DynamicSurface: re-tessellates after camera settles (6 still frames,
//! 1.0 world-unit move threshold). Screen-space criteria: 4px interior, 1.5px silhouette.

use rc3d_app::{App, CameraController, DynamicSurface};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    let mut g = rc3d_scene::SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));

    // Surface center (x=1.5, y=1.0 from grid bounds [-1,4]×[-1,3])
    let surf_center = Vec3::new(1.5, 1.0, 0.0);

    // ── Camera ──────────────────────────────────────────────────────────
    g.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(3.5, 2.5, 10.0),
            surf_center,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // ── Lights ──────────────────────────────────────────────────────────
    g.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -0.5).normalize(),
            color: Vec3::ONE,
            intensity: 1.5,
            light_group: None,
        }),
    );
    g.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.3, -0.1, 0.9).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.4,
            light_group: None,
        }),
    );

    // ── Global material for the surface ─────────────────────────────────
    g.add_child(
        root,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.25, 0.5, 0.8),
            metallic: 0.05,
            roughness: 0.5,
            ..Default::default()
        }),
    );

    // ══════════════════════════════════════════════════════════════════════
    // NURBS Curves (adaptive tessellation with tight tolerance)
    // ══════════════════════════════════════════════════════════════════════

    // ── Helix curve (2 full turns, degree 3) ────────────────────────────
    {
        let cp: Vec<Vec3> = (0..16)
            .map(|i| {
                let t = i as f32 / 15.0;
                let angle = t * std::f32::consts::TAU * 2.0;
                Vec3::new(angle.cos() * 2.0, t * 4.0, angle.sin() * 2.0)
            })
            .collect();
        let curve = rc3d_nurbs::NurbsCurve::from_points(&cp, 3);
        let samples = curve.tessellate_adaptive(0.005, 0.02);
        println!(
            "Helix: {} control pts, {} tessellated samples (angle_tol=0.02), arc={:.2}",
            cp.len(),
            samples.len(),
            curve.arc_length(256),
        );
        add_line_set(&mut g, root, &samples, 2.5);
    }

    // ── Wavy space curve (degree 3) ─────────────────────────────────────
    {
        let cp: Vec<Vec3> = (0..12)
            .map(|i| {
                let t = i as f32 / 11.0;
                Vec3::new(
                    t * 5.0,
                    (t * std::f32::consts::TAU * 3.0).sin() * 1.5,
                    (t * std::f32::consts::TAU * 1.5).cos() * 1.5,
                )
            })
            .collect();
        let curve = rc3d_nurbs::NurbsCurve::from_points(&cp, 3);
        let samples = curve.tessellate_adaptive(0.005, 0.02);
        println!(
            "Wavy curve: {} control pts, {} tessellated samples (angle_tol=0.02), arc={:.2}",
            cp.len(),
            samples.len(),
            curve.arc_length(256),
        );
        add_line_set(&mut g, root, &samples, 1.5);
    }

    // ══════════════════════════════════════════════════════════════════════
    // NURBS Surface (adaptive tessellation + smooth vertex normals)
    // ══════════════════════════════════════════════════════════════════════
    // NURBS Surface — dynamic view-dependent tessellation
    // ══════════════════════════════════════════════════════════════════════
    //
    // Uses DynamicSurface which re-tessellates when the camera moves.
    // Silhouette edges (N·view ≈ 0) get 1.5px tolerance; interior gets 4px.

    let ds = {
        let grid: Vec<Vec<Vec3>> = (0..6)
            .map(|i| {
                let u = i as f32 / 5.0;
                (0..6)
                    .map(|j| {
                        let v = j as f32 / 5.0;
                        let z = (u * std::f32::consts::TAU * 1.5).sin()
                            * (v * std::f32::consts::TAU * 1.5).cos()
                            * 1.5;
                        Vec3::new(u * 5.0 - 1.0, v * 4.0 - 1.0, z)
                    })
                    .collect()
            })
            .collect();
        let surface = rc3d_nurbs::NurbsSurface::from_points_grid(&grid, 3, 3);

        // Initial camera parameters
        let orbit_center = surf_center;
        let orbit_dist = 12.0;
        let pitch = 0.4f32;
        let yaw = 0.0f32;
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cy = yaw.cos();
        let sy = yaw.sin();
        let eye = orbit_center + Vec3::new(cp * sy, sp, cp * cy) * orbit_dist;
        let view = Mat4::look_at_rh(eye, orbit_center, Vec3::Y);
        let aspect = 800.0 / 600.0;
        let proj = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
        let mvp = proj * view;

        // Primary directional light direction for terminator detection
        let light_dir = Vec3::new(-1.0, -1.0, -0.5).normalize();

        DynamicSurface::new(
            &mut g, root, surface, eye, light_dir, &mvp, (800.0, 600.0),
            6,      // settle_frames
            1.0,    // move_threshold
            8.0,    // max_px interior
            4.0,    // silhouette_px
            2.5,    // terminator_px — tightest at N·L → 0
            0.05,   // angle_tol ≈ 3°
            5,      // max_depth
        )
    };

    // ── Run ─────────────────────────────────────────────────────────────
    let orbit = CameraController::new(surf_center, 12.0);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut App::new(g).with_camera_controller(orbit).with_dynamic_surface(ds))
        .expect("event loop");
}

/// Add a line set under a Separator (Coin3D pattern: state isolation).
fn add_line_set(
    g: &mut rc3d_scene::SceneGraph,
    parent: rc3d_core::NodeId,
    points: &[Vec3],
    line_width: f32,
) -> rc3d_core::NodeId {
    let sep = g.add_child(parent, NodeData::Separator(SeparatorNode));
    g.add_child(
        sep,
        NodeData::Coordinate3(Coordinate3Node::from_points(points.to_vec())),
    );
    let n = points.len() as i32;
    let indices: Vec<i32> = (0..n - 1).flat_map(|i| [i, i + 1]).collect();
    g.add_child(
        sep,
        NodeData::IndexedLineSet(IndexedLineSetNode {
            coord_index: indices,
            line_width,
        }),
    );
    sep
}

