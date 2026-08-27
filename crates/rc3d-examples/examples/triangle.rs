//! Triangle example — named Visual Style catalog on subtrees.
//!
//! Left to right: Wireframe, HardEdges, Perimeter, Adjacent, Silhouette, HiddenLine.
//! Ground: large receiver plane under the objects (inherits global DisplayMode)
//!
//! Usage: cargo run -p rc3d-examples --example triangle
//! Optional: --screenshot writes target/triangle_drawstyle.png then keeps the window.
//!   --screenshot-path <file>  override output path
//!   --screenshot-exit         exit after writing the PNG
//!   --global-wireframe        set global DisplayMode to Wireframe (subtree Shaded still filled)
//!   --quad                    HOOPS four-view pack (Persp/Top/Front/Right)
//!   --hlr-svg                 write target/triangle_hlr.svg after one frame
//!   --hlr-svg-path <file>     override SVG output path

use rc3d_core::math::Vec3;
use rc3d_core::{DisplayMode, VisualStyle, VisualStyleLibrary};
use rc3d_engine_api::Engine;
use rc3d_examples::common::run_example;
use rc3d_render::ibl::IblPreset;
use rc3d_scene::node_data::*;

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
    }
    None
}

fn maybe_screenshot(engine: &mut Engine) {
    if !std::env::args().any(|a| a == "--screenshot") {
        return;
    }
    engine.render();
    for dc in &engine.world.cached_draw_calls {
        eprintln!(
            "draw {} mode={:?} fill={:?} edges={:?} overlay={}",
            dc.node_type_label,
            dc.display_mode,
            dc.fill_style,
            dc.edge_style,
            dc.edge_positions.len()
        );
    }
    let quad = std::env::args().any(|a| a == "--quad");
    let (w, h, pixels) = if quad {
        engine.render_quad_pack_image(800, 600)
    } else {
        let dcs = engine.world.cached_draw_calls.clone();
        let Some(renderer) = engine.renderer.as_mut() else {
            return;
        };
        renderer.render_to_image(&dcs, &engine.world.graph, 800, 600)
    };
    let path = arg_value("--screenshot-path").unwrap_or_else(|| {
        if quad {
            "target/triangle_quad.png".to_string()
        } else {
            "target/triangle_drawstyle.png".to_string()
        }
    });
    if let Err(err) = image::save_buffer(&path, &pixels, w, h, image::ExtendedColorType::Rgba8) {
        eprintln!("screenshot failed: {err}");
        return;
    }
    eprintln!("wrote {path}");
    if std::env::args().any(|a| a == "--screenshot-exit") {
        std::process::exit(0);
    }
}

fn maybe_hlr_svg(engine: &mut Engine) {
    if !std::env::args().any(|a| a == "--hlr-svg") {
        return;
    }
    engine.render();
    let path = arg_value("--hlr-svg-path").unwrap_or_else(|| "target/triangle_hlr.svg".to_string());
    match engine.export_hidden_line_svg(&path) {
        Ok(()) => eprintln!("wrote {path}"),
        Err(err) => eprintln!("hlr svg failed: {err}"),
    }
}

fn main() {
    run_example("Triangle", |engine| {
        let global_mode = if std::env::args().any(|a| a == "--global-wireframe") {
            DisplayMode::Wireframe
        } else {
            DisplayMode::ShadedWithEdges
        };
        engine.set_display_mode(global_mode);
        // Neutral IBL keeps directional CSM from being washed out by studio fill.
        if let Some(renderer) = engine.renderer.as_mut() {
            renderer.set_ibl_preset(IblPreset::Neutral);
        }
        // Pull back and tilt down so the ground and CSM shadows stay in frame.
        engine.controller.target = Vec3::new(0.0, -0.35, 0.4);
        engine.controller.distance = 16.0;
        engine.controller.pitch = 0.48;
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 2.2, 10.0),
                Vec3::new(0.0, -0.3, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode {
                // Mostly downward, slight +Z (toward camera) so the right cube's
                // CSM blob lands on the foreground ground instead of screen center.
                // Too much +Z goes grazing and CSM loses the caster.
                direction: Vec3::new(-0.12, -1.0, 0.58),
                color: Vec3::ONE,
                intensity: 5.5,
                light_group: None,
            }),
        );

        graph.add_child(
            root,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.6, 0.8))),
        );

        let styles = VisualStyleLibrary::builtin();
        add_style_cube(graph, root, -4.0, &styles, VisualStyle::WIREFRAME);
        add_style_cube(graph, root, -2.4, &styles, VisualStyle::HARD_EDGES);

        let center = graph.add_child(root, NodeData::Separator(SeparatorNode));
        if let Some(style) = styles.get(VisualStyle::PERIMETER) {
            graph.apply_visual_style(center, style);
        }
        let verts = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-0.85, -1.0, 0.0),
            Vec3::new(0.85, -1.0, 0.0),
        ];
        graph.add_child(
            center,
            NodeData::Coordinate3(Coordinate3Node::from_points(verts)),
        );
        graph.add_child(
            center,
            NodeData::IndexedFaceSet(IndexedFaceSetNode::from_coord_index(vec![0, 1, 2])),
        );

        add_style_cube(graph, root, 1.6, &styles, VisualStyle::ADJACENT);
        add_style_cube(graph, root, 3.2, &styles, VisualStyle::SILHOUETTE);
        add_style_cube(graph, root, 4.8, &styles, VisualStyle::HIDDEN_LINE);

        // Shadow receiver: sit just below the triangle (y=-1). Do not force
        // Wireframe. Default inherits global ShadedWithEdges. Under
        // --global-wireframe keep the receiver filled so CSM is visible.
        let ground = graph.add_child(root, NodeData::Separator(SeparatorNode));
        if global_mode == DisplayMode::Wireframe {
            graph.set_display_mode(ground, DisplayMode::Shaded);
        }
        graph.add_child(
            ground,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -1.14, 0.0))),
        );
        graph.add_child(
            ground,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.38, 0.38, 0.36),
                diffuse_color: Vec3::new(0.38, 0.38, 0.36),
                metallic: 0.0,
                roughness: 0.92,
                opacity: 1.0,
                ..Default::default()
            }),
        );
        graph.add_child(
            ground,
            NodeData::Cube(CubeNode {
                width: 18.0,
                height: 0.08,
                depth: 14.0,
            }),
        );

        if std::env::args().any(|a| a == "--quad") {
            engine.apply_standard_quad_views();
        }

        maybe_screenshot(engine);
        maybe_hlr_svg(engine);
    });
}

fn add_style_cube(
    graph: &mut rc3d_scene::SceneGraph,
    root: rc3d_core::NodeId,
    x: f32,
    styles: &VisualStyleLibrary,
    name: &str,
) {
    let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    if let Some(style) = styles.get(name) {
        graph.apply_visual_style(sep, style);
    }
    graph.add_child(
        sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(x, -0.40, 0.0))),
    );
    graph.add_child(sep, NodeData::Cube(CubeNode::default()));
}
