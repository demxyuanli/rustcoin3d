//! Scene builders for the Studio case library (shared patterns from rc3d-examples).

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

use super::common::{add_camera, add_floor, add_key_light, add_pbr_sphere};

pub fn hello() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(4.0, 3.0, 6.0), Vec3::ZERO);
    add_key_light(&mut g, root, 1.2);
    add_floor(&mut g, root, 8.0);
    add_pbr_sphere(
        &mut g,
        root,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.25, 0.55, 0.9),
        0.1,
        0.4,
        0.85,
    );
    g
}

pub fn pbr_ball() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(3.5, 2.5, 5.0), Vec3::new(0.0, 1.0, 0.0));
    add_key_light(&mut g, root, 1.4);
    g.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.6, -0.2, 0.5).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.35,
            light_group: None,
        }),
    );
    add_floor(&mut g, root, 6.0);
    add_pbr_sphere(
        &mut g,
        root,
        Vec3::new(0.0, 1.1, 0.0),
        Vec3::new(0.85, 0.65, 0.25),
        0.85,
        0.25,
        1.0,
    );
    g
}

pub fn shadows() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(5.0, 5.0, 7.0), Vec3::new(0.0, 1.0, 0.0));
    add_key_light(&mut g, root, 1.3);
    let pt = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.add_child(
        pt,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(2.5, 3.5, 0.0))),
    );
    g.add_child(
        pt,
        NodeData::PointLight(PointLightNode {
            color: Vec3::new(1.0, 0.9, 0.7),
            intensity: 8.0,
            ..Default::default()
        }),
    );
    add_floor(&mut g, root, 12.0);
    for (i, x) in [-2.0_f32, 0.0, 2.0].iter().enumerate() {
        let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
        g.add_child(
            sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(*x, 0.75, 0.0))),
        );
        g.add_child(
            sep,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::new(0.3 + 0.2 * i as f32, 0.45, 0.7),
                base_color: Vec3::new(0.3 + 0.2 * i as f32, 0.45, 0.7),
                roughness: 0.5,
                ..Default::default()
            }),
        );
        g.add_child(
            sep,
            NodeData::Cube(CubeNode {
                width: 1.0,
                height: 1.5,
                depth: 1.0,
            }),
        );
    }
    g
}

pub fn pbr_grid() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    let n = 5usize;
    let spacing = 2.2_f32;
    let center = Vec3::new((n as f32 - 1.0) * spacing * 0.5, 0.5, (n as f32 - 1.0) * spacing * 0.5);
    add_camera(
        &mut g,
        root,
        center + Vec3::new(-4.0, 8.0, 12.0),
        center,
    );
    add_key_light(&mut g, root, 1.5);
    for row in 0..n {
        for col in 0..n {
            let metallic = row as f32 / (n - 1) as f32;
            let roughness = col as f32 / (n - 1) as f32;
            add_pbr_sphere(
                &mut g,
                root,
                Vec3::new(col as f32 * spacing, 0.8, row as f32 * spacing),
                Vec3::new(0.9, 0.35, 0.25),
                metallic,
                roughness.max(0.05),
                0.7,
            );
        }
    }
    g
}

pub fn area_light() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(3.0, 2.5, 5.0), Vec3::new(0.0, 0.8, 0.0));
    add_key_light(&mut g, root, 0.2);
    let al = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.add_child(
        al,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 3.0, 0.0))),
    );
    g.add_child(
        al,
        NodeData::AreaLight(AreaLightNode {
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 12.0,
            ..Default::default()
        }),
    );
    add_floor(&mut g, root, 8.0);
    add_pbr_sphere(
        &mut g,
        root,
        Vec3::new(0.0, 0.9, 0.0),
        Vec3::new(0.8, 0.8, 0.85),
        0.0,
        0.35,
        0.9,
    );
    g
}

pub fn post_fx() -> SceneGraph {
    pbr_ball()
}

pub fn volumetric() -> SceneGraph {
    let mut g = shadows();
    // Tall markers for fog depth cue.
    let root = g.roots()[0];
    for i in 0..4 {
        let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
        g.add_child(
            sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(
                -3.0 + i as f32 * 2.0,
                2.0,
                -3.0,
            ))),
        );
        g.add_child(
            sep,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::new(0.2, 0.6, 0.3),
                base_color: Vec3::new(0.2, 0.6, 0.3),
                ..Default::default()
            }),
        );
        g.add_child(
            sep,
            NodeData::Cylinder(CylinderNode {
                radius: 0.25,
                height: 4.0,
            }),
        );
    }
    g
}

pub fn wboit() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(4.0, 2.5, 6.0), Vec3::new(0.0, 1.0, 0.0));
    add_key_light(&mut g, root, 1.2);
    add_floor(&mut g, root, 8.0);
    for (i, opacity) in [0.35_f32, 0.5, 0.7].iter().enumerate() {
        let _ = opacity;
        add_pbr_sphere(
            &mut g,
            root,
            Vec3::new(-1.5 + i as f32 * 1.5, 1.0, 0.0),
            Vec3::new(0.2 + 0.3 * i as f32, 0.5, 0.9),
            0.0,
            0.2,
            0.8,
        );
    }
    // Set opacities by walking materials in order of creation (floor first).
    let mut mats = Vec::new();
    for id in g.all_node_ids() {
        if let Some(e) = g.get(id) {
            if matches!(e.data, NodeData::Material(_)) {
                mats.push(id);
            }
        }
    }
    // Skip floor (index 0); set next three.
    for (i, &id) in mats.iter().skip(1).take(3).enumerate() {
        if let Some(e) = g.get_mut(id) {
            if let NodeData::Material(m) = &mut e.data {
                m.opacity = [0.35, 0.5, 0.7][i];
            }
        }
    }
    g
}

pub fn stereo() -> SceneGraph {
    hello()
}

pub fn anim_spin() -> SceneGraph {
    let mut g = hello();
    let root = g.roots()[0];
    g.add_child(
        root,
        NodeData::Text3(Text3Node {
            string: "Case: anim_spin".into(),
            position: Vec3::new(0.0, 2.4, 0.0),
            size: 18.0,
            color: [1.0, 1.0, 1.0, 1.0],
            plane_aligned: false,
        }),
    );
    g
}

pub fn pick_demo() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(5.0, 4.0, 7.0), Vec3::ZERO);
    add_key_light(&mut g, root, 1.2);
    add_floor(&mut g, root, 10.0);
    for i in 0..6 {
        let a = i as f32 * std::f32::consts::TAU / 6.0;
        add_pbr_sphere(
            &mut g,
            root,
            Vec3::new(a.cos() * 2.5, 0.8, a.sin() * 2.5),
            Vec3::new(0.4 + 0.1 * i as f32, 0.3, 0.7),
            0.2,
            0.45,
            0.55,
        );
    }
    g
}

pub fn section_demo() -> SceneGraph {
    let mut g = pbr_ball();
    let root = g.roots()[0];
    g.add_child(
        root,
        NodeData::SectionPlane(SectionPlaneNode {
            plane: [0.0, 1.0, 0.0, -0.9],
            enabled: true,
            ..Default::default()
        }),
    );
    g
}

pub fn explode_demo() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(6.0, 4.0, 8.0), Vec3::new(0.0, 1.0, 0.0));
    add_key_light(&mut g, root, 1.2);
    add_floor(&mut g, root, 10.0);
    let parts = [
        (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.8, 0.3, 0.2)),
        (Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.3, 0.7, 0.3)),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.2, 0.4, 0.9)),
    ];
    for (i, (pos, col)) in parts.iter().enumerate() {
        let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
        g.set_name(sep, format!("part_{i}"));
        g.add_child(sep, NodeData::Transform(TransformNode::from_translation(*pos)));
        g.add_child(
            sep,
            NodeData::Material(MaterialNode {
                diffuse_color: *col,
                base_color: *col,
                roughness: 0.4,
                ..Default::default()
            }),
        );
        g.add_child(
            sep,
            NodeData::Cube(CubeNode {
                width: 1.4,
                height: 0.8,
                depth: 1.4,
            }),
        );
    }
    g
}

pub fn walk_demo() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(0.0, 1.6, 6.0), Vec3::new(0.0, 1.4, 0.0));
    add_key_light(&mut g, root, 1.1);
    add_floor(&mut g, root, 30.0);
    for z in [-8_i32, -4, 0, 4, 8] {
        for x in [-6_i32, -2, 2, 6] {
            let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
            g.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    x as f32,
                    1.0,
                    z as f32,
                ))),
            );
            g.add_child(
                sep,
                NodeData::Material(MaterialNode {
                    diffuse_color: Vec3::new(0.45, 0.45, 0.5),
                    base_color: Vec3::new(0.45, 0.45, 0.5),
                    roughness: 0.8,
                    ..Default::default()
                }),
            );
            g.add_child(
                sep,
                NodeData::Cube(CubeNode {
                    width: 1.2,
                    height: 2.0,
                    depth: 1.2,
                }),
            );
        }
    }
    g
}

pub fn torus() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(3.0, 2.5, 4.5), Vec3::ZERO);
    add_key_light(&mut g, root, 1.3);
    add_floor(&mut g, root, 6.0);
    let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.add_child(
        sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.2, 0.0))),
    );
    g.add_child(
        sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.9, 0.55, 0.2),
            base_color: Vec3::new(0.9, 0.55, 0.2),
            metallic: 0.4,
            roughness: 0.3,
            ..Default::default()
        }),
    );
    g.add_child(
        sep,
        NodeData::Torus(TorusNode {
            major_radius: 1.1,
            minor_radius: 0.35,
        }),
    );
    g
}

pub fn text3d_billboard() -> SceneGraph {
    let mut g = hello();
    let root = g.roots()[0];
    g.add_child(
        root,
        NodeData::Text3(Text3Node {
            string: "Text3 label".into(),
            position: Vec3::new(0.0, 2.2, 0.0),
            size: 22.0,
            color: [1.0, 0.95, 0.4, 1.0],
            plane_aligned: false,
        }),
    );
    let bb = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.add_child(bb, NodeData::Billboard(BillboardNode::default()));
    g.add_child(
        bb,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(2.0, 1.5, 0.0))),
    );
    g.add_child(
        bb,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.2, 0.8, 0.5),
            base_color: Vec3::new(0.2, 0.8, 0.5),
            ..Default::default()
        }),
    );
    g.add_child(bb, NodeData::Cube(CubeNode::default()));
    g
}

pub fn instancing() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(8.0, 7.0, 10.0), Vec3::ZERO);
    add_key_light(&mut g, root, 1.3);
    add_floor(&mut g, root, 16.0);
    let mut transforms = Vec::new();
    for i in 0..40 {
        let a = i as f32 * 0.4;
        let t = Mat4::from_translation(Vec3::new(a.cos() * 4.0, 0.5, a.sin() * 4.0));
        transforms.push(t.to_cols_array_2d());
    }
    let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.add_child(
        sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.3, 0.7, 0.95),
            base_color: Vec3::new(0.3, 0.7, 0.95),
            metallic: 0.3,
            roughness: 0.4,
            ..Default::default()
        }),
    );
    g.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.35 }));
    g.add_child(
        sep,
        NodeData::InstancedMesh(InstancedMeshNode { transforms }),
    );
    g
}

pub fn scene_graph_demo() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.set_name(root, "root");
    add_camera(&mut g, root, Vec3::new(5.0, 4.0, 7.0), Vec3::ZERO);
    add_key_light(&mut g, root, 1.2);
    add_floor(&mut g, root, 8.0);
    let parent = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.set_name(parent, "arm");
    g.add_child(
        parent,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.0, 0.0))),
    );
    g.add_child(
        parent,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.7, 0.3, 0.2),
            base_color: Vec3::new(0.7, 0.3, 0.2),
            ..Default::default()
        }),
    );
    g.add_child(
        parent,
        NodeData::Cube(CubeNode {
            width: 0.4,
            height: 2.0,
            depth: 0.4,
        }),
    );
    let child = g.add_child(parent, NodeData::Separator(SeparatorNode));
    g.set_name(child, "hand");
    g.add_child(
        child,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.2, 0.0))),
    );
    g.add_child(
        child,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.2, 0.5, 0.9),
            base_color: Vec3::new(0.2, 0.5, 0.9),
            ..Default::default()
        }),
    );
    g.add_child(child, NodeData::Sphere(SphereNode { radius: 0.45 }));
    g
}

pub fn material_showcase() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(0.0, 2.5, 8.0), Vec3::new(0.0, 1.0, 0.0));
    add_key_light(&mut g, root, 1.4);
    add_floor(&mut g, root, 10.0);
    let specs = [
        (Vec3::new(-3.0, 1.0, 0.0), 1.0, 0.1, Vec3::new(0.95, 0.9, 0.7)),
        (Vec3::new(-1.0, 1.0, 0.0), 0.0, 0.9, Vec3::new(0.2, 0.55, 0.3)),
        (Vec3::new(1.0, 1.0, 0.0), 0.5, 0.4, Vec3::new(0.8, 0.2, 0.2)),
        (Vec3::new(3.0, 1.0, 0.0), 0.2, 0.15, Vec3::new(0.3, 0.4, 0.95)),
    ];
    for (pos, met, rough, col) in specs {
        add_pbr_sphere(&mut g, root, pos, col, met, rough, 0.85);
    }
    g
}

pub fn env_reflect() -> SceneGraph {
    pbr_ball()
}

pub fn light_link() -> SceneGraph {
    shadows()
}

pub fn annotation_demo() -> SceneGraph {
    let mut g = hello();
    let root = g.roots()[0];
    g.add_child(
        root,
        NodeData::Text3(Text3Node {
            string: "Dim A".into(),
            position: Vec3::new(-1.2, 2.0, 0.0),
            size: 16.0,
            color: [1.0, 0.85, 0.2, 1.0],
            plane_aligned: false,
        }),
    );
    g.add_child(
        root,
        NodeData::AnnotationSet(AnnotationSetNode::default()),
    );
    g
}

pub fn lines_demo() -> SceneGraph {
    let mut g = SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    add_camera(&mut g, root, Vec3::new(4.0, 3.0, 5.0), Vec3::ZERO);
    add_key_light(&mut g, root, 1.0);
    let sep = g.add_child(root, NodeData::Separator(SeparatorNode));
    g.add_child(
        sep,
        NodeData::Coordinate3(Coordinate3Node {
            point: vec![
                Vec3::new(-2.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
        }),
    );
    g.add_child(
        sep,
        NodeData::IndexedLineSet(IndexedLineSetNode {
            coord_index: vec![0, 1, -1, 0, 2, -1, 0, 3, -1],
            ..Default::default()
        }),
    );
    g
}

pub fn nurbs_approx() -> SceneGraph {
    torus()
}
