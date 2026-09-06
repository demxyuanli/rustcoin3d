use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

#[derive(Clone, Copy, Debug)]
pub enum MeasurementMode {
    Distance,
    Angle,
    Radius,
    Diameter,
}

impl MeasurementMode {
    pub fn points_needed(self) -> usize {
        match self {
            MeasurementMode::Angle => 3,
            MeasurementMode::Distance | MeasurementMode::Radius | MeasurementMode::Diameter => 2,
        }
    }
}

pub struct MeasurementAction {
    pub mode: MeasurementMode,
    pub points: Vec<Vec3>,
    pub completed: bool,
    pub value: f32,
    pub label: String,
}

impl MeasurementAction {
    pub fn new(mode: MeasurementMode) -> Self {
        Self {
            mode,
            points: Vec::new(),
            completed: false,
            value: 0.0,
            label: String::new(),
        }
    }

    pub fn add_point(&mut self, point: Vec3) -> bool {
        self.points.push(point);
        let (needed, value, label) = match self.mode {
            MeasurementMode::Distance if self.points.len() >= 2 => {
                let d = self.points[0].distance(self.points[1]);
                (2, d, format!("{:.2} m", d))
            }
            MeasurementMode::Angle if self.points.len() >= 3 => {
                let da = self.points[0] - self.points[1];
                let db = self.points[2] - self.points[1];
                if da.length_squared() < 1e-12 || db.length_squared() < 1e-12 {
                    (3, 0.0, "0.0°".to_string())
                } else {
                    let a = da.normalize();
                    let b = db.normalize();
                    let cos_angle = a.dot(b).clamp(-1.0, 1.0);
                    let angle = cos_angle.acos().to_degrees();
                    (3, angle, format!("{:.1}°", angle))
                }
            }
            MeasurementMode::Radius if self.points.len() >= 2 => {
                let r = self.points[0].distance(self.points[1]);
                (2, r, format!("R={:.2} m", r))
            }
            MeasurementMode::Diameter if self.points.len() >= 2 => {
                let d = self.points[0].distance(self.points[1]);
                (2, d, format!("D={:.2} m", d))
            }
            _ => return false,
        };
        self.value = value;
        self.label = label;
        self.completed = self.points.len() >= needed;
        self.completed
    }

    pub fn create_node(&self, graph: &mut SceneGraph, parent: NodeId) -> NodeId {
        use rc3d_scene::node_data::*;
        let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
        if !self.points.is_empty() {
            graph.add_child(
                sep,
                NodeData::Coordinate3(Coordinate3Node::from_points(self.points.clone())),
            );
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 1.0, 0.0))),
            );
            graph.add_child(sep, NodeData::Triangle(TriangleNode));
        }
        sep
    }

    /// Persistent 3D annotation from completed measurement picks (world space).
    pub fn build_annotation_set(&self) -> Option<rc3d_scene::node_data::AnnotationSetNode> {
        use rc3d_scene::annotation::{
            world_angle_annotation, world_diameter_annotation, world_distance_annotation,
            world_radius_annotation, AnnotationStyle,
        };
        if !self.completed {
            return None;
        }
        let color = [1.0, 1.0, 0.0, 1.0];
        let style = AnnotationStyle::default();
        Some(match self.mode {
            MeasurementMode::Distance if self.points.len() >= 2 => {
                world_distance_annotation(self.points[0], self.points[1], color, style)
            }
            MeasurementMode::Angle if self.points.len() >= 3 => {
                world_angle_annotation(self.points[0], self.points[1], self.points[2], color, style)
            }
            MeasurementMode::Radius if self.points.len() >= 2 => {
                world_radius_annotation(self.points[0], self.points[1], color, style)
            }
            MeasurementMode::Diameter if self.points.len() >= 2 => {
                world_diameter_annotation(self.points[0], self.points[1], color, style)
            }
            _ => return None,
        })
    }

    /// Add `Annotation` + `AnnotationSet` under `parent` from a completed measurement.
    pub fn create_annotation_node(&self, graph: &mut SceneGraph, parent: NodeId) -> Option<NodeId> {
        use rc3d_scene::node_data::*;
        let set = self.build_annotation_set()?;
        let ann = graph.add_child(parent, NodeData::Annotation(AnnotationNode));
        let set_id = graph.add_child(ann, NodeData::AnnotationSet(set));
        Some(set_id)
    }

    pub fn points_needed(&self) -> usize {
        self.mode.points_needed()
    }
}
