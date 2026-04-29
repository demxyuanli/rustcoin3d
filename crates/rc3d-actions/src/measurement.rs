use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

#[derive(Clone, Copy, Debug)]
pub enum MeasurementMode {
    Distance,
    Angle,
    Radius,
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
}
