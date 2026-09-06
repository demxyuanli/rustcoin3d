use rc3d_actions::{MeasurementAction, MeasurementMode, Ray};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::MeasurementType;

pub fn mode_for(ty: MeasurementType) -> MeasurementMode {
    match ty {
        MeasurementType::Distance => MeasurementMode::Distance,
        MeasurementType::Angle => MeasurementMode::Angle,
        MeasurementType::Radius => MeasurementMode::Radius,
        MeasurementType::Diameter => MeasurementMode::Diameter,
    }
}

pub fn action_for(ty: MeasurementType) -> MeasurementAction {
    MeasurementAction::new(mode_for(ty))
}

pub fn ground_hit(ray: &Ray) -> Option<Vec3> {
    if ray.direction.y.abs() < 1e-6 {
        return None;
    }
    let t = -ray.origin.y / ray.direction.y;
    if t > 1e-3 {
        Some(ray.origin + ray.direction * t)
    } else {
        None
    }
}
