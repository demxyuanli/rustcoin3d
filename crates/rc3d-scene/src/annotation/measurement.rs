//! Build `AnnotationSet` nodes from interactive measurement picks (world-space points).

use rc3d_core::math::Vec3;

use crate::node_data::{AnnotationElement, AnnotationSetNode, AnnotationLabelMode};

use super::point::AnnotationPoint;
use super::types::AnnotationStyle;

fn default_offset_dir(a: Vec3, b: Vec3) -> [f32; 3] {
    let d = b - a;
    if d.length_squared() < 1e-12 {
        return [0.0, -0.3, 0.0];
    }
    let d = d.normalize();
    let up = Vec3::Y;
    let off = if d.cross(up).length_squared() > 1e-6 {
        d.cross(up).normalize()
    } else {
        d.cross(Vec3::X).normalize()
    };
    (off * 0.3).into()
}

/// Distance measurement in world space → `AnnotationSet` (place under `Annotation` with identity transform).
pub fn world_distance_annotation(
    a: Vec3,
    b: Vec3,
    color: [f32; 4],
    style: AnnotationStyle,
) -> AnnotationSetNode {
    AnnotationSetNode {
        style,
        elements: vec![AnnotationElement::Dimension {
            start: AnnotationPoint::local(a.into()),
            end: AnnotationPoint::local(b.into()),
            offset_dir: default_offset_dir(a, b),
            extension_len: 0.3,
            arrow_size: 0.15,
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            color,
        }],
        visible: true,
    }
}

/// Angle measurement (3 world points: arm0, center, arm2) → `AnnotationSet`.
pub fn world_angle_annotation(
    arm0: Vec3,
    center: Vec3,
    arm2: Vec3,
    color: [f32; 4],
    style: AnnotationStyle,
) -> AnnotationSetNode {
    let r = (arm0 - center).length().max((arm2 - center).length()) * 0.35;
    AnnotationSetNode {
        style,
        elements: vec![AnnotationElement::AngleDimension {
            center: AnnotationPoint::local(center.into()),
            arm1: AnnotationPoint::local(arm0.into()),
            arm2: AnnotationPoint::local(arm2.into()),
            radius: r.max(0.1),
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            color,
        }],
        visible: true,
    }
}

/// Radius measurement (center, perimeter) → `AnnotationSet`.
pub fn world_radius_annotation(
    center: Vec3,
    perimeter: Vec3,
    color: [f32; 4],
    style: AnnotationStyle,
) -> AnnotationSetNode {
    AnnotationSetNode {
        style,
        elements: vec![AnnotationElement::RadialDimension {
            center: AnnotationPoint::local(center.into()),
            perimeter: AnnotationPoint::local(perimeter.into()),
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            arrow_size: 0.15,
            color,
        }],
        visible: true,
    }
}

/// Diameter measurement (p1, p2 with midpoint center) → `AnnotationSet`.
pub fn world_diameter_annotation(
    p1: Vec3,
    p2: Vec3,
    color: [f32; 4],
    style: AnnotationStyle,
) -> AnnotationSetNode {
    let center = (p1 + p2) * 0.5;
    AnnotationSetNode {
        style,
        elements: vec![AnnotationElement::DiameterDimension {
            center: AnnotationPoint::local(center.into()),
            p1: AnnotationPoint::local(p1.into()),
            p2: AnnotationPoint::local(p2.into()),
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            arrow_size: 0.15,
            color,
        }],
        visible: true,
    }
}
