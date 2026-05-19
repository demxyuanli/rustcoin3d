//! Resolve display labels from geometry and label mode.

use super::format::{format_angle_degrees, format_diameter, format_length, format_radius};
use super::types::{AnnotationLabelMode, AnnotationStyle};

pub fn resolve_label(
    label: &str,
    mode: &AnnotationLabelMode,
    auto_value: impl FnOnce() -> String,
) -> String {
    match mode {
        AnnotationLabelMode::Fixed => label.to_string(),
        AnnotationLabelMode::Prefix => {
            let measured = auto_value();
            if label.is_empty() {
                measured
            } else {
                format!("{label}{measured}")
            }
        }
        AnnotationLabelMode::Auto => {
            if label.is_empty() {
                auto_value()
            } else {
                label.to_string()
            }
        }
    }
}

pub fn resolve_length_label(
    label: &str,
    mode: &AnnotationLabelMode,
    distance: f32,
    style: &AnnotationStyle,
) -> String {
    resolve_label(label, mode, || format_length(distance, style))
}

pub fn resolve_angle_label(
    label: &str,
    mode: &AnnotationLabelMode,
    degrees: f32,
    style: &AnnotationStyle,
) -> String {
    resolve_label(label, mode, || format_angle_degrees(degrees, style))
}

pub fn resolve_radius_label(
    label: &str,
    mode: &AnnotationLabelMode,
    radius: f32,
    style: &AnnotationStyle,
) -> String {
    resolve_label(label, mode, || format_radius(radius, style))
}

pub fn resolve_diameter_label(
    label: &str,
    mode: &AnnotationLabelMode,
    diameter: f32,
    style: &AnnotationStyle,
) -> String {
    resolve_label(label, mode, || format_diameter(diameter, style))
}
