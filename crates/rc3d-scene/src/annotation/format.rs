//! Formatting helpers for annotation labels.

use super::types::AnnotationStyle;

pub fn format_length(value: f32, style: &AnnotationStyle) -> String {
    let pow = 10f32.powi(style.decimals as i32);
    let rounded = (value * pow).round() / pow;
    if style.decimals == 0 {
        format!("{}{}", rounded as i32, style.unit_suffix)
    } else {
        format!("{:.prec$}{}", rounded, style.unit_suffix, prec = style.decimals as usize)
    }
}

pub fn format_angle_degrees(value_deg: f32, style: &AnnotationStyle) -> String {
    let pow = 10f32.powi(style.decimals as i32);
    let rounded = (value_deg * pow).round() / pow;
    if style.decimals == 0 {
        format!("{:.0}°", rounded)
    } else {
        format!("{:.prec$}°", rounded, prec = style.decimals as usize)
    }
}

pub fn format_radius(value: f32, style: &AnnotationStyle) -> String {
    format!("R{}", format_length(value, style))
}

pub fn format_diameter(value: f32, style: &AnnotationStyle) -> String {
    format!("D{}", format_length(value, style))
}
