//! Shared types for 3D viewport annotations.

use serde::{Deserialize, Serialize};

/// How the measurement label string is produced.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum AnnotationLabelMode {
    /// Use explicit `label` when non-empty; otherwise format from geometry.
    #[default]
    Auto,
    /// Always use the explicit `label` field.
    Fixed,
    /// `prefix` + formatted measurement (e.g. `"W="` + `"1.50"`).
    Prefix,
}

/// Default drawing parameters for an [`AnnotationSet`](crate::node_data::AnnotationSetNode).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnnotationStyle {
    pub extension_len: f32,
    pub arrow_size: f32,
    /// Minimum projected label height in screen pixels (readability floor).
    pub font_size: f32,
    /// World label height = `extension_len * label_height_factor` before screen floor.
    #[serde(default = "default_label_height_factor")]
    pub label_height_factor: f32,
    pub decimals: u32,
    /// Appended to auto-formatted lengths (e.g. `" mm"`).
    pub unit_suffix: String,
    /// Number of segments for circular arcs (angle dimensions, callouts).
    pub arc_segments: u32,
    /// Model/world units per pixel for Leader/Callout `label_offset` (camera-independent).
    pub leader_offset_scale: f32,
}

fn default_label_height_factor() -> f32 {
    0.5
}

impl Default for AnnotationStyle {
    fn default() -> Self {
        Self {
            extension_len: 0.3,
            arrow_size: 0.15,
            font_size: 14.0,
            label_height_factor: default_label_height_factor(),
            decimals: 2,
            unit_suffix: String::new(),
            arc_segments: 24,
            leader_offset_scale: 0.015,
        }
    }
}

impl AnnotationStyle {
    pub fn merge_with_element(
        &self,
        extension_len: Option<f32>,
        arrow_size: Option<f32>,
    ) -> Self {
        let mut s = self.clone();
        if let Some(v) = extension_len {
            s.extension_len = v;
        }
        if let Some(v) = arrow_size {
            s.arrow_size = v;
        }
        s
    }
}
