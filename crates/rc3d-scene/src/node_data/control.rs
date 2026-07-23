//! Traversal control, text, measurement, markup, and annotation set nodes.
use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use serde::{Deserialize, Serialize};

/// Event callback node: marker for scene-graph event routing.
///
/// When HandleEventAction encounters this node during traversal,
/// the application-level handler decides whether to consume the event.
/// The `enabled` flag controls whether the node participates in routing.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EventCallbackNode {
    pub enabled: bool,
}

impl Default for EventCallbackNode {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Pick style: controls whether this node (and its children) can be picked.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PickStyleNode {
    pub pickable: bool,
}

impl Default for PickStyleNode {
    fn default() -> Self {
        Self { pickable: true }
    }
}

/// One LOD level: a group of children rendered at this detail level.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LodLevel {
    pub children: Vec<NodeId>,
    pub max_distance: f32,
}

/// LOD switch node (Coin3D SoLOD / SoLevelOfDetail pattern).
/// Selects one child group based on camera distance.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[derive(Default)]
pub struct LodNode {
    pub levels: Vec<LodLevel>,
    pub current_level: usize,
}

impl LodNode {
    /// Select the LOD level based on camera distance.
    /// Each `LodLevel.max_distance` defines the threshold: pick the first level
    /// whose `max_distance >= distance`, or the last level if none matches.
    pub fn select_level(&mut self, distance: f32) {
        for (i, level) in self.levels.iter().enumerate() {
            if distance <= level.max_distance {
                self.current_level = i;
                return;
            }
        }
        self.current_level = self.levels.len().saturating_sub(1);
    }
}


/// Section/cutting plane node (Coin3D SoClipPlane pattern).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SectionPlaneNode {
    pub plane: [f32; 4],
    pub enabled: bool,
    /// RGBA color for the cap surface at the cut boundary.
    pub cap_color: [f32; 4],
    /// When true, render a filled cap at the clip plane intersection.
    pub cap_enabled: bool,
}

impl Default for SectionPlaneNode {
    fn default() -> Self {
        Self {
            plane: [0.0, 1.0, 0.0, 0.0],
            enabled: true,
            cap_color: [0.5, 0.5, 0.5, 1.0],
            cap_enabled: false,
        }
    }
}

/// Switch node: traverses one child based on index (Coin3D SoSwitch pattern).
/// which_child: -1 = all, -2 = none, 0..N = specific child.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwitchNode {
    pub which_child: i32,
    pub children: Vec<rc3d_core::NodeId>,
}

impl Default for SwitchNode {
    fn default() -> Self {
        Self { which_child: -1, children: Vec::new() }
    }
}

/// MultipleCopy node: repeats child traversal with offset transforms
/// (Coin3D SoMultipleCopy pattern).
#[derive(Serialize, Deserialize, Clone, Debug)]
#[derive(Default)]
pub struct MultipleCopyNode {
    pub copies: Vec<rc3d_core::math::Mat4>,
    pub children: Vec<rc3d_core::NodeId>,
}


/// Screen-space 2D text label (Coin3D SoText2 pattern).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Text2Node {
    pub string: String,
    pub position: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Text2Node {
    fn default() -> Self {
        Self { string: String::new(), position: [0.0, 0.0], size: 16.0, color: [1.0, 1.0, 1.0, 1.0] }
    }
}

/// World-space 3D text label (Coin3D SoText3 pattern).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Text3Node {
    pub string: String,
    pub position: Vec3,
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Text3Node {
    fn default() -> Self {
        Self { string: String::new(), position: Vec3::ZERO, size: 16.0, color: [1.0, 1.0, 1.0, 1.0] }
    }
}

/// Central node type enum.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum MeasurementType {
    Distance,
    Angle,
    Radius,
    Diameter,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MeasurementNode {
    pub points: Vec<rc3d_core::math::Vec3>,
    pub measurement_type: MeasurementType,
    pub label: String,
    pub color: [f32; 4],
    pub value: f32,
}

impl Default for MeasurementNode {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            measurement_type: MeasurementType::Distance,
            label: String::new(),
            color: [1.0, 1.0, 0.0, 1.0],
            value: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum MarkupElement {
    Line {
        start: [f32; 2],
        end: [f32; 2],
        color: [f32; 4],
        width: f32,
    },
    Rect {
        origin: [f32; 2],
        size: [f32; 2],
        color: [f32; 4],
        filled: bool,
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        color: [f32; 4],
    },
    Freehand {
        points: Vec<[f32; 2]>,
        color: [f32; 4],
        width: f32,
    },
    Text {
        position: [f32; 2],
        string: String,
        size: f32,
        color: [f32; 4],
    },
    /// Dimension line with extension lines and arrowheads.
    Dimension {
        start: [f32; 2],
        end: [f32; 2],
        /// Offset direction (normalized) for extension lines.
        offset_dir: [f32; 2],
        /// Extension line length.
        extension_len: f32,
        /// Arrowhead size.
        arrow_size: f32,
        /// Measurement label (e.g., "12.34 m").
        label: String,
        color: [f32; 4],
    },
    /// Angle dimension: arc between two rays from a center point.
    AngleDimension {
        center: [f32; 2],
        arm1: [f32; 2],
        arm2: [f32; 2],
        radius: f32,
        color: [f32; 4],
        label: String,
    },
    /// Radial dimension: line from center to circumference point.
    RadialDimension {
        center: [f32; 2],
        perimeter: [f32; 2],
        color: [f32; 4],
        label: String,
    },
    /// Diameter dimension: line through center between two perimeter points.
    DiameterDimension {
        p1: [f32; 2],
        p2: [f32; 2],
        center: [f32; 2],
        color: [f32; 4],
        label: String,
    },
    /// Leader line from a point to a text label.
    Leader {
        anchor: [f32; 2],
        label_pos: [f32; 2],
        text: String,
        color: [f32; 4],
    },
    /// Callout bubble: leader line + filled circle with text.
    Callout {
        anchor: [f32; 2],
        label_pos: [f32; 2],
        text: String,
        radius: f32,
        color: [f32; 4],
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MarkupNode {
    pub elements: Vec<MarkupElement>,
    pub layer_name: String,
    pub visible: bool,
}

// ── 3D Annotation Elements (world-space, projected to screen each frame) ──

pub use crate::annotation::{AnnotationLabelMode, AnnotationPoint, AnnotationStyle};

/// GD&T geometric characteristic symbol.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GdtSymbol {
    /// — Straightness
    Straightness,
    /// ⌓ Flatness
    #[default]
    Flatness,
    /// ○ Circularity
    Circularity,
    /// ⌭ Cylindricity
    Cylindricity,
    /// ⌒ Profile of a line
    ProfileOfLine,
    /// ⌓ Profile of a surface (closed)
    ProfileOfSurface,
    /// ∠ Angularity
    Angularity,
    /// ⊥ Perpendicularity
    Perpendicularity,
    /// ∥ Parallelism
    Parallelism,
    /// ⌖ Position
    Position,
    /// ◎ Concentricity
    Concentricity,
    /// ⌯ Symmetry
    Symmetry,
    /// ↗ Circular runout
    CircularRunout,
    /// ↗ Total runout (double arrow)
    TotalRunout,
}

/// GD&T material condition modifier.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdtMaterialCondition {
    /// Ⓜ Maximum material condition
    MaximumMaterial,
    /// Ⓛ Least material condition
    LeastMaterial,
    /// Ⓢ Regardless of feature size
    RegardlessOfFeature,
}

/// Datum target type indicator.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DatumTargetType {
    /// Crosshair with circle
    #[default]
    Point,
    /// Line with circles at ends
    Line,
    /// Hatched rectangle
    Area,
}

/// Welding symbol type (ISO 2553 / AWS A2.4).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub enum WeldType {
    #[default]
    Fillet,           // triangle
    SquareGroove,     // vertical lines
    VGroove,          // V
    BevelGroove,      // single bevel
    UGroove,          // U
    JGroove,          // J
    Plug,             // square
    Spot,             // filled circle
    Seam,             // open circles
}

/// A 3D annotation element positioned in world space.
/// Projected to screen coordinates each frame using the camera VP matrix.
///
/// Each element lies on one coordinate-axis-parallel plane; geometry is built in 3D
/// then projected once per frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AnnotationElement {
    /// Linear dimension between two points (extension lines + arrows).
    Dimension {
        start: AnnotationPoint,
        end: AnnotationPoint,
        /// Offset direction in set-local space (magnitude = offset distance).
        offset_dir: [f32; 3],
        #[serde(default = "default_annotation_extension_len")]
        extension_len: f32,
        #[serde(default = "default_annotation_arrow_size")]
        arrow_size: f32,
        #[serde(default)]
        label: String,
        #[serde(default)]
        label_mode: AnnotationLabelMode,
        color: [f32; 4],
    },
    /// Angular dimension: arc between rays `center`→`arm1` and `center`→`arm2`.
    AngleDimension {
        center: AnnotationPoint,
        arm1: AnnotationPoint,
        arm2: AnnotationPoint,
        radius: f32,
        #[serde(default)]
        label: String,
        #[serde(default)]
        label_mode: AnnotationLabelMode,
        color: [f32; 4],
    },
    /// Radial dimension from `center` to `perimeter`.
    RadialDimension {
        center: AnnotationPoint,
        perimeter: AnnotationPoint,
        #[serde(default)]
        label: String,
        #[serde(default)]
        label_mode: AnnotationLabelMode,
        #[serde(default = "default_annotation_arrow_size")]
        arrow_size: f32,
        color: [f32; 4],
    },
    /// Diameter dimension through `center` between `p1` and `p2`.
    DiameterDimension {
        center: AnnotationPoint,
        p1: AnnotationPoint,
        p2: AnnotationPoint,
        #[serde(default)]
        label: String,
        #[serde(default)]
        label_mode: AnnotationLabelMode,
        #[serde(default = "default_annotation_arrow_size")]
        arrow_size: f32,
        color: [f32; 4],
    },
    /// Leader line from a 3D anchor to a label (pixel offset from anchor).
    Leader {
        anchor: AnnotationPoint,
        label_offset: [f32; 2],
        text: String,
        color: [f32; 4],
    },
    /// Leader + circular callout at the label.
    Callout {
        anchor: AnnotationPoint,
        label_offset: [f32; 2],
        text: String,
        radius: f32,
        color: [f32; 4],
    },
    /// Datum cross at a 3D point.
    Datum {
        position: AnnotationPoint,
        size: f32,
        color: [f32; 4],
    },
    /// GD&T feature control frame (e.g. ⌓ 0.05 A)
    GdtFeatureControlFrame {
        /// Geometric characteristic symbol (e.g. "flatness", "position").
        symbol: GdtSymbol,
        /// Tolerance value (e.g. 0.05).
        tolerance: f32,
        /// Optional diameter modifier prefix Ø.
        diameter: bool,
        /// Primary datum reference.
        datum_primary: Option<String>,
        /// Secondary datum reference.
        datum_secondary: Option<String>,
        /// Material condition modifier.
        material_condition: Option<GdtMaterialCondition>,
        /// Position in set-local space (top-left of the frame).
        position: AnnotationPoint,
        /// Direction of the leader line from frame to feature (None = no leader).
        leader_target: Option<AnnotationPoint>,
        /// Display color.
        color: [f32; 4],
    },
    /// Datum target: point/line/area marker with label (e.g. A1, B2).
    GdtDatumTarget {
        position: AnnotationPoint,
        /// Target label (e.g. "A1").
        label: String,
        /// Target type indicator.
        target_type: DatumTargetType,
        size: f32,
        color: [f32; 4],
    },
    /// Chamfer dimension (C X 45° format).
    ChamferDimension {
        /// Chamfer start point.
        start: AnnotationPoint,
        /// Chamfer end point.
        end: AnnotationPoint,
        /// Offset direction for the dimension line.
        offset_dir: [f32; 3],
        extension_len: f32,
        arrow_size: f32,
        label: String,
        label_mode: AnnotationLabelMode,
        color: [f32; 4],
    },
    /// Ordinate (baseline) dimension: single jogged leader to a datum plane.
    OrdinateDimension {
        /// Feature point to dimension.
        feature: AnnotationPoint,
        /// Datum plane origin.
        datum: AnnotationPoint,
        /// Direction of the coordinate axis (should be axis-aligned).
        axis_dir: [f32; 3],
        /// Jog position (distance from feature along axis).
        jog_length: f32,
        /// Offset perpendicular to axis for the leader.
        offset: f32,
        label: String,
        label_mode: AnnotationLabelMode,
        color: [f32; 4],
    },
    /// Surface finish / roughness annotation (ISO 1302).
    /// e.g. Ra 3.2 with a check mark symbol.
    SurfaceFinish {
        /// Ra value in micrometers (e.g. 3.2).
        ra_value: f32,
        /// Optional additional text (e.g. machining method).
        note: Option<String>,
        /// Anchor point on the surface.
        position: AnnotationPoint,
        /// Direction the leader points (from annotation to surface).
        direction: [f32; 3],
        /// Display color.
        color: [f32; 4],
    },
    /// Welding symbol annotation (ISO 2553 / AWS A2.4).
    WeldSymbol {
        /// Weld type (fillet, groove, plug, etc.).
        weld_type: WeldType,
        /// Weld size (e.g. leg length for fillet).
        size: Option<f32>,
        /// Weld length.
        length: Option<f32>,
        /// Field or shop weld.
        field_weld: bool,
        /// Arrow side text.
        arrow_side_text: Option<String>,
        /// Other side text.
        other_side_text: Option<String>,
        /// Arrow anchor point on the joint.
        position: AnnotationPoint,
        /// Direction the arrow points.
        arrow_dir: [f32; 3],
        color: [f32; 4],
    },
    /// Datum identifier triangle (ISO 5459) -- filled or open triangle with letter.
    DatumIdentifier {
        /// Datum letter (e.g. "A").
        label: String,
        /// Triangle base center position.
        position: AnnotationPoint,
        /// Triangle size.
        size: f32,
        /// Whether the triangle is filled.
        filled: bool,
        color: [f32; 4],
    },
}

fn default_annotation_extension_len() -> f32 {
    0.3
}

fn default_annotation_arrow_size() -> f32 {
    0.15
}

impl AnnotationElement {
    /// Linear dimension with auto label from measured distance.
    pub fn linear_auto(
        start: [f32; 3],
        end: [f32; 3],
        offset_dir: [f32; 3],
        color: [f32; 4],
    ) -> Self {
        Self::Dimension {
            start: AnnotationPoint::local(start),
            end: AnnotationPoint::local(end),
            offset_dir,
            extension_len: default_annotation_extension_len(),
            arrow_size: default_annotation_arrow_size(),
            label: String::new(),
            label_mode: AnnotationLabelMode::Auto,
            color,
        }
    }

    /// Leader with text at pixel offset from projected anchor.
    pub fn leader(anchor: [f32; 3], label_offset: [f32; 2], text: impl Into<String>, color: [f32; 4]) -> Self {
        Self::Leader {
            anchor: AnnotationPoint::local(anchor),
            label_offset,
            text: text.into(),
            color,
        }
    }
}

/// Annotation set: leaf node of 3D elements under an [`Annotation`](AnnotationNode) group.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnnotationSetNode {
    pub elements: Vec<AnnotationElement>,
    pub visible: bool,
    #[serde(default)]
    pub style: AnnotationStyle,
}

impl Default for AnnotationSetNode {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            visible: true,
            style: AnnotationStyle::default(),
        }
    }
}

impl Default for MarkupNode {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            layer_name: String::new(),
            visible: true,
        }
    }
}
