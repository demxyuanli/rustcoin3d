use serde::{Deserialize, Serialize};

/// Combined convenience enum (Coin3D `SoDrawStyle` presets).
///
/// Fill and edges are orthogonal ([`FillStyle`] x [`EdgeStyle`]). These six
/// values remain the global / serialized presets; per-node
/// `fill_style` / `edge_style` can compose combinations that have no variant
/// (e.g. shaded fill + full topology edges).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisplayMode {
    #[default]
    ShadedWithEdges,
    Shaded,
    Wireframe,
    HiddenLine,
    Flat,
    FlatWithEdge,
}

/// Face fill / shading (HOOPS `VisibilityControl.SetFaces` + rendering mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FillStyle {
    #[default]
    Shaded,
    Flat,
    HiddenLine,
    /// No filled rasterization (wireframe-only when paired with [`EdgeStyle::Full`]).
    None,
}

/// Edge overlay independent of fill (HOOPS edge visibility + classification).
///
/// HOOPS maps: `Hard` = dihedral crease, `Perimeter` = boundary,
/// `Adjacent` = interior non-crease, `Silhouette` = view contour (+ boundary),
/// `Full` = mesh (all unique edges), `Crease` = Hard + Perimeter (CAD default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeStyle {
    #[default]
    None,
    /// Boundary + dihedral crease (default 12 deg). HOOPS hard+perimeter.
    Crease,
    /// View-dependent silhouette (plus boundary).
    Silhouette,
    /// All unique mesh edges, including coplanar face diagonals.
    Full,
    /// Boundary edges only (one adjacent face).
    Perimeter,
    /// Interior dihedral crease only (no boundary).
    Hard,
    /// Interior non-crease edges (smooth shared faces).
    Adjacent,
}

/// Resolved fill + edges for one draw / one inherited state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Appearance {
    pub fill: FillStyle,
    pub edges: EdgeStyle,
}

impl Appearance {
    pub fn from_display_mode(mode: DisplayMode) -> Self {
        Self {
            fill: mode.fill(),
            edges: mode.edges(),
        }
    }

    /// `display_mode` replaces both axes; `fill_style` / `edge_style` then override.
    pub fn resolve(
        inherited: Self,
        display_mode: Option<DisplayMode>,
        fill_style: Option<FillStyle>,
        edge_style: Option<EdgeStyle>,
    ) -> Self {
        let mut app = inherited;
        if let Some(mode) = display_mode {
            app = Self::from_display_mode(mode);
        }
        if let Some(fill) = fill_style {
            app.fill = fill;
        }
        if let Some(edges) = edge_style {
            app.edges = edges;
        }
        app
    }

    /// Nearest [`DisplayMode`] preset (HUD / sort keys). Combos without a
    /// variant collapse to the closest FILLED / FILLED+LINES / LINES value.
    pub fn to_display_mode(self) -> DisplayMode {
        match (self.fill, self.edges) {
            (FillStyle::None, _) => DisplayMode::Wireframe,
            (FillStyle::Shaded, EdgeStyle::None) => DisplayMode::Shaded,
            (FillStyle::Shaded, _) => DisplayMode::ShadedWithEdges,
            (FillStyle::Flat, EdgeStyle::None) => DisplayMode::Flat,
            (FillStyle::Flat, _) => DisplayMode::FlatWithEdge,
            (FillStyle::HiddenLine, _) => DisplayMode::HiddenLine,
        }
    }

    pub fn wants_lit_solid(self) -> bool {
        self.fill == FillStyle::Shaded
    }

    pub fn is_flat_fill(self) -> bool {
        matches!(self.fill, FillStyle::Flat | FillStyle::HiddenLine)
    }

    pub fn wants_filled(self) -> bool {
        self.wants_lit_solid() || self.is_flat_fill()
    }

    pub fn wants_feature_edges(self) -> bool {
        self.edges == EdgeStyle::Crease
    }

    pub fn wants_silhouette(self) -> bool {
        self.edges == EdgeStyle::Silhouette
    }

    /// Overlay that cannot use the shared GPU crease buffer (per-draw line list).
    pub fn wants_cpu_edge_overlay(self) -> bool {
        matches!(
            self.edges,
            EdgeStyle::Silhouette | EdgeStyle::Perimeter | EdgeStyle::Hard | EdgeStyle::Adjacent
        )
    }

    /// Crease or classified overlay (depth-tested line pass).
    pub fn wants_edge_overlay(self) -> bool {
        self.wants_feature_edges() || self.wants_cpu_edge_overlay()
    }

    pub fn wants_full_edges(self) -> bool {
        self.edges == EdgeStyle::Full
    }

    /// No fill, full topology (classic wireframe).
    pub fn wants_wireframe_only(self) -> bool {
        self.fill == FillStyle::None && self.edges == EdgeStyle::Full
    }

    /// Fast hidden-line: dashed occluded edges (inverted depth, not analytic HLR).
    pub fn wants_hidden_dashes(self) -> bool {
        self.fill == FillStyle::HiddenLine && self.edges != EdgeStyle::None
    }
}

impl DisplayMode {
    pub fn fill(self) -> FillStyle {
        match self {
            DisplayMode::Shaded | DisplayMode::ShadedWithEdges => FillStyle::Shaded,
            DisplayMode::Flat | DisplayMode::FlatWithEdge => FillStyle::Flat,
            DisplayMode::HiddenLine => FillStyle::HiddenLine,
            DisplayMode::Wireframe => FillStyle::None,
        }
    }

    pub fn edges(self) -> EdgeStyle {
        match self {
            DisplayMode::Shaded | DisplayMode::Flat => EdgeStyle::None,
            DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine | DisplayMode::FlatWithEdge => {
                EdgeStyle::Crease
            }
            DisplayMode::Wireframe => EdgeStyle::Full,
        }
    }

    pub fn appearance(self) -> Appearance {
        Appearance::from_display_mode(self)
    }

    /// PBR/Phong filled pass (Coin3D `FILLED`, without the unlit Flat path).
    pub fn wants_lit_solid(self) -> bool {
        self.appearance().wants_lit_solid()
    }

    /// Unlit filled pass (`Flat` / `FlatWithEdge`).
    pub fn is_flat_fill(self) -> bool {
        self.appearance().is_flat_fill()
    }

    /// Any filled rasterization (not wireframe-only).
    pub fn wants_filled(self) -> bool {
        self.appearance().wants_filled()
    }

    /// Feature-edge overlay (Coin3D `FILLED | LINES` crease).
    pub fn wants_feature_edges(self) -> bool {
        self.appearance().wants_feature_edges()
    }

    /// Wireframe-only (Coin3D `LINES`).
    pub fn wants_wireframe_only(self) -> bool {
        self.appearance().wants_wireframe_only()
    }
}

/// Named HOOPS-style visual style (fill x edges) for a segment / subtree.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisualStyle {
    pub name: String,
    pub appearance: Appearance,
}

impl VisualStyle {
    pub const SHADED: &'static str = "Shaded";
    pub const SHADED_WITH_EDGES: &'static str = "ShadedWithEdges";
    pub const WIREFRAME: &'static str = "Wireframe";
    pub const HIDDEN_LINE: &'static str = "HiddenLine";
    pub const FLAT: &'static str = "Flat";
    pub const FLAT_WITH_EDGES: &'static str = "FlatWithEdges";
    pub const SILHOUETTE: &'static str = "Silhouette";
    pub const HARD_EDGES: &'static str = "HardEdges";
    pub const PERIMETER: &'static str = "Perimeter";
    pub const ADJACENT: &'static str = "Adjacent";
    pub const INSPECTION: &'static str = "Inspection";

    pub fn new(name: impl Into<String>, fill: FillStyle, edges: EdgeStyle) -> Self {
        Self {
            name: name.into(),
            appearance: Appearance { fill, edges },
        }
    }

    pub fn from_display_mode(name: impl Into<String>, mode: DisplayMode) -> Self {
        Self {
            name: name.into(),
            appearance: Appearance::from_display_mode(mode),
        }
    }
}

/// Catalog of named styles that can be registered and applied to a subtree.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VisualStyleLibrary {
    styles: Vec<VisualStyle>,
}

impl VisualStyleLibrary {
    pub fn new() -> Self {
        Self {
            styles: Vec::new(),
        }
    }

    /// Built-in CAD catalog (DisplayMode presets + edge-class styles).
    pub fn builtin() -> Self {
        let mut lib = Self::new();
        lib.register(VisualStyle::from_display_mode(
            VisualStyle::SHADED,
            DisplayMode::Shaded,
        ));
        lib.register(VisualStyle::from_display_mode(
            VisualStyle::SHADED_WITH_EDGES,
            DisplayMode::ShadedWithEdges,
        ));
        lib.register(VisualStyle::from_display_mode(
            VisualStyle::WIREFRAME,
            DisplayMode::Wireframe,
        ));
        lib.register(VisualStyle::from_display_mode(
            VisualStyle::HIDDEN_LINE,
            DisplayMode::HiddenLine,
        ));
        lib.register(VisualStyle::from_display_mode(
            VisualStyle::FLAT,
            DisplayMode::Flat,
        ));
        lib.register(VisualStyle::from_display_mode(
            VisualStyle::FLAT_WITH_EDGES,
            DisplayMode::FlatWithEdge,
        ));
        lib.register(VisualStyle::new(
            VisualStyle::SILHOUETTE,
            FillStyle::Shaded,
            EdgeStyle::Silhouette,
        ));
        lib.register(VisualStyle::new(
            VisualStyle::HARD_EDGES,
            FillStyle::Shaded,
            EdgeStyle::Hard,
        ));
        lib.register(VisualStyle::new(
            VisualStyle::PERIMETER,
            FillStyle::Shaded,
            EdgeStyle::Perimeter,
        ));
        lib.register(VisualStyle::new(
            VisualStyle::ADJACENT,
            FillStyle::Shaded,
            EdgeStyle::Adjacent,
        ));
        lib.register(VisualStyle::new(
            VisualStyle::INSPECTION,
            FillStyle::Shaded,
            EdgeStyle::Hard,
        ));
        lib
    }

    pub fn register(&mut self, style: VisualStyle) {
        if let Some(existing) = self.styles.iter_mut().find(|s| s.name == style.name) {
            *existing = style;
        } else {
            self.styles.push(style);
        }
    }

    pub fn get(&self, name: &str) -> Option<&VisualStyle> {
        self.styles.iter().find(|s| s.name == name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.styles.iter().map(|s| s.name.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = &VisualStyle> {
        self.styles.iter()
    }
}
