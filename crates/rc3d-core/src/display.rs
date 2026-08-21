use serde::{Deserialize, Serialize};

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

impl DisplayMode {
    /// PBR/Phong filled pass (Coin3D `FILLED`, without the unlit Flat path).
    pub fn wants_lit_solid(self) -> bool {
        matches!(
            self,
            DisplayMode::Shaded | DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine
        )
    }

    /// Unlit filled pass (`Flat` / `FlatWithEdge`).
    pub fn is_flat_fill(self) -> bool {
        matches!(self, DisplayMode::Flat | DisplayMode::FlatWithEdge)
    }

    /// Any filled rasterization (not wireframe-only).
    pub fn wants_filled(self) -> bool {
        self.wants_lit_solid() || self.is_flat_fill()
    }

    /// Feature-edge overlay (Coin3D `FILLED | LINES`).
    pub fn wants_feature_edges(self) -> bool {
        matches!(
            self,
            DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine | DisplayMode::FlatWithEdge
        )
    }

    /// Wireframe-only (Coin3D `LINES`).
    pub fn wants_wireframe_only(self) -> bool {
        matches!(self, DisplayMode::Wireframe)
    }
}
