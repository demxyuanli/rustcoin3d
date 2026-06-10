//! PMI data types for AP242 annotations — surface finishes and extended data sets.

use rc3d_core::math::Vec3;

/// Surface finish / roughness annotation (ISO 1302).
#[derive(Debug, Clone)]
pub struct PmiSurfaceFinish {
    /// Ra (arithmetic average roughness) in micrometers.
    pub ra_value: Option<f32>,
    /// Rz (average maximum height) in micrometers.
    pub rz_value: Option<f32>,
    /// Symbol type indicating the machining requirement.
    pub symbol: FinishSymbol,
    /// 3D anchor point on the surface.
    pub anchor_point: Vec3,
    /// Optional additional note text (e.g. machining method).
    pub note: Option<String>,
}

/// Surface finish symbol type per ISO 1302.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum FinishSymbol {
    /// Basic machining symbol (√).
    Basic,
    /// Material removal required (⊿).
    MaterialRemoval,
    /// No material removal allowed (⊿ with circle).
    NoRemoval,
    /// Any process acceptable (no restriction).
    #[default]
    Any,
}


/// Extended PMI data set including surface finishes.
/// This wraps the existing `PmiData` fields and adds surface finish data.
#[derive(Debug, Default)]
pub struct PmiDataSet {
    /// Dimensional annotations (from pmi_extract).
    pub dimensions: Vec<super::pmi_extract::PmiDimension>,
    /// Datum identifiers (from pmi_extract).
    pub datums: Vec<super::pmi_extract::PmiDatum>,
    /// GD&T tolerance frames (from pmi_extract).
    pub tolerances: Vec<super::pmi_extract::PmiToleranceFrame>,
    /// Surface finish annotations (new in pmi_types).
    pub surface_finishes: Vec<PmiSurfaceFinish>,
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_finish_construction() {
        let finish = PmiSurfaceFinish {
            ra_value: Some(3.2),
            rz_value: Some(12.5),
            symbol: FinishSymbol::MaterialRemoval,
            anchor_point: Vec3::new(10.0, 20.0, 0.0),
            note: Some("Milled".to_string()),
        };
        assert_eq!(finish.symbol, FinishSymbol::MaterialRemoval);
        assert!((finish.ra_value.unwrap() - 3.2).abs() < 1e-6);
        assert_eq!(finish.note.as_deref(), Some("Milled"));
    }

    #[test]
    fn test_finish_symbol_default() {
        let sym = FinishSymbol::default();
        assert_eq!(sym, FinishSymbol::Any);
    }

    #[test]
    fn test_pmi_data_set_default() {
        let ds = PmiDataSet::default();
        assert!(ds.dimensions.is_empty());
        assert!(ds.datums.is_empty());
        assert!(ds.tolerances.is_empty());
        assert!(ds.surface_finishes.is_empty());
    }
}
