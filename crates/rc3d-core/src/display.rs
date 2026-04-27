#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisplayMode {
    Shaded,
    Wireframe,
    ShadedWithEdges,
    HiddenLine,
}

impl Default for DisplayMode {
    fn default() -> Self {
        Self::ShadedWithEdges
    }
}
