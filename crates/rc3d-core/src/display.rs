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
