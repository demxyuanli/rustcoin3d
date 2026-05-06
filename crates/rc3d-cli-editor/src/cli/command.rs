use rc3d_core::NodeId;

#[derive(Debug, Clone)]
pub enum CliCommand {
    SceneLoad(String),
    SceneReset,
    TestRun(Option<String>),
    TestStop,
    CameraOrbit { dx: f32, dy: f32 },
    CameraPan { dx: f32, dy: f32 },
    CameraZoom(f32),
    CameraFit,
    Select(NodeId),
    SelectClear,
    PropSet {
        node: NodeId,
        field: String,
        value: String,
    },
    DisplayMode(String),
    LogFilter(String),
    LogClear,
    Help,
    Quit,
}
