use rc3d_core::NodeId;
use super::command::CliCommand;

pub fn parse(input: &str) -> Result<CliCommand, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("empty command".into());
    }

    let mut parts: Vec<&str> = trimmed.split_whitespace().collect();
    let cmd = parts.remove(0).to_lowercase();

    match cmd.as_str() {
        "scene" => parse_scene(&parts),
        "test" => parse_test(&parts),
        "camera" => parse_camera(&parts),
        "select" => parse_select(&parts),
        "prop" => parse_prop(&parts),
        "display" => parse_display(&parts),
        "log" => parse_log(&parts),
        "help" => Ok(CliCommand::Help),
        "quit" | "exit" => Ok(CliCommand::Quit),
        _ => Err(format!("unknown command: {cmd}")),
    }
}

fn parse_scene(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("load") => {
            let path = parts.get(1).ok_or("usage: scene load <path>")?;
            Ok(CliCommand::SceneLoad(path.to_string()))
        }
        Some("reset") => Ok(CliCommand::SceneReset),
        _ => Err("usage: scene load <path> | scene reset".into()),
    }
}

fn parse_test(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("run") => Ok(CliCommand::TestRun(parts.get(1).map(|s| s.to_string()))),
        Some("stop") => Ok(CliCommand::TestStop),
        _ => Err("usage: test run [suite] | test stop".into()),
    }
}

fn parse_camera(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("orbit") => {
            let dx: f32 = parts.get(1).unwrap_or(&"0").parse().map_err(|_| "invalid dx")?;
            let dy: f32 = parts.get(2).unwrap_or(&"0").parse().map_err(|_| "invalid dy")?;
            Ok(CliCommand::CameraOrbit { dx, dy })
        }
        Some("pan") => {
            let dx: f32 = parts.get(1).unwrap_or(&"0").parse().map_err(|_| "invalid dx")?;
            let dy: f32 = parts.get(2).unwrap_or(&"0").parse().map_err(|_| "invalid dy")?;
            Ok(CliCommand::CameraPan { dx, dy })
        }
        Some("zoom") => {
            let amount: f32 = parts.get(1).unwrap_or(&"1").parse().map_err(|_| "invalid zoom")?;
            Ok(CliCommand::CameraZoom(amount))
        }
        Some("fit") => Ok(CliCommand::CameraFit),
        _ => Err("usage: camera orbit|pan|zoom|fit <params>".into()),
    }
}

fn parse_select(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("clear") => Ok(CliCommand::SelectClear),
        Some(id_str) => {
            let id: u64 = id_str.parse().map_err(|_| "invalid node id")?;
            Ok(CliCommand::Select(slotmap::KeyData::from_ffi(id).into()))
        }
        None => Err("usage: select <id> | select clear".into()),
    }
}

fn parse_prop(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("set") => {
            let node_str = parts.get(1).ok_or("usage: prop set <node> <field> <value>")?;
            let field = parts.get(2).ok_or("missing field")?;
            let value = parts.get(3).ok_or("missing value")?;
            let node_id: u64 = node_str.parse().map_err(|_| "invalid node id")?;
            Ok(CliCommand::PropSet {
                node: slotmap::KeyData::from_ffi(node_id).into(),
                field: field.to_string(),
                value: value.to_string(),
            })
        }
        _ => Err("usage: prop set <node> <field> <value>".into()),
    }
}

fn parse_display(parts: &[&str]) -> Result<CliCommand, String> {
    let mode = parts.first().ok_or("usage: display <wireframe|shaded|edges|hidden>")?;
    Ok(CliCommand::DisplayMode(mode.to_string()))
}

fn parse_log(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("filter") => {
            Ok(CliCommand::LogFilter(parts.get(1).unwrap_or(&"info").to_string()))
        }
        Some("clear") => Ok(CliCommand::LogClear),
        _ => Err("usage: log filter <level> | log clear".into()),
    }
}
