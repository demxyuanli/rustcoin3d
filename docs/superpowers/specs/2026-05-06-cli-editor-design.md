# CLI Editor for Test Engine — Design Spec

> **Goal:** Build a new standalone editor (new crate, not modifying existing editor) driven by a CLI command engine, with a GPU window for 3D plus egui diagnostics, and a **terminal TUI** (ratatui) as the primary command surface.

> **Tech Stack:** Rust, winit, wgpu, egui 0.31 (viewport-adjacent panels only), ratatui + crossterm (CLI TUI thread), rc3d-render, rc3d-scene, rc3d-core

---

## 1. Architecture Overview

Single process, **two UI surfaces**:

1. **Terminal (stdout):** Alternate-screen ratatui shell — scrollback log, `rc3d>` input line, Tab completion on command prefixes, history navigation (↑/↓), Enter to run parser + executor against shared `EditorSession`.
2. **GPU window:** winit + wgpu — full-frame 3D via `rc3d-render`, with egui overlay for central viewport caption, properties (right), and diagnostics (bottom). No egui CLI panel.

`EditorSession` holds `RwLock<EngineState>` shared between the **TUI thread** and **winit main thread**. Quit from TUI notifies the GPU loop via `EventLoopProxy<AppEvent>` so the window closes cleanly.

**Target rendering model (incremental toward spec):** 3D to swapchain behind egui overlay today; migrating to center **viewport texture + egui Image** remains planned.

```
   Terminal (TTY)                         GPU window (winit + egui)
┌──────────────────────┐                 ┌─────────────────────────────┐
│ command log          │                 │ viewport | props | diag     │
│ ...                  │                 │ + 3D on surface             │
└──────────────────────┘                 └─────────────────────────────┘
 │ rc3d> input                                                   ▲
 └──────────────► parse → execute ──► RwLock<EngineState> ────────┘
```

## 2. CLI Engine (Core)

All editor actions are commands. The CLI engine is the backbone.

```
Command string → Parser → Command enum → Executor → (mutates EngineState, emits Events)
```

### Command Categories
- `scene load <path>` / `scene reset` — scene lifecycle
- `test run <suite>` / `test run all` / `test stop` — test execution
- `camera orbit|pan|zoom <params>` — view control
- `select <node_id>` / `select clear` — selection
- `prop set <node> <field> <value>` — property mutation
- `log filter <level>` / `log clear` — log control

### Key Types
```rust
enum CliCommand {
    SceneLoad(PathBuf),
    TestRun(Option<String>),
    Select(NodeId),
    PropSet { node: NodeId, field: String, value: String },
    // ...
}

struct EngineState {
    scene: SceneGraph,
    selection: HashSet<NodeId>,
    active_test: Option<TestRun>,
    camera: CameraState,
    log_entries: VecDeque<LogEntry>,
}

enum EngineEvent {
    SceneLoaded,
    TestStarted(String),
    TestPassed,
    TestFailed(String),
    SelectionChanged(HashSet<NodeId>),
    LogAppended(LogEntry),
    // ...
}
```

Events fan out to the TUI (echo + scrollback) and to egui panels (read-only views of shared state).

### Terminal TUI (`src/tui/`)
- Alternate screen, bordered command log + input areas
- Command history (↑/↓) and Tab completion on static command prefixes
- Runs only when stdout is a TTY; otherwise GPU-only session with CLI disabled (`log::warn`)

## 3. Window Panels (egui)

### Center — 3D Viewport
- Renders `EngineState.scene` via rc3d-render to a wgpu texture (target: egui Image in center; current: full-surface + overlay)
- Camera controlled by CLI commands or mouse (orbit/pan/zoom)
- Selection highlights rendered as outlines
- Test result markers overlaid on nodes (pass/fail)
- Shows selected node's properties in scrollable form
- Each field is a labeled editor (text, color picker, dropdown)
- Changes emit `PropSet` commands back to CLI engine
- Empty state: scene summary (node count, draw calls, FPS)

### Bottom — Log/Diagnostics Panel
- Streaming log output, filterable by level
- Test run progress bar + summary stats
- Performance metrics (frame time, GPU memory)
- Tabbed: Log | Tests | Perf | GPU

## 4. Data Flow

```
User input (terminal TUI or widget)
    → CliCommand (via `cli::submit`)
    → Executor mutates EngineState
    → Emits EngineEvent(s)
    → TUI echoes events; egui panels read RwLock<EngineState>
    → 3D viewport re-renders, properties refreshes, logs append
```

EngineState is single source of truth. TUI writes through the parser/executor; egui panels read the same `EditorSession` (`RwLock`).

## 5. Crate Structure

```
crates/rc3d-cli-editor/
├── Cargo.toml
├── src/
│   ├── main.rs           # winit + AppEvent proxy; spawns TUI thread
│   ├── session.rs        # EditorSession = shared RwLock<EngineState>
│   ├── egui_paint.rs    # overlay painter
│   ├── tui/
│   │   └── shell.rs      # ratatui thread, EventLoopProxy quit
│   ├── cli/
│   │   ├── mod.rs
│   │   ├── parser.rs     # string → CliCommand
│   │   ├── command.rs    # CliCommand enum
│   │   ├── dispatch.rs   # submit → execute + formatted event lines
│   │   └── executor.rs   # command → state mutation + events
│   ├── engine/
│   │   ├── mod.rs
│   │   └── state.rs      # EngineState, EngineEvent, TestRun
│   ├── panels/
│   │   ├── mod.rs
│   │   ├── viewport.rs
│   │   ├── properties.rs
│   │   └── diagnostics.rs
│   └── render.rs         # wgpu setup, 3D viewport rendering
```

## 6. Dependencies

- `rc3d-render` — 3D rendering
- `rc3d-scene` — scene graph
- `rc3d-core` — math, NodeId
- `egui` + `egui-wgpu` + `egui-winit` — UI panels only
- `winit` — window/event loop
- `wgpu` — GPU access
- `ratatui` + `crossterm` — terminal TUI for commands
- `env_logger` / `log` — logging

## 7. Non-Goals

- Not modifying existing `rc3d-app` editor
- No scripting/plugin system in initial version
- No network-based remote control
