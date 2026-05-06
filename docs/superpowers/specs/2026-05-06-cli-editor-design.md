# CLI Editor for Test Engine — Design Spec

> **Goal:** Build a new standalone editor (new crate, not modifying existing editor) driven by a CLI command engine, with a 4-panel layout: CLI panel (left), 3D viewport (center), properties panel (right), log/diagnostics panel (bottom).

> **Tech Stack:** Rust, winit, wgpu, egui 0.31 (panels only), rc3d-render, rc3d-scene, rc3d-core

---

## 1. Architecture Overview

Single process, single winit window. egui manages 4-panel split layout. 3D viewport renders via rc3d-render directly to a wgpu texture, displayed as an egui Image in the central area.

```
┌─────────────────────────────────────────────────┐
│  rc3d-cli-editor (new crate)                     │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ CLI panel │  │  3D      │  │ Props    │       │
│  │ (left)   │  │ viewport │  │ panel    │       │
│  │          │  │ (center) │  │ (right)  │       │
│  └──────────┘  └──────────┘  └──────────┘       │
│  ┌─────────────────────────────────────────┐     │
│  │  Log / diagnostics panel (bottom)       │     │
│  └─────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
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

Events fan out to panels. CLI panel emits commands; other panels react to events.

## 3. Four Panels

### Left — CLI Panel
- Multi-line output area (command history + results)
- Single-line input at bottom
- Auto-complete (commands, node IDs, file paths)
- Command history buffer (up-arrow recall)
- Hand-rolled tokenizer for parsing (command set is small and known)

### Center — 3D Viewport
- Renders `EngineState.scene` via rc3d-render to a wgpu texture
- Camera controlled by CLI commands or mouse (orbit/pan/zoom)
- Selection highlights rendered as outlines
- Test result markers overlaid on nodes (pass/fail)

### Right — Properties Panel
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
User input (CLI or widget)
    → CliCommand
    → Executor mutates EngineState
    → Emits EngineEvent(s)
    → All panels read EngineState (shared ref)
    → 3D viewport re-renders, properties refreshes, logs append
```

EngineState is single source of truth. Panels are read-only views + input producers. No panel-to-panel direct communication.

## 5. Crate Structure

```
crates/rc3d-cli-editor/
├── Cargo.toml
├── src/
│   ├── main.rs           # winit event loop, window setup
│   ├── cli/
│   │   ├── mod.rs
│   │   ├── parser.rs     # string → CliCommand
│   │   ├── command.rs    # CliCommand enum
│   │   └── executor.rs   # command → state mutation + events
│   ├── engine/
│   │   ├── mod.rs
│   │   └── state.rs      # EngineState, EngineEvent, TestRun
│   ├── panels/
│   │   ├── mod.rs
│   │   ├── cli_panel.rs
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
- `env_logger` / `log` — logging

## 7. Non-Goals

- Not modifying existing `rc3d-app` editor
- No scripting/plugin system in initial version
- No network-based remote control
