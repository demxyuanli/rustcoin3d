# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Compositor node graph** (`rc3d-render/compositor` + GPU passes): Blender-aligned
  node set — RGB/Value inputs, Math (14 ops), Exposure, Gamma, Hue/Saturation/Value,
  Invert, Alpha Over, Dilate/Erode, Translate/Rotate/Scale/Crop, Mix with 16 blend
  modes (Screen, Divide, Difference, Darken, Lighten, Overlay, Dodge, Burn, Hue,
  Saturation, Value, Color, Subtract…), CAD preset passes. WGSL execution via
  ping-pong buffers; node collapse state persists across sessions.
- **CAD display tiers** (`CadDisplayTier`): Visualization / IndustrialDisplay /
  ProductRendering with GPU capability clamping, orbit-time downgrade
  (Viz<-Industrial) and cooldown recovery; `--cad-matrix` verification matrix in
  `rc3d-studio` (table-driven rows, 12 scenarios).
- **Studio case library**: 24 parameter/process demo cases embedded in the Assets
  tab with filtering, parameter sliders and step-by-step guidance (replaces the
  floating browser window).
- **Editor keymap** (`rc3d-editor/keymap`): default shortcut table (builder-style
  `KeyChord::new().ctrl()…` constants) with per-user overrides persisted in prefs.
- **Studio split panel**: Hierarchy tab now hosts the scene tree plus the selected
  node's property table with a draggable divider (ratio persisted).
- **i18n catalogs**: 450-key en/zh-Hans catalogs audited for dead/missing keys.

### Changed
- **UI restructure**: `menus.rs` split into `menus/` (file, view, display, render,
  create, labels) with shared helpers; `draw.rs` split into `docks.rs` (dock
  geometry, resize, tab bars, hierarchy split) and `shell.rs` (status bar, dialogs,
  context menu); compositor UI split into `viewer.rs`/`widgets.rs`/`labels.rs`/
  `sync.rs`; inspector split into `fields.rs`/`render.rs`.
- **Studio host split**: `app.rs` reduced to the event loop; frame presentation,
  redraw scheduling, and command/prefs handling moved to `present.rs`, `redraw.rs`,
  `host_cmds.rs` with a `HostState` bundle.
- **Docks**: manual layout for side/bottom docks guarantees 3D-viewport alignment
  at any sidebar width (replaces egui Panel with one-frame lag); Blender-style
  hover-visible row buttons in the Hierarchy tree.
- **Editor apply layer**: `apply.rs` split into `apply/` (io, scene, render, tools).
- Dependency upgrades: wgpu 30, egui/eframe 0.36, glyphon 0.12, MSRV 1.95.

### Removed
- Dead paths: `rc3d-app` crate, editor `console` tab, `selection.rs`, cases
  browser window, `viewport_film.rs`, unused `EditorContext` fields, unused
  compositor `ADD_OPS` variants, 10 dead i18n keys.

## [2026-08-30]

### Changed
- Upgraded wgpu 30, egui 0.36, glyphon 0.12; MSRV raised to 1.95.
- Split render collector; unified viewport pick matrices.

### Added
- `rc3d-studio` desktop editor with Fluent chrome, i18n (en/zh-Hans), and
  `egui_ltreeview` hierarchy.

## [2026-08] — Coin3D/Three.js feature parity wave

### Added
- Lighting/materials: HemisphereLight, LightProbe (L2 SH), KHR anisotropy/
  clearcoat/sheen/iridescence/transmission (IBL refraction), toon/normal/depth
  visualization materials, ShaderMaterial (`custom_wgsl`).
- Geometry/nodes: Sprite, InstancedMesh, BatchedMesh, StereoCamera, TransformManip
  + Dragger hierarchy, node kits, Rotation/RotationXYZ, Font (SDF labels).
- IO: FBX import, Draco decode, KTX2/BasisU transcode, per-face materials (OBJ
  `usemtl`), render-to-image headless API.
- Post FX: volumetric fog, halftone/glitch stylize passes.
- Optimization: GPU frustum culling + indirect draw, cluster-tree LOD culling,
  LightSetTable light dedup (1280B → 4B/draw), streaming mesh pool with LRU budget,
  static-frame fast path, LightProbe SH irradiance.

[Unreleased]: https://example.invalid/compare/v0.0.0...HEAD
