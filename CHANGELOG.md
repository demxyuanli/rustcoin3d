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
- **Interactive 3D PDF export** (`rc3d-pdf`): real U3D (ECMA-363) embedding —
  hand-rolled binary writer (file header, node/resource modifier chains, CLOD
  mesh declaration + base-mesh continuation) wrapped in a PDF 1.7 `/3D`
  annotation with camera framing. `export_u3d_pdf(scene)` flattens supported
  shapes to world space via the shared scene traversal (Separator/Transform
  hierarchies honored); each shape is a separate model node so parts stay
  selectable, and `Material` nodes / `IndexedFaceSet.material_groups`
  surface as per-part diffuse colours via lit-texture shader + material
  resources (palette deduplicated). `export_mesh_u3d_pdf` takes an explicit
  triangle mesh.
- **3D PDF materials, assembly tree & view options** (`rc3d-pdf`): material
  `albedo_texture` images are PNG-embedded as U3D texture resources (declaration
  + continuation blocks, UVs ride in the base mesh) and material `opacity`
  maps to the material opacity channel; named Separators become U3D
  GroupNodes (0xFFFFFF21) so Acrobat's model tree mirrors the scene assembly
  (parent/child name links, identity transforms, per-part world-space
  geometry, duplicate labels auto-renamed); new `export_u3d_pdf_opts` +
  `PdfOptions` configure the initial 3D view — `/RM` render mode, `/LS`
  lighting scheme (non-White schemes use the PDF 1.7 `/3DLightingScheme`
  dictionary), `/BG` solid DeviceRGB background, `/P` field of view, camera
  zoom and framing-center override. Studio File menu “Export 3D PDF (U3D)…”
  wires the exporter into the editor command bus.
- **Studio 3D PDF export options dialog**: “Export 3D PDF (U3D)…” now opens a
  lightweight options dialog (title, render mode, lighting, background colour,
  FOV/zoom) backed by `PdfOptions` before writing the file; the command bus
  `Export3dPdf` command carries the chosen options through to
  `export_u3d_pdf_opts`.

### Changed
- **3D PDF export dialog remembers its context**: the file picker starts in
  the last-used folder (the editor's `file_dialog_dir`, persisted across
  sessions) with a default `export.pdf` name, and the confirmed `PdfOptions`
  seed the next export instead of resetting to defaults.
- **Settings relocated to title bar**: the View menu “Settings” item was
  replaced by a gear button on the right side of the caption bar that toggles a
  top-right anchored settings panel (shared body with the old menu); the menu
  item stays only when the caption bar is disabled.
- **`rc3d-pdf` module restructure**: `u3d.rs` (1009 L) split into a `u3d/`
  module directory — `u3d/mod.rs` holds the production writer and
  `u3d/tests.rs` the unit-test suite; byte-for-byte identical output.
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
