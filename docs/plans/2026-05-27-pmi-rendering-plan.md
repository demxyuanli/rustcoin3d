# PMI Rendering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge extracted STEP PMI data into the existing 3D annotation rendering pipeline, with fixed geometry resolution and 8 new tolerance entity types.

**Architecture:** Fix `extract_pmi()` geometry placeholders → add 8 tolerance EntityType variants → map `PmiData` to `AnnotationElement` (using `GdtFeatureControlFrame` for tolerances, `Dimension`/`Datum` for annotations) → wrap in `AnnotationSetNode` → attach to SceneGraph. Render handled by existing `pass_markup`.

**Tech Stack:** Rust, existing `rc3d-scene/annotation/` system, `AnnotationElement::GdtFeatureControlFrame`

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `crates/rc3d-io/src/step/entity_types.rs` | Modify | +8 tolerance/datum EntityType variants |
| `crates/rc3d-io/src/step/pmi/pmi_extract.rs` | Modify | Fix geometry extraction + tolerance subtypes |
| `crates/rc3d-io/src/step/pmi/pmi_render.rs` | Modify | PmiData → AnnotationSetNode bridge |
| `crates/rc3d-io/src/step/mod.rs` | Modify | Wire into import pipeline |
| `crates/rc3d-io/Cargo.toml` | Modify | `pmi` feature default on |

---

### Task 1: Add 8 tolerance/datum EntityType variants

**Files:**
- Modify: `crates/rc3d-io/src/step/entity_types.rs`

- [ ] **Step 1: Add 8 variants to EntityType enum**

In `entity_types.rs`, in the PMI section (after `GeometricTolerance,` around line 81), add:

```rust
    // Tolerance subtypes
    FlatnessTolerance,
    PositionTolerance,
    ProfileTolerance,
    ParallelismTolerance,
    PerpendicularityTolerance,
    RunoffTolerance,
    StraightnessTolerance,
    // Datum
    DatumReferenceElement,
```

- [ ] **Step 2: Add string mappings to from_name()**

In the `from_name` match block, after `"GEOMETRIC_TOLERANCE" => Self::GeometricTolerance,` (around line 170), add:

```rust
            "FLATNESS_TOLERANCE" => Self::FlatnessTolerance,
            "POSITION_TOLERANCE" => Self::PositionTolerance,
            "LINE_PROFILE_TOLERANCE" | "SURFACE_PROFILE_TOLERANCE" | "PROFILE_TOLERANCE" => Self::ProfileTolerance,
            "PARALLELISM_TOLERANCE" => Self::ParallelismTolerance,
            "PERPENDICULARITY_TOLERANCE" => Self::PerpendicularityTolerance,
            "CIRCULAR_RUNOUT_TOLERANCE" | "TOTAL_RUNOUT_TOLERANCE" => Self::RunoffTolerance,
            "STRAIGHTNESS_TOLERANCE" => Self::StraightnessTolerance,
            "DATUM_REFERENCE_ELEMENT" => Self::DatumReferenceElement,
```

- [ ] **Step 3: Build check**

```bash
rtk cargo check -p rc3d-io
```

Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-io/src/step/entity_types.rs
rtk git commit -m "feat(step): add 8 tolerance/datum EntityType variants for PMI"
```

---

### Task 2: Fix geometry extraction in pmi_extract.rs

**Files:**
- Modify: `crates/rc3d-io/src/step/value.rs`
- Modify: `crates/rc3d-io/src/step/pmi/pmi_extract.rs`

**Why:** Current `extract_dimension`, `extract_datum`, `extract_tolerance` return placeholder geometry (Vec3::ZERO). Need real 3D positions from STEP entity chains. Also need to add `as_string()` to StepValue — existing pmi code uses it but the method doesn't exist (pmi feature was never compiled).

- [ ] **Step 0: Add as_string() method to StepValue**

In `crates/rc3d-io/src/step/value.rs`, add after `as_int()`:

```rust
    pub fn as_string(&self) -> Option<&str> {
        match self {
            StepValue::String(s) => Some(s),
            StepValue::Typed(_, inner) => inner.as_string(),
            _ => None,
        }
    }
```

- [ ] **Step 1: Add helper to resolve points from entity references**

After the `extract_tolerance` function, add:

```rust
/// Resolve a 3D point from a STEP entity reference (CARTESIAN_POINT or AXIS2_PLACEMENT_3D.origin).
fn resolve_pmi_point(ref_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let rec = entities.get(&ref_id)?;
    match rec.entity_type {
        EntityType::CartesianPoint => {
            let coords = rec.params.nth_param(0)?;
            let list = coords.as_list()?;
            let x = list.first().and_then(|v| v.as_real())? as f32;
            let y = list.get(1).and_then(|v| v.as_real())? as f32;
            let z = list.get(2).and_then(|v| v.as_real())? as f32;
            Some(Vec3::new(x, y, z))
        }
        _ => None,
    }
}

/// Resolve first CARTESIAN_POINT from a list of entity references.
fn resolve_pmi_points(ref_ids: &[u64], entities: &EntityIndex) -> Vec<Vec3> {
    ref_ids.iter().filter_map(|&id| resolve_pmi_point(id, entities)).collect()
}
```

- [ ] **Step 2: Fix extract_dimension() — resolve real start/end positions**

Replace the placeholder return in `extract_dimension()` (lines 93-99):

```rust
    // Resolve ANNOTATION_OCCURRENCE to get the STYLED_ITEM chain and reference points
    let mut start = Vec3::ZERO;
    let mut end = Vec3::new(1.0, 0.0, 0.0);
    let mut found = false;

    for (_, anno_rec) in entities.iter() {
        if anno_rec.entity_type != EntityType::AnnotationOccurrence {
            continue;
        }
        // ANNOTATION_OCCURRENCE(name, item, styled_item)
        if let Some(item_id) = anno_rec.params.nth_param(1).and_then(|v| v.as_ref_id()) {
            if item_id == id_of_current_dimension {
                // Extract reference points from ANNOTATION_OCCURRENCE params[3] (if present)
                if let Some(ref_pts) = anno_rec.params.nth_param(3).and_then(|v| v.as_list()) {
                    let ref_ids: Vec<u64> = ref_pts.iter().filter_map(|v| v.as_ref_id()).collect();
                    let pts = resolve_pmi_points(&ref_ids, entities);
                    if pts.len() >= 2 {
                        start = pts[0];
                        end = pts[1];
                        found = true;
                    } else if pts.len() == 1 {
                        start = pts[0];
                    }
                }
                break;
            }
        }
    }

    if !found {
        // Fallback: search for CARTESIAN_POINT refs in the entity's own params
        let point_ids: Vec<u64> = params
            .iter_list()
            .flat_map(|v| v.iter())
            .filter_map(|v| v.as_ref_id())
            .filter(|&id| {
                entities.get(&id)
                    .map(|r| r.entity_type == EntityType::CartesianPoint)
                    .unwrap_or(false)
            })
            .collect();
        let pts = resolve_pmi_points(&point_ids, entities);
        if pts.len() >= 2 {
            start = pts[0];
            end = pts[1];
        }
    }

    let offset_dir = Vec3::Y;
    Some(PmiDimension { start, end, offset_dir, text })
```

Note: Replace `id_of_current_dimension` with the actual entity ID. Pass it as a parameter to `extract_dimension`.

- [ ] **Step 3: Update extract_dimension() signature to accept entity_id**

Change signature from:
```rust
fn extract_dimension(params: &StepValue, entities: &EntityIndex) -> Option<PmiDimension>
```
to:
```rust
fn extract_dimension(entity_id: u64, params: &StepValue, entities: &EntityIndex) -> Option<PmiDimension>
```

Update call site in `extract_pmi()`:
```rust
EntityType::DimensionalSize => {
    if let Some(dim) = extract_dimension(eid, &record.params, entities) {
        pmi.dimensions.push(dim);
    }
}
```

Where `eid` is the entity ID from the loop iteration.

- [ ] **Step 4: Fix extract_datum() — resolve origin from DATUM_FEATURE/AXIS2_PLACEMENT_3D**

Replace the `extract_datum` function body (lines 102-127):

```rust
fn extract_datum(params: &StepValue, entities: &EntityIndex) -> Option<PmiDatum> {
    let label = params.nth_param(1)
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "DATUM".to_string());

    let mut origin = Vec3::ZERO;
    let mut normal = Vec3::Z;

    // Walk: DATUM → DATUM_FEATURE → AXIS2_PLACEMENT_3D
    if let Some(ref_list) = params.nth_param(2).and_then(|v| v.as_list()) {
        for rv in ref_list {
            if let Some(ref_id) = rv.as_ref_id() {
                if let Some(rec) = entities.get(&ref_id) {
                    match rec.entity_type {
                        EntityType::DatumFeature => {
                            // DATUM_FEATURE(name, label, geometry)
                            if let Some(geom_id) = rec.params.nth_param(2).and_then(|v| v.as_ref_id()) {
                                if let Some(geom_rec) = entities.get(&geom_id) {
                                    if geom_rec.entity_type == EntityType::Axis2Placement3D {
                                        if let Some((o, axis, _)) = super::super::topology::resolve_placement(
                                            geom_id, entities
                                        ) {
                                            origin = o;
                                            normal = axis;
                                        }
                                    }
                                }
                            }
                        }
                        EntityType::Axis2Placement3D => {
                            if let Some((o, axis, _)) = super::super::topology::resolve_placement(
                                ref_id, entities
                            ) {
                                origin = o;
                                normal = axis;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Some(PmiDatum { origin, normal, label })
}
```

- [ ] **Step 5: Add PMI point resolution test**

In `pmi_extract.rs` tests module (around line 147), add:

```rust
    #[test]
    fn test_extract_pmi_dimension_with_points() {
        let input = "\
ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('pt1', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('pt2', (10.0, 0.0, 0.0));
#3 = DIMENSIONAL_CHARACTERISTIC_REPRESENTATION('', 10.0);
#4 = DIMENSIONAL_SIZE('dist', '', #3);
#5 = ANNOTATION_OCCURRENCE('', #4, $, (#1, #2));
ENDSEC;
END-ISO-10303-21;
";
        let ex = parser::parse_exchange(input).unwrap();
        let pmi = extract_pmi(&ex.entities);
        assert_eq!(pmi.dimensions.len(), 1);
        let dim = &pmi.dimensions[0];
        assert!((dim.start.x - 0.0).abs() < 1e-6);
        assert!((dim.end.x - 10.0).abs() < 1e-6);
    }
```

- [ ] **Step 6: Run tests**

```bash
rtk cargo test -p rc3d-io -- pmi
```

Expected: 2 tests pass (existing + new).

- [ ] **Step 7: Commit**

```bash
rtk git add crates/rc3d-io/src/step/pmi/pmi_extract.rs
rtk git commit -m "fix(step): resolve real PMI geometry from STEP entity chains"
```

---

### Task 3: Add tolerance subtype extraction

**Files:**
- Modify: `crates/rc3d-io/src/step/pmi/pmi_extract.rs`

- [ ] **Step 1: Add extract_pmi match arms for 8 new tolerance types**

In `extract_pmi()`, after the `EntityType::GeometricTolerance` match arm (around line 57), add:

```rust
            EntityType::FlatnessTolerance
            | EntityType::PositionTolerance
            | EntityType::ProfileTolerance
            | EntityType::ParallelismTolerance
            | EntityType::PerpendicularityTolerance
            | EntityType::RunoffTolerance
            | EntityType::StraightnessTolerance => {
                if let Some(tol) = extract_tolerance(&record.params, entities) {
                    pmi.tolerances.push(tol);
                }
            }
```

- [ ] **Step 2: Add GDT symbol mapping helper**

After `extract_tolerance()` function, add:

```rust
use rc3d_scene::node_data::GdtSymbol;

fn gdt_symbol_for_entity(entity_type: EntityType) -> GdtSymbol {
    match entity_type {
        EntityType::FlatnessTolerance | EntityType::GeometricTolerance => GdtSymbol::Flatness,
        EntityType::PositionTolerance => GdtSymbol::Position,
        EntityType::ProfileTolerance => GdtSymbol::ProfileOfSurface,
        EntityType::ParallelismTolerance => GdtSymbol::Parallelism,
        EntityType::PerpendicularityTolerance => GdtSymbol::Perpendicularity,
        EntityType::RunoffTolerance => GdtSymbol::CircularRunout,
        EntityType::StraightnessTolerance => GdtSymbol::Straightness,
        _ => GdtSymbol::Flatness,
    }
}
```

- [ ] **Step 3: Build check**

```bash
rtk cargo check -p rc3d-io
```

Expected: compiles cleanly. (Requires `rc3d-scene` dependency — already present in `Cargo.toml`.)

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-io/src/step/pmi/pmi_extract.rs
rtk git commit -m "feat(step): extract tolerance subtypes with GDT symbol mapping"
```

---

### Task 4: Build PMI → AnnotationSetNode bridge

**Files:**
- Modify: `crates/rc3d-io/src/step/pmi/pmi_render.rs`

- [ ] **Step 1: Replace pmi_render.rs stub with bridge implementation**

Replace the entire file content:

```rust
//! Convert PMI data to scene graph annotation nodes.
//! Maps PmiData → AnnotationElement[] → AnnotationSetNode → SceneGraph.
//! Rendering handled by existing pass_markup + plane_text pipeline.

use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;
use rc3d_scene::node_data::{
    AnnotationElement, AnnotationSetNode, NodeData,
};
use rc3d_scene::annotation::{
    AnnotationLabelMode, AnnotationPoint, AnnotationStyle,
};

use super::pmi_extract::{PmiData, PmiDimension, PmiDatum, PmiToleranceFrame};

/// Default annotation style for imported PMI.
fn pmi_style() -> AnnotationStyle {
    AnnotationStyle {
        decimals: 3,
        unit_suffix: " mm".to_string(),
        font_size: 14.0,
        ..AnnotationStyle::default()
    }
}

/// Convert a PmiDimension to an AnnotationElement.
fn dimension_to_element(dim: &PmiDimension) -> AnnotationElement {
    AnnotationElement::Dimension {
        start: AnnotationPoint::local([dim.start.x, dim.start.y, dim.start.z]),
        end: AnnotationPoint::local([dim.end.x, dim.end.y, dim.end.z]),
        offset_dir: [dim.offset_dir.x, dim.offset_dir.y, dim.offset_dir.z],
        extension_len: 0.3,
        arrow_size: 0.15,
        label: dim.text.clone(),
        label_mode: AnnotationLabelMode::Fixed,
        color: [1.0, 1.0, 0.0, 1.0], // yellow
    }
}

/// Convert a PmiDatum to an AnnotationElement.
fn datum_to_element(datum: &PmiDatum) -> AnnotationElement {
    AnnotationElement::Datum {
        position: AnnotationPoint::local([datum.origin.x, datum.origin.y, datum.origin.z]),
        size: 0.3,
        color: [1.0, 0.5, 0.0, 1.0], // orange
    }
}

/// Convert a PmiToleranceFrame to an AnnotationElement (GdtFeatureControlFrame).
fn tolerance_to_element(tol: &PmiToleranceFrame) -> AnnotationElement {
    AnnotationElement::GdtFeatureControlFrame {
        symbol: tol.symbol,
        tolerance: tol.value,
        diameter: tol.diameter,
        datum_primary: tol.datum_primary.clone(),
        datum_secondary: tol.datum_secondary.clone(),
        material_condition: tol.material_condition,
        position: AnnotationPoint::local([tol.origin.x, tol.origin.y, tol.origin.z]),
        leader_target: tol.leader_points.first().map(|&p| {
            AnnotationPoint::local([p.x, p.y, p.z])
        }),
        color: [0.3, 0.8, 1.0, 1.0], // blue
    }
}

/// Convert all PMI data into a Vec<AnnotationElement>.
fn pmi_to_elements(pmi: &PmiData) -> Vec<AnnotationElement> {
    let mut elements = Vec::new();

    for dim in &pmi.dimensions {
        elements.push(dimension_to_element(dim));
    }
    for datum in &pmi.datums {
        elements.push(datum_to_element(datum));
    }
    for tol in &pmi.tolerances {
        elements.push(tolerance_to_element(tol));
    }

    elements
}

/// Attach PMI annotations to the scene graph as an AnnotationSet node.
/// Returns the NodeId of the created AnnotationSet node.
pub fn attach_pmi_to_scene(
    graph: &mut SceneGraph,
    parent: NodeId,
    pmi: &PmiData,
) -> NodeId {
    let elements = pmi_to_elements(pmi);
    if elements.is_empty() {
        return parent;
    }
    let set = AnnotationSetNode {
        elements,
        visible: true,
        style: pmi_style(),
    };
    graph.add_child(parent, NodeData::AnnotationSet(set))
}
```

- [ ] **Step 2: Update PmiToleranceFrame struct to include GdtFeatureControlFrame fields**

In `pmi_extract.rs`, replace the `PmiToleranceFrame` struct:

```rust
/// Geometric tolerance frame with GD&T typed fields.
#[derive(Debug, Clone)]
pub struct PmiToleranceFrame {
    pub origin: Vec3,
    pub leader_points: Vec<Vec3>,
    pub text: String,
    pub symbol: rc3d_scene::node_data::GdtSymbol,
    pub value: f32,
    pub diameter: bool,
    pub datum_primary: Option<String>,
    pub datum_secondary: Option<String>,
    pub material_condition: Option<rc3d_scene::node_data::GdtMaterialCondition>,
}
```

- [ ] **Step 3: Update extract_tolerance() to fill new GDT fields**

In `pmi_extract.rs`, replace the `extract_tolerance` function:

```rust
fn extract_tolerance(
    entity_type: EntityType,
    params: &super::super::value::StepValue,
    entities: &EntityIndex,
) -> Option<PmiToleranceFrame> {
    let text = params.nth_param(1)
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "TOL".to_string());

    let mut origin = Vec3::ZERO;
    let mut leader_points = vec![];
    let mut value = 0.0f32;
    let mut diameter = false;
    let datum_primary: Option<String> = None;
    let datum_secondary: Option<String> = None;

    // Resolve tolerance value from DIMENSIONAL_CHARACTERISTIC_REPRESENTATION
    if let Some(nom_val) = params.nth_param(2) {
        if let Some(id) = nom_val.as_ref_id() {
            if let Some(rec) = entities.get(&id) {
                if let Some(v) = rec.params.nth_param(1) {
                    value = v.as_real().unwrap_or(0.0) as f32;
                }
            }
        }
    }

    // Resolve position: search ANNOTATION_OCCURRENCE for anchor points
    for (_, anno_rec) in entities.iter() {
        if anno_rec.entity_type != EntityType::AnnotationOccurrence {
            continue;
        }
        if let Some(ref_pts) = anno_rec.params.nth_param(3).and_then(|v| v.as_list()) {
            let ref_ids: Vec<u64> = ref_pts.iter().filter_map(|v| v.as_ref_id()).collect();
            let pts = resolve_pmi_points(&ref_ids, entities);
            if let Some(&first) = pts.first() {
                origin = first;
            }
            leader_points = pts;
        }
    }

    Some(PmiToleranceFrame {
        origin,
        leader_points,
        text,
        symbol: gdt_symbol_for_entity(entity_type),
        value,
        diameter,
        datum_primary,
        datum_secondary,
        material_condition: None,
    })
}
```

- [ ] **Step 4: Build check**

```bash
rtk cargo check -p rc3d-io
```

Expected: compiles cleanly.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-io/src/step/pmi/pmi_extract.rs crates/rc3d-io/src/step/pmi/pmi_render.rs
rtk git commit -m "feat(step): PMI→AnnotationSetNode bridge with GdtFeatureControlFrame"
```

---

### Task 5: Wire into import pipeline + feature flag default

**Files:**
- Modify: `crates/rc3d-io/src/step/mod.rs`
- Modify: `crates/rc3d-io/Cargo.toml`

- [ ] **Step 1: Enable pmi feature by default in Cargo.toml**

Read `crates/rc3d-io/Cargo.toml`. Find the `[features]` section. If `default = [...]` exists, add `"pmi"` to it. If not, add:

```toml
[features]
default = ["pmi"]
pmi = []
```

- [ ] **Step 2: Wire PMI into exchange_to_scene_graph()**

In `mod.rs`, inside `exchange_to_scene_graph()`, before the final `log::info!(...)` (around line 358), add:

```rust
    // PMI annotations
    {
        let pmi_data = pmi::pmi_extract::extract_pmi(&exchange.entities);
        let has_pmi = !pmi_data.dimensions.is_empty()
            || !pmi_data.datums.is_empty()
            || !pmi_data.tolerances.is_empty();
        if has_pmi {
            log::info!(
                "[STEP] PMI: {} dims, {} datums, {} tolerances",
                pmi_data.dimensions.len(),
                pmi_data.datums.len(),
                pmi_data.tolerances.len(),
            );
            pmi::pmi_render::attach_pmi_to_scene(&mut graph, root, &pmi_data);
        }
    }
```

- [ ] **Step 3: Remove cfg(feature = "pmi") from mod.rs header**

In `mod.rs`, remove the `#[cfg(feature = "pmi")]` gate from line 3:

Change:
```rust
#[cfg(feature = "pmi")]
pub mod pmi;
```
To:
```rust
pub mod pmi;
```

- [ ] **Step 4: Build check + run tests**

```bash
rtk cargo check -p rc3d-io
rtk cargo test -p rc3d-io
```

Expected: compiles cleanly. 281+ tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-io/src/step/mod.rs crates/rc3d-io/Cargo.toml
rtk git commit -m "feat(step): wire PMI annotations into import pipeline, default pmi feature"
```

---

### Task 6: Integration test with PMI STEP data

**Files:**
- Modify: `crates/rc3d-io/tests/step_files.rs`

- [ ] **Step 1: Add PMI extraction test**

In `step_files.rs`, add a test:

```rust
#[test]
fn test_pmi_extraction_from_step_snippet() {
    let input = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('pt1', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('pt2', (10.0, 0.0, 0.0));
#3 = DIMENSIONAL_CHARACTERISTIC_REPRESENTATION('', 10.0);
#4 = DIMENSIONAL_SIZE('dist', '', #3);
#5 = ANNOTATION_OCCURRENCE('', #4, $, (#1, #2));
#10 = CARTESIAN_POINT('org', (5.0, 0.0, 0.0));
#11 = AXIS2_PLACEMENT_3D('', #10, #12, #13);
#12 = DIRECTION('', (0.0, 0.0, 1.0));
#13 = DIRECTION('', (1.0, 0.0, 0.0));
#14 = DATUM('', 'A', (#11));
#20 = CARTESIAN_POINT('org2', (5.0, 2.0, 0.0));
#21 = GEOMETRIC_TOLERANCE('', '', #3);
#22 = ANNOTATION_OCCURRENCE('', #21, $, (#20));
ENDSEC;
END-ISO-10303-21;
    ";
    let graph = rc3d_io::step::parse_step(input).expect("parse");
    // Walk scene graph to find AnnotationSet node
    let mut found_annot = false;
    for (nid, entry) in graph.iter() {
        if matches!(&entry.data, rc3d_scene::node_data::NodeData::AnnotationSet(s) if !s.elements.is_empty()) {
            found_annot = true;
            break;
        }
    }
    assert!(found_annot, "PMI AnnotationSet node should exist in parsed scene graph");
}
```

- [ ] **Step 2: Run test**

```bash
rtk cargo test -p rc3d-io -- step_files test_pmi_extraction
```

Expected: test passes.

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-io/tests/step_files.rs
rtk git commit -m "test(step): verify PMI extraction and annotation node creation"
```

