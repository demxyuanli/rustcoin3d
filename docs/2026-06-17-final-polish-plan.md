# Phase 5: Final Polish — Canonical + NotchedEdges/Tails + Quality

Date: 2026-06-17.

## E1: Canonical Cylinder/Sphere detection
**File**: `heal/canonical.rs` (extend)
**Goal**: Complete the canonical recognition — detect BSpline cylinders and spheres.
**Algorithm**:
- Cylinder: check if CPs have constant distance from a fitted axis line
- Sphere: check if CPs have constant distance from a fitted center point
**Verify**: BSpline cylinder → detected, BSpline sphere → detected

## E2: FixNotchedEdges
**File**: `heal/wire_ops.rs` (extend)
**Goal**: Detect V-shaped notches in wires and fix them.
**Algorithm**:
- Find consecutive edges with angle < threshold at shared vertex
- Merge the two edges into one (remove the notch vertex)
**OCC**: `ShapeFix_Wire::FixNotchedEdges()`
**Verify**: Wire with V-notch → notch removed

## E3: FixTails
**File**: `heal/wire_ops.rs` (extend)
**Goal**: Remove small "tail" edges at wire junctions.
**Algorithm**:
- Detect edges with one vertex connected to only one other edge (dead-end)
- If edge length < threshold: remove edge and its isolated vertex
**OCC**: `ShapeFix_Wire::FixTails()`
**Verify**: Wire with tail → tail removed

## E4: Test hardening
**Goal**: Add missing edge-case tests for recent features.
- UnifySameDomain: merge 3+ coplanar faces
- FixSmallFace: merge into neighbor with inner wires
- ModelHealer: gap between 3+ meshes
- Binary: BSpline roundtrip with Polyline fallback
- IGES: composite curve import
