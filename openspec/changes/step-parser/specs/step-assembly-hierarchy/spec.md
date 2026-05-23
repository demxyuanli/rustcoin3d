## ADDED Requirements

### Requirement: Extract assembly transforms from NAUO
The system SHALL parse `NEXT_ASSEMBLY_USAGE_OCCURRENCE` entities to extract product placement transforms.

#### Scenario: Single-level assembly
- **WHEN** a `NEXT_ASSEMBLY_USAGE_OCCURRENCE` references a child product and an `ITEM_DEFINED_TRANSFORMATION` with `AXIS2_PLACEMENT_3D`
- **THEN** the system computes a 4×4 transform matrix from the placement (origin, X-axis, Z-axis)

### Requirement: Build product-to-shape mapping
The system SHALL trace `PRODUCT_DEFINITION_SHAPE` → `SHAPE_DEFINITION_REPRESENTATION` → `SHAPE_REPRESENTATION` to map products to their geometric items.

#### Scenario: Product with single shape
- **WHEN** `PRODUCT_DEFINITION_SHAPE` references a `SHAPE_DEFINITION_REPRESENTATION` containing a `SHAPE_REPRESENTATION` with shell items
- **THEN** the system maps the product to its shell IDs

### Requirement: Build multi-level assembly tree
The system SHALL reconstruct the full assembly hierarchy with nested transforms.

#### Scenario: Two-level assembly
- **WHEN** assembly A contains sub-assembly B placed by transform T1, and B contains part C placed by transform T2
- **THEN** part C's geometry is transformed by T1 × T2

#### Scenario: Part reused in multiple assemblies
- **WHEN** the same product definition is referenced by two different `NEXT_ASSEMBLY_USAGE_OCCURRENCE` with different transforms
- **THEN** the geometry is instantiated twice, once for each placement

### Requirement: Apply transforms to tessellated vertices
The system SHALL transform tessellated vertex positions by the accumulated assembly transform matrix before adding to the vertex buffer.

#### Scenario: Translated part
- **WHEN** a part's `AXIS2_PLACEMENT_3D` has location (100, 0, 0)
- **THEN** all vertices of that part are offset by (100, 0, 0)

### Requirement: Build SceneGraph with assembly hierarchy
The system SHALL output a SceneGraph where each assembly node is a `Separator` with a child `Transform` node, under which the part's geometry (`Coordinate3` + `IndexedFaceSet`) is placed.

#### Scenario: Assembly with two parts
- **WHEN** the assembly has two parts with different transforms
- **THEN** the SceneGraph contains:
  - Root Separator
    - Separator (part 1) → Transform (T1) → Coordinate3 + IndexedFaceSet
    - Separator (part 2) → Transform (T2) → Coordinate3 + IndexedFaceSet

### Requirement: Handle missing transforms gracefully
The system SHALL use identity transform when a product has no explicit placement.

#### Scenario: Root-level product without transform
- **WHEN** a product definition shape is not referenced by any `NEXT_ASSEMBLY_USAGE_OCCURRENCE`
- **THEN** its geometry is placed at origin with identity transform
