## ADDED Requirements

### Requirement: Parse ISO 10303-21 exchange structure
The system SHALL parse ASCII STEP files conforming to ISO 10303-21 into indexed entity records without external STEP libraries.

#### Scenario: Parse simple exchange file
- **WHEN** input contains HEADER and DATA sections with entity instances (`#1 = CARTESIAN_POINT('', (1.0, 2.0, 3.0));`)
- **THEN** the parser returns a map of entity ID to entity name and parameter list

#### Scenario: Parse nested parameter lists
- **WHEN** input contains `#1 = ENTITY('name', (#2, #3, (4.0, 5.0)));`
- **THEN** the parser produces a `List` containing `Ref(2)`, `Ref(3)`, and a nested `List([Real(4.0), Real(5.0)])`

#### Scenario: Parse typed parameters
- **WHEN** input contains `LENGTH_MEASURE(0.001)` or `SI_UNIT(.MILLI., .METRE.)`
- **THEN** the parser produces `Typed("LENGTH_MEASURE", Real(0.001))` and `Typed("SI_UNIT", List([Enum(".MILLI."), Enum(".METRE.")]))`

#### Scenario: Handle omitted parameters
- **WHEN** input contains `$` or `*` as parameter values
- **THEN** the parser produces `Omitted` for each

#### Scenario: Parse empty data section
- **WHEN** input contains `DATA; ENDSEC;` with no entities
- **THEN** the parser returns an empty entity map

#### Scenario: Reject malformed syntax
- **WHEN** input contains an unclosed string `'unterminated` or missing semicolon after entity
- **THEN** the parser returns a `StepError::Parse` error with a descriptive message

### Requirement: Preserve entity IDs
The system SHALL preserve the numeric entity instance IDs (`#1`, `#42`) as keys in the entity map for cross-reference resolution.

#### Scenario: Entity ID preservation
- **WHEN** parsing `#42 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));`
- **THEN** the entity map contains key `42` with name `"CARTESIAN_POINT"`

### Requirement: Handle file encoding
The system SHALL accept ASCII, UTF-8, and Latin-1 encoded STEP files by detecting encoding and converting to Rust `String`.

#### Scenario: UTF-8 file
- **WHEN** input is valid UTF-8 STEP text
- **THEN** parsing proceeds normally

#### Scenario: Non-UTF-8 fallback
- **WHEN** input contains non-UTF-8 bytes
- **THEN** the parser attempts Latin-1 decoding and proceeds if successful
