//! VRML 2.0 (VRML97) geometry reader.
//!
//! Minimal parser covering the 95% case: `Coordinate`, `IndexedFaceSet`,
//! `Transform`, `Shape`, and `Material`. Unsupported nodes (lights, cameras,
//! textures, scripts, sensors) are silently skipped.
//!
//! ## Parsing strategy
//! 1. Tokenize: strip `#`-comments, split on whitespace / `{` `}` `[` `]` `,`
//! 2. Recursive descent: each node is `NodeName { field* }`
//! 3. Transform stack: compose `translation * rotation * scale` in f64,
//!    pass down to children; convert to f32 `Mat4` on output
//! 4. Geometry collection: when `Shape` → `IndexedFaceSet` → `Coordinate`
//!    is found, emit `(MeshResult, Option<Mat4>)`

use std::path::Path;

use rc3d_core::math::{Mat4, PMat4, PQuat, PVec3, PVec4, Real};
use rc3d_shape::mesh_result::MeshResult;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a VRML file and return mesh data, one entry per `Shape`.
///
/// Each entry is `(mesh, optional_world_transform)`. The transform maps
/// the mesh from its local coordinate frame into the file's world frame.
/// `None` means identity.
pub fn import_vrml(path: &Path) -> Result<Vec<(MeshResult, Option<Mat4>)>, VrmlError> {
    let text = std::fs::read_to_string(path)?;
    parse_vrml_str(&text)
}

/// Parse a VRML string into mesh data.
pub fn parse_vrml_str(input: &str) -> Result<Vec<(MeshResult, Option<Mat4>)>, VrmlError> {
    let tokens = tokenize(input);
    let mut pos = 0usize;
    let mut results = Vec::new();

    // Skip VRML header: `#VRML V2.0 utf8` (no braces, just bare tokens).
    if pos < tokens.len() && tokens[pos].starts_with("#VRML") {
        pos += 1; // skip #VRML
        while pos < tokens.len() && tokens[pos] != "{" {
            pos += 1;
        }
        // pos now points to '{' of first node (or EOF)
    }

    while pos < tokens.len() {
        collect_meshes(&tokens, &mut pos, &PMat4::IDENTITY, &mut results)?;
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum VrmlError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Tokenize VRML source into a flat list of tokens.
///
/// Rules:
/// - `#` starts a comment → rest of line ignored
/// - `{` `}` `[` `]` `,` become their own tokens
/// - Whitespace separates tokens
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    for line in input.lines() {
        // Strip comments
        let content = match line.find('#') {
            Some(pos) => &line[..pos],
            None => line,
        };

        let mut current = String::new();
        let flush = |cur: &mut String, out: &mut Vec<String>| {
            if !cur.is_empty() {
                out.push(cur.clone());
                cur.clear();
            }
        };

        for ch in content.chars() {
            match ch {
                '{' | '}' | '[' | ']' | ',' => {
                    flush(&mut current, &mut tokens);
                    tokens.push(ch.to_string());
                }
                c if c.is_whitespace() => {
                    flush(&mut current, &mut tokens);
                }
                _ => {
                    current.push(ch);
                }
            }
        }
        flush(&mut current, &mut tokens);
    }

    // Second pass: merge consecutive commas (some VRML files have ",,")
    // and filter empty tokens.
    tokens.retain(|t| !t.is_empty());
    tokens.retain(|t| *t != ",");

    tokens
}

// ---------------------------------------------------------------------------
// Recursive descent parser
// ---------------------------------------------------------------------------

/// Parse a single node (name already consumed in some contexts) and push any
/// meshes found into `results`. Advances `pos` past the closing `}`.
///
/// Returns `Ok(())` even when the node is unsupported (it is skipped).
fn collect_meshes(
    tokens: &[String],
    pos: &mut usize,
    transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    if *pos >= tokens.len() {
        return Ok(());
    }

    // Read node name. If the next token is `{` or `}` (happens with stray
    // braces from DEF/USE or header), skip it.
    let token = &tokens[*pos];
    if token == "}" || token == "]" {
        *pos += 1;
        return Ok(());
    }
    if token == "{" {
        // Stray open brace — skip its body
        *pos += 1;
        skip_balanced(&tokens, pos, "{", "}")?;
        return Ok(());
    }

    let node_name = tokens[*pos].to_lowercase();
    *pos += 1;

    // VRML allows `DEF Name NodeType { ... }` and `USE Name`.
    // Handle DEF: skip the name and re-read node type.
    if node_name == "def" {
        if *pos < tokens.len() {
            *pos += 1; // skip the DEF name
        }
        if *pos < tokens.len() {
            let real_name = tokens[*pos].to_lowercase();
            *pos += 1;
            return collect_node_body(&real_name, tokens, pos, transform, results);
        }
        return Ok(());
    }
    if node_name == "use" {
        // USE references aren't resolved — skip the name and return
        if *pos < tokens.len() {
            *pos += 1;
        }
        return Ok(());
    }

    collect_node_body(&node_name, tokens, pos, transform, results)
}

/// Parse the `{ field* }` body of a node whose name has already been consumed.
fn collect_node_body(
    node_name: &str,
    tokens: &[String],
    pos: &mut usize,
    transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    // Expect '{'
    if *pos < tokens.len() && tokens[*pos] == "{" {
        *pos += 1;
    }

    match node_name {
        "transform" => parse_transform_fields(tokens, pos, transform, results),
        "shape" => parse_shape_fields(tokens, pos, transform, results),
        "group" => {
            // Group is like Transform without the transform — just process children
            parse_group_fields(tokens, pos, transform, results)
        }
        "separator" => {
            // Separator is like Group (push/pop from Inventor — treat as Group)
            parse_group_fields(tokens, pos, transform, results)
        }
        _ => {
            // Unknown node — skip its body
            skip_to_matching_brace(tokens, pos)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Node-specific field parsers
// ---------------------------------------------------------------------------

/// Parse `Transform { translation ... rotation ... scale ... children [...] }`.
fn parse_transform_fields(
    tokens: &[String],
    pos: &mut usize,
    parent_transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    let mut translation = PVec3::ZERO;
    let mut rotation_axis = PVec3::Z;
    let mut rotation_angle: Real = 0.0;
    let mut has_rotation = false;
    let mut scale = PVec3::new(1.0, 1.0, 1.0);

    loop {
        if *pos >= tokens.len() {
            return Ok(());
        }
        let token = &tokens[*pos];
        match token.as_str() {
            "}" => {
                *pos += 1;
                return Ok(());
            }
            "]" => {
                // We're in a children array that was consumed; pop out
                *pos += 1;
                return Ok(());
            }
            _ => {}
        }

        let field = tokens[*pos].to_lowercase();
        *pos += 1;

        match field.as_str() {
            "translation" => {
                translation = parse_pvec3(tokens, pos)?;
            }
            "rotation" => {
                rotation_axis = parse_pvec3(tokens, pos)?;
                rotation_angle = parse_real(tokens, pos)?;
                has_rotation = true;
            }
            "scale" => {
                scale = parse_pvec3(tokens, pos)?;
            }
            "center" => {
                // Consume 3 floats (center point)
                let _ = parse_pvec3(tokens, pos);
            }
            "scaleorientation" => {
                // Consume 4 floats (axis + angle)
                let _ = parse_pvec3(tokens, pos);
                let _ = parse_real(tokens, pos);
            }
            "bboxcenter" | "bboxsize" => {
                // Consume 3 floats
                let _ = parse_pvec3(tokens, pos);
            }
            "children" => {
                let local = build_transform(translation, rotation_axis, rotation_angle, scale, has_rotation);
                let composed = *parent_transform * local;
                parse_children_array(tokens, pos, &composed, results)?;
            }
            _ => {
                // Unknown field — skip its value
                skip_field_value(tokens, pos)?;
            }
        }
    }
}

/// Parse `Group { children [...] }` — or `Separator { Shape ... }`.
///
/// In strict VRML 2.0, Group always uses `children [...]`.  In practice many
/// files (especially Open Inventor legacy) place child nodes directly inside
/// Separator / Group, so we treat any unknown field that looks like a node
/// (followed by `{`) as a direct child.
fn parse_group_fields(
    tokens: &[String],
    pos: &mut usize,
    parent_transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    loop {
        if *pos >= tokens.len() {
            return Ok(());
        }
        match tokens[*pos].as_str() {
            "}" => { *pos += 1; return Ok(()); }
            "]" => { *pos += 1; return Ok(()); }
            _ => {}
        }

        let field = tokens[*pos].to_lowercase();

        match field.as_str() {
            "children" => {
                *pos += 1;
                parse_children_array(tokens, pos, parent_transform, results)?;
            }
            "bboxcenter" | "bboxsize" => {
                *pos += 1;
                let _ = parse_pvec3(tokens, pos);
            }
            _ => {
                // Unknown field — if it looks like a node (next token is `{`),
                // treat it as a direct child node. Otherwise skip the value.
                if *pos + 1 < tokens.len() && tokens[*pos + 1] == "{" {
                    collect_meshes(tokens, pos, parent_transform, results)?;
                } else {
                    *pos += 1;
                    skip_field_value(tokens, pos)?;
                }
            }
        }
    }
}

/// Parse `Shape { appearance ... geometry ... }`.
fn parse_shape_fields(
    tokens: &[String],
    pos: &mut usize,
    transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    let mut coords: Option<Vec<PVec3>> = None;
    let mut coord_index: Option<Vec<i32>> = None;

    loop {
        if *pos >= tokens.len() {
            return emit_mesh(coords, coord_index, transform, results);
        }
        let token = &tokens[*pos];
        match token.as_str() {
            "}" => {
                *pos += 1;
                return emit_mesh(coords, coord_index, transform, results);
            }
            "]" => {
                *pos += 1;
                return emit_mesh(coords, coord_index, transform, results);
            }
            _ => {}
        }

        let field = tokens[*pos].to_lowercase();
        *pos += 1;

        match field.as_str() {
            "appearance" => {
                // Skip Appearance node (and its Material child)
                skip_node_value(tokens, pos)?;
            }
            "geometry" => {
                // Expect IndexedFaceSet (or other geometry node)
                if *pos < tokens.len() && tokens[*pos] != "{" {
                    let geo_name = tokens[*pos].to_lowercase();
                    *pos += 1;
                    if geo_name == "indexedfaceset" || geo_name == "indexedlineset" {
                        let (c, ci) = parse_indexed_face_set(tokens, pos)?;
                        coords = c;
                        coord_index = ci;
                    } else if geo_name == "box" || geo_name == "sphere" || geo_name == "cone"
                        || geo_name == "cylinder" || geo_name == "text"
                        || geo_name == "elevationgrid" || geo_name == "extrusion"
                        || geo_name == "pointset"
                    {
                        // Primitives — skip for now (would need tessellation)
                        skip_balanced(tokens, pos, "{", "}")?;
                    } else {
                        skip_node_value_body(tokens, pos)?;
                    }
                } else {
                    skip_field_value(tokens, pos)?;
                }
            }
            _ => {
                skip_field_value(tokens, pos)?;
            }
        }
    }
}

/// Parse `IndexedFaceSet { coord Coordinate {...} coordIndex [...] ... }`.
/// Returns `(coords, coord_index)`.
fn parse_indexed_face_set(
    tokens: &[String],
    pos: &mut usize,
) -> Result<(Option<Vec<PVec3>>, Option<Vec<i32>>), VrmlError> {
    let mut coords: Option<Vec<PVec3>> = None;
    let mut coord_index: Option<Vec<i32>> = None;

    // Expect '{'
    if *pos < tokens.len() && tokens[*pos] == "{" {
        *pos += 1;
    }

    loop {
        if *pos >= tokens.len() {
            return Ok((coords, coord_index));
        }
        if tokens[*pos] == "}" {
            *pos += 1;
            return Ok((coords, coord_index));
        }

        let field = tokens[*pos].to_lowercase();
        *pos += 1;

        match field.as_str() {
            "coord" => {
                // Expect `Coordinate { point [...] }` (or Coordinate3)
                coords = parse_coordinate_node(tokens, pos)?;
            }
            "coordindex" => {
                coord_index = parse_int_array(tokens, pos)?;
            }
            "color" | "normal" | "texcoord" => {
                // Skip sub-node
                skip_node_value(tokens, pos)?;
            }
            "colorindex" | "normalindex" | "texcoordindex" => {
                let _ = parse_int_array(tokens, pos);
            }
            "solid" | "ccw" | "convex" | "creaseangle" => {
                // Single-value fields
                skip_field_value(tokens, pos)?;
            }
            _ => {
                skip_field_value(tokens, pos)?;
            }
        }
    }
}

/// Parse `Coordinate { point [ x y z, x y z, ... ] }` → `Vec<PVec3>`.
/// Also accepts the older `Coordinate3` node name synonym.
///
/// Handles `DEF Name Coordinate { ... }` and `DEF Name Coordinate3 { ... }`.
fn parse_coordinate_node(
    tokens: &[String],
    pos: &mut usize,
) -> Result<Option<Vec<PVec3>>, VrmlError> {
    // Read node name — may be `Coordinate`, `Coordinate3`, `DEF`, or already consumed.
    // Resolve DEF/USE prefix.
    if *pos < tokens.len() && tokens[*pos] != "{" {
        let name = tokens[*pos].to_lowercase();
        *pos += 1;

        let resolved = match name.as_str() {
            "def" => {
                // Skip DEF name, read actual node type
                if *pos < tokens.len() {
                    *pos += 1; // skip the DEF name
                }
                if *pos < tokens.len() {
                    let real = tokens[*pos].to_lowercase();
                    *pos += 1;
                    real
                } else {
                    return Ok(None);
                }
            }
            "use" => {
                // USE reference — skip name, no coordinate data
                if *pos < tokens.len() {
                    *pos += 1;
                }
                return Ok(None);
            }
            other => other.to_string(),
        };

        if resolved != "coordinate" && resolved != "coordinate3" {
            // Not what we expected — skip the node body
            if *pos < tokens.len() && tokens[*pos] == "{" {
                *pos += 1;
                skip_to_matching_brace(tokens, pos)?;
            }
            return Ok(None);
        }
    }

    // Expect '{'
    if *pos < tokens.len() && tokens[*pos] == "{" {
        *pos += 1;
    }

    let mut points: Option<Vec<PVec3>> = None;

    loop {
        if *pos >= tokens.len() {
            return Ok(points);
        }
        if tokens[*pos] == "}" {
            *pos += 1;
            return Ok(points);
        }

        let field = tokens[*pos].to_lowercase();
        *pos += 1;

        match field.as_str() {
            "point" => {
                points = Some(parse_pvec3_array(tokens, pos)?);
            }
            _ => {
                skip_field_value(tokens, pos)?;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Children array
// ---------------------------------------------------------------------------

/// Parse `children [...]` — iterate nodes inside the array and collect meshes.
fn parse_children_array(
    tokens: &[String],
    pos: &mut usize,
    transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    // Expect '['
    if *pos < tokens.len() && tokens[*pos] == "[" {
        *pos += 1;
    }

    while *pos < tokens.len() {
        if tokens[*pos] == "]" {
            *pos += 1;
            return Ok(());
        }
        if tokens[*pos] == "}" {
            // Parent closed before children array ended
            return Ok(());
        }
        collect_meshes(tokens, pos, transform, results)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Value parsers
// ---------------------------------------------------------------------------

/// Parse three consecutive floats into a `PVec3`.
fn parse_pvec3(tokens: &[String], pos: &mut usize) -> Result<PVec3, VrmlError> {
    let x = parse_real(tokens, pos)?;
    let y = parse_real(tokens, pos)?;
    let z = parse_real(tokens, pos)?;
    Ok(PVec3::new(x, y, z))
}

/// Parse a single f64 value. Accepts scientific notation.
fn parse_real(tokens: &[String], pos: &mut usize) -> Result<Real, VrmlError> {
    if *pos >= tokens.len() {
        return Err(VrmlError::Parse("unexpected end of input, expected number".into()));
    }
    let token = &tokens[*pos];
    // Some VRML files put commas before numbers — try to handle gracefully
    if token == "," {
        *pos += 1;
        return parse_real(tokens, pos);
    }
    let val = token
        .parse::<Real>()
        .map_err(|_| VrmlError::Parse(format!("expected number, got '{}'", token)))?;
    *pos += 1;
    Ok(val)
}

/// Parse a single i32 value.
fn parse_int(tokens: &[String], pos: &mut usize) -> Result<i32, VrmlError> {
    if *pos >= tokens.len() {
        return Err(VrmlError::Parse("unexpected end of input, expected integer".into()));
    }
    let token = &tokens[*pos];
    let val = token
        .parse::<i32>()
        .map_err(|_| VrmlError::Parse(format!("expected integer, got '{}'", token)))?;
    *pos += 1;
    Ok(val)
}

/// Parse a `[ ... ]` array of PVec3 values (groups of 3 floats).
fn parse_pvec3_array(tokens: &[String], pos: &mut usize) -> Result<Vec<PVec3>, VrmlError> {
    expect_open_bracket(tokens, pos)?;
    let mut points = Vec::new();
    // Collect all raw values inside the brackets
    let mut buf = Vec::new();
    while *pos < tokens.len() && tokens[*pos] != "]" {
        buf.push(tokens[*pos].clone());
        *pos += 1;
    }
    if *pos < tokens.len() {
        *pos += 1; // skip ']'
    }

    // Parse in groups of 3
    let nums: Vec<Real> = buf
        .iter()
        .map(|t| {
            t.parse::<Real>()
                .map_err(|_| VrmlError::Parse(format!("expected number, got '{}'", t)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    for chunk in nums.chunks(3) {
        if chunk.len() == 3 {
            points.push(PVec3::new(chunk[0], chunk[1], chunk[2]));
        }
    }
    Ok(points)
}

/// Parse a `[ ... ]` array of i32 values. Triangulates quads:
/// [a,b,c,d,-1] → [a,b,c,-1, a,c,d,-1].
fn parse_int_array(tokens: &[String], pos: &mut usize) -> Result<Option<Vec<i32>>, VrmlError> {
    if *pos >= tokens.len() {
        return Ok(None);
    }
    if tokens[*pos] != "[" {
        // Might be a single value like `coordIndex 0` (degenerate, skip)
        return Ok(None);
    }
    expect_open_bracket(tokens, pos)?;

    let mut raw: Vec<i32> = Vec::new();
    while *pos < tokens.len() && tokens[*pos] != "]" {
        raw.push(parse_int(tokens, pos)?);
    }
    if *pos < tokens.len() {
        *pos += 1; // skip ']'
    }

    // Triangulate: group faces by -1 sentinels.
    let mut out: Vec<i32> = Vec::new();
    let mut face: Vec<i32> = Vec::new();
    for &idx in &raw {
        if idx == -1 {
            if face.len() >= 3 {
                // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
                for i in 1..face.len() - 1 {
                    out.extend_from_slice(&[face[0], face[i], face[i + 1], -1]);
                }
            }
            face.clear();
        } else {
            face.push(idx);
        }
    }
    // Remaining face without trailing -1
    if face.len() >= 3 {
        for i in 1..face.len() - 1 {
            out.extend_from_slice(&[face[0], face[i], face[i + 1], -1]);
        }
    }

    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// Skipping helpers
// ---------------------------------------------------------------------------

/// Skip a field value: could be a single token, a bracketed array, or a node.
fn skip_field_value(tokens: &[String], pos: &mut usize) -> Result<(), VrmlError> {
    if *pos >= tokens.len() {
        return Ok(());
    }
    match tokens[*pos].as_str() {
        "[" => skip_balanced(tokens, pos, "[", "]")?,
        "{" => skip_balanced(tokens, pos, "{", "}")?,
        _ => {
            // Skip a node value like `Appearance { ... }` or `NULL`
            if tokens[*pos].to_lowercase() == "null" {
                *pos += 1;
                return Ok(());
            }
            // It might be a node name followed by '{'
            let maybe_node = *pos;
            *pos += 1;
            if *pos < tokens.len() && tokens[*pos] == "{"
                && tokens[maybe_node] != "translation"
                && tokens[maybe_node] != "rotation"
                && tokens[maybe_node] != "scale"
                && tokens[maybe_node] != "center"
                && tokens[maybe_node] != "children"
                && tokens[maybe_node] != "geometry"
                && tokens[maybe_node] != "appearance"
                && tokens[maybe_node] != "coord"
                && tokens[maybe_node] != "coordindex"
                && tokens[maybe_node] != "point"
                && tokens[maybe_node] != "bboxcenter"
                && tokens[maybe_node] != "bboxsize"
                && tokens[maybe_node] != "scaleorientation"
            {
                *pos += 1; // skip '{'
                skip_to_matching_brace(tokens, pos)?;
            }
            // Otherwise it's a simple value token — already skipped by *pos += 1
        }
    }
    Ok(())
}

/// Skip a node value when we know a node name was just consumed.
fn skip_node_value(tokens: &[String], pos: &mut usize) -> Result<(), VrmlError> {
    if *pos < tokens.len() && tokens[*pos] != "{" {
        // Skip the node name
        *pos += 1;
    }
    if *pos < tokens.len() && tokens[*pos] == "{" {
        *pos += 1;
        skip_to_matching_brace(tokens, pos)?;
    }
    Ok(())
}

/// Skip a node body when we know the name was consumed but `{` may or may not
/// have been consumed yet.
fn skip_node_value_body(tokens: &[String], pos: &mut usize) -> Result<(), VrmlError> {
    if *pos < tokens.len() && tokens[*pos] == "{" {
        *pos += 1;
        skip_to_matching_brace(tokens, pos)?;
    }
    Ok(())
}

/// Skip balanced delimiters: `[`...`]` or `{`...`}`.
fn skip_balanced(
    tokens: &[String],
    pos: &mut usize,
    open: &str,
    close: &str,
) -> Result<(), VrmlError> {
    // Expect open delimiter (already checked by caller in most cases)
    if *pos < tokens.len() && tokens[*pos] == open {
        *pos += 1;
    }
    let mut depth: i32 = 1;
    while depth > 0 && *pos < tokens.len() {
        if tokens[*pos] == open {
            depth += 1;
        } else if tokens[*pos] == close {
            depth -= 1;
        }
        *pos += 1;
    }
    if depth != 0 {
        return Err(VrmlError::Parse(format!(
            "unmatched '{}' / '{}'",
            open, close
        )));
    }
    Ok(())
}

/// Skip tokens until we've matched the current `{...}` nesting level.
/// Assumes we have already consumed the opening `{`.
fn skip_to_matching_brace(tokens: &[String], pos: &mut usize) -> Result<(), VrmlError> {
    let mut depth: i32 = 1;
    while depth > 0 && *pos < tokens.len() {
        match tokens[*pos].as_str() {
            "{" => depth += 1,
            "}" => depth -= 1,
            _ => {}
        }
        *pos += 1;
    }
    if depth != 0 {
        return Err(VrmlError::Parse("unmatched '{'".into()));
    }
    Ok(())
}

/// Expect and consume `[`.
fn expect_open_bracket(tokens: &[String], pos: &mut usize) -> Result<(), VrmlError> {
    if *pos >= tokens.len() || tokens[*pos] != "[" {
        return Err(VrmlError::Parse("expected '['".into()));
    }
    *pos += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

/// Build a local f64 transform matrix from VRML translation, rotation, scale.
///
/// VRML order: translate, then rotate, then scale.
/// In matrix terms: `M = T * R * S`.
fn build_transform(
    translation: PVec3,
    rotation_axis: PVec3,
    rotation_angle: Real,
    scale: PVec3,
    has_rotation: bool,
) -> PMat4 {
    let t = PMat4::from_translation(translation);
    let s = PMat4::from_scale(scale);
    let r = if has_rotation && rotation_axis.length_squared() > 1e-12 {
        let axis = rotation_axis.normalize();
        let quat = PQuat::from_axis_angle(axis, rotation_angle as f64);
        PMat4::from_quat(quat)
    } else {
        PMat4::IDENTITY
    };
    t * r * s
}

/// Convert optional PMat4 to Mat4 (f32). Returns `None` if identity.
fn to_mat4(m: &PMat4) -> Option<Mat4> {
    // Check if matrix is effectively identity
    let is_identity = m.x_axis.abs_diff_eq(PVec4::X, 1e-7)
        && m.y_axis.abs_diff_eq(PVec4::Y, 1e-7)
        && m.z_axis.abs_diff_eq(PVec4::Z, 1e-7)
        && m.w_axis.abs_diff_eq(PVec4::W, 1e-7);
    if is_identity {
        return None;
    }
    let cols: [[f32; 4]; 4] = [
        [m.x_axis.x as f32, m.x_axis.y as f32, m.x_axis.z as f32, m.x_axis.w as f32],
        [m.y_axis.x as f32, m.y_axis.y as f32, m.y_axis.z as f32, m.y_axis.w as f32],
        [m.z_axis.x as f32, m.z_axis.y as f32, m.z_axis.z as f32, m.z_axis.w as f32],
        [m.w_axis.x as f32, m.w_axis.y as f32, m.w_axis.z as f32, m.w_axis.w as f32],
    ];
    Some(Mat4::from_cols_array_2d(&cols))
}

/// Emit a `(MeshResult, Option<Mat4>)` from parsed coords + indices.
fn emit_mesh(
    coords: Option<Vec<PVec3>>,
    coord_index: Option<Vec<i32>>,
    transform: &PMat4,
    results: &mut Vec<(MeshResult, Option<Mat4>)>,
) -> Result<(), VrmlError> {
    match (coords, coord_index) {
        (Some(verts), Some(indices)) if !verts.is_empty() && !indices.is_empty() => {
            let mut mesh = MeshResult {
                vertices: verts,
                normals: Vec::new(),
                indices,
            };
            mesh.compute_normals();
            results.push((mesh, to_mat4(transform)));
        }
        _ => {
            // No geometry in this shape — skip
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_vrml_triangles() {
        let input = r#"#VRML V2.0 utf8
Transform {
    children [
        Shape {
            geometry IndexedFaceSet {
                coord Coordinate {
                    point [
                        0 0 0, 1 0 0, 0 1 0, 0 0 1
                    ]
                }
                coordIndex [
                    0, 1, 2, -1,
                    0, 2, 3, -1,
                    0, 3, 1, -1,
                    1, 3, 2, -1,
                ]
            }
        }
    ]
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1);
        let (ref mesh, ref transform) = result[0];
        assert_eq!(mesh.vertices.len(), 4);
        assert!(mesh.indices.len() >= 12); // 4 triangles * 4 (3 idx + -1)
        assert!(transform.is_none()); // identity transform
    }

    #[test]
    fn test_transform() {
        let input = r#"#VRML V2.0 utf8
Transform {
    translation 10 20 30
    rotation 0 0 1 1.5708
    scale 2 2 2
    children [
        Shape {
            geometry IndexedFaceSet {
                coord Coordinate {
                    point [ 0 0 0, 1 0 0, 0 1 0 ]
                }
                coordIndex [ 0, 1, 2, -1 ]
            }
        }
    ]
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1);
        let (_mesh, transform) = &result[0];
        assert!(transform.is_some(), "non-identity transform should be Some");
        let m = transform.unwrap();
        // Translation component
        assert!((m.w_axis.x - 10.0).abs() < 0.01, "tx should be 10");
        assert!((m.w_axis.y - 20.0).abs() < 0.01, "ty should be 20");
        assert!((m.w_axis.z - 30.0).abs() < 0.01, "tz should be 30");
    }

    #[test]
    fn test_nested_transform() {
        let input = r#"#VRML V2.0 utf8
Transform {
    translation 1 0 0
    children [
        Transform {
            translation 0 2 0
            children [
                Shape {
                    geometry IndexedFaceSet {
                        coord Coordinate {
                            point [ 0 0 0, 1 0 0, 0 1 0 ]
                        }
                        coordIndex [ 0, 1, 2, -1 ]
                    }
                }
            ]
        }
    ]
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1);
        let (_mesh, transform) = &result[0];
        assert!(transform.is_some());
        let m = transform.unwrap();
        // Translation should be (1+0, 0+2, 0+0) = (1, 2, 0)
        assert!((m.w_axis.x - 1.0).abs() < 0.01);
        assert!((m.w_axis.y - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_quad_triangulation() {
        let input = r#"#VRML V2.0 utf8
Shape {
    geometry IndexedFaceSet {
        coord Coordinate {
            point [ 0 0 0, 1 0 0, 1 1 0, 0 1 0 ]
        }
        coordIndex [ 0, 1, 2, 3, -1 ]
    }
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1);
        let (ref mesh, _) = result[0];
        // Quad triangulated to 2 triangles: (0,1,2) + (0,2,3)
        // Each triangle: 3 indices + -1 sentinel = 4 tokens
        assert_eq!(mesh.indices.len(), 8, "2 triangles * 4 = 8 tokens");
        assert_eq!(mesh.indices[0], 0);
        assert_eq!(mesh.indices[1], 1);
        assert_eq!(mesh.indices[2], 2);
        assert_eq!(mesh.indices[3], -1);
        assert_eq!(mesh.indices[4], 0);
        assert_eq!(mesh.indices[5], 2);
        assert_eq!(mesh.indices[6], 3);
        assert_eq!(mesh.indices[7], -1);
    }

    #[test]
    fn test_skip_unknown_nodes() {
        let input = r#"#VRML V2.0 utf8
Transform {
    children [
        DirectionalLight { direction 0 0 -1 }
        Shape {
            geometry IndexedFaceSet {
                coord Coordinate {
                    point [ 0 0 0, 1 0 0, 0 1 0 ]
                }
                coordIndex [ 0, 1, 2, -1 ]
            }
        }
    ]
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1, "unknown nodes should be skipped");
    }

    #[test]
    fn test_empty_input() {
        let result = parse_vrml_str("#VRML V2.0 utf8\n").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_def_use_skipped() {
        let input = r#"#VRML V2.0 utf8
DEF MyShape Shape {
    geometry IndexedFaceSet {
        coord DEF MyCoords Coordinate {
            point [ 0 0 0, 1 0 0, 0 1 0 ]
        }
        coordIndex [ 0, 1, 2, -1 ]
    }
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_separator_group_nodes() {
        let input = r#"#VRML V2.0 utf8
Separator {
    Shape {
        geometry IndexedFaceSet {
            coord Coordinate {
                point [ 0 0 0, 1 0 0, 0 1 0 ]
            }
            coordIndex [ 0, 1, 2, -1 ]
        }
    }
}
"#;
        let result = parse_vrml_str(input).unwrap();
        assert_eq!(result.len(), 1);
    }
}
