//! BREP diagnostic parser — validates OCC BREP ASCII format.
//! Run: cargo test -p rc3d-io --test brep_parse_diag -- --nocapture

use rc3d_core::math::Real;
use std::fs;
use std::path::Path;

#[derive(Debug, Default)]
struct BrepDiag {
    locations: usize,
    curve2ds_count: usize,
    curves_count: usize,
    surfaces_count: usize,
    tshapes_count: usize,

    // TShapes counts
    vertices: usize, edges: usize, wires: usize, faces: usize,
    shells: usize, solids: usize, compounds: usize,

    // Content validation
    surface_info: Vec<String>,
    curve_info: Vec<String>,
    edge_tolerances: Vec<Real>,
    face_orientations: Vec<i32>,

    errors: Vec<String>,
    warnings: Vec<String>,
}

fn count_numbers(line: &str) -> usize {
    line.split_whitespace()
        .filter(|s| s.chars().any(|c| c.is_ascii_digit() || c == '-' || c == '.'))
        .count()
}

fn parse_brep(path: &Path) -> BrepDiag {
    let mut d = BrepDiag::default();
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { d.errors.push(format!("cannot read: {}", e)); return d; }
    };
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() || lines[0].trim() != "DBRep_DrawableShape" {
        d.errors.push("missing DBRep_DrawableShape header".into());
        return d;
    }

    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() { i += 1; continue; }

        if line.starts_with("Locations") {
            d.locations = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            i += 1;
            for _ in 0..d.locations { i += 1; }
            continue;
        }
        if line.starts_with("Curve2ds") {
            d.curve2ds_count = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            i += 1;
            let mut parsed = 0;
            while i < lines.len() && parsed < d.curve2ds_count {
                let l = lines[i].trim();
                if l.is_empty() { i += 1; continue; }
                if is_section_header(l) { break; }
                parsed += 1;
                i += 1;
            }
            continue;
        }
        if line.starts_with("Curves") {
            d.curves_count = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            i += 1;
            let mut parsed = 0;
            while i < lines.len() && parsed < d.curves_count {
                let l = lines[i].trim();
                if l.is_empty() { i += 1; continue; }
                if is_section_header(l) { break; }
                let nums = count_numbers(l);
                let typ = l.split_whitespace().next().unwrap_or("?");
                let expected = match typ {
                    "1" => (7, "Line"),
                    "2" => (14, "Circle"),
                    "3" => (14, "Ellipse"),
                    "7" => (14, "BSpline3D-header"),
                    "8" => (2, "Line-params"),
                    _ => (0, "?"),
                };
                let label = format!("Curve#{}({}) {}nums", parsed + 1, expected.1, nums);
                if expected.0 > 0 && nums != expected.0 {
                    d.warnings.push(format!("{}: expected {} nums, got {}", label, expected.0, nums));
                }
                d.curve_info.push(label);
                parsed += 1;
                i += 1;
            }
            continue;
        }
        if line.starts_with("Surfaces") {
            d.surfaces_count = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            i += 1;
            let mut parsed = 0;
            while i < lines.len() && parsed < d.surfaces_count {
                let l = lines[i].trim();
                if l.is_empty() { i += 1; continue; }
                if is_section_header(l) { break; }
                let nums = count_numbers(l);
                let typ = l.split_whitespace().next().unwrap_or("?");
                let expected = match typ {
                    "1" => (13, "Plane"),
                    "2" => (14, "Cylinder"),
                    "3" => (14, "Cone"),
                    "4" => (14, "Sphere"),
                    "5" => (15, "Torus"),
                    "6" => (7, "Extrusion"),
                    "7" => (8, "Revolution"),
                    "8" => (0, "BSpline-var"),
                    "9" => (0, "Offset-var"),
                    _ => (0, "?"),
                };
                let label = format!("Surface#{}({}) {}nums", parsed + 1, expected.1, nums);
                if expected.0 > 0 && nums != expected.0 {
                    d.errors.push(format!("{}: expected {} nums, got {}", label, expected.0, nums));
                }
                d.surface_info.push(label);
                parsed += 1;
                i += 1;
            }
            continue;
        }
        if line.starts_with("TShapes") {
            d.tshapes_count = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            i += 1;
            let mut tshape_idx = 0;
            while i < lines.len() && tshape_idx < d.tshapes_count {
                let l = lines[i].trim();
                if l.is_empty() { i += 1; continue; }
                match l {
                    "Ve" => {
                        d.vertices += 1; tshape_idx += 1; i += 1;
                        // tolerance line
                        if i < lines.len() && !lines[i].trim().is_empty() { i += 1; }
                        // position line
                        if i < lines.len() && !lines[i].trim().is_empty() { i += 1; }
                        // flags: skip until * or next section
                    }
                    "Ed" => {
                        d.edges += 1; tshape_idx += 1; i += 1;
                        // tolerance+flags line: numbers
                        if i < lines.len() {
                            let tol_line = lines[i].trim();
                            if let Some(tol_str) = tol_line.split_whitespace().next() {
                                if let Ok(tol) = tol_str.parse::<Real>() {
                                    d.edge_tolerances.push(tol);
                                }
                            }
                            i += 1;
                        }
                        // curve ref line
                        if i < lines.len() && !lines[i].trim().is_empty() { i += 1; }
                        // PCurve lines or "0" terminator
                        while i < lines.len() && !lines[i].trim().is_empty() && lines[i].trim() != "0101000" {
                            i += 1;
                        }
                    }
                    "Fa" => {
                        d.faces += 1; tshape_idx += 1; i += 1;
                        if i < lines.len() {
                            let fa_line = lines[i].trim();
                            let parts: Vec<&str> = fa_line.split_whitespace().collect();
                            if parts.len() >= 4 {
                                if let Ok(orient) = parts[3].parse::<i32>() {
                                    d.face_orientations.push(orient);
                                }
                            }
                            i += 1;
                        }
                    }
                    "Wi" => { d.wires += 1; tshape_idx += 1; i += 1; }
                    "Sh" => { d.shells += 1; tshape_idx += 1; i += 1; }
                    "So" => { d.solids += 1; tshape_idx += 1; i += 1; }
                    "Co" => { d.compounds += 1; tshape_idx += 1; i += 1; }
                    _ => { i += 1; }
                }
            }
            continue;
        }
        // Skip PolygonOnTriangulations, Polygon3D, Triangulations headers
        if line.starts_with("Polygon") || line.starts_with("Triangulations") {
            i += 1;
            continue;
        }
        i += 1;
    }

    // Validate
    let total = d.vertices + d.edges + d.wires + d.faces + d.shells + d.solids + d.compounds;
    if total != d.tshapes_count && d.tshapes_count > 0 {
        d.errors.push(format!("TShapes count mismatch: header={} counted={}", d.tshapes_count, total));
    }

    d
}

fn is_section_header(l: &str) -> bool {
    l.starts_with("Curves") || l.starts_with("Surfaces") || l.starts_with("TShapes")
        || l.starts_with("Polygon") || l.starts_with("Triangulations")
        || l.starts_with("Curve2ds") || l.starts_with("Locations")
}

fn print_diag(d: &BrepDiag) {
    println!("  Locs={}  C2ds={}  Curves={}  Surfs={}  TShapes={}",
        d.locations, d.curve2ds_count, d.curves_count, d.surfaces_count, d.tshapes_count);
    println!("  V{} E{} W{} F{} Sh{} So{} Co{}",
        d.vertices, d.edges, d.wires, d.faces, d.shells, d.solids, d.compounds);
    if !d.surface_info.is_empty() {
        println!("  Surfaces: {}", d.surface_info.join(" | "));
    }
    if !d.curve_info.is_empty() {
        println!("  Curves: {}", d.curve_info.join(" | "));
    }
    if !d.face_orientations.is_empty() {
        println!("  Fa orient: {:?}", d.face_orientations);
    }
    let tol_max = d.edge_tolerances.iter().cloned().fold(0.0_f64, Real::max);
    let tol_big = d.edge_tolerances.iter().filter(|&&t| t > 0.01).count();
    println!("  Ed tolerances: max={:.6} big(>.01)={}/{}", tol_max, tol_big, d.edge_tolerances.len());
    for e in &d.errors { println!("  ERROR: {}", e); }
    for w in &d.warnings { println!("  WARN: {}", w); }
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
#[ignore = "diagnostic tool: no assertions, manual inspection only"]
fn diag_cube() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output/brep/Cube.brep");
    if !p.exists() { eprintln!("SKIP"); return; }
    let d = parse_brep(&p);
    println!("\n=== Cube.brep ==="); print_diag(&d);
}

#[test]
#[ignore = "diagnostic tool: no assertions, manual inspection only"]
fn diag_cs() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output/brep/cs.brep");
    if !p.exists() { eprintln!("SKIP"); return; }
    let d = parse_brep(&p);
    println!("\n=== cs.brep ==="); print_diag(&d);
}

#[test]
#[ignore = "diagnostic tool: no assertions, manual inspection only"]
fn diag_all() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output/brep");
    let Ok(entries) = fs::read_dir(&dir) else { return; };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().map(|x| x == "brep").unwrap_or(false) {
            let d = parse_brep(&p);
            println!("\n=== {} ===", p.file_name().unwrap().to_string_lossy()); print_diag(&d);
        }
    }
}

#[test]
#[ignore = "diagnostic tool: no assertions, manual inspection only"]
fn diag_occ() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../steps/comp");
    let Ok(entries) = fs::read_dir(&dir) else { return; };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().map(|x| x == "brep").unwrap_or(false) {
            let d = parse_brep(&p);
            println!("\n=== OCC {} ===", p.file_name().unwrap().to_string_lossy()); print_diag(&d);
        }
    }
}
