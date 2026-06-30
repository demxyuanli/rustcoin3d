//! BREP structural validator — checks reference integrity without OCC.
//! Compile: rustc validate_brep.rs -o validate_brep.exe
//! Usage:   validate_brep.exe compare/out/step-*.brep

use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("Usage: {} <brep>...", args[0]); std::process::exit(1); }
    let (mut ok, mut fail) = (0usize, 0usize);
    for p in &args[1..] {
        match check(Path::new(p)) {
            Ok(()) => { ok += 1; println!("  OK  {}", p); }
            Err(e) => { fail += 1; eprintln!("  FAIL {}: {}", p, e); }
        }
    }
    println!("{}/{} valid", ok, ok + fail);
    if fail > 0 { std::process::exit(1); }
}

fn check(path: &Path) -> Result<(), String> {
    let s = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
    if !s.contains("DBRep_DrawableShape") { return Err("missing header".into()); }

    // Count TShapes
    let total: usize = s.lines().find(|l| l.trim().starts_with("TShapes"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .ok_or("missing TShapes count")?;

    // Collect reference lines (end with *) and their preceding type markers
    let lines: Vec<&str> = s.lines().collect();
    let mut last_type = "";
    let mut refs: Vec<(&str, &str)> = Vec::new(); // (type, ref_line)
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        if t == "Ve" || t == "Ed" || t == "Wi" || t == "Fa" || t == "Sh" || t == "So" {
            last_type = t;
        }
        if t.ends_with('*') && !t.starts_with("010") && !t.starts_with("011") && !t.starts_with("110") {
            refs.push((last_type, t));
        }
    }

    // Validate references. Format: [+-]TShapeNum LocationIdx [+-]TShapeNum LocationIdx ... *
    // Location indices are 0..N (0 = identity). TShape references are reverse-indexed.
    for (typ, rline) in &refs {
        let tokens: Vec<&str> = rline.split_whitespace()
            .filter(|s| *s != "*")
            .collect();
        // Process in pairs: (tshape_ref, location_idx)
        for chunk in tokens.chunks(2) {
            let rev_n: usize = chunk.first()
                .and_then(|s| s.trim_start_matches(&['+', '-'] as &[_]).parse().ok())
                .unwrap_or(0);
            if rev_n == 0 { continue; }
            if rev_n > total {
                return Err(format!("{} ref {} out of range (max {}) in '{}'",
                    typ, rev_n, total, rline.trim()));
            }
        }
    }

    // Basic presence checks
    if !refs.iter().any(|(t, _)| *t == "Ve") { return Err("no vertices".into()); }
    if !refs.iter().any(|(t, _)| *t == "Ed") { return Err("no edges".into()); }
    if !refs.iter().any(|(t, _)| *t == "Fa") { return Err("no faces".into()); }
    if !refs.iter().any(|(t, _)| *t == "Sh") { return Err("no shells".into()); }
    if !refs.iter().any(|(t, _)| *t == "So") { return Err("no solids".into()); }

    Ok(())
}
