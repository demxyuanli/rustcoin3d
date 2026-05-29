//! Part21 entity inventory baseline for `test_data/Shape.step`.
//!
//! Format: one line per instance `id,KEYWORD` (sorted by id).
//! Optional OCC oracle: export with OCCT StepDump or in-house tool using the same format.
//!
//! Regenerate fixture:
//! ```powershell
//! $env:RC3D_UPDATE_ENTITY_INVENTORY='1'
//! rtk cargo test -p rc3d-io --test step_entity_inventory update_shape_entity_inventory -- --nocapture
//! ```

use std::collections::HashSet;
use std::path::PathBuf;

use rc3d_io::step::part21::read::read_exchange;
use rc3d_io::step::adapter::AdapterMode;
use rc3d_io::step::schema::ap242::instance_keyword;

fn test_data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn inventory_lines(step_text: &str) -> Vec<String> {
    let model = read_exchange(step_text).expect("part21 read");
    let mut lines: Vec<String> = model
        .instances()
        .filter_map(|inst| {
            let kw = instance_keyword(inst, AdapterMode::CompatMerge)?;
            Some(format!("{},{}", inst.id, kw))
        })
        .collect();
    lines.sort_by_key(|line| {
        line.split(',')
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    });
    lines
}

#[test]
fn shape_entity_inventory_within_one_percent() {
    let path = test_data("Shape.step");
    if !path.exists() {
        eprintln!("SKIP: Shape.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read Shape.step");
    let current = inventory_lines(&text);
    assert!(!current.is_empty(), "Shape.step inventory must not be empty");

    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shape_entity_inventory.txt");
    let baseline_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", fixture_path.display()));
    let baseline: HashSet<String> = baseline_text
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let current_set: HashSet<String> = current.iter().cloned().collect();

    let only_baseline: usize = baseline.difference(&current_set).count();
    let only_current: usize = current_set.difference(&baseline).count();
    let diff = only_baseline + only_current;
    let base_len = baseline.len().max(1);
    let ratio = diff as f64 / base_len as f64;

    if diff > 0 {
        eprintln!(
            "inventory diff: {diff} lines ({ratio:.4} of {base_len}), +current={only_current}, -baseline={only_baseline}"
        );
        for line in baseline.difference(&current_set).take(5) {
            eprintln!("  missing in current: {line}");
        }
        for line in current_set.difference(&baseline).take(5) {
            eprintln!("  extra in current: {line}");
        }
    }

    assert!(
        ratio <= 0.01,
        "Shape.step entity inventory drift {ratio:.4} > 1% (diff {diff}/{base_len})"
    );
}

#[test]
fn update_shape_entity_inventory() {
    if std::env::var("RC3D_UPDATE_ENTITY_INVENTORY").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set RC3D_UPDATE_ENTITY_INVENTORY=1 to regenerate fixture");
        return;
    }
    let path = test_data("Shape.step");
    let text = std::fs::read_to_string(&path).expect("read");
    let lines = inventory_lines(&text);
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shape_entity_inventory.txt");
    if let Some(parent) = fixture_path.parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    let body = format!("{}\n", lines.join("\n"));
    std::fs::write(&fixture_path, body).expect("write fixture");
    eprintln!(
        "wrote {} lines to {}",
        lines.len(),
        fixture_path.display()
    );
}
