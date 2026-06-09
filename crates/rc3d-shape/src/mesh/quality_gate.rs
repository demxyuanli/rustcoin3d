//! Per-face and shell-level mesh quality checks.
//!
//! Applied after meshing, before shell merge. Thresholds vary by
//! `TessellationTier` (Preview relaxed, Precision strict).

use rc3d_core::math::Vec3;

use crate::mesh::config::{MeshQualityGate, TessellationTier};

/// Quality verdict for a single face's mesh output.
#[derive(Debug, Clone)]
pub enum FaceQualityVerdict {
    /// Mesh passes all quality checks.
    Pass,
    /// Minor issues — retry with hint might fix.
    Retry(String),
    /// Quality violation — skip face or fail per tier.
    Fail(String),
}

/// Quality check applied to a chunk of face triangles.
pub fn check_face_quality(
    vertices: &[Vec3],
    tri_first: usize,
    tri_count: usize,
    indices: &[i32],
    gate: &MeshQualityGate,
    tier: TessellationTier,
) -> FaceQualityVerdict {
    let (degenerate, max_aspect, min_area) =
        scan_triangles(vertices, tri_first, tri_count, indices);

    let total = tri_count.max(1);
    let degen_rate = degenerate as f32 / total as f32;

    // Degenerate triangle check
    if degen_rate > gate.max_degenerate_rate {
        return match tier {
            TessellationTier::Preview => FaceQualityVerdict::Retry(format!(
                "degenerate rate {:.1}% > {:.1}%",
                degen_rate * 100.0,
                gate.max_degenerate_rate * 100.0
            )),
            TessellationTier::Standard => FaceQualityVerdict::Retry(format!(
                "degenerate rate {:.1}% > {:.1}%",
                degen_rate * 100.0,
                gate.max_degenerate_rate * 100.0
            )),
            TessellationTier::Precision => FaceQualityVerdict::Fail(format!(
                "degenerate rate {:.1}% exceeds limit {:.1}%",
                degen_rate * 100.0,
                gate.max_degenerate_rate * 100.0
            )),
        };
    }

    // Aspect ratio check (only when we have valid triangles)
    if max_aspect > gate.max_aspect_ratio && min_area > 1e-12 {
        return match tier {
            TessellationTier::Preview => FaceQualityVerdict::Pass, // Preview: skip aspect ratio
            TessellationTier::Standard => FaceQualityVerdict::Retry(format!(
                "max aspect {:.1} > {:.1}",
                max_aspect, gate.max_aspect_ratio
            )),
            TessellationTier::Precision => FaceQualityVerdict::Fail(format!(
                "max aspect {:.1} exceeds limit {:.1}",
                max_aspect, gate.max_aspect_ratio
            )),
        };
    }

    FaceQualityVerdict::Pass
}

/// Scan a triangle range and return (degenerate_count, max_aspect_ratio, min_area).
fn scan_triangles(
    vertices: &[Vec3],
    tri_first: usize,
    tri_count: usize,
    indices: &[i32],
) -> (usize, f32, f32) {
    let mut degenerate = 0usize;
    let mut max_aspect = 0.0f32;
    let mut min_area = f32::MAX;

    for ti in tri_first..tri_first.saturating_add(tri_count) {
        let base = ti * 4;
        if base + 3 >= indices.len() || indices[base + 3] != -1 {
            continue;
        }
        let i0 = indices[base] as usize;
        let i1 = indices[base + 1] as usize;
        let i2 = indices[base + 2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];
        let ab = b - a;
        let ac = c - a;
        let bc = c - b;
        let ab_len = ab.length();
        let ac_len = ac.length();
        let bc_len = bc.length();
        let area = ab.cross(ac).length() * 0.5;
        if area < 1e-12 {
            degenerate += 1;
            continue;
        }
        min_area = min_area.min(area);
        let max_edge = ab_len.max(ac_len).max(bc_len);
        let min_edge = ab_len.min(ac_len).min(bc_len).max(1e-12);
        let aspect = max_edge / min_edge;
        max_aspect = max_aspect.max(aspect);
    }

    if min_area == f32::MAX {
        min_area = 0.0;
    }
    (degenerate, max_aspect, min_area)
}

// ── Shell-level quality ──────────────────────────────────────────────

/// Aggregate quality check over an entire shell after merge.
#[derive(Debug, Clone, Default)]
pub struct ShellQualityResult {
    pub total_tris: usize,
    pub degenerate_tris: usize,
    pub max_aspect_ratio: f32,
    pub passed: bool,
}

/// Run shell-level quality checks after merging all face meshes.
pub fn check_shell_quality(
    vertices: &[Vec3],
    indices: &[i32],
    gate: &MeshQualityGate,
    tier: TessellationTier,
) -> ShellQualityResult {
    // Count actual triangles (handle short buffers gracefully).
    let raw_count = indices.len() / 4;
    let (degenerate, max_aspect, _) = scan_triangles(vertices, 0, raw_count, indices);
    let total = raw_count;
    let degen_rate = if total > 0 {
        degenerate as f32 / total as f32
    } else {
        1.0 // empty mesh = all "degenerate" for quality purposes
    };

    let passed = match tier {
        TessellationTier::Preview => degen_rate <= gate.max_degenerate_rate,
        TessellationTier::Standard => {
            degen_rate <= gate.max_degenerate_rate && max_aspect <= gate.max_aspect_ratio
        }
        TessellationTier::Precision => {
            degen_rate <= gate.max_degenerate_rate
                && max_aspect <= gate.max_aspect_ratio
                && total > 0
        }
    };

    ShellQualityResult {
        total_tris: total,
        degenerate_tris: degenerate,
        max_aspect_ratio: max_aspect,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tri(indices: &[i32; 3]) -> Vec<i32> {
        vec![indices[0], indices[1], indices[2], -1]
    }

    #[test]
    fn test_quality_gate_valid_triangle() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = make_tri(&[0, 1, 2]);
        let gate = MeshQualityGate::default();
        let v = check_face_quality(&verts, 0, 1, &indices, &gate, TessellationTier::Standard);
        assert!(matches!(v, FaceQualityVerdict::Pass));
    }

    #[test]
    fn test_quality_gate_degenerate_triangle() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let indices = make_tri(&[0, 1, 2]);
        let gate = MeshQualityGate {
            max_degenerate_rate: 0.0,
            ..Default::default()
        };
        let v = check_face_quality(&verts, 0, 1, &indices, &gate, TessellationTier::Precision);
        assert!(matches!(v, FaceQualityVerdict::Fail(_)));
    }

    #[test]
    fn test_quality_gate_preview_relaxed() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = make_tri(&[0, 1, 2]);
        let gate = MeshQualityGate::default();
        // Preview skips aspect ratio check
        let v = check_face_quality(&verts, 0, 1, &indices, &gate, TessellationTier::Preview);
        assert!(matches!(v, FaceQualityVerdict::Pass));
    }

    #[test]
    fn test_shell_quality_empty() {
        let gate = MeshQualityGate::default();
        let r = check_shell_quality(&[], &[], &gate, TessellationTier::Precision);
        assert!(!r.passed);
        assert_eq!(r.total_tris, 0);
    }

    #[test]
    fn test_shell_quality_preview_allows_moderate_degenerate() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0), // degenerate
            Vec3::new(0.0, 0.0, 0.0), // degenerate
            Vec3::new(0.0, 0.0, 0.0), // degenerate
        ];
        // 1 valid tri + 1 degenerate tri = 50% degenerate > 15% limit
        // But Preview allows up to 15% — with 2 tris, 1 degen = 50% → should fail
        let mut indices = make_tri(&[0, 1, 2]);
        indices.extend(make_tri(&[3, 4, 5]));
        let gate = MeshQualityGate::default();
        let r = check_shell_quality(&verts, &indices, &gate, TessellationTier::Preview);
        assert!(!r.passed); // 50% degenerate exceeds 15% Preview limit
        assert_eq!(r.degenerate_tris, 1);
    }
}
