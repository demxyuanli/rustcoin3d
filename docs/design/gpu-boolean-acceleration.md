# GPU-Accelerated Boolean Operations — Design Study

> 2026-06-05 | Deep research follow-up Q5

## Motivation

The current CPU boolean pipeline (`rc3d-shape/src/bool/`) scales as O(F²) for PaveFiller
face-pair testing with AABB pre-filtering. For models > 10k faces, the face-pair
intersection testing becomes the dominant cost. GPU acceleration could reduce this
to O(F log F) via compute-shader BVH traversal.

## Why Not Conservative Rasterization

Conservative rasterization (VK_EXT_conservative_rasterization) ensures no false
negatives in triangle-triangle overlap detection, but:

1. **wgpu does not expose it**: No `wgpu::Features` flag for conservative rasterization
   as of wgpu 0.20. Would require raw Vulkan/DX12 interop.
2. **Hardware fragmentation**: NVIDIA Maxwell+ (2014), AMD GCN3+ — not universal.
3. **Resolution-dependent accuracy**: Even with conservative rasterization, the
   precision of detected overlaps is limited by the framebuffer resolution.
   Sub-pixel intersections may still be missed or over-reported.

## Proposed: Hybrid CPU/GPU with Compute Shaders

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0 (GPU): Build BVH over all face AABBs               │
│ Phase 1 (GPU): Compute shader: AABB overlap → candidates   │
│ Phase 2 (CPU): For each candidate pair, exact SSI via      │
│                marching + Newton (existing pipeline)        │
│ Phase 3-5 (CPU): Split, classify, select, stitch           │
└─────────────────────────────────────────────────────────────┘
```

### GPU Kernel Design

```wgsl
// Candidate pair detection compute shader
struct GpuAABB {
    min: vec3<f32>,
    max: vec3<f32>,
    face_id: u32,
    _pad: u32, // alignment to 32 bytes
}

@group(0) @binding(0) var<storage, read> bboxes_a: array<GpuAABB>;
@group(0) @binding(1) var<storage, read> bboxes_b: array<GpuAABB>;
@group(0) @binding(2) var<storage, read_write> candidate_pairs: array<u32>; // (face_a, face_b) pairs
@group(0) @binding(3) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(4) var<uniform> params: CullParams;

struct CullParams {
    count_a: u32,
    count_b: u32,
    max_pairs: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count_a { return; }
    let bb_a = bboxes_a[i];

    for (var j = 0u; j < params.count_b; j++) {
        let bb_b = bboxes_b[j];
        if (aabb_overlaps(bb_a, bb_b)) {
            let idx = atomicAdd(&pair_count, 1u);
            if idx < params.max_pairs {
                candidate_pairs[idx * 2u] = bb_a.face_id;
                candidate_pairs[idx * 2u + 1u] = bb_b.face_id;
            }
        }
    }
}

fn aabb_overlaps(a: GpuAABB, b: GpuAABB) -> bool {
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y
        && a.min.z <= b.max.z && a.max.z >= b.min.z;
}
```

### Data Flow

1. Extract face AABBs from BRepStore → GPU buffer (32 bytes per face)
2. Dispatch compute shader (N * M / 64 workgroups)
3. Read back `pair_count` atomic → allocate CPU buffer
4. Dispatch compact shader (if needed) to pack dense candidate pairs
5. Map GPU buffer → CPU for exact intersection computation via existing SSI pipeline

### Performance Estimate

| Model Size | CPU (current) | GPU (estimated) | Speedup |
|-----------|---------------|-----------------|---------|
| 100 faces | < 1ms | < 1ms + ~0.5ms GPU overhead | 0.5x (slower) |
| 1k faces | ~10ms | ~5ms | 2x |
| 10k faces | ~500ms | ~30ms | 15x |
| 100k faces | ~50s | ~200ms | 250x |

GPU overhead includes buffer allocation, upload, dispatch, fence, and readback (~0.5ms total for small workloads). Breakeven is approximately 500 faces.

### Implementation Plan (if pursued)

1. **Extract face AABB generation** into a reusable function (`face_vertex_bbox` already exists in `aabb.rs`; needs GPU-buffer-friendly layout)
2. **Create GPU BVH builder and traversal shader** — for O(F log F) vs O(F²) AABB testing
3. **Create candidate pair compact shader** — packs sparse results into dense buffer
4. **Integrate with `pave_filler.rs`**: replace the nested-loop `test_pair` closure with GPU path
5. **Fallback to CPU** when GPU is unavailable or model is too small

### Open Questions

1. **Dynamic topology**: After face splitting (Phase 2), the face set changes. Can the GPU path
   handle incremental updates to the AABB set?
2. **Full GPU PaveFiller**: Can we move the entire intersection computation (marching + Newton)
   to compute shaders? Newton iteration maps well to GPU but surface evaluation is complex.
3. **Readback latency**: Is the GPU→CPU buffer readback latency acceptable for
   interactive boolean operations? Typical PCIe 4.0 x16 readback is ~25 GB/s,
   meaning 1M candidate pairs (8 MB) takes ~0.3ms.

## Recommendation

**Defer implementation until:**
1. CPU pipeline is fully working and benchmarked on models > 10k faces
2. A concrete use case requires interactive boolean on large assemblies
3. wgpu compute shader support is stable for the target platforms

**Preparatory work that can be done now:**
- Make the AABB extraction function GPU-buffer-friendly (32-byte aligned struct)
- Keep the PaveFiller interface abstract enough to swap in a GPU implementation
- Benchmark the CPU pipeline to establish a baseline for future comparison
