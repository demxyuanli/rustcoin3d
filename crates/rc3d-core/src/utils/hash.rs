//! Float hashing and composite-key helpers (f32 + f64).

// ── f32 (rendering) ──

/// Convert a slice of `f32` to a `Vec<u32>` via `to_bits()`.
pub fn f32_slice_to_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Convert a `[f32; 3]` to `[u32; 3]` for use as a hash/comparison key.
#[inline]
pub fn f32x3_to_bits(v: [f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

/// Quantized hash key for vertex deduplication (f32).
#[inline]
pub fn f32x3_quantized_bits(v: [f32; 3]) -> [u32; 3] {
    let q = |x: f32| (x * 1e5).round().to_bits();
    [q(v[0]), q(v[1]), q(v[2])]
}

/// Convert a `[f32; 4]` to `[u32; 4]`.
#[inline]
pub fn f32x4_to_bits(v: [f32; 4]) -> [u32; 4] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()]
}

/// Total-order key for `f32` (handles NaN and -0.0).
#[inline]
pub fn f32_total_key(v: f32) -> u32 {
    let bits = v.to_bits();
    bits ^ (((bits as i32) >> 31) as u32)
}

// ── f64 (geometry kernel) ──

/// Convert a slice of `f64` to a `Vec<u64>` via `to_bits()`.
pub fn f64_slice_to_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Convert a `[f64; 3]` to `[u64; 3]` for use as a hash/comparison key.
#[inline]
pub fn f64x3_to_bits(v: [f64; 3]) -> [u64; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

/// Quantized hash key for vertex deduplication (f64).
#[inline]
pub fn f64x3_quantized_bits(v: [f64; 3]) -> [u64; 3] {
    let q = |x: f64| (x * 1e5).round().to_bits();
    [q(v[0]), q(v[1]), q(v[2])]
}

/// Convert a `[f64; 4]` to `[u64; 4]`.
#[inline]
pub fn f64x4_to_bits(v: [f64; 4]) -> [u64; 4] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()]
}

/// Total-order key for `f64` (handles NaN and -0.0).
#[inline]
pub fn f64_total_key(v: f64) -> u64 {
    let bits = v.to_bits();
    bits ^ (((bits as i64) >> 63) as u64)
}
