//! Float hashing and composite-key helpers.
//!
//! Rendering pipelines and caches frequently need to construct deterministic
//! hash keys from floating-point data. Using `f32::to_bits()` gives a stable
//! `u32` representation that preserves NaN payloads and maps `-0.0` to `0`.

/// Convert a slice of `f32` to a `Vec<u32>` via `to_bits()`.
///
/// Used for building cache keys from coordinate / normal / texcoord arrays.
pub fn f32_slice_to_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Convert a `[f32; N]` to `[u32; N]` for use as a hash/comparison key.
#[inline]
pub fn f32x3_to_bits(v: [f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

/// Quantized hash key for vertex deduplication.
/// Rounds coordinates to 1e-5 precision before hashing, so nearly-equal
/// vertices (e.g. from different edge samplings that meet at the same point)
/// map to the same key.
#[inline]
pub fn f32x3_quantized_bits(v: [f32; 3]) -> [u32; 3] {
    let q = |x: f32| (x * 1e5).round().to_bits();
    [q(v[0]), q(v[1]), q(v[2])]
}

/// Convert a `[f32; 4]` to `[u32; 4]` for use as a hash/comparison key.
#[inline]
pub fn f32x4_to_bits(v: [f32; 4]) -> [u32; 4] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()]
}

/// Total-order key for `f32` suitable for sorting (handles NaN and -0.0).
///
/// Unlike `to_bits()`, this produces a key where the numeric ordering
/// matches the bitwise ordering (for non-NaN values).
///
/// See: <https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp>
#[inline]
pub fn f32_total_key(v: f32) -> u32 {
    let bits = v.to_bits();
    bits ^ (((bits as i32) >> 31) as u32)
}
