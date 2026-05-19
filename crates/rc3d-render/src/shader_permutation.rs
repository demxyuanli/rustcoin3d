//! Shader permutation system via WGSL source preprocessing.
//!
//! Expands `#ifdef` / `#ifndef` / `#else` / `#endif` directives in WGSL source
//! before compilation, enabling compile-time feature stripping for:
//! - Normal mapping (`#ifdef HAS_NORMAL_MAP`)
//! - Shadow mapping (`#ifdef HAS_SHADOW`)
//! - Albedo texture (`#ifdef HAS_ALBEDO_TEX`)
//! - IBL lighting (`#ifdef HAS_IBL`)
//!
//! Caches compiled shader modules by permutation key to avoid re-compilation.

use std::collections::HashMap;

/// Shader feature flags used as permutation key bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderFeatures {
    pub bits: u32,
}

impl ShaderFeatures {
    pub const NONE: Self = Self { bits: 0 };
    pub const HAS_NORMAL_MAP: Self = Self { bits: 1 << 0 };
    pub const HAS_SHADOW: Self = Self { bits: 1 << 1 };
    pub const HAS_ALBEDO_TEX: Self = Self { bits: 1 << 2 };
    pub const HAS_IBL: Self = Self { bits: 1 << 3 };
    pub const HAS_MR_TEX: Self = Self { bits: 1 << 4 };
    pub const HAS_EMISSIVE_TEX: Self = Self { bits: 1 << 5 };
    pub const HAS_OCCLUSION_TEX: Self = Self { bits: 1 << 6 };

    pub fn contains(&self, other: ShaderFeatures) -> bool {
        self.bits & other.bits == other.bits
    }

    pub fn with(mut self, flag: ShaderFeatures) -> Self {
        self.bits |= flag.bits;
        self
    }

    pub const ALL_NAMED: &[(&str, ShaderFeatures)] = &[
        ("HAS_NORMAL_MAP", ShaderFeatures::HAS_NORMAL_MAP),
        ("HAS_SHADOW", ShaderFeatures::HAS_SHADOW),
        ("HAS_ALBEDO_TEX", ShaderFeatures::HAS_ALBEDO_TEX),
        ("HAS_IBL", ShaderFeatures::HAS_IBL),
        ("HAS_MR_TEX", ShaderFeatures::HAS_MR_TEX),
        ("HAS_EMISSIVE_TEX", ShaderFeatures::HAS_EMISSIVE_TEX),
        ("HAS_OCCLUSION_TEX", ShaderFeatures::HAS_OCCLUSION_TEX),
    ];
}

pub type PermutationKey = u32;

/// Preprocess WGSL source: expand `#ifdef` / `#ifndef` / `#else` / `#endif`.
///
/// Lines in excluded blocks are dropped. Emitted lines preserve their original
/// content (minus the directive line itself, which becomes a `//` comment).
pub fn preprocess_wgsl(source: &str, features: ShaderFeatures) -> String {
    let active: [bool; 7] = [
        features.contains(ShaderFeatures::HAS_NORMAL_MAP),
        features.contains(ShaderFeatures::HAS_SHADOW),
        features.contains(ShaderFeatures::HAS_ALBEDO_TEX),
        features.contains(ShaderFeatures::HAS_IBL),
        features.contains(ShaderFeatures::HAS_MR_TEX),
        features.contains(ShaderFeatures::HAS_EMISSIVE_TEX),
        features.contains(ShaderFeatures::HAS_OCCLUSION_TEX),
    ];
    let names: [&str; 7] = [
        "HAS_NORMAL_MAP", "HAS_SHADOW", "HAS_ALBEDO_TEX", "HAS_IBL",
        "HAS_MR_TEX", "HAS_EMISSIVE_TEX", "HAS_OCCLUSION_TEX",
    ];

    let mut result = String::with_capacity(source.len());
    // Stack of (skip_lines, invert_on_else) per nested level
    let mut stack: Vec<(bool, bool)> = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();

        if let Some(feature) = trimmed.strip_prefix("#ifdef ") {
            let feature = feature.trim();
            let on = names.iter()
                .position(|&n| n == feature)
                .map(|i| active[i])
                .unwrap_or(false);
            let parent_skip = stack.last().map(|s| s.0).unwrap_or(false);
            stack.push((parent_skip || !on, on));
            continue;
        }
        if let Some(feature) = trimmed.strip_prefix("#ifndef ") {
            let feature = feature.trim();
            let off = names.iter()
                .position(|&n| n == feature)
                .map(|i| !active[i])
                .unwrap_or(false);
            let parent_skip = stack.last().map(|s| s.0).unwrap_or(false);
            let invert = off;
            stack.push((parent_skip || !off, invert));
            continue;
        }
        if trimmed == "#else" {
            let parent_skip = if stack.len() >= 2 {
                stack[stack.len() - 2].0
            } else {
                false
            };
            if let Some((skip, inverted)) = stack.last_mut() {
                *skip = parent_skip || *inverted;
                *inverted = !*inverted;
            }
            continue;
        }
        if trimmed == "#endif" {
            stack.pop();
            continue;
        }

        if !stack.last().map(|s| s.0).unwrap_or(false) {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// Manages compiled shader module caching keyed by (permutation_key, source_kind).
pub struct ShaderVariantCache {
    cache: HashMap<(PermutationKey, &'static str), ShaderModuleEntry>,
}

struct ShaderModuleEntry {
    module: wgpu::ShaderModule,
}

impl ShaderVariantCache {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    /// Fetch or compile a shader variant.
    ///
    /// `source_kind` distinguishes different shader source files ("pbr", "flat", etc.).
    pub fn get_module(
        &mut self,
        device: &wgpu::Device,
        key: PermutationKey,
        source_kind: &'static str,
        source: &str,
    ) -> &wgpu::ShaderModule {
        let entry_key = (key, source_kind);
        self.cache.entry(entry_key).or_insert_with(|| {
            let features = ShaderFeatures { bits: key };
            let processed = preprocess_wgsl(source, features);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("perm"),
                source: wgpu::ShaderSource::Wgsl(processed.into()),
            });
            ShaderModuleEntry { module }
        });
        &self.cache[&entry_key].module
    }

    /// Request recompilation by removing a cached entry.
    pub fn invalidate(&mut self, key: PermutationKey, source_kind: &'static str) {
        self.cache.remove(&(key, source_kind));
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for ShaderVariantCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_empty() {
        let src = "fn main() {}\n";
        let out = preprocess_wgsl(src, ShaderFeatures::NONE);
        assert_eq!(out, src);
    }

    #[test]
    fn preprocess_ifdef_active() {
        let src = "#ifdef HAS_NORMAL_MAP\nlet x = 1;\n#endif\nlet y = 2;\n";
        let out = preprocess_wgsl(src, ShaderFeatures::HAS_NORMAL_MAP);
        assert_eq!(out, "let x = 1;\nlet y = 2;\n");
    }

    #[test]
    fn preprocess_ifdef_inactive() {
        let src = "#ifdef HAS_SHADOW\nlet x = 1;\n#endif\nlet y = 2;\n";
        let out = preprocess_wgsl(src, ShaderFeatures::NONE);
        assert_eq!(out, "let y = 2;\n");
    }

    #[test]
    fn preprocess_ifndef() {
        let src = "#ifndef HAS_IBL\nno ibl branch\n#endif\n";
        let out = preprocess_wgsl(src, ShaderFeatures::HAS_IBL);
        assert_eq!(out, ""); // HAS_IBL is set, so #ifndef block is excluded

        let out2 = preprocess_wgsl(src, ShaderFeatures::NONE);
        assert_eq!(out2, "no ibl branch\n");
    }

    #[test]
    fn preprocess_ifdef_else() {
        let src = "#ifdef HAS_NORMAL_MAP\nnormal on\n#else\nnormal off\n#endif\ntail\n";
        let out = preprocess_wgsl(src, ShaderFeatures::HAS_NORMAL_MAP);
        assert_eq!(out, "normal on\ntail\n");
        let out2 = preprocess_wgsl(src, ShaderFeatures::NONE);
        assert_eq!(out2, "normal off\ntail\n");
    }

    #[test]
    fn preprocess_nested() {
        let src = "#ifdef HAS_SHADOW\n  #ifdef HAS_NORMAL_MAP\n    shadow + normal\n  #endif\nshadow only\n#endif\n";
        let out = preprocess_wgsl(src, ShaderFeatures::HAS_SHADOW.with(ShaderFeatures::HAS_NORMAL_MAP));
        assert_eq!(out, "    shadow + normal\nshadow only\n");
    }
}
