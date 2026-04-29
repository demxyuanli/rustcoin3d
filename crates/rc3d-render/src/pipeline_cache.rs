use std::path::{Path, PathBuf};

/// Manages persistent GPU pipeline cache data for faster cold starts.
///
/// On supporting backends (Vulkan, DX12), compiled pipeline data is saved to disk.
/// On next startup, it's loaded back to skip recompilation.
pub struct PipelineCacheManager {
    path: PathBuf,
    cache: Option<wgpu::PipelineCache>,
}

impl PipelineCacheManager {
    pub fn new(device: &wgpu::Device, cache_dir: &Path) -> Self {
        let path = cache_dir.join("pipeline_cache.bin");

        if !device
            .features()
            .contains(wgpu::Features::PIPELINE_CACHE)
        {
            log::info!("Pipeline cache: PIPELINE_CACHE not enabled on device, skipping");
            return Self {
                path,
                cache: None,
            };
        }

        let initial_data = std::fs::read(&path).unwrap_or_default();

        // SAFETY: `create_pipeline_cache` is unsafe because invalid cache data
        // may cause undefined behavior. We load data previously saved by this
        // same code via `get_data()`. If the data is invalid (e.g. driver update),
        // `fallback: true` instructs wgpu to start fresh.
        let cache = if initial_data.is_empty() {
            log::info!("Pipeline cache: no previous cache found, starting fresh");
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("Pipeline Cache"),
                    data: None,
                    fallback: true,
                })
            }
        } else {
            log::info!(
                "Pipeline cache: loaded {} bytes from disk",
                initial_data.len()
            );
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("Pipeline Cache"),
                    data: Some(&initial_data),
                    fallback: true,
                })
            }
        };

        Self {
            path,
            cache: Some(cache),
        }
    }

    /// Returns the PipelineCache reference for use in pipeline creation descriptors.
    pub fn as_ref(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref()
    }

    /// Save the accumulated pipeline cache data to disk.
    /// Returns true if data was written.
    pub fn save_to_disk(&mut self) -> bool {
        let Some(ref cache) = self.cache else {
            return false;
        };
        let Some(data) = cache.get_data() else {
            return false;
        };

        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match std::fs::write(&self.path, &data) {
            Ok(_) => {
                log::info!(
                    "Pipeline cache: saved {} bytes to {:?}",
                    data.len(),
                    self.path
                );
                true
            }
            Err(e) => {
                log::warn!("Pipeline cache: failed to save: {e}");
                false
            }
        }
    }

    /// Discard the cache (use when GPU driver changes or cache becomes invalid).
    pub fn invalidate(&mut self) {
        self.cache = None;
        let _ = std::fs::remove_file(&self.path);
    }

    /// Returns the cache directory path.
    pub fn cache_path(&self) -> &Path {
        &self.path
    }
}
