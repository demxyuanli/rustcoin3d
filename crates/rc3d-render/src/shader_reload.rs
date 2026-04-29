use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Per-shader metadata for change detection.
#[derive(Clone, Debug)]
struct ShaderEntry {
    path: PathBuf,
    modified: SystemTime,
}

/// Simple poll-based shader file watcher for runtime hot-reload.
///
/// Call `check_and_reload()` each frame (or on demand) to detect
/// modified .wgsl files and trigger pipeline rebuilds.
pub struct ShaderHotReload {
    shaders: HashMap<String, ShaderEntry>,
    /// Callback invoked when any shader changes. Returns true if pipelines were rebuilt.
    on_change: Option<Box<dyn FnMut(&str) -> bool + Send>>,
    poll_interval_frames: u64,
    frame_since_check: u64,
}

impl ShaderHotReload {
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            on_change: None,
            poll_interval_frames: 30, // Check every 30 frames (~0.5s at 60fps)
            frame_since_check: 0,
        }
    }

    /// Register a shader to watch. `name` is an identifier (e.g. "pbr", "shadow_depth").
    /// `path` is the source .wgsl file path.
    pub fn watch(&mut self, name: &str, path: &Path) {
        let modified = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        self.shaders.insert(
            name.to_string(),
            ShaderEntry {
                path: path.to_path_buf(),
                modified,
            },
        );
    }

    /// Register multiple shaders from a directory. Discovers all .wgsl files.
    pub fn watch_directory(&mut self, dir: &Path) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    self.watch(&name, &path);
                }
            }
        }
    }

    /// Set the callback invoked when a shader changes.
    /// The closure receives the shader name and should return true if the rebuild succeeded.
    pub fn set_on_change(&mut self, cb: impl FnMut(&str) -> bool + Send + 'static) {
        self.on_change = Some(Box::new(cb));
    }

    /// Set how often (in frames) to check for shader changes.
    pub fn set_poll_interval(&mut self, frames: u64) {
        self.poll_interval_frames = frames.max(1);
    }

    /// Call each frame. Checks for shader file changes and triggers the callback.
    /// Returns a list of shader names that changed this frame.
    pub fn check_and_reload(&mut self) -> Vec<String> {
        self.frame_since_check += 1;
        if self.frame_since_check < self.poll_interval_frames {
            return Vec::new();
        }
        self.frame_since_check = 0;

        let mut changed = Vec::new();
        for (name, entry) in &mut self.shaders {
            if let Ok(meta) = std::fs::metadata(&entry.path) {
                if let Ok(mtime) = meta.modified() {
                    if mtime > entry.modified {
                        entry.modified = mtime;
                        changed.push(name.clone());
                    }
                }
            }
        }

        if !changed.is_empty() {
            log::info!(
                "Shader hot-reload: {} shader(s) changed: {:?}",
                changed.len(),
                changed
            );
            if let Some(ref mut cb) = self.on_change {
                for name in &changed {
                    if cb(name) {
                        log::info!("Shader '{}' reloaded successfully", name);
                    } else {
                        log::error!("Failed to reload shader '{}'", name);
                    }
                }
            }
        }

        changed
    }

    /// Force a reload of all registered shaders (useful for initial load or manual trigger).
    pub fn force_reload_all(&mut self) {
        for entry in self.shaders.values_mut() {
            if let Ok(meta) = std::fs::metadata(&entry.path) {
                if let Ok(mtime) = meta.modified() {
                    entry.modified = mtime;
                }
            }
        }
        self.frame_since_check = self.poll_interval_frames;
        self.check_and_reload();
    }

    /// Return the path for a registered shader.
    pub fn shader_path(&self, name: &str) -> Option<&Path> {
        self.shaders.get(name).map(|e| e.path.as_path())
    }
}

impl Default for ShaderHotReload {
    fn default() -> Self {
        Self::new()
    }
}
