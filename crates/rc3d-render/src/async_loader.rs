use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use rc3d_scene::SceneGraph;

/// Handle to a potentially not-yet-loaded asset.
pub struct AssetHandle<T> {
    state: Arc<Mutex<AssetState<T>>>,
}

#[derive(Clone)]
enum AssetState<T> {
    Loading,
    Loaded(T),
    Failed(String),
}

impl<T: Clone> AssetHandle<T> {
    pub fn get(&self) -> Option<T> {
        match &*self.state.lock().unwrap() {
            AssetState::Loaded(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn is_loading(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), AssetState::Loading)
    }

    pub fn is_loaded(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), AssetState::Loaded(_))
    }

    pub fn error(&self) -> Option<String> {
        match &*self.state.lock().unwrap() {
            AssetState::Failed(e) => Some(e.clone()),
            _ => None,
        }
    }
}

/// A completed load result from a background thread.
enum LoadResult {
    Scene {
        path: PathBuf,
        result: Result<SceneGraph, String>,
    },
}

/// Request to load an asset on a background thread.
struct LoadRequest {
    path: PathBuf,
    kind: LoadKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LoadKind {
    Scene,
}

/// Manages background asset loading with a thread pool.
/// File I/O and parsing happen on background threads;
/// GPU upload happens on the main thread via `collect_loaded()`.
pub struct AsyncAssetManager {
    _request_tx: mpsc::Sender<LoadRequest>,
    result_rx: mpsc::Receiver<LoadResult>,
    pending_scenes: HashMap<PathBuf, Arc<Mutex<AssetState<SceneGraph>>>>,
    /// Number of background worker threads.
    worker_count: usize,
}

impl AsyncAssetManager {
    /// Create a new async asset manager with `worker_count` background threads.
    pub fn new(worker_count: usize) -> Self {
        let workers = worker_count.max(1);
        let (request_tx, request_rx) = mpsc::channel::<LoadRequest>();
        let (result_tx, result_rx) = mpsc::channel::<LoadResult>();

        let rx = Arc::new(Mutex::new(request_rx));
        for i in 0..workers {
            let rx = Arc::clone(&rx);
            let tx = result_tx.clone();
            thread::Builder::new()
                .name(format!("asset-loader-{i}"))
                .spawn(move || {
                    loop {
                        let req = {
                            let lock = rx.lock().unwrap();
                            match lock.recv() {
                                Ok(req) => req,
                                Err(_) => break, // Channel closed
                            }
                        };
                        match req.kind {
                            LoadKind::Scene => {
                                let result = rc3d_io::import_file(&req.path)
                                    .map_err(|e| e.to_string());
                                let _ = tx.send(LoadResult::Scene {
                                    path: req.path,
                                    result,
                                });
                            }
                        }
                    }
                })
                .expect("failed to spawn asset loader thread");
        }

        Self {
            _request_tx: request_tx,
            result_rx,
            pending_scenes: HashMap::new(),
            worker_count: workers,
        }
    }

    /// Request loading a scene file asynchronously.
    /// Returns a handle that will contain the SceneGraph once loaded.
    pub fn load_scene(&mut self, path: &Path) -> AssetHandle<SceneGraph> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        if let Some(state) = self.pending_scenes.get(&canonical) {
            return AssetHandle {
                state: Arc::clone(state),
            };
        }

        let state = Arc::new(Mutex::new(AssetState::Loading));
        self.pending_scenes
            .insert(canonical.clone(), Arc::clone(&state));

        let _ = self._request_tx.send(LoadRequest {
            path: canonical,
            kind: LoadKind::Scene,
        });

        AssetHandle { state }
    }

    /// Poll for completed loads. Returns paths of newly loaded assets.
    /// Call this each frame. GPU upload of any resulting data should
    /// happen on the calling (main) thread.
    pub fn collect_loaded(&mut self) -> Vec<PathBuf> {
        let mut completed = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            match result {
                LoadResult::Scene { path, result } => {
                    if let Some(state) = self.pending_scenes.get(&path) {
                        let mut lock = state.lock().unwrap();
                        match result {
                            Ok(scene) => {
                                *lock = AssetState::Loaded(scene);
                            }
                            Err(e) => {
                                *lock = AssetState::Failed(e);
                            }
                        }
                    }
                    completed.push(path);
                }
            }
        }
        completed
    }

    /// Check if a scene has finished loading (without collecting).
    pub fn scene_loaded(&self, path: &Path) -> bool {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.pending_scenes
            .get(&canonical)
            .map(|s| matches!(&*s.lock().unwrap(), AssetState::Loaded(_)))
            .unwrap_or(false)
    }

    pub fn worker_count(&self) -> usize {
        self.worker_count
    }
}
