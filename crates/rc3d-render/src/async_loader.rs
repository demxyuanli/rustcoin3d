use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use rc3d_core::{EngineError, EngineResult};
use rc3d_scene::SceneGraph;

/// Handle to a potentially not-yet-loaded asset.
pub struct AssetHandle<T> {
    state: Arc<Mutex<AssetState<T>>>,
    cancel_token: Arc<AtomicBool>,
}

#[derive(Clone)]
enum AssetState<T> {
    Loading,
    Loaded(T),
    Failed(String),
    Cancelled,
}

impl<T> AssetHandle<T> {
    pub fn is_loading(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), AssetState::Loading)
    }

    pub fn is_loaded(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), AssetState::Loaded(_))
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.load(Ordering::Acquire)
    }

    /// Cancel this load. If already in-flight, the worker thread will
    /// drop the result. If cancelled, `get()` returns `None`.
    pub fn cancel(&self) {
        self.cancel_token.store(true, Ordering::Release);
        if let Ok(mut lock) = self.state.lock() {
            if matches!(&*lock, AssetState::Loading) {
                *lock = AssetState::Cancelled;
            }
        }
    }

    pub(crate) fn new(state: Arc<Mutex<AssetState<T>>>, cancel_token: Arc<AtomicBool>) -> Self {
        Self { state, cancel_token }
    }
}

impl<T: Clone> AssetHandle<T> {
    pub fn get(&self) -> Option<T> {
        match &*self.state.lock().unwrap() {
            AssetState::Loaded(v) => Some(v.clone()),
            _ => None,
        }
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
        result: EngineResult<SceneGraph>,
    },
    Cancelled {
        path: PathBuf,
    },
}

/// Request to load an asset on a background thread.
struct LoadRequest {
    path: PathBuf,
    kind: LoadKind,
    cancel_token: Arc<AtomicBool>,
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
                                Err(_) => break,
                            }
                        };
                        // Check cancel token before starting work
                        if req.cancel_token.load(Ordering::Acquire) {
                            let _ = tx.send(LoadResult::Cancelled {
                                path: req.path,
                            });
                            continue;
                        }
                        match req.kind {
                            LoadKind::Scene => {
                                let result = rc3d_io::import_file(&req.path)
                                    .map_err(|e| EngineError::Parse(e.to_string()));
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
            // Re-use existing pending load; return a new handle with a fresh cancel token
            return AssetHandle::new(
                Arc::clone(state),
                Arc::new(AtomicBool::new(false)),
            );
        }

        let state = Arc::new(Mutex::new(AssetState::Loading));
        let cancel_token = Arc::new(AtomicBool::new(false));
        self.pending_scenes
            .insert(canonical.clone(), Arc::clone(&state));

        let _ = self._request_tx.send(LoadRequest {
            path: canonical,
            kind: LoadKind::Scene,
            cancel_token: Arc::clone(&cancel_token),
        });

        AssetHandle::new(state, cancel_token)
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
                                *lock = AssetState::Failed(e.to_string());
                            }
                        }
                    }
                    completed.push(path);
                }
                LoadResult::Cancelled { path } => {
                    if let Some(state) = self.pending_scenes.get(&path) {
                        if let Ok(mut lock) = state.lock() {
                            if matches!(&*lock, AssetState::Loading) {
                                *lock = AssetState::Cancelled;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_cancel_sets_state() {
        let state = Arc::new(Mutex::new(AssetState::<()>::Loading));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let handle = AssetHandle::new(Arc::clone(&state), Arc::clone(&cancel_token));

        handle.cancel();
        assert!(handle.is_cancelled());
        assert!(matches!(
            &*state.lock().unwrap(),
            AssetState::Cancelled
        ));
    }

    #[test]
    fn test_cancel_before_loaded_returns_none() {
        let state = Arc::new(Mutex::new(AssetState::<String>::Loading));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let handle = AssetHandle::new(Arc::clone(&state), Arc::clone(&cancel_token));

        handle.cancel();
        assert!(handle.get().is_none());
        assert!(!handle.is_loading());
    }

    #[test]
    fn test_get_after_load() {
        let state = Arc::new(Mutex::new(AssetState::<String>::Loaded("hello".into())));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let handle = AssetHandle::new(Arc::clone(&state), Arc::clone(&cancel_token));

        assert!(handle.is_loaded());
        assert_eq!(handle.get(), Some("hello".to_string()));
    }

    #[test]
    fn test_error_state() {
        let state = Arc::new(Mutex::new(AssetState::<String>::Failed("oops".into())));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let handle = AssetHandle::new(Arc::clone(&state), Arc::clone(&cancel_token));

        assert!(!handle.is_loaded());
        assert!(!handle.is_loading());
        assert_eq!(handle.error(), Some("oops".to_string()));
    }

    #[test]
    fn test_cancel_does_not_affect_already_loaded() {
        let state = Arc::new(Mutex::new(AssetState::<u32>::Loaded(42)));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let handle = AssetHandle::new(Arc::clone(&state), Arc::clone(&cancel_token));

        handle.cancel();
        // Already loaded data should survive cancel
        assert!(handle.is_loaded());
        assert_eq!(handle.get(), Some(42));
    }
}
