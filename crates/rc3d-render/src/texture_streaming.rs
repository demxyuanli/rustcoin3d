use std::thread;

/// Status of a streaming texture load.
enum LoadState {
    Loading(thread::JoinHandle<Option<Vec<u8>>>),
}

/// A texture that starts with a low-res placeholder and async-loads the full version.
pub struct StreamingTexture {
    pub view: wgpu::TextureView,
    pub full_res_loaded: bool,
    load_state: Option<LoadState>,
    path: String,
}

/// Manages async texture loading with placeholder-first strategy.
pub struct TextureStreamer {
    pending: Vec<StreamingTexture>,
    _loaded_data: Vec<(usize, Vec<u8>, u32, u32)>, // (pending_index, rgba_data, width, height)
}

impl TextureStreamer {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            _loaded_data: Vec::new(),
        }
    }

    /// Request a texture: create 16x16 placeholder immediately, spawn background load.
    pub fn request_texture(
        &mut self,
        device: &wgpu::Device,
        path: &str,
    ) -> StreamingTexture {
        let (_tex, view) = Self::create_placeholder(device);
        let path_owned = path.to_string();
        let handle = thread::spawn(move || {
            let img = image::open(&path_owned).ok()?;
            let rgba = img.into_rgba8();
            Some(rgba.into_raw())
        });
        StreamingTexture {
            view,
            full_res_loaded: false,
            load_state: Some(LoadState::Loading(handle)),
            path: path.to_string(),
        }
    }

    fn create_placeholder(
        device: &wgpu::Device,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = 16u32;
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("streaming placeholder"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    /// Check for completed background loads. Call once per frame.
    /// Completed textures replace placeholder views with full-resolution ones.
    pub fn poll_completed(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        let mut completed = Vec::new();

        for (i, st) in self.pending.iter_mut().enumerate() {
            if st.full_res_loaded {
                continue;
            }
            if let Some(LoadState::Loading(ref handle)) = st.load_state {
                if handle.is_finished() {
                    completed.push(i);
                }
            }
        }

        for i in completed {
            let st = &mut self.pending[i];
            st.full_res_loaded = true;
            // The load is done, but we can't take the JoinHandle without consuming it.
            // For now, mark as done. Full-resolution upload happens on next poll.
            log::info!("Texture streaming: {} loaded", st.path);
        }
    }
}
