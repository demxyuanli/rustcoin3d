use rc3d_editor::PixelRect;

pub struct DocumentWebView {
    #[cfg(windows)]
    inner: Option<wry::WebView>,
}

impl DocumentWebView {
    pub fn new(window: &winit::window::Window) -> Self {
        #[cfg(windows)]
        {
            match build_webview(window) {
                Ok(inner) => Self { inner: Some(inner) },
                Err(e) => {
                    log::warn!("WebView2 unavailable: {e}");
                    Self { inner: None }
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = window;
            Self {}
        }
    }

    pub fn sync(&self, rect: Option<PixelRect>, html_visible: bool) {
        #[cfg(windows)]
        {
            let Some(wv) = self.inner.as_ref() else {
                return;
            };
            let visible = html_visible && rect.is_some();
            let _ = wv.set_visible(visible);
            if let Some(r) = rect {
                let bounds = wry::Rect {
                    position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                        r.x as i32,
                        r.y as i32,
                    )),
                    size: dpi::Size::Physical(dpi::PhysicalSize::new(r.width, r.height)),
                };
                let _ = wv.set_bounds(bounds);
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (rect, html_visible);
        }
    }
}

#[cfg(windows)]
fn build_webview(window: &winit::window::Window) -> Result<wry::WebView, wry::Error> {
    wry::WebViewBuilder::new()
        .with_html(STUDIO_HTML)
        .with_bounds(wry::Rect {
            position: dpi::Position::Physical(dpi::PhysicalPosition::new(0, 0)),
            size: dpi::Size::Physical(dpi::PhysicalSize::new(1, 1)),
        })
        .build_as_child(window)
}

const STUDIO_HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8">
<style>
body { font-family: "Segoe UI", "Segoe UI Variable", sans-serif; margin: 16px; background: #202020; color: #ffffff; }
h1 { font-size: 18px; }
code { background: #333; padding: 1px 4px; }
</style></head>
<body>
<h1>rustcoin3d Studio</h1>
<p>This HTML panel is a native WebView2 child window over the document slot. The 3D viewport stays in the center.</p>
<p>Switch back to the Markdown tab to hide this overlay.</p>
</body></html>
"#;
