//! PDF 1.7 packaging of a U3D stream as an interactive 3D annotation.
//!
//! A single-page document whose full-page `/Subtype /3D` annotation binds the
//! embedded U3D stream (`/3DD`), an activation dictionary (`/3DA`, activate on
//! import), and one `/Type /3DView` camera framed on the model bounding box
//! (Acrobat 7.1+ / Reader).

use std::io::Write as _;

/// One interactive 3D model: a U3D stream plus framing metadata.
pub struct PdfScene {
    /// U3D byte stream (see [`crate::u3d::encode_u3d`]).
    pub u3d: Vec<u8>,
    /// Model title, used as the 3D view name.
    pub title: String,
    /// Bounding sphere center (model units).
    pub center: [f32; 3],
    /// Bounding sphere radius around `center` (model units).
    pub radius: f32,
    /// Per-view presentation options (render mode, lighting, background…).
    pub options: PdfOptions,
}

/// Initial-view presentation knobs that map onto the PDF 1.7 3D view
/// dictionary. `Default` reproduces the original export byte-for-byte.
#[derive(Clone, Copy, Debug)]
pub struct PdfOptions {
    /// Render mode dictionary (`/RM`). `None` leaves the artwork default.
    pub render_mode: Option<PdfRenderMode>,
    /// Lighting scheme. `White` keeps the legacy `/Lights` form used by the
    /// original validated export; any other scheme switches to the PDF 1.7
    /// `/3DLightingScheme` dictionary.
    pub lighting: PdfLighting,
    /// Solid `DeviceRGB` background (`/BG`); `None` = viewer default.
    pub background: Option<[f32; 3]>,
    /// Perspective field of view in degrees (`/P /FOV`); default 45.
    pub fov_degrees: Option<f32>,
    /// Scale factor applied to the framing camera distance, i.e. zoom out
    /// (`> 1`) or in (`< 1`) around the orbit center.
    pub zoom: Option<f32>,
    /// Override the orbit / framing center (world coordinates); default is
    /// the scene bounding-sphere center.
    pub center: Option<[f32; 3]>,
}

impl Default for PdfOptions {
    fn default() -> Self {
        Self {
            render_mode: None,
            lighting: PdfLighting::White,
            background: None,
            fov_degrees: None,
            zoom: None,
            center: None,
        }
    }
}

/// PDF 1.7 render-mode dictionary subtypes (ISO 32000-1 Table 319 names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdfRenderMode {
    Solid,
    SolidWireframe,
    Transparent,
    TransparentWireframe,
    Wireframe,
    Vertices,
    BoundingBox,
    TransparentBoundingBox,
    SolidOutline,
}

impl PdfRenderMode {
    /// Name-object value written under `/RM /Subtype`.
    pub fn pdf_name(self) -> &'static str {
        match self {
            Self::Solid => "Solid",
            Self::SolidWireframe => "SolidWireframe",
            Self::Transparent => "Transparent",
            Self::TransparentWireframe => "TransparentWireframe",
            Self::Wireframe => "Wireframe",
            Self::Vertices => "Vertices",
            Self::BoundingBox => "BoundingBox",
            Self::TransparentBoundingBox => "TransparentBoundingBox",
            Self::SolidOutline => "SolidOutline",
        }
    }
}

/// PDF 1.7 lighting-scheme dictionary subtypes (ISO 32000-1 Table 320 names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdfLighting {
    White,
    Artwork,
    None,
    Day,
    Night,
    Red,
    Blue,
    Cad,
    Headlamp,
}

impl PdfLighting {
    /// Name-object value written under `/LS /Subtype` (non-White schemes).
    pub fn pdf_name(self) -> &'static str {
        match self {
            Self::White => "White",
            Self::Artwork => "Artwork",
            Self::None => "None",
            Self::Day => "Day",
            Self::Night => "Night",
            Self::Red => "Red",
            Self::Blue => "Blue",
            Self::Cad => "CAD",
            Self::Headlamp => "Headlamp",
        }
    }
}

/// Byte-level PDF assembler. Text primitives are pushed ASCII; the raw U3D
/// stream is appended verbatim so byte offsets in the xref stay exact.
struct PdfBuf {
    out: Vec<u8>,
    offsets: Vec<usize>,
}

impl PdfBuf {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.out.extend_from_slice(bytes);
    }

    fn text(&mut self, s: &str) {
        self.out.extend_from_slice(s.as_bytes());
    }

    fn line(&mut self, s: &str) {
        self.text(s);
        self.out.push(b'\n');
    }

    /// Open `n 0 obj`, recording the byte offset for the xref table.
    fn begin_obj(&mut self, n: u32) {
        self.offsets.push(self.out.len());
        let _ = writeln!(self.out, "{n} 0 obj");
    }

    fn end_obj(&mut self) {
        self.line("endobj");
    }
}

/// Assemble a single-page interactive 3D PDF around a U3D stream.
pub fn encode_3d_pdf(scene: &PdfScene) -> Vec<u8> {
    let mut pdf = PdfBuf::new();
    pdf.text("%PDF-1.7\n%");
    pdf.raw(&[0xE2, 0xE3, 0xCF, 0xD3]); // binary comment marker
    pdf.line("");

    // Object 1: catalog.
    pdf.begin_obj(1);
    pdf.line("<< /Type /Catalog /Pages 2 0 R >>");
    pdf.end_obj();

    // Object 2: page tree.
    pdf.begin_obj(2);
    pdf.line("<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    pdf.end_obj();

    // Object 3: single page; the 3D annotation covers it entirely.
    pdf.begin_obj(3);
    pdf.line("<< /Type /Page /Parent 2 0 R");
    pdf.line("/MediaBox [0 0 612 792]");
    pdf.line("/Contents 4 0 R");
    pdf.line("/Annots [5 0 R]");
    pdf.line(">>");
    pdf.end_obj();

    // Object 4: empty content stream (nothing drawn in page space).
    pdf.begin_obj(4);
    pdf.line("<< /Length 2 >>");
    pdf.line("stream");
    pdf.raw(b"q Q");
    pdf.line("");
    pdf.line("endstream");
    pdf.end_obj();

    // Object 5: full-page 3D annotation. /3DA activates the view on import
    // and shows it; /3DV is the initial view below.
    let camera = camera_for(scene);
    pdf.begin_obj(5);
    pdf.line("<< /Type /Annot /Subtype /3D");
    pdf.line("/Rect [0 0 612 792]");
    pdf.line("/F 4");
    pdf.line("/3DD 6 0 R");
    pdf.line("/3DV 7 0 R");
    pdf.line("/3DA << /AIS true /DIS /D >>");
    pdf.line("/P 3 0 R");
    pdf.line(">>");
    pdf.end_obj();

    // Object 6: the U3D stream (referenced by /3DD).
    let u3d = &scene.u3d;
    pdf.offsets.push(pdf.out.len());
    pdf.text("6 0 obj\n<< /Type /3D /Subtype /U3D\n");
    let _ = writeln!(pdf.out, "/Length {}", u3d.len());
    pdf.text(">>\nstream\n");
    pdf.raw(u3d);
    pdf.text("\nendstream\nendobj\n");

    // Object 7: initial 3D view (referenced by /3DV). C2W is the 12-element
    // camera-to-world matrix: rows right/up/forward (row-major) followed by
    // the camera position in world coordinates. Optional /RM /LS /BG /P
    // entries are appended only when requested, keeping the default export
    // byte-identical to earlier builds.
    let opts = scene.options;
    let fov = opts.fov_degrees.unwrap_or(45.0);
    let c2w = camera.c2w;
    pdf.begin_obj(7);
    pdf.line("<< /Type /3DView");
    pdf.line(&format!("/XN ({})", esc(&scene.title)));
    pdf.line(&format!("/IN ({})", esc(&scene.title)));
    pdf.line("/MS /M");
    pdf.line(&format!("/C2W [{}]", num_row(&c2w)));
    pdf.line(&format!("/CO {}", fmt3(camera.dist)));
    pdf.line(&format!("/P << /Type /Projection /Subtype /Perspective /FOV {} >>", fmt3(fov)));
    if opts.lighting == PdfLighting::White {
        // Original validated form (Acrobat 7.x era lights dictionary).
        pdf.line("/LS << /Type /Lights /Scheme /White >>");
    } else {
        pdf.line(&format!(
            "/LS << /Type /3DLightingScheme /Subtype /{} >>",
            opts.lighting.pdf_name()
        ));
    }
    if let Some(rm) = opts.render_mode {
        pdf.line(&format!(
            "/RM << /Type /3DRenderMode /Subtype /{} >>",
            rm.pdf_name()
        ));
    }
    if let Some(bg) = opts.background {
        pdf.line(&format!(
            "/BG << /Type /3DBG /CS /DeviceRGB /C [{}] >>",
            num3(bg)
        ));
    }
    pdf.line(">>");
    pdf.end_obj();

    // Cross-reference table.
    let xref_at = pdf.out.len();
    let count = pdf.offsets.len() + 1; // + entry 0
    let _ = write!(pdf.out, "xref\n0 {count}\n");
    pdf.line("0000000000 65535 f ");
    for &off in &pdf.offsets {
        let _ = writeln!(pdf.out, "{off:010} 00000 n ");
    }
    let _ = write!(pdf.out, "trailer\n<< /Size {count} /Root 1 0 R >>\n");
    let _ = write!(pdf.out, "startxref\n{xref_at}\n%%EOF\n");
    pdf.out
}

/// A camera whose basis rows are right / up / forward and whose translation
/// is the eye position; the model center sits `dist` in front of the eye.
struct Camera {
    c2w: [[f32; 3]; 4],
    dist: f32,
}

/// Escape a literal string for a PDF parenthesised name.
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)")
}

fn fmt3(v: f32) -> String {
    if v.abs() < 1e-5 {
        return "0".to_string();
    }
    format!("{v:.4}")
}

fn num3(v: [f32; 3]) -> String {
    format!("{} {} {}", fmt3(v[0]), fmt3(v[1]), fmt3(v[2]))
}

fn num_row(v: &[[f32; 3]; 4]) -> String {
    [
        num3(v[0]),
        num3(v[1]),
        num3(v[2]),
        num3(v[3]),
    ]
    .join(" ")
}

/// Frame the model bounding sphere with a perspective camera honoring the
/// view options (FOV, zoom, framing-center override).
///
/// The camera sits `dist` along -Z from the center and looks toward +Z, so
/// with world up +Y the right/up/forward rows are un-mirrored. C2W third row
/// is the forward vector (eye -> target), matching Acrobat's convention
/// (verified against Adobe-generated 3D PDFs).
fn camera_for(scene: &PdfScene) -> Camera {
    let c = scene.options.center.unwrap_or(scene.center);
    let r = scene.radius.max(1e-4);
    let fov = scene.options.fov_degrees.unwrap_or(45.0);
    let zoom = scene.options.zoom.unwrap_or(1.0);
    let dist = (r / (fov.to_radians() * 0.5).tan()) * 1.25 * zoom;
    let eye = [c[0], c[1], c[2] - dist];

    let fwd = norm3(sub3(c, eye)); // eye -> target
    let up_world = [0.0, 1.0, 0.0];
    let right = norm3(cross3(up_world, fwd));
    let up = cross3(fwd, right);

    Camera {
        c2w: [right, up, fwd, eye],
        dist,
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / l, v[1] / l, v[2] / l]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_scene() -> PdfScene {
        PdfScene {
            u3d: vec![0x55, 0x33, 0x44, 0x00, 0x20, 0x00, 0x00, 0x00],
            title: "Cube".into(),
            center: [0.5, 0.5, 0.5],
            radius: 0.87,
            options: PdfOptions::default(),
        }
    }

    #[test]
    fn annotation_wiring_is_consistent() {
        let bytes = encode_3d_pdf(&sample_scene());
        let text = String::from_utf8_lossy(&bytes);
        // The annotation must reference stream 6 and view 7, and the view
        // object must actually declare /Type /3DView.
        let ann = text.find("/Subtype /3D").expect("annotation");
        let ann_end = text[ann..].find("endobj").unwrap() + ann;
        let ann_dict = &text[ann..ann_end];
        assert!(ann_dict.contains("/3DD 6 0 R"), "/3DD -> stream");
        assert!(ann_dict.contains("/3DV 7 0 R"), "/3DV -> view");
        assert!(ann_dict.contains("/AIS true"), "activation dict present");
        let view_at = text.find("/Type /3DView").expect("3DView object");
        let view_end = (view_at + 600).min(text.len());
        let view_dict = &text[view_at..view_end];
        assert!(view_dict.contains("/MS /M"), "matrix view mode");
        assert!(view_dict.contains("/C2W ["), "camera matrix present");
        assert!(view_dict.contains("/FOV 45"), "perspective projection");
        // No bogus /Type /3DD object type anywhere.
        assert!(!text.contains("/Type /3DD"), "no fake 3DD object type");
    }

    #[test]
    fn camera_points_forward_at_center() {
        let cam = camera_for(&sample_scene());
        // Forward row (index 2) must point from eye toward the center.
        let f = cam.c2w[2];
        let target_dir = sub3(sample_scene().center, cam.c2w[3]);
        let dot = f[0] * target_dir[0] + f[1] * target_dir[1] + f[2] * target_dir[2];
        assert!(dot > 0.99, "third basis row aims at the model");
    }

    /// Requested render mode / background / lighting / FOV all surface in
    /// the 3D view dictionary; default options must not add any of them.
    #[test]
    fn optional_view_entries_roundtrip() {
        let base = encode_3d_pdf(&sample_scene());
        let base = String::from_utf8_lossy(&base).into_owned();
        assert!(!base.contains("/3DLightingScheme"), "default keeps legacy lights dict");
        assert!(!base.contains("/3DRenderMode"), "no render-mode dict by default");
        assert!(!base.contains("/BG <<"), "no background by default");
        assert!(base.contains("/FOV 45"), "default 45 degree FOV");

        let mut scene = sample_scene();
        scene.options.render_mode = Some(PdfRenderMode::TransparentWireframe);
        scene.options.background = Some([1.0, 0.0, 0.5]);
        scene.options.lighting = PdfLighting::Cad;
        scene.options.fov_degrees = Some(60.0);
        scene.options.zoom = Some(0.5);
        let bytes = encode_3d_pdf(&scene);
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(text.contains("/Subtype /TransparentWireframe"), "render mode name");
        assert!(text.contains("/3DRenderMode"), "render-mode dict present");
        assert!(text.contains("/3DLightingScheme /Subtype /CAD"), "modern lighting dict");
        assert!(!text.contains("/Type /Lights"), "legacy lights replaced for custom scheme");
        assert!(text.contains("/BG << /Type /3DBG /CS /DeviceRGB /C [1.0000 0 0.5000] >>"), "background color");
        assert!(text.contains("/FOV 60"), "fov override applied");

        // Zoom and center override re-frame the camera: closer camera when
        // zooming in, and orbiting a moved center.
        let cam = camera_for(&scene);
        let tight = camera_for(&sample_scene());
        assert!(cam.dist < tight.dist, "zoom < 1 pulls the camera closer");
        let moved = {
            let mut s = sample_scene();
            s.options.center = Some([10.0, 0.0, 0.0]);
            camera_for(&s)
        };
        assert!((moved.c2w[3][0] - 10.0).abs() < 1e-3, "eye X follows the center override");
    }
}
