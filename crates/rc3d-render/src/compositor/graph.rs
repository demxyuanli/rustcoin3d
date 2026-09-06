//! Compositor node graph: CAD look flags + Color sockets, Viewer reachability, topo execution.

use std::collections::{HashMap, HashSet, VecDeque};

pub type CompNodeId = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompOp {
    RenderLayers,
    Mix,
    Blur,
    BrightContrast,
    ColorRamp,
    /// Constant color source (Blender RGB node).
    Rgb,
    /// Constant scalar source (Blender Value node).
    Value,
    /// Scalar math with an operator (Blender Math node).
    Math,
    /// Exposure stop adjustment (Blender Exposure node).
    Exposure,
    /// Gamma correction (Blender Gamma node).
    Gamma,
    /// Hue / Saturation / Value adjust (Blender Hue Saturation Value node).
    HueSat,
    /// Invert colors (Blender Invert node).
    Invert,
    /// Alpha-weighted overlay with the second image (Blender Alpha Over).
    AlphaOver,
    /// Pixel offsets in UV space (Blender Translate node).
    Translate,
    /// Rotate image about its center (Blender Rotate node).
    Rotate,
    /// Scale image about its center (Blender Scale node).
    Scale,
    /// Crop the image to a normalized rect (Blender Crop node).
    Crop,
    /// Morphological min/max filter (Blender Dilate/Erode node).
    DilateErode,
    Ssao,
    Fxaa,
    Taa,
    ColorGrade,
    Bloom,
    Dof,
    Ssr,
    Fog,
    Edges,
    HiddenLine,
    Xray,
    Grid,
    Shadows,
    Viewer,
}

impl CompOp {
    pub fn is_cad_pass(self) -> bool {
        matches!(
            self,
            Self::Ssao
                | Self::Fxaa
                | Self::Taa
                | Self::ColorGrade
                | Self::Bloom
                | Self::Dof
                | Self::Ssr
                | Self::Fog
                | Self::Edges
                | Self::HiddenLine
                | Self::Xray
                | Self::Grid
                | Self::Shadows
        )
    }

    /// Group shown in the Add menu (Blender-style categories).
    pub fn menu_group(self) -> CompMenuGroup {
        match self {
            Self::RenderLayers | Self::Rgb | Self::Value => CompMenuGroup::Input,
            Self::Mix
            | Self::AlphaOver
            | Self::BrightContrast
            | Self::Exposure
            | Self::Gamma
            | Self::HueSat
            | Self::Invert
            | Self::ColorRamp => CompMenuGroup::Color,
            Self::Blur | Self::DilateErode => CompMenuGroup::Filter,
            Self::Translate | Self::Rotate | Self::Scale | Self::Crop => CompMenuGroup::Transform,
            Self::Math => CompMenuGroup::Converter,
            _ => CompMenuGroup::CadPass,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompMenuGroup {
    Input,
    Color,
    Filter,
    Transform,
    Converter,
    CadPass,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CadLook {
    pub active: bool,
    pub ssao: bool,
    pub fxaa: bool,
    pub taa: bool,
    pub color_grading: bool,
    pub bloom: bool,
    pub bloom_str: f32,
    pub dof: bool,
    pub ssr: bool,
    pub fog: bool,
    pub edges: bool,
    pub hidden_line: bool,
    pub xray: bool,
    pub grid: bool,
    pub shadows: bool,
}

impl CadLook {
    pub fn needs_hdr(self) -> bool {
        self.ssao
            || self.taa
            || self.color_grading
            || self.bloom
            || self.dof
            || self.ssr
            || self.fog
            || self.xray
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixBlend {
    Mix,
    Add,
    Multiply,
    Screen,
    Divide,
    Difference,
    Darken,
    Lighten,
    Overlay,
    Dodge,
    Burn,
    Hue,
    Saturation,
    Value,
    Color,
    Subtract,
}

impl MixBlend {
    /// GPU op code (matches `mix_color` in compositor.wgsl).
    pub fn gpu_code(self) -> u32 {
        match self {
            Self::Mix => 0,
            Self::Add => 1,
            Self::Multiply => 2,
            Self::Screen => 3,
            Self::Divide => 4,
            Self::Difference => 5,
            Self::Darken => 6,
            Self::Lighten => 7,
            Self::Overlay => 8,
            Self::Dodge => 9,
            Self::Burn => 10,
            Self::Hue => 11,
            Self::Saturation => 12,
            Self::Value => 13,
            Self::Color => 14,
            Self::Subtract => 15,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Minimum,
    Maximum,
    LessThan,
    GreaterThan,
    Absolute,
    Floor,
    Ceiling,
    Sine,
    Cosine,
}

impl MathOp {
    pub fn gpu_code(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Subtract => 1,
            Self::Multiply => 2,
            Self::Divide => 3,
            Self::Power => 4,
            Self::Minimum => 5,
            Self::Maximum => 6,
            Self::LessThan => 7,
            Self::GreaterThan => 8,
            Self::Absolute => 9,
            Self::Floor => 10,
            Self::Ceiling => 11,
            Self::Sine => 12,
            Self::Cosine => 13,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompNode {
    pub id: CompNodeId,
    pub op: CompOp,
    pub pos: [f32; 2],
    pub mix_blend: MixBlend,
    pub fac: f32,
    pub blur_radius: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub ramp: [(f32, [f32; 3]); 4],
    pub ramp_count: u32,
    /// Constant color for Rgb sources (linear RGBA).
    pub color: [f32; 4],
    /// Scalar: Value source, Math operand B, Exposure stops, Gamma, Rotate angle, Scale factor.
    pub value: f32,
    /// Math node operator.
    pub math_op: MathOp,
    /// Constant scalar for Math operand A when its image input is missing.
    pub value_a: f32,
    /// Transform offsets in UV: [tx, ty, rot(rad), scale].
    pub transform: [f32; 4],
    /// Crop rect in normalized UV (x0, y0, x1, y1).
    pub crop: [f32; 4],
    /// Dilate (positive) / erode (negative) pixel distance.
    pub morph_amount: i32,
    /// HueSat: hue shift (0..1), saturation (0..2), value (0..2).
    pub hsv: [f32; 3],
    /// Invert factor (0 = no change, 1 = full invert).
    pub invert_fac: f32,
    /// UI-only: node body expanded (Blender fold state), not sent to GPU.
    pub open: bool,
}

impl CompNode {
    pub fn new(id: CompNodeId, op: CompOp, pos: [f32; 2]) -> Self {
        Self {
            id,
            op,
            pos,
            mix_blend: MixBlend::Mix,
            fac: 0.5,
            blur_radius: 4.0,
            brightness: 0.0,
            contrast: 0.0,
            ramp: [
                (0.0, [0.0, 0.0, 0.0]),
                (1.0, [1.0, 1.0, 1.0]),
                (1.0, [1.0, 1.0, 1.0]),
                (1.0, [1.0, 1.0, 1.0]),
            ],
            ramp_count: 2,
            color: [1.0, 1.0, 1.0, 1.0],
            value: 0.5,
            math_op: MathOp::Add,
            value_a: 0.5,
            transform: [0.0, 0.0, 0.0, 1.0],
            crop: [0.0, 0.0, 1.0, 1.0],
            morph_amount: 1,
            hsv: [0.5, 1.0, 1.0],
            invert_fac: 1.0,
            open: true,
        }
    }

    pub fn input_count(&self) -> u8 {
        match self.op {
            CompOp::RenderLayers | CompOp::Rgb | CompOp::Value => 0,
            CompOp::Mix | CompOp::AlphaOver | CompOp::Math => 2,
            _ => 1,
        }
    }

    pub fn label(&self) -> &'static str {
        match self.op {
            CompOp::RenderLayers => "Beauty",
            CompOp::Mix => "Mix",
            CompOp::Blur => "Blur",
            CompOp::BrightContrast => "Bright/Contrast",
            CompOp::ColorRamp => "ColorRamp",
            CompOp::Rgb => "RGB",
            CompOp::Value => "Value",
            CompOp::Math => "Math",
            CompOp::Exposure => "Exposure",
            CompOp::Gamma => "Gamma",
            CompOp::HueSat => "Hue/Sat",
            CompOp::Invert => "Invert",
            CompOp::AlphaOver => "Alpha Over",
            CompOp::Translate => "Translate",
            CompOp::Rotate => "Rotate",
            CompOp::Scale => "Scale",
            CompOp::Crop => "Crop",
            CompOp::DilateErode => "Dilate/Erode",
            CompOp::Ssao => "SSAO",
            CompOp::Fxaa => "FXAA",
            CompOp::Taa => "TAA",
            CompOp::ColorGrade => "Color Grade",
            CompOp::Bloom => "Bloom",
            CompOp::Dof => "DOF",
            CompOp::Ssr => "SSR",
            CompOp::Fog => "Fog",
            CompOp::Edges => "Feature Edges",
            CompOp::HiddenLine => "Hidden Line",
            CompOp::Xray => "X-Ray",
            CompOp::Grid => "Grid",
            CompOp::Shadows => "Shadows",
            CompOp::Viewer => "Viewer",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompEdge {
    pub from: CompNodeId,
    pub to: CompNodeId,
    pub to_slot: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompPing {
    Scene,
    Black,
    A,
    B,
}

#[derive(Clone, Copy, Debug)]
pub struct CompExecStep {
    pub op_code: u32,
    pub mix_mode: u32,
    pub fac: f32,
    pub radius: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub dir: [f32; 2],
    pub stop_count: u32,
    pub stops: [[f32; 4]; 4],
    /// Constant RGBA (Rgb node).
    pub color: [f32; 4],
    /// Scalars packed: [value, value_a, transform_tx, transform_ty].
    pub scalars0: [f32; 4],
    /// Scalars packed: [rot(rad), scale, gamma, exposure].
    pub scalars1: [f32; 4],
    /// Scalars packed: [hue, sat, val, invert_fac].
    pub scalars2: [f32; 4],
    /// Crop rect normalized UV.
    pub crop: [f32; 4],
    /// Morph pixel distance (signed).
    pub morph: i32,
    pub src_a: CompPing,
    pub src_b: CompPing,
    pub dst_is_b: bool,
    pub write_preview: bool,
}

#[derive(Clone, Debug)]
pub struct CompositorGraph {
    pub nodes: Vec<CompNode>,
    pub edges: Vec<CompEdge>,
    next_id: CompNodeId,
    pub has_cycle: bool,
}

impl Default for CompositorGraph {
    fn default() -> Self {
        Self::identity()
    }
}

impl CompositorGraph {
    pub fn identity() -> Self {
        Self::from_chain(&[])
    }

    pub fn preset_industrial() -> Self {
        Self::from_chain(&[CompOp::Ssao, CompOp::Taa, CompOp::ColorGrade, CompOp::Shadows])
    }

    pub fn preset_product() -> Self {
        Self::from_chain(&[
            CompOp::Ssao,
            CompOp::Taa,
            CompOp::ColorGrade,
            CompOp::Ssr,
            CompOp::Fog,
            CompOp::Dof,
            CompOp::Shadows,
        ])
    }

    pub fn preset_hidden_line() -> Self {
        Self::from_chain(&[CompOp::HiddenLine])
    }

    pub fn preset_xray_edges() -> Self {
        Self::from_chain(&[CompOp::Xray, CompOp::Edges])
    }

    pub fn preset_ssao_taa() -> Self {
        Self::from_chain(&[CompOp::Ssao, CompOp::Taa])
    }

    pub fn preset_edges_only() -> Self {
        Self::from_chain(&[CompOp::Edges])
    }

    fn from_chain(ops: &[CompOp]) -> Self {
        let mut g = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_id: 1,
            has_cycle: false,
        };
        let beauty = g.add_node(CompOp::RenderLayers, [40.0, 80.0]);
        let mut prev = beauty;
        let mut x = 260.0;
        for &op in ops {
            let id = g.add_node(op, [x, 80.0]);
            g.connect(prev, id, 0);
            prev = id;
            x += 220.0;
        }
        let viewer = g.add_node(CompOp::Viewer, [x, 80.0]);
        g.connect(prev, viewer, 0);
        g
    }

    pub fn node(&self, id: CompNodeId) -> Option<&CompNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: CompNodeId) -> Option<&mut CompNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn replace(&mut self, nodes: Vec<CompNode>, edges: Vec<CompEdge>) {
        let max_id = nodes.iter().map(|n| n.id).max().unwrap_or(0);
        self.next_id = max_id.saturating_add(1).max(1);
        self.nodes = nodes;
        self.edges = edges;
    }

    pub fn add_node(&mut self, op: CompOp, pos: [f32; 2]) -> CompNodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(CompNode::new(id, op, pos));
        id
    }

    pub fn remove_node(&mut self, id: CompNodeId) {
        if self
            .node(id)
            .is_some_and(|n| matches!(n.op, CompOp::RenderLayers | CompOp::Viewer))
        {
            return;
        }
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    pub fn connect(&mut self, from: CompNodeId, to: CompNodeId, to_slot: u8) {
        if from == to {
            return;
        }
        let Some(dst) = self.node(to) else {
            return;
        };
        if to_slot >= dst.input_count() {
            return;
        }
        self.edges.retain(|e| !(e.to == to && e.to_slot == to_slot));
        self.edges.push(CompEdge { from, to, to_slot });
    }

    pub fn disconnect_input(&mut self, to: CompNodeId, to_slot: u8) {
        self.edges.retain(|e| !(e.to == to && e.to_slot == to_slot));
    }

    pub fn input_source(&self, to: CompNodeId, to_slot: u8) -> Option<CompNodeId> {
        self.edges
            .iter()
            .find(|e| e.to == to && e.to_slot == to_slot)
            .map(|e| e.from)
    }

    pub fn cad_look(&self) -> CadLook {
        let Some(viewer) = self
            .nodes
            .iter()
            .find(|n| n.op == CompOp::Viewer)
            .map(|n| n.id)
        else {
            return CadLook::default();
        };
        let reachable = self.ancestors_of(viewer);
        let mut look = CadLook::default();
        for n in &self.nodes {
            if !reachable.contains(&n.id) {
                continue;
            }
            match n.op {
                CompOp::Ssao => {
                    look.active = true;
                    look.ssao = true;
                }
                CompOp::Fxaa => {
                    look.active = true;
                    look.fxaa = true;
                }
                CompOp::Taa => {
                    look.active = true;
                    look.taa = true;
                }
                CompOp::ColorGrade => {
                    look.active = true;
                    look.color_grading = true;
                }
                CompOp::Bloom => {
                    look.active = true;
                    look.bloom = true;
                    look.bloom_str = n.fac.clamp(0.0, 1.0);
                }
                CompOp::Dof => {
                    look.active = true;
                    look.dof = true;
                }
                CompOp::Ssr => {
                    look.active = true;
                    look.ssr = true;
                }
                CompOp::Fog => {
                    look.active = true;
                    look.fog = true;
                }
                CompOp::Edges => {
                    look.active = true;
                    look.edges = true;
                }
                CompOp::HiddenLine => {
                    look.active = true;
                    look.hidden_line = true;
                }
                CompOp::Xray => {
                    look.active = true;
                    look.xray = true;
                }
                CompOp::Grid => {
                    look.active = true;
                    look.grid = true;
                }
                CompOp::Shadows => {
                    look.active = true;
                    look.shadows = true;
                }
                _ => {}
            }
        }
        look
    }

    pub fn compile(&mut self) -> Vec<CompExecStep> {
        let Some(viewer) = self
            .nodes
            .iter()
            .find(|n| n.op == CompOp::Viewer)
            .map(|n| n.id)
        else {
            self.has_cycle = false;
            return Vec::new();
        };
        let reachable = self.ancestors_of(viewer);
        match self.topo_reachable(&reachable) {
            Some(order) => {
                self.has_cycle = false;
                self.build_steps(&order, viewer)
            }
            None => {
                self.has_cycle = true;
                Vec::new()
            }
        }
    }

    fn ancestors_of(&self, sink: CompNodeId) -> HashSet<CompNodeId> {
        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        q.push_back(sink);
        seen.insert(sink);
        while let Some(id) = q.pop_front() {
            for e in &self.edges {
                if e.to == id && seen.insert(e.from) {
                    q.push_back(e.from);
                }
            }
        }
        seen
    }

    fn topo_reachable(&self, keep: &HashSet<CompNodeId>) -> Option<Vec<CompNodeId>> {
        let mut indeg: HashMap<CompNodeId, u32> = HashMap::new();
        for id in keep {
            indeg.insert(*id, 0);
        }
        for e in &self.edges {
            if keep.contains(&e.from) && keep.contains(&e.to) {
                *indeg.entry(e.to).or_insert(0) += 1;
            }
        }
        let mut q: VecDeque<CompNodeId> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut out = Vec::with_capacity(keep.len());
        while let Some(id) = q.pop_front() {
            out.push(id);
            for e in &self.edges {
                if e.from == id && keep.contains(&e.to) {
                    if let Some(d) = indeg.get_mut(&e.to) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            q.push_back(e.to);
                        }
                    }
                }
            }
        }
        if out.len() == keep.len() {
            Some(out)
        } else {
            None
        }
    }

    fn build_steps(&self, order: &[CompNodeId], viewer: CompNodeId) -> Vec<CompExecStep> {
        let mut written: HashMap<CompNodeId, CompPing> = HashMap::new();
        let mut use_b = false;
        let mut steps = Vec::new();
        for &id in order {
            let Some(n) = self.node(id) else {
                continue;
            };
            if n.op == CompOp::Viewer {
                let src = self
                    .input_source(id, 0)
                    .and_then(|s| written.get(&s).copied())
                    .unwrap_or(CompPing::Black);
                steps.push(preview_step(src));
                continue;
            }
            if n.op.is_cad_pass() {
                let src = self
                    .input_source(id, 0)
                    .and_then(|s| written.get(&s).copied())
                    .unwrap_or(CompPing::Black);
                written.insert(id, src);
                continue;
            }
            let dst = if use_b { CompPing::B } else { CompPing::A };
            use_b = !use_b;
            match n.op {
                CompOp::RenderLayers => {
                    steps.push(copy_step(CompPing::Scene, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Mix => {
                    let a = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    let b = self
                        .input_source(id, 1)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(mix_step(a, b, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Blur => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    let mid = dst;
                    steps.push(blur_step(src, mid, n, [1.0, 0.0]));
                    let dst2 = if use_b { CompPing::B } else { CompPing::A };
                    use_b = !use_b;
                    steps.push(blur_step(mid, dst2, n, [0.0, 1.0]));
                    written.insert(id, dst2);
                }
                CompOp::BrightContrast => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(bc_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::ColorRamp => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(ramp_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Rgb => {
                    steps.push(rgb_step(dst, n));
                    written.insert(id, dst);
                }
                CompOp::Value => {
                    steps.push(value_step(dst, n));
                    written.insert(id, dst);
                }
                CompOp::Math => {
                    let a = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    let b = self
                        .input_source(id, 1)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(math_step(a, b, dst, n));
                    written.insert(id, dst);
                }
                CompOp::AlphaOver => {
                    let a = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    let b = self
                        .input_source(id, 1)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(alpha_over_step(a, b, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Exposure => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(exposure_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Gamma => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(gamma_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::HueSat => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(hue_sat_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Invert => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(invert_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Translate => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(translate_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Rotate => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(rotate_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Scale => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(scale_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Crop => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(crop_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::DilateErode => {
                    let src = self
                        .input_source(id, 0)
                        .and_then(|s| written.get(&s).copied())
                        .unwrap_or(CompPing::Black);
                    steps.push(morph_step(src, dst, n));
                    written.insert(id, dst);
                }
                CompOp::Viewer | CompOp::Ssao | CompOp::Fxaa | CompOp::Taa | CompOp::ColorGrade
                | CompOp::Bloom | CompOp::Dof | CompOp::Ssr | CompOp::Fog | CompOp::Edges
                | CompOp::HiddenLine | CompOp::Xray | CompOp::Grid | CompOp::Shadows => {}
            }
        }
        if !steps.iter().any(|s| s.write_preview) {
            let src = written.get(&viewer).copied().unwrap_or(CompPing::Black);
            steps.push(preview_step(src));
        }
        steps
    }
}

fn copy_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    base_step(0, src, CompPing::Black, dst, n)
}

fn mix_step(a: CompPing, b: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(1, a, b, dst, n);
    s.mix_mode = n.mix_blend.gpu_code();
    s
}

fn blur_step(src: CompPing, dst: CompPing, n: &CompNode, dir: [f32; 2]) -> CompExecStep {
    let mut s = base_step(2, src, CompPing::Black, dst, n);
    s.dir = dir;
    s
}

fn bc_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    base_step(3, src, CompPing::Black, dst, n)
}

fn ramp_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(4, src, CompPing::Black, dst, n);
    s.stop_count = n.ramp_count.clamp(2, 4);
    for i in 0..4 {
        let (t, rgb) = n.ramp[i];
        s.stops[i] = [rgb[0], rgb[1], rgb[2], t];
    }
    s
}

/// Op codes 6.. mirror `hdr_main` in compositor.wgsl.
fn rgb_step(dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(6, CompPing::Black, CompPing::Black, dst, n);
    s.color = n.color;
    s
}

fn value_step(dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(7, CompPing::Black, CompPing::Black, dst, n);
    s.scalars0 = [n.value, 0.0, 0.0, 0.0];
    s
}

fn math_step(a: CompPing, b: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(8, a, b, dst, n);
    s.mix_mode = n.math_op.gpu_code();
    s.scalars0 = [n.value, n.value_a, 0.0, 0.0];
    s
}

fn alpha_over_step(a: CompPing, b: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(9, a, b, dst, n);
    s.fac = n.fac.clamp(0.0, 1.0);
    s
}

fn exposure_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(10, src, CompPing::Black, dst, n);
    s.scalars1 = [0.0, 0.0, 1.0, n.value];
    s
}

fn gamma_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(11, src, CompPing::Black, dst, n);
    s.scalars1 = [0.0, 0.0, n.value.max(1.0e-4), 0.0];
    s
}

fn hue_sat_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(12, src, CompPing::Black, dst, n);
    s.scalars2 = [n.hsv[0], n.hsv[1], n.hsv[2], 0.0];
    s
}

fn invert_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(13, src, CompPing::Black, dst, n);
    s.scalars2 = [0.0, 0.0, 0.0, n.invert_fac.clamp(0.0, 1.0)];
    s
}

fn translate_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(14, src, CompPing::Black, dst, n);
    s.scalars0 = [n.transform[0], n.transform[1], 0.0, 0.0];
    s
}

fn rotate_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(15, src, CompPing::Black, dst, n);
    s.scalars1 = [n.transform[2], 1.0, 0.0, 0.0];
    s
}

fn scale_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(16, src, CompPing::Black, dst, n);
    s.scalars1 = [0.0, n.transform[3].max(1.0e-4), 0.0, 0.0];
    s
}

fn crop_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(17, src, CompPing::Black, dst, n);
    s.crop = [
        n.crop[0].clamp(0.0, 1.0),
        n.crop[1].clamp(0.0, 1.0),
        n.crop[2].clamp(0.0, 1.0),
        n.crop[3].clamp(0.0, 1.0),
    ];
    s
}

fn morph_step(src: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    let mut s = base_step(18, src, CompPing::Black, dst, n);
    s.morph = n.morph_amount.clamp(-16, 16);
    s
}

fn preview_step(src: CompPing) -> CompExecStep {
    CompExecStep {
        op_code: 5,
        mix_mode: 0,
        fac: 0.0,
        radius: 0.0,
        brightness: 0.0,
        contrast: 0.0,
        dir: [0.0, 0.0],
        stop_count: 0,
        stops: [[0.0; 4]; 4],
        color: [0.0; 4],
        scalars0: [0.0; 4],
        scalars1: [0.0; 4],
        scalars2: [0.0; 4],
        crop: [0.0; 4],
        morph: 0,
        src_a: src,
        src_b: CompPing::Black,
        dst_is_b: false,
        write_preview: true,
    }
}

fn base_step(op: u32, a: CompPing, b: CompPing, dst: CompPing, n: &CompNode) -> CompExecStep {
    CompExecStep {
        op_code: op,
        mix_mode: 0,
        fac: n.fac.clamp(0.0, 1.0),
        radius: n.blur_radius.max(0.0),
        brightness: n.brightness.clamp(-1.0, 1.0),
        contrast: n.contrast.clamp(-1.0, 1.0),
        dir: [0.0, 0.0],
        stop_count: n.ramp_count,
        stops: [[0.0; 4]; 4],
        color: [0.0; 4],
        scalars0: [0.0; 4],
        scalars1: [0.0; 4],
        scalars2: [0.0; 4],
        crop: [0.0; 4],
        morph: 0,
        src_a: a,
        src_b: b,
        dst_is_b: matches!(dst, CompPing::B),
        write_preview: false,
    }
}
