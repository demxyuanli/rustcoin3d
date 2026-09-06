pub mod graph;
pub mod gpu;

pub use graph::{
    CadLook, CompEdge, CompMenuGroup, CompNode, CompNodeId, CompOp, CompositorGraph, MathOp,
    MixBlend,
};
pub use gpu::CompositorGpu;

use graph::{CompExecStep, CompPing};

impl crate::renderer::Renderer {
    pub fn set_compositor(&mut self, graph: &CompositorGraph) {
        self.compositor_enabled = true;
        self.compositor_graph = graph.clone();
    }

    pub(crate) fn encode_compositor(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dest: &wgpu::TextureView,
    ) {
        if !self.compositor_enabled || self.overlay_pass {
            return;
        }
        let mut graph = self.compositor_graph.clone();
        let steps = graph.compile();
        self.compositor_graph.has_cycle = graph.has_cycle;
        if graph.has_cycle || film_passthrough(&steps) {
            return;
        }
        let src = if self.hdr_post_processing {
            self.gpu.post_fx.as_ref().map(|fx| fx.post_ldr_view.clone())
        } else {
            self.gpu.ldr_shade_view.clone()
        };
        let Some(src) = src else {
            return;
        };
        if self.gpu.compositor.is_none() {
            self.gpu.compositor = Some(CompositorGpu::new(&self.device, self.config.format));
        }
        let (w, h) = self.pass_target_size;
        let region = self.scene_region.clamped_to(w, h);
        let mut gpu = self.gpu.compositor.take();
        if let Some(ref mut gpu) = gpu {
            gpu.ensure_size(&self.device, w, h);
            gpu.encode(
                &self.device,
                &self.queue,
                encoder,
                Some(&src),
                &steps,
                dest,
                [region.x, region.y, region.width, region.height],
            );
        }
        self.gpu.compositor = gpu;
    }

    pub(crate) fn compositor_needs_ldr_film(&self) -> bool {
        if !self.compositor_enabled || self.hdr_post_processing || self.overlay_pass {
            return false;
        }
        let mut graph = self.compositor_graph.clone();
        let steps = graph.compile();
        !graph.has_cycle && !film_passthrough(&steps)
    }
}

fn film_passthrough(steps: &[CompExecStep]) -> bool {
    let mut copies = 0u32;
    let mut previews = 0u32;
    for s in steps {
        if s.write_preview {
            previews += 1;
            if !matches!(s.src_a, CompPing::A | CompPing::B) {
                return false;
            }
        } else if s.op_code == 0 && s.src_a == CompPing::Scene {
            copies += 1;
        } else {
            return false;
        }
    }
    copies == 1 && previews == 1
}
