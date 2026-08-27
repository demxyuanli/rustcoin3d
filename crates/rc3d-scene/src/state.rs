use crate::element::{
    CoordinateElement, DisplayModeElement, Element, ElementId, LightData, LightElement,
    MaterialElement, ModelMatrixElement, NormalElement, ProjectionMatrixElement,
    TextureCoordinate2Element, ViewMatrixElement, NUM_ELEMENT_TYPES,
};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{Appearance, DisplayMode};
use crate::{MorphTargetNode, SkinnedMeshNode};

pub struct State {
    stacks: Vec<Vec<Box<dyn Element>>>,
    /// Current morph target data (set by MorphTargetNode traversal).
    pub morph_targets: Option<MorphTargetNode>,
    /// Current skinning payload (set by SkinnedMeshNode traversal).
    pub skinned_mesh: Option<SkinnedMeshNode>,
}

impl State {
    pub fn new() -> Self {
        let stacks: Vec<Vec<Box<dyn Element>>> = vec![
            vec![Box::new(ModelMatrixElement::default()) as Box<dyn Element>],
            vec![Box::new(ViewMatrixElement::default()) as Box<dyn Element>],
            vec![Box::new(ProjectionMatrixElement::default()) as Box<dyn Element>],
            vec![Box::new(CoordinateElement::default()) as Box<dyn Element>],
            vec![Box::new(NormalElement::default()) as Box<dyn Element>],
            vec![Box::new(MaterialElement::default()) as Box<dyn Element>],
            vec![Box::new(LightElement::default()) as Box<dyn Element>],
            vec![Box::new(TextureCoordinate2Element::default()) as Box<dyn Element>],
            vec![Box::new(DisplayModeElement::default()) as Box<dyn Element>],
        ];
        Self {
            stacks,
            morph_targets: None,
            skinned_mesh: None,
        }
    }

    pub fn push(&mut self, element_id: ElementId) {
        if let Some(stack) = self.stacks.get_mut(element_id.0 as usize) {
            if let Some(top) = stack.last() {
                stack.push(top.clone());
            }
        }
    }

    pub fn pop(&mut self, element_id: ElementId) {
        if let Some(stack) = self.stacks.get_mut(element_id.0 as usize) {
            if stack.len() > 1 {
                stack.pop();
            }
        }
    }

    pub fn push_all(&mut self) {
        for i in 0..NUM_ELEMENT_TYPES {
            self.push(ElementId(i as u16));
        }
    }

    pub fn pop_all(&mut self) {
        for i in (0..NUM_ELEMENT_TYPES).rev() {
            self.pop(ElementId(i as u16));
        }
    }

    // --- Typed accessors ---

    pub fn model_matrix(&self) -> Mat4 {
        self.get_el::<ModelMatrixElement>(ElementId(0)).expect("element type mismatch").matrix
    }

    pub fn set_model_matrix(&mut self, matrix: Mat4) {
        self.get_el_mut::<ModelMatrixElement>(ElementId(0)).matrix = matrix;
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.get_el::<ViewMatrixElement>(ElementId(1)).expect("element type mismatch").matrix
    }

    pub fn set_view_matrix(&mut self, matrix: Mat4) {
        self.get_el_mut::<ViewMatrixElement>(ElementId(1)).matrix = matrix;
    }

    pub fn projection_matrix(&self) -> Mat4 {
        self.get_el::<ProjectionMatrixElement>(ElementId(2)).expect("element type mismatch").matrix
    }

    pub fn set_projection_matrix(&mut self, matrix: Mat4) {
        self.get_el_mut::<ProjectionMatrixElement>(ElementId(2)).matrix = matrix;
    }

    pub fn coordinate(&self) -> &CoordinateElement {
        self.get_el_ref::<CoordinateElement>(ElementId(3))
    }

    pub fn set_coordinate(&mut self, points: Vec<Vec3>) {
        self.get_el_mut::<CoordinateElement>(ElementId(3)).points = points;
    }

    pub fn normal(&self) -> &NormalElement {
        self.get_el_ref::<NormalElement>(ElementId(4))
    }

    pub fn set_normal(&mut self, vectors: Vec<Vec3>) {
        self.get_el_mut::<NormalElement>(ElementId(4)).vectors = vectors;
    }

    pub fn material(&self) -> &MaterialElement {
        self.get_el_ref::<MaterialElement>(ElementId(5))
    }

    pub fn set_material(&mut self, mat: MaterialElement) {
        *self.get_el_mut::<MaterialElement>(ElementId(5)) = mat;
    }

    pub fn lights(&self) -> &[LightData] {
        &self.get_el_ref::<LightElement>(ElementId(6)).lights
    }

    pub fn add_light(&mut self, light: LightData) {
        self.get_el_mut::<LightElement>(ElementId(6)).lights.push(light);
    }

    pub fn texture_coordinate2(&self) -> &TextureCoordinate2Element {
        self.get_el_ref::<TextureCoordinate2Element>(ElementId(7))
    }

    pub fn set_texture_coordinate2(&mut self, coords: Vec<[f32; 2]>) {
        self.get_el_mut::<TextureCoordinate2Element>(ElementId(7)).coords = coords;
    }

    pub fn display_mode(&self) -> DisplayMode {
        self.get_el_ref::<DisplayModeElement>(ElementId(8)).mode
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.get_el_mut::<DisplayModeElement>(ElementId(8))
            .set_display_mode(mode);
    }

    pub fn appearance(&self) -> Appearance {
        self.get_el_ref::<DisplayModeElement>(ElementId(8)).appearance()
    }

    pub fn set_appearance(&mut self, app: Appearance) {
        self.get_el_mut::<DisplayModeElement>(ElementId(8))
            .set_appearance(app);
    }

    pub fn morph_targets(&self) -> Option<&MorphTargetNode> {
        self.morph_targets.as_ref()
    }

    pub fn set_morph_targets(&mut self, mt: MorphTargetNode) {
        self.morph_targets = Some(mt);
    }

    pub fn clear_morph_targets(&mut self) {
        self.morph_targets = None;
    }

    pub fn skinned_mesh(&self) -> Option<&SkinnedMeshNode> {
        self.skinned_mesh.as_ref()
    }

    pub fn set_skinned_mesh(&mut self, sm: SkinnedMeshNode) {
        self.skinned_mesh = Some(sm);
    }

    pub fn clear_skinned_mesh(&mut self) {
        self.skinned_mesh = None;
    }

    fn get_el<T: 'static>(&self, id: ElementId) -> Option<&T> {
        self.stacks
            .get(id.0 as usize)
            .and_then(|s| s.last())
            .and_then(|e| e.as_any().downcast_ref::<T>())
    }

    fn get_el_ref<T: 'static>(&self, id: ElementId) -> &T {
        self.get_el(id).expect("element type mismatch")
    }

    fn get_el_mut<T: 'static>(&mut self, id: ElementId) -> &mut T {
        self.stacks
            .get_mut(id.0 as usize)
            .and_then(|s| s.last_mut())
            .and_then(|e| e.as_any_mut().downcast_mut::<T>())
            .expect("element type mismatch")
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
