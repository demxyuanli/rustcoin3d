//! Minimal U3D (ECMA-363 4th edition) binary writer for embedding static
//! meshes in a 3D PDF.
//!
//! Block layout mirrors the canonical structure produced by the Intel/RH
//! U3D converter (`box_2shaders.u3d`, `TextureLayers.u3d`):
//!
//! ```text
//! File Header
//!   -> Node Modifier Chains      (Model Node + optional Shading Modifier)
//!   -> Resource Modifier Chains  (CLOD Mesh Declaration)
//!   -> Lit Texture Shaders       (one per palette entry / material name)
//!   -> Material Resources        (one per palette entry)
//!   -> Texture Resource Declarations (PNG image per active texture)
//!   -> CLOD Base Mesh Continuations   (positions / normals / uv / faces)
//!   -> Texture Continuations          (PNG image bytes)
//! ```
//!
//! All blocks are little-endian, 32-bit aligned. Metadata sections are
//! always empty. Each mesh carries per-vertex normals and one shading
//! description. Optional texture mapping uses the single texture layer
//! (TM_NONE, unit transform) that the PDF viewers bind onto model UVs.

/// Embedded texture payload (PNG-encoded image).
#[derive(Clone, Debug)]
pub struct U3dTexture {
    /// Resource label; the encoder renames it to a unique `Tex{n}` id.
    pub name: String,
    /// Texture image width in pixels.
    pub width: u32,
    /// Texture image height in pixels.
    pub height: u32,
    /// Whether the image carries an alpha channel (RGBA vs RGB pixels).
    pub has_alpha: bool,
    /// PNG-encoded image bytes (U3D compression type 0x02).
    pub png: Vec<u8>,
}

/// Mesh payload consumed by the U3D encoder.
#[derive(Clone, Debug, Default)]
pub struct U3dMesh {
    /// Node / resource chain name.
    pub name: String,
    /// World-space positions.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex unit normals, same length as `positions`.
    pub normals: Vec<[f32; 3]>,
    /// One `[a, b, c]` index triple per triangle into `positions`.
    pub triangles: Vec<[u32; 3]>,
    /// Optional diffuse RGBA. When present the mesh is bound to a lit
    /// texture shader whose material carries this color; `None` renders
    /// with the viewer's default (unlit white) shader.
    pub diffuse: Option<[f32; 4]>,
    /// Per-vertex texture coordinates, one `[u, v]` pair per position.
    /// Meaningful only when `texture` is present and length matches
    /// `positions`.
    pub texcoords: Vec<[f32; 2]>,
    /// Optional texture resource bound to the mesh's shader layer 0.
    pub texture: Option<U3dTexture>,
    /// Named GroupNode ancestry (outermost first) for the U3D scene graph.
    /// Empty = the mesh hangs directly off the world (default node).
    pub assembly: Vec<String>,
}

impl U3dMesh {
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.triangles.is_empty()
    }

    /// True when UV data is complete enough to bind the texture.
    fn has_usable_uv(&self) -> bool {
        self.texture.is_some() && !self.texcoords.is_empty() && self.texcoords.len() == self.positions.len()
    }
}

// ── low-level byte writer ──────────────────────────────────────────────────

const HEADER_TYPE: u32 = 0x0044_3355; // "U3D\x00"
const CHAIN_TYPE: u32 = 0xFFFF_FF14; // Modifier Chain
const MODEL_NODE_TYPE: u32 = 0xFFFF_FF22; // Model Node
const GROUP_NODE_TYPE: u32 = 0xFFFF_FF21; // Group Node
const CLOD_DECL_TYPE: u32 = 0xFFFF_FF31; // CLOD Mesh Declaration
const CLOD_BASE_TYPE: u32 = 0xFFFF_FF3B; // CLOD Base Mesh Continuation
const SHADING_MODIFIER_TYPE: u32 = 0xFFFF_FF45; // Shading Modifier
const LIT_TEXTURE_SHADER_TYPE: u32 = 0xFFFF_FF53; // Lit Texture Shader
const MATERIAL_TYPE: u32 = 0xFFFF_FF54; // Material Resource
const TEXTURE_DECL_TYPE: u32 = 0xFFFF_FF55; // Texture Resource Declaration
const TEXTURE_CONT_TYPE: u32 = 0xFFFF_FF5C; // Texture Continuation

const CHAIN_NODE: u32 = 0; // node modifier chain
const CHAIN_MODEL_RESOURCE: u32 = 1; // model resource modifier chain

struct U3dWriter {
    out: Vec<u8>,
}

impl U3dWriter {
    fn new() -> Self {
        Self { out: Vec::new() }
    }

    fn len(&self) -> usize {
        self.out.len()
    }

    fn i16(&mut self, v: i16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn f32(&mut self, v: f32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn f64(&mut self, v: f64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    /// U3D string: U16 byte count followed by the UTF-8 bytes.
    fn str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        debug_assert!(bytes.len() <= u16::MAX as usize);
        self.u16(bytes.len() as u16);
        self.out.extend_from_slice(bytes);
    }

    fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    /// Pad the block payload so the next block starts on a 4-byte boundary.
    fn align4(&mut self) {
        while !self.out.len().is_multiple_of(4) {
            self.out.push(0);
        }
    }

    /// Start a block: writes the type and reserves the two size fields.
    /// Returns the offset of the block start for [`Self::end_block`].
    fn begin_block(&mut self, block_type: u32) -> usize {
        let start = self.out.len();
        self.u32(block_type);
        self.u32(0); // data size (patched on end)
        self.u32(0); // metadata size: always 0
        start
    }

    /// Patch the data size and align the block end to 4 bytes.
    fn end_block(&mut self, start: usize) {
        let data_len = (self.out.len() - start - 12) as u32;
        self.out[start + 4..start + 8].copy_from_slice(&data_len.to_le_bytes());
        self.align4();
    }

    fn bytes(&self) -> Vec<u8> {
        self.out.clone()
    }
}

// ── U3D document assembly ──────────────────────────────────────────────────

/// Encode `mesh` as a complete standalone U3D file.
pub fn encode_u3d(mesh: &U3dMesh) -> Vec<u8> {
    encode_u3d_many(std::slice::from_ref(mesh))
}

/// One shaded look (material colour + optional texture layer) shared by
/// one or more meshes; a shader and a material are emitted per entry.
struct ShadingEntry {
    /// Shader name == material resource name.
    material: String,
    /// Material diffuse colour (clamped before writing).
    color: [f32; 4],
    /// Texture resource name bound to shader channel 0, when textured.
    texture: Option<String>,
}

/// Encode several meshes into one U3D file.
///
/// Every mesh becomes its own model node + CLOD resource chain, so each
/// part can be selected / rotated independently in Acrobat. Non-empty
/// [`U3dMesh::assembly`] chains create GroupNodes (ECMA-363 0xFFFFFF21) so
/// parts appear under a parent/child assembly tree that mirrors the scene's
/// named Separator nesting; group and node transforms are identity because
/// the geometry is already in world space. Meshes with a diffuse colour or
/// a texture are bound (through a shading modifier) to a lit-texture
/// shader + material resource pair; equal colours share one palette entry.
/// Textured meshes add a PNG texture resource declared in the declaration
/// section and delivered in its own continuation block; their UVs ride in
/// the base mesh.
pub fn encode_u3d_many(meshes: &[U3dMesh]) -> Vec<u8> {
    if meshes.is_empty() {
        return Vec::new();
    }

    // De-duplicate diffuse colours into a palette of shader/material names
    // and collect distinct textures (one resource per unique source name).
    let mut palette: Vec<ShadingEntry> = Vec::new();
    let mut palette_key: std::collections::HashMap<(u32, Option<String>), usize> =
        std::collections::HashMap::new();
    let mut textures: Vec<&U3dTexture> = Vec::new();
    let mut tex_key: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut shader_of: Vec<Option<String>> = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        let tex = mesh.texture.as_ref().filter(|_| mesh.has_usable_uv());
        if mesh.diffuse.is_none() && tex.is_none() {
            shader_of.push(None);
            continue;
        }
        let tex_name = tex.map(|t| {
            let idx = *tex_key.entry(t.name.as_str()).or_insert_with(|| {
                textures.push(t);
                textures.len() - 1
            });
            format!("Tex{idx}")
        });
        // A textured mesh without an explicit colour uses a white material
        // so the multiply-blended texture alone determines the surface.
        let color = mesh.diffuse.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        let key = (color_key(color), tex_name.clone());
        let pe = *palette_key.entry(key).or_insert_with(|| {
            palette.push(ShadingEntry {
                material: format!("Mat{}", palette.len()),
                color,
                texture: tex_name.clone(),
            });
            palette.len() - 1
        });
        shader_of.push(Some(palette[pe].material.clone()));
    }

    // Distinct assembly groups in document order (outermost first), each with
    // a globally unique node name so U3D's name-based parent/child links stay
    // unambiguous even when two Separators share a user label.
    struct GroupNode {
        name: String,
        parent: Option<String>,
    }
    let mut groups: Vec<GroupNode> = Vec::new();
    let mut path_to_group: std::collections::HashMap<Vec<String>, String> =
        std::collections::HashMap::new();
    let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mesh in meshes {
        let mut prefix: Vec<String> = Vec::with_capacity(mesh.assembly.len());
        for part in &mesh.assembly {
            prefix.push(part.clone());
            if path_to_group.contains_key(&prefix) {
                continue;
            }
            let display = part.clone();
            let mut name = display.clone();
            let mut k = 2;
            while !used_names.insert(name.clone()) {
                name = format!("{display}_{k}");
                k += 1;
            }
            let parent = (prefix.len() > 1)
                .then(|| path_to_group.get(&prefix[..prefix.len() - 1]).cloned())
                .flatten();
            groups.push(GroupNode {
                name: name.clone(),
                parent,
            });
            path_to_group.insert(prefix.clone(), name);
        }
    }
    let parent_of = |mesh: &U3dMesh| -> Option<String> {
        if mesh.assembly.is_empty() {
            None
        } else {
            path_to_group.get(&mesh.assembly).cloned()
        }
    };

    let mut w = U3dWriter::new();

    // File Header Block. Data: version(4) profile(4) decl-size(4) file-size(8)
    // encoding(4) units-scale(8). Profile 0x0C = no compression + defined units.
    let header = w.begin_block(HEADER_TYPE);
    w.i16(0);
    w.i16(0);
    w.u32(0x0000_000C);
    let decl_size_at = w.len();
    w.u32(0); // declaration size, patched below
    let file_size_at = w.len();
    w.u64(0); // total file size, patched below
    w.u32(106); // MIBEnum 106 = UTF-8
    w.f64(1.0); // units scaling factor (model units == meters)
    w.end_block(header);

    let decl_start = w.len();
    // Group nodes first: children reference parents by name, and readers may
    // resolve them eagerly while walking the declaration section.
    for group in &groups {
        write_group_node_chain(&mut w, &group.name, group.parent.as_deref());
    }
    for (i, mesh) in meshes.iter().enumerate() {
        let parent = parent_of(mesh);
        write_node_chain(&mut w, mesh, shader_of[i].as_deref(), parent.as_deref());
    }
    for mesh in meshes {
        write_resource_chain(&mut w, mesh);
    }
    for entry in &palette {
        write_lit_shader(&mut w, &entry.material, &entry.material, entry.texture.as_deref());
    }
    for entry in &palette {
        write_material(&mut w, &entry.material, entry.color);
    }
    for (i, tex) in textures.iter().enumerate() {
        write_texture_declaration(&mut w, &format!("Tex{i}"), tex);
    }
    let decl_end = w.len();

    // Continuation blocks live after the declaration section.
    for mesh in meshes {
        write_base_mesh_continuation(&mut w, mesh);
    }
    for (i, tex) in textures.iter().enumerate() {
        write_texture_continuation(&mut w, &format!("Tex{i}"), tex);
    }

    let total = w.out.len() as u64;
    let decl_size = (decl_end - decl_start) as u32;
    w.out[decl_size_at..decl_size_at + 4].copy_from_slice(&decl_size.to_le_bytes());
    w.out[file_size_at..file_size_at + 8].copy_from_slice(&total.to_le_bytes());

    w.bytes()
}

/// Quantise an RGBA colour into a palette-dedup key.
fn color_key(c: [f32; 4]) -> u32 {
    let q = |v: f32| -> u32 { (v.clamp(0.0, 1.0) * 255.0).round() as u32 };
    q(c[0]) | (q(c[1]) << 8) | (q(c[2]) << 16) | (q(c[3]) << 24)
}

/// Node Modifier Chain (type 0) holding one Model Node and, for coloured
/// meshes, its shading modifier. `parent` names the GroupNode this part
/// belongs to; `None` keeps the part at the world root (as before).
fn write_node_chain(
    w: &mut U3dWriter,
    mesh: &U3dMesh,
    shader: Option<&str>,
    parent: Option<&str>,
) {
    let chain = w.begin_block(CHAIN_TYPE);
    w.str(&mesh.name);
    w.u32(CHAIN_NODE);
    w.u32(0); // chain attributes
    w.align4();
    w.u32(1 + u32::from(shader.is_some())); // modifier count

    // Model Node declaration (first modifier of the node chain).
    let model = w.begin_block(MODEL_NODE_TYPE);
    w.str(&mesh.name);
    write_parent_data(w, parent);
    w.str(&mesh.name); // resource modifier chain name (== node name)
    w.u32(3); // visibility: front and back
    w.end_block(model);

    if let Some(shader_name) = shader {
        write_shading_modifier(w, &mesh.name, shader_name);
    }

    w.end_block(chain);
}

/// Group Node Modifier Chain: a pure assembly node (no resource). Children
/// parts list this chain's name in their own Model Node parent data, so the
/// Acrobat model tree mirrors the scene's named Separator nesting. The
/// group transform is identity, matching the world-space geometry export.
fn write_group_node_chain(w: &mut U3dWriter, name: &str, parent: Option<&str>) {
    let chain = w.begin_block(CHAIN_TYPE);
    w.str(name);
    w.u32(CHAIN_NODE);
    w.u32(0); // chain attributes
    w.align4();
    w.u32(1); // modifier count

    let group = w.begin_block(GROUP_NODE_TYPE);
    w.str(name);
    write_parent_data(w, parent);
    w.end_block(group);

    w.end_block(chain);
}

/// ECMA-363 9.5.1.2 Parent Node Data: parent count followed by, for each
/// parent, its name string and a relative transform. All transforms written
/// here are identity because extracted geometry is already in world space;
/// the hierarchy is purely organisational.
fn write_parent_data(w: &mut U3dWriter, parent: Option<&str>) {
    let Some(parent) = parent else {
        w.u32(0); // parent node count: root
        return;
    };
    w.u32(1); // parent node count
    w.str(parent);
    // Identity matrix written in the standard "alphabetic" order
    // (A E I M / B F J N / C G K O / D H L P), identical to an identity in
    // any convention.
    for i in 0..16 {
        w.f32(if i % 5 == 0 { 1.0 } else { 0.0 });
    }
}

/// Shading Modifier: binds the model's renderable mesh group to one shader
/// list (the per-mesh lit shader carrying its diffuse material).
fn write_shading_modifier(w: &mut U3dWriter, node_name: &str, shader_name: &str) {
    let blk = w.begin_block(SHADING_MODIFIER_TYPE);
    w.str(node_name);
    w.u32(1); // chain index: the model node occupies index 0
    w.u32(0x0F); // shading attributes: mesh | line | point | glyph groups
    w.u32(1); // shader list count
    w.u32(1); // shader count
    w.str(shader_name);
    w.end_block(blk);
}

/// Lit Texture Shader with lighting enabled, optional single texture layer
/// (channel 0) and no alpha test. Its only channel binds the material
/// resource below; when a texture is given, one 140-byte texture-information
/// block follows the material name (matches the Intel converter output).
fn write_lit_shader(
    w: &mut U3dWriter,
    shader_name: &str,
    material_name: &str,
    texture: Option<&str>,
) {
    let blk = w.begin_block(LIT_TEXTURE_SHADER_TYPE);
    w.str(shader_name);
    w.u32(0x0000_0001); // attributes: lighting enabled
    w.f32(0.0); // alpha test reference
    w.u32(0x0000_0617); // alpha test function: ALWAYS
    w.u32(0x0000_0606); // color blend function: FB_ALPHA_BLEND
    w.u32(0x0000_0001); // render pass enabled flags: pass 0
    w.u32(u32::from(texture.is_some())); // shader channels (bit 0 = texture layer)
    w.u32(0); // alpha texture channels
    w.str(material_name);
    if let Some(tex_name) = texture {
        write_texture_info(w, tex_name);
    }
    w.end_block(blk);
}

/// One texture-layer information block: identity transform, TM_NONE, plain
/// multiply-style tiling as used by the reference converter.
fn write_texture_info(w: &mut U3dWriter, tex_name: &str) {
    w.str(tex_name);
    w.f32(1.0); // texture intensity
    w.u8(0x02); // blend function: replace (previous shading result)
    w.u8(0x01); // blend source: blending constant
    w.f32(0.5); // blend constant
    w.u8(0x00); // texture mode: TM_NONE (use model texture coordinates)
    // Texture transform matrix + wrap matrix, both identity, written in the
    // alphabetic order A E I M / B F J N / C G K O / D H L P (diagonal at
    // float indices 0, 5, 10, 15).
    for _ in 0..2 {
        for i in 0..16 {
            w.f32(if i % 5 == 0 { 1.0 } else { 0.0 });
        }
    }
    w.u8(0x03); // texture repeat: tile in both coordinate dimensions
}

/// Texture Resource Declaration: PNG image header + continuation metadata.
fn write_texture_declaration(w: &mut U3dWriter, name: &str, tex: &U3dTexture) {
    let blk = w.begin_block(TEXTURE_DECL_TYPE);
    w.str(name);
    w.u32(tex.height);
    w.u32(tex.width);
    w.u8(if tex.has_alpha { 0x0F } else { 0x0E }); // RGBA / RGB image type
    w.u32(1); // continuation image count
    w.u8(0x02); // compression type: PNG
    w.u8(if tex.has_alpha { 0x0F } else { 0x0E }); // channels: R|G|B(|A)
    w.u16(0); // continuation image attributes
    w.u32(tex.png.len() as u32); // image data byte count
    w.end_block(blk);
}

/// Texture Continuation: the PNG image bytes for image index 0.
fn write_texture_continuation(w: &mut U3dWriter, name: &str, tex: &U3dTexture) {
    let blk = w.begin_block(TEXTURE_CONT_TYPE);
    w.str(name);
    w.u32(0); // continuation image index
    w.out.extend_from_slice(&tex.png);
    w.end_block(blk);
}

/// Material Resource: ambient/diffuse/specular/emissive + reflectivity +
/// opacity. Ambient is a dimmed copy of the diffuse for a baseline tint.
fn write_material(w: &mut U3dWriter, name: &str, rgba: [f32; 4]) {
    let rgb = [
        rgba[0].clamp(0.0, 1.0),
        rgba[1].clamp(0.0, 1.0),
        rgba[2].clamp(0.0, 1.0),
    ];
    let blk = w.begin_block(MATERIAL_TYPE);
    w.str(name);
    w.u32(0x3F); // attributes: ambient|diffuse|specular|emissive|reflectivity|opacity
    for v in rgb {
        w.f32(v * 0.25); // ambient color
    }
    for v in rgb {
        w.f32(v); // diffuse color
    }
    for _ in 0..3 {
        w.f32(0.07); // specular color
    }
    for _ in 0..3 {
        w.f32(0.0); // emissive color
    }
    w.f32(0.1); // reflectivity
    w.f32(rgba[3].clamp(0.0, 1.0)); // opacity
    w.end_block(blk);
}

/// Resource Modifier Chain (type 1) holding the CLOD Mesh Declaration.
fn write_resource_chain(w: &mut U3dWriter, mesh: &U3dMesh) {
    let chain = w.begin_block(CHAIN_TYPE);
    w.str(&mesh.name);
    w.u32(CHAIN_MODEL_RESOURCE);
    w.u32(0); // chain attributes
    w.align4();
    w.u32(1); // modifier count

    write_clod_declaration(w, mesh);
    w.end_block(chain);
}

/// CLOD Mesh Declaration: max mesh description + CLOD + resource description.
fn write_clod_declaration(w: &mut U3dWriter, mesh: &U3dMesh) {
    let n_pos = mesh.positions.len() as u32;
    let n_nrm = mesh.normals.len() as u32;
    let n_face = mesh.triangles.len() as u32;
    let textured = mesh.has_usable_uv();

    let blk = w.begin_block(CLOD_DECL_TYPE);
    w.str(&mesh.name);
    w.u32(0); // chain index

    // Max Mesh Description.
    w.u32(0); // mesh attributes: faces carry a normal index per corner
    w.u32(n_face);
    w.u32(n_pos);
    w.u32(n_nrm);
    w.u32(0); // diffuse color count
    w.u32(0); // specular color count
    w.u32(if textured { n_pos } else { 0 }); // texture coord count
    w.u32(1); // shading description count
    w.u32(0); // shading attributes: no per-vertex colors
    w.u32(u32::from(textured)); // texture layer count
    if textured {
        w.u32(2); // texture coord dimensions (2D uv per layer)
    }
    w.u32(0); // original shading id

    // CLOD Description: single resolution covering the whole mesh.
    w.u32(n_pos); // minimum resolution
    w.u32(n_pos); // final maximum resolution

    // Resource Description (3 quality factors + 8 reals).
    w.u32(0); // position quality factor
    w.u32(0); // normal quality factor
    w.u32(0); // texture coord quality factor
    w.f32(1.0); // position inverse quant
    w.f32(1.0); // normal inverse quant
    w.f32(1.0); // texture coord inverse quant
    w.f32(1.0); // diffuse color inverse quant
    w.f32(1.0); // specular color inverse quant
    w.f32(0.0); // normal crease parameter
    w.f32(0.0); // normal update parameter
    w.f32(0.0); // normal tolerance parameter

    w.u32(0); // bone count (no skeleton)
    w.end_block(blk);
}

/// CLOD Base Mesh Continuation: the actual positions, normals, UVs and faces.
fn write_base_mesh_continuation(w: &mut U3dWriter, mesh: &U3dMesh) {
    let textured = mesh.has_usable_uv();
    let blk = w.begin_block(CLOD_BASE_TYPE);
    w.str(&mesh.name);
    w.u32(0); // chain index
    w.u32(mesh.triangles.len() as u32); // base face count
    w.u32(mesh.positions.len() as u32);
    w.u32(mesh.normals.len() as u32);
    w.u32(0); // diffuse
    w.u32(0); // specular
    w.u32(if textured { mesh.positions.len() as u32 } else { 0 }); // texture coords

    for p in &mesh.positions {
        w.f32(p[0]);
        w.f32(p[1]);
        w.f32(p[2]);
    }
    for n in &mesh.normals {
        w.f32(n[0]);
        w.f32(n[1]);
        w.f32(n[2]);
    }
    if textured {
        for uv in &mesh.texcoords {
            w.f32(uv[0]);
            w.f32(uv[1]);
        }
    }
    for tri in &mesh.triangles {
        w.u32(0); // shading id
        for &vi in tri {
            w.u32(vi); // position index
            w.u32(vi); // normal index (normals are 1:1 with positions)
            if textured {
                w.u32(vi); // texture coord index (uvs are 1:1 with positions)
            }
        }
    }
    w.end_block(blk);
}

#[cfg(test)]
mod tests;
