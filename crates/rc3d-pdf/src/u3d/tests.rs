/// Human-readable summary of a U3D byte stream (for diagnostics).
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn describe(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "u3d {} bytes", bytes.len());
    if bytes.len() < 52 {
        return s;
    }
    let _ = writeln!(s, "  magic      {:08X}", u32_at(bytes, 0));
    let _ = writeln!(s, "  data size  {}", u32_at(bytes, 4));
    let _ = writeln!(s, "  profile    {:08X}", u32_at(bytes, 16));
    let _ = writeln!(s, "  decl size  {}", u32_at(bytes, 20));
    let _ = writeln!(s, "  file size  {}", u64_at(bytes, 24));
    let _ = writeln!(s, "  encoding   {}", u32_at(bytes, 32));
    s
}

#[cfg(test)]
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

#[cfg(test)]
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}


    use super::*;

    fn demo_mesh() -> U3dMesh {
        // Tetrahedron: 4 positions, 4 faces.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let normals = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let triangles = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        U3dMesh {
            name: "MeshResource".into(),
            positions,
            normals,
            triangles,
            diffuse: None,
            ..Default::default()
        }
    }

    fn colored_mesh(name: &str, c: [f32; 4]) -> U3dMesh {
        let mut m = demo_mesh();
        m.name = name.into();
        m.diffuse = Some(c);
        m
    }

    /// Walk every block from the end of the header, returning block types.
    fn walk_types(bytes: &[u8]) -> Vec<u32> {
        let mut pos = 44; // end of file header block (12 header + 32 data)
        let mut seen = Vec::new();
        while pos + 12 <= bytes.len() {
            let ty = u32_at(bytes, pos);
            let data = u32_at(bytes, pos + 4) as usize;
            let meta = u32_at(bytes, pos + 8) as usize;
            assert_eq!(meta, 0, "no metadata in minimal files");
            seen.push(ty);
            pos += 12 + data + meta;
            while pos % 4 != 0 {
                pos += 1;
            }
        }
        assert_eq!(pos, bytes.len(), "walk consumes the whole file");
        seen
    }

    #[test]
    fn header_fields_are_patched() {
        let bytes = encode_u3d(&demo_mesh());
        assert_eq!(u32_at(&bytes, 0), 0x0044_3355, "U3D magic");
        assert_eq!(u32_at(&bytes, 4), 32, "header data size");
        assert_eq!(u64_at(&bytes, 24), bytes.len() as u64, "file size");
        // The declared declaration size equals the distance from the end of
        // the file header (44) to the continuation block that follows the
        // two declaration chains.
        let decl = u32_at(&bytes, 20) as usize;
        assert_eq!(44 + decl, find_type(&bytes, CLOD_BASE_TYPE), "decl ends exactly before continuation");
    }

    fn find_type(bytes: &[u8], wanted: u32) -> usize {
        let mut pos = 44;
        while pos + 12 <= bytes.len() {
            let ty = u32_at(bytes, pos);
            if ty == wanted {
                return pos;
            }
            let data = u32_at(bytes, pos + 4) as usize;
            pos += 12 + data;
            while pos % 4 != 0 {
                pos += 1;
            }
        }
        bytes.len()
    }

    /// Every block must start 4-byte aligned and the walk must terminate
    /// exactly at EOF, proving sizes, alignment and patched lengths agree.
    #[test]
    fn block_walk_is_aligned_and_exhaustive() {
        let seen = walk_types(&encode_u3d(&demo_mesh()));
        assert_eq!(
            seen,
            vec![CHAIN_TYPE, CHAIN_TYPE, CLOD_BASE_TYPE],
            "block type order after header: node chain, resource chain, continuation"
        );
    }

    /// Coloured meshes: node chain per mesh, resource chain per mesh, one
    /// shader + material declaration per distinct colour, then a base mesh
    /// continuation per mesh. Equal colours share one palette entry.
    #[test]
    fn colored_multi_mesh_layout_matches_reference() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        let meshes = vec![
            colored_mesh("Cube", red),
            colored_mesh("Sphere", blue),
            colored_mesh("CubeCopy", red),
        ];
        let bytes = encode_u3d_many(&meshes);
        let seen = walk_types(&bytes);
        // 3 node chains + 3 resource chains, 2 shaders + 2 materials,
        // then 3 base mesh continuations.
        assert_eq!(
            seen,
            vec![
                CHAIN_TYPE,
                CHAIN_TYPE,
                CHAIN_TYPE, // node chains
                CHAIN_TYPE,
                CHAIN_TYPE,
                CHAIN_TYPE, // model resource chains
                LIT_TEXTURE_SHADER_TYPE,
                LIT_TEXTURE_SHADER_TYPE, // shaders (red + blue)
                MATERIAL_TYPE,
                MATERIAL_TYPE, // materials (red + blue)
                CLOD_BASE_TYPE,
                CLOD_BASE_TYPE,
                CLOD_BASE_TYPE, // base mesh continuations
            ],
            "coloured meshes carry one shader/material per distinct colour"
        );
        // Declared declaration section ends exactly where continuations start.
        let decl = u32_at(&bytes, 20) as usize;
        assert_eq!(
            44 + decl,
            find_type(&bytes, CLOD_BASE_TYPE),
            "shader + material blocks live inside the declaration section"
        );
        assert_eq!(u64_at(&bytes, 24), bytes.len() as u64, "file size patched");
    }

    /// Shader and material declarations follow the resource chains in
    /// declaration order and appear before the base mesh continuations.
    #[test]
    fn shader_material_order_matches_reference_file() {
        let meshes = vec![colored_mesh("Cube", [0.9, 0.2, 0.1, 1.0])];
        let bytes = encode_u3d_many(&meshes);
        let node_chain = find_type(&bytes, CHAIN_TYPE);
        let shader = find_type(&bytes, LIT_TEXTURE_SHADER_TYPE);
        let material = find_type(&bytes, MATERIAL_TYPE);
        let base = find_type(&bytes, CLOD_BASE_TYPE);
        assert!(node_chain < shader, "chains precede shader resources");
        assert!(shader < material, "shader precedes its material");
        assert!(material < base, "materials precede continuations");
    }

    fn tiny_png() -> Vec<u8> {
        use image::codecs::png::PngEncoder;
        use image::{ExtendedColorType, ImageEncoder as _};
        let mut out = Vec::new();
        let px = [
            200u8, 60, 60, // (0,0)
            30, 160, 90, // (1,0)
            255, 240, 220, // (0,1)
            120, 40, 180, // (1,1)
        ];
        PngEncoder::new(&mut out)
            .write_image(&px, 2, 2, ExtendedColorType::Rgb8)
            .unwrap();
        out
    }

    fn textured_mesh(name: &str) -> U3dMesh {
        let mut m = demo_mesh();
        m.name = name.into();
        m.texcoords = m.positions.iter().map(|p| [p[0], p[1]]).collect();
        m.texture = Some(U3dTexture {
            name: "checker".into(),
            width: 2,
            height: 2,
            has_alpha: false,
            png: tiny_png(),
        });
        m
    }

    /// A textured mesh emits: node chain, resource chain, one shader +
    /// material (white, no explicit colour), one texture declaration, then
    /// the base-mesh and texture continuations — mirroring the reference
    /// converter's block order. The PNG payload and the `Tex0` resource
    /// name must survive in the stream.
    #[test]
    fn textured_mesh_block_layout_matches_reference() {
        let bytes = encode_u3d(&textured_mesh("Tile"));
        let seen = walk_types(&bytes);
        assert_eq!(
            seen,
            vec![
                CHAIN_TYPE,
                CHAIN_TYPE,
                LIT_TEXTURE_SHADER_TYPE,
                MATERIAL_TYPE,
                TEXTURE_DECL_TYPE,
                CLOD_BASE_TYPE,
                TEXTURE_CONT_TYPE,
            ],
            "textured mesh layout follows the reference block order"
        );
        let decl = u32_at(&bytes, 20) as usize;
        assert_eq!(
            44 + decl,
            find_type(&bytes, CLOD_BASE_TYPE),
            "texture declaration stays inside the declaration section"
        );
        assert!(bytes.windows(4).any(|w| w == [0x89, 0x50, 0x4E, 0x47]), "PNG signature embedded");
        assert!(String::from_utf8_lossy(&bytes).contains("Tex0"), "texture resource named Tex0");
    }

    /// Two meshes sharing one texture source share the texture resource but
    /// keep separate model chains and base-mesh continuations.
    #[test]
    fn shared_texture_writes_single_resource() {
        let meshes = vec![textured_mesh("A"), textured_mesh("B")];
        let seen = walk_types(&encode_u3d_many(&meshes));
        assert_eq!(
            seen,
            vec![
                CHAIN_TYPE,
                CHAIN_TYPE, // node chains x2
                CHAIN_TYPE,
                CHAIN_TYPE, // resource chains x2
                LIT_TEXTURE_SHADER_TYPE,
                MATERIAL_TYPE,
                TEXTURE_DECL_TYPE, // one shared white shading entry + texture
                CLOD_BASE_TYPE,
                CLOD_BASE_TYPE,
                TEXTURE_CONT_TYPE,
            ]
        );
    }

    /// A texture without matching UVs is ignored: the mesh stays unshaded
    /// (no shader / material / texture blocks at all).
    #[test]
    fn texture_without_uv_is_ignored() {
        let mut m = demo_mesh();
        m.texture = Some(U3dTexture {
            name: "orphan".into(),
            width: 2,
            height: 2,
            has_alpha: false,
            png: tiny_png(),
        });
        let seen = walk_types(&encode_u3d(&m));
        assert_eq!(seen, vec![CHAIN_TYPE, CHAIN_TYPE, CLOD_BASE_TYPE]);
    }

    /// The material resource carries opacity: last float of its data.
    #[test]
    fn material_opacity_roundtrips() {
        let bytes = encode_u3d(&colored_mesh("Glass", [0.0, 0.5, 1.0, 0.5]));
        let at = find_type(&bytes, MATERIAL_TYPE);
        assert_ne!(at, bytes.len(), "material block present");
        let data_len = u32_at(&bytes, at + 4) as usize;
        let data = &bytes[at + 12..at + 12 + data_len];
        let opacity = f32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        assert_eq!(opacity, 0.5, "opacity float lands at the end of the material");
    }

    /// Flattened part chains (no assembly) stay byte-identical in layout to
    /// the pre-group format: each node chain declares one root-level model.
    #[test]
    fn flat_mesh_writes_no_group_chains() {
        let meshes = vec![colored_mesh("A", [1.0, 0.0, 0.0, 1.0])];
        let seen = walk_types(&encode_u3d_many(&meshes));
        assert_eq!(
            seen,
            vec![CHAIN_TYPE, CHAIN_TYPE, LIT_TEXTURE_SHADER_TYPE, MATERIAL_TYPE, CLOD_BASE_TYPE],
            "flat export keeps exactly node + resource + shader + material chains"
        );
    }

    /// Parts with an assembly path emit one GroupNode chain per distinct
    /// prefix (outermost first), then model chains whose parent data names
    /// the innermost group, then the usual resource/shader/continuation
    /// blocks. Group transforms are identity (16 floats: 1, 0, 0, 1, ...).
    #[test]
    fn assembly_groups_shape_the_model_tree() {
        let mut wheel_left = colored_mesh("PartA", [1.0, 0.0, 0.0, 1.0]);
        wheel_left.assembly = vec!["Car".into(), "WheelLeft".into()];
        let mut wheel_right = colored_mesh("PartB", [0.0, 0.0, 1.0, 1.0]);
        wheel_right.assembly = vec!["Car".into(), "WheelRight".into()];
        let mut hood = colored_mesh("PartC", [0.0, 1.0, 0.0, 1.0]);
        hood.assembly = vec!["Car".into(), "Hood".into()];
        let bytes = encode_u3d_many(&[wheel_left, wheel_right, hood]);

        // GroupNode chains lead the declaration section; one per distinct
        // path prefix ("Car", "Car/WheelLeft", "Car/WheelRight", "Car/Hood").
        let chain_types = chain_names(&bytes);
        // 4 group node chains + 3 part node chains + 3 resource chains.
        assert_eq!(chain_types.len(), 10, "4 groups + 3 part + 3 resource chains");
        assert_eq!(&chain_types[0..7], &["Car", "WheelLeft", "WheelRight", "Hood", "PartA", "PartB", "PartC"]);

        // Every part's model node also references its group by name, so the
        // group declaration bytes appear once per distinct prefix.
        let group_le = GROUP_NODE_TYPE.to_le_bytes();
        let group_blocks = bytes.windows(4).filter(|w| *w == group_le).count();
        assert_eq!(group_blocks, 4, "four group node declarations");
    }

    /// The same label under different parents produces unique, resolvable
    /// U3D nodes ("Wheel" under Left, "Wheel_2" under Right), while the same
    /// path shares one group.
    #[test]
    fn duplicate_group_labels_are_disambiguated() {
        let mut a = colored_mesh("PartA", [1.0, 0.0, 0.0, 1.0]);
        a.assembly = vec!["Left".into(), "Wheel".into()];
        let mut b = colored_mesh("PartB", [0.0, 0.0, 1.0, 1.0]);
        b.assembly = vec!["Right".into(), "Wheel".into()];
        let bytes = encode_u3d_many(&[a, b]);
        let names = chain_names(&bytes);
        // Left group, Left/Wheel group, Right group, Right/Wheel (renamed).
        assert_eq!(
            &names[0..6],
            &["Left", "Wheel", "Right", "Wheel_2", "PartA", "PartB"]
        );
        assert!(String::from_utf8_lossy(&bytes).contains("Wheel_2"), "second group got a unique node name");
    }

    /// Read the chain name of every top-level CHAIN_TYPE block (chain data
    /// begins with the chain's U16 length-prefixed name right after the
    /// 12-byte block header).
    fn chain_names(bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut pos = 44;
        while pos + 12 <= bytes.len() {
            let ty = u32_at(bytes, pos);
            let data = u32_at(bytes, pos + 4) as usize;
            if ty == CHAIN_TYPE {
                let len = u16::from_le_bytes(bytes[pos + 12..pos + 14].try_into().unwrap()) as usize;
                out.push(String::from_utf8_lossy(&bytes[pos + 14..pos + 14 + len]).into_owned());
            }
            pos += 12 + data;
            while pos % 4 != 0 {
                pos += 1;
            }
        }
        out
    }
