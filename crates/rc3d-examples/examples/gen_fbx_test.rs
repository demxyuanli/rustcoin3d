//! Generates a minimal FBX test file (a colored cube) for verifying the FBX importer.
//!
//! Usage: cargo run -p rc3d-examples --example gen_fbx_test [output_path]

use std::io::Cursor;

use fbxcel::low::FbxVersion;
use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cache/generated_models/test_cube.fbx".to_string());

    let data = generate_cube_fbx();
    std::fs::write(&path, &data).expect("Failed to write FBX file");
    println!("Generated FBX test file: {path}");
}

fn generate_cube_fbx() -> Vec<u8> {
    let sink = Cursor::new(Vec::new());
    let mut w = Writer::new(sink, FbxVersion::V7_4).expect("Failed to create writer");

    // FBX header extension
    {
        let mut aw = w.new_node("FBXHeaderExtension").expect("node");
        aw.append_i64(1).expect("attr");
        drop(aw);

        {
            let mut aw = w.new_node("FBXVersion").expect("node");
            aw.append_i32(7400).expect("attr");
            drop(aw);
            w.close_node().expect("close");
        }

        w.close_node().expect("close");
    }

    // File ID
    {
        let mut aw = w.new_node("FileId").expect("node");
        aw.append_i64(0).expect("attr");
        drop(aw);
        w.close_node().expect("close");
    }

    // CreationTime
    {
        let mut aw = w.new_node("CreationTime").expect("node");
        aw.append_string_direct("2026-01-01 00:00:00:000")
            .expect("attr");
        drop(aw);
        w.close_node().expect("close");
    }

    // Creator
    {
        let mut aw = w.new_node("Creator").expect("node");
        aw.append_string_direct("rc3d FBX test generator")
            .expect("attr");
        drop(aw);
        w.close_node().expect("close");
    }

    // GlobalSettings
    {
        let mut aw = w.new_node("GlobalSettings").expect("node");
        aw.append_i64(1).expect("attr");
        drop(aw);

        {
            let mut aw = w.new_node("Version").expect("node");
            aw.append_i32(1000).expect("attr");
            drop(aw);
            w.close_node().expect("close");
        }

        w.close_node().expect("close");
    }

    // Objects
    let geometry_id: i64 = 1;
    let material_id: i64 = 2;
    let model_id: i64 = 3;
    {
        let aw = w.new_node("Objects").expect("node");
        drop(aw);

        // Geometry
        {
            let mut aw = w.new_node("Geometry").expect("node");
            aw.append_i64(geometry_id).expect("id");
            aw.append_string_direct("Geometry::Cube").expect("name");
            aw.append_string_direct("Mesh").expect("type");
            drop(aw);

            // Cube vertices (8 vertices = 24 floats)
            {
                let verts: &[f32] = &[
                    -0.5, -0.5, 0.5, // 0: front bottom left
                    0.5, -0.5, 0.5, // 1: front bottom right
                    0.5, 0.5, 0.5, // 2: front top right
                    -0.5, 0.5, 0.5, // 3: front top left
                    -0.5, -0.5, -0.5, // 4: back bottom left
                    0.5, -0.5, -0.5, // 5: back bottom right
                    0.5, 0.5, -0.5, // 6: back top right
                    -0.5, 0.5, -0.5, // 7: back top left
                ];
                let mut aw2 = w.new_node("Vertices").expect("node");
                aw2.append_i32(verts.len() as i32).expect("len");
                aw2.append_arr_f32_from_iter(None, verts.iter().cloned())
                    .expect("data");
                drop(aw2);
                w.close_node().expect("close");
            }

            // PolygonVertexIndex
            {
                let indices: &[i32] = &[
                    0, 1, 2, -4, // Front face
                    5, 4, 7, -7, // Back face
                    4, 0, 3, -8, // Left face
                    1, 5, 6, -3, // Right face
                    3, 2, 6, -8, // Top face
                    4, 5, 1, -1, // Bottom face
                ];
                let mut aw2 = w.new_node("PolygonVertexIndex").expect("node");
                aw2.append_i32(indices.len() as i32).expect("len");
                aw2.append_arr_i32_from_iter(None, indices.iter().cloned())
                    .expect("data");
                drop(aw2);
                w.close_node().expect("close");
            }

            // Normals
            {
                let normals: &[f32] = &[
                    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
                    -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
                ];
                let mut aw2 = w.new_node("Normals").expect("node");
                aw2.append_i32(normals.len() as i32).expect("len");
                aw2.append_arr_f32_from_iter(None, normals.iter().cloned())
                    .expect("data");
                drop(aw2);
                w.close_node().expect("close");
            }

            w.close_node().expect("close Geometry");
        }

        // Material
        {
            let mut aw = w.new_node("Material").expect("node");
            aw.append_i64(material_id).expect("id");
            aw.append_string_direct("Material::CubeMat").expect("name");
            aw.append_string_direct("").expect("type");
            drop(aw);

            {
                let mut aw2 = w.new_node("DiffuseColor").expect("node");
                aw2.append_i32(3).expect("len");
                aw2.append_f32(0.2).expect("r");
                aw2.append_f32(0.6).expect("g");
                aw2.append_f32(0.9).expect("b");
                drop(aw2);
                w.close_node().expect("close");
            }

            {
                let mut aw2 = w.new_node("SpecularColor").expect("node");
                aw2.append_i32(3).expect("len");
                aw2.append_f32(1.0).expect("r");
                aw2.append_f32(1.0).expect("g");
                aw2.append_f32(1.0).expect("b");
                drop(aw2);
                w.close_node().expect("close");
            }

            {
                let mut aw2 = w.new_node("Shininess").expect("node");
                aw2.append_i32(1).expect("len");
                aw2.append_f32(50.0).expect("value");
                drop(aw2);
                w.close_node().expect("close");
            }

            w.close_node().expect("close Material");
        }

        // Model (container for geometry)
        {
            let mut aw = w.new_node("Model").expect("node");
            aw.append_i64(model_id).expect("id");
            aw.append_string_direct("Model::Cube").expect("name");
            aw.append_string_direct("Mesh").expect("type");
            drop(aw);
            w.close_node().expect("close Model");
        }

        w.close_node().expect("close Objects");
    }

    // Connections
    {
        let aw = w.new_node("Connections").expect("node");
        drop(aw);

        {
            let mut aw2 = w.new_node("C").expect("node");
            aw2.append_string_direct("OO").expect("type");
            aw2.append_i64(geometry_id).expect("child");
            aw2.append_i64(model_id).expect("parent");
            drop(aw2);
            w.close_node().expect("close");
        }

        {
            let mut aw2 = w.new_node("C").expect("node");
            aw2.append_string_direct("OO").expect("type");
            aw2.append_i64(material_id).expect("child");
            aw2.append_i64(model_id).expect("parent");
            drop(aw2);
            w.close_node().expect("close");
        }

        w.close_node().expect("close Connections");
    }

    let sink = w.finalize(&FbxFooter::default()).expect("finalize");
    sink.into_inner()
}
