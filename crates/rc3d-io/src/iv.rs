use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IvError {
    #[error("parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unknown node type: {0}")]
    UnknownNode(String),
}

/// Parse an Inventor .iv file into a SceneGraph.
pub fn parse_iv(input: &str) -> Result<SceneGraph, IvError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(&tokens);
    parser.parse()
}

/// Write a SceneGraph to .iv format.
pub fn write_iv(graph: &SceneGraph) -> String {
    let mut out = String::from("#Inventor V2.1 ascii\n\n");
    for &root in graph.roots() {
        write_node(graph, root, &mut out, 0);
    }
    out
}

// --- Tokenizer ---

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String),
    Number(f32),
    Int(i32),
    LBrace,
    RBrace,
    LBracket,
    RBracket,
}

fn tokenize(input: &str) -> Result<Vec<(Token, usize)>, IvError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();
    loop {
        let (i, ch) = match chars.peek() {
            Some(&x) => x,
            None => break,
        };
        match ch {
            '{' => { tokens.push((Token::LBrace, i)); chars.next(); }
            '}' => { tokens.push((Token::RBrace, i)); chars.next(); }
            '[' => { tokens.push((Token::LBracket, i)); chars.next(); }
            ']' => { tokens.push((Token::RBracket, i)); chars.next(); }
            '#' => { while chars.next_if(|&(_, c)| c != '\n').is_some() {} }
            c if c.is_whitespace() => { chars.next(); }
            c if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() => {
                let mut num_str = String::new();
                if c == '-' || c == '+' { num_str.push(c); chars.next(); }
                while let Some(&(_, c)) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '-' {
                        num_str.push(c); chars.next();
                    } else { break; }
                }
                let line = input[..i].lines().count();
                if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                    let v: f32 = num_str.parse().map_err(|_| IvError::Parse {
                        line, message: format!("invalid float: {num_str}"),
                    })?;
                    tokens.push((Token::Number(v), i));
                } else if num_str == "-" || num_str == "+" {
                    tokens.push((Token::Ident(num_str), i));
                } else {
                    let v: i32 = num_str.parse().map_err(|_| IvError::Parse {
                        line, message: format!("invalid int: {num_str}"),
                    })?;
                    tokens.push((Token::Int(v), i));
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut ident = String::new();
                while let Some(&(_, c)) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' { ident.push(c); chars.next(); }
                    else { break; }
                }
                tokens.push((Token::Ident(ident), i));
            }
            _ => { chars.next(); }
        }
    }
    Ok(tokens)
}

// --- Parser ---

struct Parser<'a> {
    tokens: &'a [(Token, usize)],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [(Token, usize)]) -> Self { Self { tokens, pos: 0 } }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn next_tok(&mut self) -> Result<&Token, IvError> {
        if self.pos >= self.tokens.len() { return Err(IvError::UnexpectedEof); }
        let tok = &self.tokens[self.pos].0;
        self.pos += 1;
        Ok(tok)
    }

    fn expect(&mut self, expected: &Token) -> Result<(), IvError> {
        let tok = self.next_tok()?;
        if tok == expected { Ok(()) }
        else { Err(IvError::Parse { line: 0, message: format!("expected {expected:?}, got {tok:?}") }) }
    }

    fn read_float(&mut self) -> Result<f32, IvError> {
        match self.next_tok()? {
            Token::Number(f) => Ok(*f),
            Token::Int(i) => Ok(*i as f32),
            other => Err(IvError::Parse { line: 0, message: format!("expected number, got {other:?}") }),
        }
    }

    fn read_vec3(&mut self) -> Result<Vec3, IvError> {
        Ok(Vec3::new(self.read_float()?, self.read_float()?, self.read_float()?))
    }

    fn parse(&mut self) -> Result<SceneGraph, IvError> {
        let mut graph = SceneGraph::new();
        while self.pos < self.tokens.len() {
            if let Token::Ident(name) = self.peek().cloned().unwrap_or(Token::Ident(String::new())) {
                self.parse_top_level_node(&mut graph, None)?;
            } else {
                self.pos += 1;
            }
        }
        Ok(graph)
    }

    /// Parse a node and add it to the graph as child of `parent` (or as root if None).
    fn parse_top_level_node(&mut self, graph: &mut SceneGraph, parent: Option<rc3d_core::NodeId>) -> Result<rc3d_core::NodeId, IvError> {
        let name = match self.peek() {
            Some(Token::Ident(s)) => s.clone(),
            _ => return Err(IvError::Parse { line: 0, message: "expected node name".into() }),
        };
        self.pos += 1; // skip ident
        self.expect(&Token::LBrace)?;

        let node_id = match name.as_str() {
            "Separator" | "Group" => {
                let data = if name == "Separator" { NodeData::Separator(SeparatorNode) }
                           else { NodeData::Group(GroupNode) };
                let id = match parent {
                    Some(p) => graph.add_child(p, data),
                    None => graph.add_root(data),
                };
                // Parse children until }
                while let Some(Token::Ident(child_name)) = self.peek().cloned() {
                    // Check if this ident is a field name or a child node type
                    if is_field_name(&child_name) {
                        break;
                    }
                    self.parse_top_level_node(graph, Some(id))?;
                }
                id
            }
            _ => {
                let data = self.parse_node_fields(&name)?;
                match parent {
                    Some(p) => graph.add_child(p, data),
                    None => graph.add_root(data),
                }
            }
        };

        self.expect(&Token::RBrace)?;
        Ok(node_id)
    }

    fn parse_node_fields(&mut self, type_name: &str) -> Result<NodeData, IvError> {
        match type_name {
            "Transform" => self.parse_transform(),
            "Material" => self.parse_material(),
            "Cube" => self.parse_cube(),
            "Sphere" => self.parse_sphere(),
            "Cone" => self.parse_cone(),
            "Cylinder" => self.parse_cylinder(),
            "PerspectiveCamera" => self.parse_perspective_camera(),
            "DirectionalLight" => self.parse_directional_light(),
            "Coordinate3" => self.parse_coordinate3(),
            _ => {
                // Skip unknown fields until }
                while let Some(tok) = self.peek() {
                    if tok == &Token::RBrace { break; }
                    self.pos += 1;
                }
                Ok(NodeData::Separator(SeparatorNode))
            }
        }
    }

    fn parse_transform(&mut self) -> Result<NodeData, IvError> {
        let mut t = TransformNode::default();
        while let Some(Token::Ident(field)) = self.peek().cloned() {
            match field.as_str() {
                "translation" => { self.pos += 1; t.translation = self.read_vec3()?; }
                "scaleFactor" => { self.pos += 1; t.scale = self.read_vec3()?; }
                "rotation" => {
                    self.pos += 1;
                    let axis = self.read_vec3()?;
                    let angle = self.read_float()?;
                    t.rotation = rc3d_core::math::Mat4::from_axis_angle(axis.normalize(), angle);
                }
                "center" => { self.pos += 1; t.center = self.read_vec3()?; }
                _ => break,
            }
        }
        Ok(NodeData::Transform(t))
    }

    fn parse_material(&mut self) -> Result<NodeData, IvError> {
        let mut mat = MaterialNode::default();
        while let Some(Token::Ident(field)) = self.peek().cloned() {
            match field.as_str() {
                "diffuseColor" => { self.pos += 1; mat.diffuse_color = self.read_vec3()?; }
                "ambientColor" => { self.pos += 1; mat.ambient_color = self.read_vec3()?; }
                "specularColor" => { self.pos += 1; mat.specular_color = self.read_vec3()?; }
                "shininess" => { self.pos += 1; mat.shininess = self.read_float()?; }
                _ => break,
            }
        }
        Ok(NodeData::Material(mat))
    }

    fn parse_cube(&mut self) -> Result<NodeData, IvError> {
        let mut cube = CubeNode::default();
        while let Some(Token::Ident(field)) = self.peek().cloned() {
            match field.as_str() {
                "width" => { self.pos += 1; cube.width = self.read_float()?; }
                "height" => { self.pos += 1; cube.height = self.read_float()?; }
                "depth" => { self.pos += 1; cube.depth = self.read_float()?; }
                _ => break,
            }
        }
        Ok(NodeData::Cube(cube))
    }

    fn parse_sphere(&mut self) -> Result<NodeData, IvError> {
        let mut s = SphereNode::default();
        while let Some(Token::Ident(f)) = self.peek().cloned() {
            if f == "radius" { self.pos += 1; s.radius = self.read_float()?; } else { break; }
        }
        Ok(NodeData::Sphere(s))
    }

    fn parse_cone(&mut self) -> Result<NodeData, IvError> {
        let mut c = ConeNode::default();
        while let Some(Token::Ident(f)) = self.peek().cloned() {
            match f.as_str() {
                "bottomRadius" => { self.pos += 1; c.bottom_radius = self.read_float()?; }
                "height" => { self.pos += 1; c.height = self.read_float()?; }
                _ => break,
            }
        }
        Ok(NodeData::Cone(c))
    }

    fn parse_cylinder(&mut self) -> Result<NodeData, IvError> {
        let mut c = CylinderNode::default();
        while let Some(Token::Ident(f)) = self.peek().cloned() {
            match f.as_str() {
                "radius" => { self.pos += 1; c.radius = self.read_float()?; }
                "height" => { self.pos += 1; c.height = self.read_float()?; }
                _ => break,
            }
        }
        Ok(NodeData::Cylinder(c))
    }

    fn parse_perspective_camera(&mut self) -> Result<NodeData, IvError> {
        let mut cam = PerspectiveCameraNode::default();
        while let Some(Token::Ident(f)) = self.peek().cloned() {
            match f.as_str() {
                "position" => { self.pos += 1; cam.position = self.read_vec3()?; }
                "nearDistance" => { self.pos += 1; cam.near = self.read_float()?; }
                "farDistance" => { self.pos += 1; cam.far = self.read_float()?; }
                "heightAngle" => { self.pos += 1; cam.fov = self.read_float()?; }
                _ => break,
            }
        }
        cam.aspect = 800.0 / 600.0;
        cam.orientation = rc3d_core::math::Mat4::look_at_rh(cam.position, Vec3::ZERO, Vec3::Y);
        Ok(NodeData::PerspectiveCamera(cam))
    }

    fn parse_directional_light(&mut self) -> Result<NodeData, IvError> {
        let mut l = DirectionalLightNode::default();
        while let Some(Token::Ident(f)) = self.peek().cloned() {
            match f.as_str() {
                "direction" => { self.pos += 1; l.direction = self.read_vec3()?; }
                "color" => { self.pos += 1; l.color = self.read_vec3()?; }
                "intensity" => { self.pos += 1; l.intensity = self.read_float()?; }
                _ => break,
            }
        }
        Ok(NodeData::DirectionalLight(l))
    }

    fn parse_coordinate3(&mut self) -> Result<NodeData, IvError> {
        let mut points = Vec::new();
        while let Some(Token::Ident(f)) = self.peek().cloned() {
            if f == "point" {
                self.pos += 1;
                if let Some(Token::LBracket) = self.peek() { self.pos += 1; }
                while matches!(self.peek(), Some(Token::Number(_)) | Some(Token::Int(_))) {
                    points.push(self.read_vec3()?);
                }
                if let Some(Token::RBracket) = self.peek() { self.pos += 1; }
            } else { break; }
        }
        Ok(NodeData::Coordinate3(Coordinate3Node::from_points(points)))
    }
}

/// Known field names — anything else is treated as a child node type.
fn is_field_name(name: &str) -> bool {
    matches!(name,
        "translation" | "rotation" | "scaleFactor" | "center" |
        "diffuseColor" | "ambientColor" | "specularColor" | "shininess" | "emissiveColor" |
        "width" | "height" | "depth" | "radius" | "bottomRadius" |
        "position" | "nearDistance" | "farDistance" | "heightAngle" | "focalDistance" |
        "direction" | "color" | "intensity" | "location" | "cutOffAngle" | "dropOffRate" |
        "point" | "vector" | "coordIndex" | "normalIndex" |
        "renderCaching" | "boundingBoxCaching" | "renderCulling" | "pickCulling"
    )
}

// --- Writer ---

fn write_node(graph: &SceneGraph, node: rc3d_core::NodeId, out: &mut String, indent: usize) {
    let Some(entry) = graph.get(node) else { return };
    let pad = "  ".repeat(indent);
    match &entry.data {
        NodeData::Separator(_) => {
            out.push_str(&format!("{pad}Separator {{\n"));
            for &child in &entry.children { write_node(graph, child, out, indent + 1); }
            out.push_str(&format!("{pad}}}\n"));
        }
        NodeData::Group(_) => {
            out.push_str(&format!("{pad}Group {{\n"));
            for &child in &entry.children { write_node(graph, child, out, indent + 1); }
            out.push_str(&format!("{pad}}}\n"));
        }
        NodeData::Transform(t) => {
            out.push_str(&format!("{pad}Transform {{\n"));
            out.push_str(&format!("{pad}  translation {} {} {}\n", t.translation.x, t.translation.y, t.translation.z));
            out.push_str(&format!("{pad}  scaleFactor {} {} {}\n", t.scale.x, t.scale.y, t.scale.z));
            out.push_str(&format!("{pad}}}\n"));
        }
        NodeData::Material(m) => {
            out.push_str(&format!("{pad}Material {{ diffuseColor {} {} {} }}\n",
                m.diffuse_color.x, m.diffuse_color.y, m.diffuse_color.z));
        }
        NodeData::Cube(c) => {
            out.push_str(&format!("{pad}Cube {{ width {} height {} depth {} }}\n", c.width, c.height, c.depth));
        }
        NodeData::Sphere(s) => {
            out.push_str(&format!("{pad}Sphere {{ radius {} }}\n", s.radius));
        }
        NodeData::Cone(c) => {
            out.push_str(&format!("{pad}Cone {{ bottomRadius {} height {} }}\n", c.bottom_radius, c.height));
        }
        NodeData::Cylinder(c) => {
            out.push_str(&format!("{pad}Cylinder {{ radius {} height {} }}\n", c.radius, c.height));
        }
        NodeData::PerspectiveCamera(c) => {
            out.push_str(&format!("{pad}PerspectiveCamera {{ position {} {} {} heightAngle {} }}\n",
                c.position.x, c.position.y, c.position.z, c.fov));
        }
        NodeData::DirectionalLight(l) => {
            out.push_str(&format!("{pad}DirectionalLight {{ direction {} {} {} }}\n",
                l.direction.x, l.direction.y, l.direction.z));
        }
        _ => { out.push_str(&format!("{pad}# unhandled node\n")); }
    }
}
