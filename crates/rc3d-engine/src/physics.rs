//! Physics integration via Rapier3D.
//!
//! Wraps Rapier's rigid body + collider pipeline and syncs transforms
//! back to the scene graph each frame.

use rapier3d::prelude::*;
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;

/// Represents a physics body attached to a scene graph node.
/// Static bodies (no scene attachment) have `scene_node == None`.
#[derive(Clone, Copy, Debug)]
pub struct PhysicsBody {
    pub rigid_body_handle: RigidBodyHandle,
    pub collider_handle: Option<ColliderHandle>,
    pub scene_node: Option<NodeId>,
}

/// Physics world wrapping Rapier's simulation.
pub struct PhysicsWorld {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    island_manager: IslandManager,
    broad_phase: BroadPhaseMultiSap,
    narrow_phase: NarrowPhase,
    ccd_solver: CCDSolver,
    query_pipeline: Option<QueryPipeline>,
    gravity: Vec3,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    /// Tracked physics bodies linked to scene nodes.
    bodies: Vec<PhysicsBody>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhaseMultiSap::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: None,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            bodies: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, g: Vec3) {
        self.gravity = g;
    }

    /// Add a dynamic rigid body with a collider attached to a scene node.
    pub fn add_dynamic_body(
        &mut self,
        scene_node: NodeId,
        translation: Vec3,
        shape: SharedShape,
    ) -> PhysicsBody {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![translation.x, translation.y, translation.z])
            .build();
        let rb_handle = self.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::new(shape).build();
        let collider_handle = self.collider_set.insert_with_parent(
            collider,
            rb_handle,
            &mut self.rigid_body_set,
        );

        let body = PhysicsBody {
            rigid_body_handle: rb_handle,
            collider_handle: Some(collider_handle),
            scene_node: Some(scene_node),
        };
        self.bodies.push(body);
        body
    }

    /// Add a static rigid body (floor, wall, etc.) with a collider.
    pub fn add_static_body(
        &mut self,
        translation: Vec3,
        shape: SharedShape,
    ) -> PhysicsBody {
        let rigid_body = RigidBodyBuilder::fixed()
            .translation(vector![translation.x, translation.y, translation.z])
            .build();
        let rb_handle = self.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::new(shape).build();
        let collider_handle = self.collider_set.insert_with_parent(
            collider,
            rb_handle,
            &mut self.rigid_body_set,
        );

        let body = PhysicsBody {
            rigid_body_handle: rb_handle,
            collider_handle: Some(collider_handle),
            scene_node: None,
        };
        self.bodies.push(body);
        body
    }

    /// Step the physics simulation by `delta_time` seconds.
    /// Uses fixed sub-stepping for stability.
    pub fn step(&mut self, delta_time: f32) {
        self.integration_parameters.dt = delta_time;

        let gravity = vector![self.gravity.x, self.gravity.y, self.gravity.z];
        self.physics_pipeline.step(
            &gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            self.query_pipeline.as_mut(),
            &(),
            &(),
        );
    }

    /// Sync physics transforms back to the scene graph.
    pub fn sync_to_scene(&self, graph: &mut SceneGraph) {
        for body in &self.bodies {
            let Some(scene_node) = body.scene_node else { continue };
            let rb = match self.rigid_body_set.get(body.rigid_body_handle) {
                Some(rb) => rb,
                None => continue,
            };
            let pos = rb.translation();
            let rot = rb.rotation();

            if let Some(entry) = graph.get_mut(scene_node) {
                if let NodeData::Transform(t) = &mut entry.data {
                    let rot4 = rc3d_core::math::Quat::from_xyzw(rot.i, rot.j, rot.k, rot.w);
                    t.translation = Vec3::new(pos.x, pos.y, pos.z);
                    t.rotation = rc3d_core::math::Mat4::from_quat(rot4);
                } else {
                    log::warn!("Physics sync: node is not Transform, skipped");
                }
            }
        }
    }

    /// Apply velocity or force to a rigid body.
    pub fn apply_impulse(&mut self, body: PhysicsBody, impulse: Vec3) {
        if let Some(rb) = self.rigid_body_set.get_mut(body.rigid_body_handle) {
            rb.apply_impulse(vector![impulse.x, impulse.y, impulse.z], true);
        }
    }

    /// Note: stale handles from removed rigid bodies may accumulate. Call cleanup periodically.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn rigid_body_position(&self, body: PhysicsBody) -> Option<Vec3> {
        self.rigid_body_set
            .get(body.rigid_body_handle)
            .map(|rb| {
                let p = rb.translation();
                Vec3::new(p.x, p.y, p.z)
            })
    }

    /// Walk the scene graph and create Rapier colliders for shape nodes (Cube, Sphere, Cylinder).
    /// Nodes already tracked are skipped. Call once after loading a scene, or after structural edits.
    pub fn sync_colliders_from_scene(&mut self, graph: &SceneGraph) {
        let roots = graph.roots().to_vec();
        for &root in &roots {
            self.sync_subtree(graph, root, &Mat4::IDENTITY);
        }
    }

    fn sync_subtree(&mut self, graph: &SceneGraph, node: NodeId, parent_model: &Mat4) {
        use rc3d_core::math::Mat4;
        let entry = match graph.get(node) {
            Some(e) => e,
            None => return,
        };
        let local_model = match &entry.data {
            NodeData::Transform(t) => {
                *parent_model
                    * (Mat4::from_translation(-t.center)
                        * Mat4::from_scale(t.scale)
                        * t.rotation
                        * Mat4::from_translation(t.center + t.translation))
            }
            NodeData::Rotation(r) => *parent_model * r.to_matrix(),
            NodeData::RotationXYZ(r) => *parent_model * r.to_matrix(),
            _ => *parent_model,
        };
        let shape: Option<SharedShape> = match &entry.data {
            NodeData::Cube(c) => Some(shapes::cube([c.width / 2.0, c.height / 2.0, c.depth / 2.0])),
            NodeData::Sphere(s) => Some(shapes::sphere(s.radius)),
            NodeData::Cylinder(c) => Some(shapes::cylinder(c.height / 2.0, c.radius)),
            _ => None,
        };
        if let Some(s) = shape {
            let (_, _, trans) = local_model.to_scale_rotation_translation();
            if !self.bodies.iter().any(|b| b.scene_node == Some(node)) {
                self.add_dynamic_body(node, trans, s);
            }
        }
        let children = entry.children.to_vec();
        for child in children {
            self.sync_subtree(graph, child, &local_model);
        }
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

/// Create common collider shapes.
pub mod shapes {
    use rapier3d::prelude::*;

    pub fn sphere(radius: f32) -> SharedShape {
        SharedShape::ball(radius)
    }
    pub fn cube(half_extents: [f32; 3]) -> SharedShape {
        SharedShape::cuboid(half_extents[0], half_extents[1], half_extents[2])
    }
    pub fn capsule(half_height: f32, radius: f32) -> SharedShape {
        SharedShape::capsule_y(half_height, radius)
    }
    pub fn cylinder(half_height: f32, radius: f32) -> SharedShape {
        SharedShape::cylinder(half_height, radius)
    }
}
