//! High-level animation builders on top of the `Engine` trait.
//!
//! These provide a composable, declarative API for common animation patterns
//! without requiring direct `Engine` implementation.

use rc3d_core::math::Vec3;
use rc3d_engine::{Engine, EngineRegistry, ElapsedTimeEngine, SineField, SineOscillatorEngine};

use crate::scene::NodeHandle;

/// Rotation speed + axis for `Rotator`.
#[derive(Clone, Debug)]
pub struct RotationConfig {
    pub speed: f32,       // radians per second
    pub axis: Vec3,       // rotation axis (normalized)
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self { speed: 1.0, axis: Vec3::Y }
    }
}

/// Oscillation config for `Oscillator`.
#[derive(Clone, Debug)]
pub enum OscillationMode {
    ScaleX,
    ScaleY,
    ScaleZ,
    TranslationY,
    TranslationX,
    TranslationZ,
}

impl From<OscillationMode> for SineField {
    fn from(mode: OscillationMode) -> Self {
        match mode {
            OscillationMode::ScaleX => SineField::ScaleX,
            OscillationMode::ScaleY => SineField::ScaleY,
            OscillationMode::ScaleZ => SineField::ScaleZ,
            OscillationMode::TranslationX => SineField::TranslationX,
            OscillationMode::TranslationY => SineField::TranslationY,
            OscillationMode::TranslationZ => SineField::TranslationZ,
        }
    }
}

/// Continuous rotation animation.
pub struct Rotator;

impl Rotator {
    /// Create an `ElapsedTimeEngine` that rotates the given node.
    pub fn over(handle: NodeHandle, config: RotationConfig) -> ElapsedTimeEngine {
        ElapsedTimeEngine::new(handle.id(), config.speed, config.axis)
    }
}

/// Sine-wave oscillation animation.
pub struct Oscillator;

impl Oscillator {
    /// Create a `SineOscillatorEngine` for the given node.
    pub fn over(
        handle: NodeHandle,
        frequency: f32,
        amplitude: f32,
        mode: OscillationMode,
    ) -> SineOscillatorEngine {
        SineOscillatorEngine::new(handle.id(), frequency, amplitude, mode.into())
    }
}

/// Builder for creating and registering animations.
///
/// ```ignore
/// let mut anim = Animator::new();
/// anim.rotate(cube_handle, RotationConfig { speed: 0.5, axis: Vec3::Y });
/// anim.oscillate(sphere_handle, 2.0, 0.3, OscillationMode::ScaleX);
/// let registry = anim.build();
/// ```
pub struct Animator {
    engines: Vec<Box<dyn Engine>>,
}

impl Default for Animator {
    fn default() -> Self {
        Self::new()
    }
}

impl Animator {
    pub fn new() -> Self {
        Self { engines: Vec::new() }
    }

    /// Add a rotation animation.
    pub fn rotate(&mut self, handle: NodeHandle, config: RotationConfig) -> &mut Self {
        self.engines
            .push(Box::new(Rotator::over(handle, config)));
        self
    }

    /// Add an oscillation animation.
    pub fn oscillate(
        &mut self,
        handle: NodeHandle,
        frequency: f32,
        amplitude: f32,
        mode: OscillationMode,
    ) -> &mut Self {
        self.engines
            .push(Box::new(Oscillator::over(handle, frequency, amplitude, mode)));
        self
    }

    /// Add any custom Engine.
    pub fn add_engine(&mut self, engine: Box<dyn Engine>) -> &mut Self {
        self.engines.push(engine);
        self
    }

    /// Number of registered engines.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Build into an `EngineRegistry` for use with `App::with_engines()`.
    pub fn build(self) -> EngineRegistry {
        EngineRegistry { engines: self.engines }
    }
}
