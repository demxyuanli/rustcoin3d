# Engine System

## 1. Overview

The engine system provides per-frame simulation capabilities analogous to Coin3D's `SoEngine` family. Engines evaluate on each frame, reading from input fields and writing to output fields, which can be connected to scene graph node fields for animation and simulation.

## 2. Architecture

```
EngineRegistry ─── Vec<Box<dyn Engine>>
    │              Vec<EngineConnection>   (engine ports + node fields)
    │
    ├── ElapsedTimeEngine   (time-driven rotation; timeOut)
    ├── SineOscillatorEngine (waveform animation; value)
    ├── CalculatorEngine    (expression evaluation; iA..iH / oA..oH)
    ├── ComposeMatrixEngine (TRS matrix composition)
    ├── InterpolateVec3Engine
    ├── InterpolateFloatEngine
    ├── InterpolateRotationEngine
    ├── ComposeVec3fEngine  (x/y/z -> vector)
    ├── DecomposeVec3fEngine
    ├── GateEngine          (enable / trigger pass-through)
    ├── ConcatenateEngine
    ├── SelectOneEngine
    ├── BoolOperationEngine
    ├── OnOffEngine         (toggle state machine)
    ├── OneShotEngine       (single-trigger timer)
    ├── CounterEngine       (integer counter)
    ├── TimeCounterEngine
    └── TriggerAnyEngine    (fire-once trigger)
```

Empty `connections` keeps insertion-order `evaluate`. Non-empty graphs pull `set_input`, evaluate in Kahn topological order (`toposort_linear`; cycle → insertion order), then write engine outputs onto node fields (`Transform.translation` = field 0, etc.).

`SceneGraph` also holds a **cross-node field graph** (`FieldRef` edges, `field_sources` / `field_targets`). `World::evaluate_engines` runs engines then `propagate_fields`.

## 3. Core Trait

```rust
pub trait Engine: Any + Debug {
    /// Called once per frame with current graph and simulation time.
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64);

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn set_input(&mut self, port: &str, value: FieldValue) {}
    fn output(&self, port: &str) -> Option<FieldValue> { None }
}
```

### 3.1 Connection graph

```rust
let sine = registry.add(SineOscillatorEngine::unbound(1.2, 1.0));
let calc = registry.add(CalculatorEngine::from_expr("oA = iA * 0.5"));
let compose = registry.add(ComposeVec3fEngine::unbound_xyz(2.0, 0.0, 0.0));
registry.connect_engines(sine, "value", calc, "iA");
registry.connect_engines(calc, "oA", compose, "y");
registry.connect_to_node(compose, "vector", bounce_tf, 0); // Transform.translation
```

Demo: `rotating_cube` (ElapsedTime + Gate/Decompose/Concatenate chain + `connect_fields` rotation copy).

## 4. Engine Reference

### 4.1 ElapsedTimeEngine

Drives rotation based on elapsed time.

```
Input:  speed (float)   — rotation speed in radians/second
Output: rotation (float) — accumulated rotation angle [0, 2π)
```

```rust
ElapsedTimeEngine {
    speed: 1.0,           // rad/s
    target: NodeId,       // Transform node to rotate
    field: SineField::Rotation,  // Which transform field to modify
}
```

Algorithm: `angle = (time * speed) % (2π)`, applied to target node's transform field.

### 4.2 SineOscillatorEngine

Generates sine wave outputs for animation.

```
Input:  frequency (float) — Hz
        amplitude (float) — peak value
        offset (float)    — DC offset
Output: value (float)     — amplitude * sin(2π * frequency * time) + offset
```

```rust
SineOscillatorEngine {
    frequency: 1.0,
    amplitude: 1.0,
    offset: 0.0,
    target: NodeId,
    field: SineField::ScaleX,  // ScaleX, ScaleY, ScaleZ, TranslationX, etc.
}
```

### 4.3 CalculatorEngine

Expression-based field calculator (Coin3D SoCalculator pattern).

```
Input:  iA, iB, iC, iD ... (float inputs, named by letter)
Output: oA, oB, oC, oD ... (float outputs)
```

```rust
CalculatorEngine {
    expressions: vec![
        "oA = sin(iA) * 3.0",
        "oB = iA + iB",
    ],
    inputs: HashMap<char, Vec<FieldRef>>,
    outputs: HashMap<char, Vec<FieldRef>>,
}
```

Supported operations: `+`, `-`, `*`, `/`, `sin`, `cos`, `tan`, `sqrt`, `abs`, `pow`, `min`, `max`, `clamp`, `lerp`.

### 4.4 ComposeMatrixEngine

Composes a 4×4 transformation matrix from separate TRS components (SoComposeMatrix).

```
Input:  translation (Vec3)
        rotation (Vec4 quaternion)
        scaleFactor (Vec3)
        center (Vec3)
Output: matrix (Mat4)
```

```rust
ComposeMatrixEngine {
    translation: Option<FieldRef>,
    rotation: Option<FieldRef>,
    scale_factor: Option<FieldRef>,
    center: Option<FieldRef>,
    output: FieldRef,
}
```

### 4.5 Interpolation Engines

#### InterpolateVec3Engine

```
alpha: f32 (interpolation factor, typically 0.0-1.0)
keyValue: Vec<Vec3> (keyframe values)
key: Vec<f32> (keyframe positions, 0.0-1.0)
output: Vec3
```

#### InterpolateFloatEngine

```
alpha: f32
keyValue: Vec<f32>
key: Vec<f32>
output: f32
```

#### InterpolateRotationEngine

Uses spherical linear interpolation (slerp) for rotation quaternions:

```
alpha: f32
keyValue: Vec<Quat>
key: Vec<f32>
output: Quat
```

### 4.6 OnOffEngine

State machine toggling (Coin3D SoOnOff).

```
Input:  trigger (bool) — rising edge triggers toggle
Output: whichChild (i32) — -1 = all children visible, -2 = none visible
```

```rust
OnOffEngine {
    state: bool,           // Current on/off state
    last_trigger: bool,    // For edge detection
    target: NodeId,        // Switch node to control
}
```

### 4.7 OneShotEngine

Single-trigger timer (Coin3D SoOneShot).

```
Input:  trigger (bool) — rising edge starts timer
        duration (float) — timer duration in seconds
Output: ramp (float) — linear ramp 0→1 over duration
        isActive (bool) — true while timer running
```

```rust
OneShotEngine {
    duration: f32,
    elapsed: f64,
    active: bool,
    last_trigger: bool,
}
```

### 4.8 CounterEngine

Integer counter with increment/decrement and wrap-around.

```
Input:  trigger (bool)
        min (i32)
        max (i32)
        step (i32)
Output: output (i32)
```

```rust
CounterEngine {
    value: i32,
    min: i32,
    max: i32,
    step: i32,
    last_trigger: bool,
}
```

### 4.9 ComposeVec3fEngine

Composes a Vec3 from three separate float inputs.

```
Input:  x (float), y (float), z (float)
Output: vector (Vec3)
```

### 4.10 TriggerAnyEngine

Fires once on any trigger input change.

```
Input:  trigger (bool)
Output: output (bool) — true for one frame when triggered
```

## 5. Simulation Scheduler

The `SimulationScheduler` provides ordered per-frame callbacks:

```rust
pub struct SimulationScheduler {
    pre_engines: Vec<FrametimeCallback>,
    post_engines: Vec<FrametimeCallback>,
}

type FrametimeCallback = Box<dyn FnMut(&mut SceneGraph, f64) + Send>;
```

Execution order:
1. Pre-engine callbacks (before EngineRegistry::evaluate_all)
2. Engine evaluation (each engine's `evaluate()`)
3. Post-engine callbacks (after engines)
4. Sensor queue dispatch

## 6. Time Management

```rust
pub struct TimeManager {
    pub elapsed: f64,        // Total elapsed time (seconds)
    pub delta: f32,          // Frame delta (seconds)
    pub time_scale: f32,     // Speed multiplier (1.0 = realtime)
    start: Instant,
}
```

- Monotonic clock via `std::time::Instant`
- Adjustable time scale for slow-motion or fast-forward
- `delta` is clamped to prevent spiral of death on long frames

## 7. Physics Integration

```rust
pub struct PhysicsBody {
    pub mass: f32,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub forces: Vec3,
    pub torques: Vec3,
    pub damping: f32,
}

pub struct PhysicsWorld {
    pub gravity: Vec3,
    pub time_step: f32,
    pub bodies: HashMap<NodeId, PhysicsBody>,
    pub constraints: Vec<Constraint>,
}
```

- Semi-implicit Euler integration
- Simple collision detection (AABB overlap)
- Constraint solver for joints

## 8. Sensors

### 8.1 Sensor Types

| Sensor | Purpose |
|--------|---------|
| `AlarmSensor` | Delayed callback after N seconds |
| `TimerSensor` | Periodic callback every N seconds |
| `SensorQueue` | Queued sensor events for deferred dispatch |
| `FieldChangeCallback` | React to field value changes |
| `NodeDeleteCallback` | React to node removal |

### 8.2 Sensor Registry

```rust
pub struct SensorRegistry {
    field_listeners: HashMap<(NodeId, FieldIndex), Vec<Box<dyn FieldChangeCallback>>>,
    node_delete_listeners: HashMap<NodeId, Vec<Box<dyn NodeDeleteCallback>>>,
}
```

## 9. Engine Integration Example

```rust
// Set up a sine-driven rotation
let mut graph = SceneGraph::new();

// Create a transform to animate
let transform = graph.add_root(NodeData::Transform(TransformNode::default()));

// Create engine that drives the transform's rotation
let engine = SineOscillatorEngine {
    frequency: 2.0,      // 2 Hz
    amplitude: 3.14,      // ±π radians
    offset: 0.0,
    target: transform,
    field: SineField::RotationZ,
};

// Register and run
let mut registry = EngineRegistry::new();
registry.register(Box::new(engine));

// Per frame:
let time = time_manager.elapsed;
registry.evaluate_all(&mut graph, time);
```
