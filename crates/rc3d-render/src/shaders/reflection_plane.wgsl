// Planar reflection rendering (HOOPS ReflectionPlane equivalent).
// Renders geometry mirrored across a plane using stencil buffer.
//
// Implementation requires two passes:
// Pass 1 (Stencil): Render reflection geometry with stencil write to mark mirror region
// Pass 2 (Reflect): Render reflected scene with stencil test + inverted view matrix
//
// Integration path (post_processor.rs or dedicated pass):
// 1. Enable stencil in depth attachment descriptor
// 2. Pass 1: render a fullscreen quad at the reflection plane with stencil write (REPLACE, ref=1)
// 3. Pass 2: render reflected scene with:
//    - stencil test: EQUAL, ref=1
//    - view matrix mirrored across the plane normal
//    - face culling reversed
// 4. Clear stencil after pass 2
//
// This file serves as documentation of the algorithm.
// Actual implementation deferred to wgpu stencil buffer integration.

// See wgpu::RenderPassDepthStencilAttachment for stencil configuration:
// stencil_ops: Some(wgpu::Operations {
//     load: LoadOp::Clear(0),
//     store: StoreOp::Store,
// })

// Mirror view matrix across a plane with normal n and origin o:
// fn mirror_view_matrix(view: mat4x4<f32>, n: vec3<f32>, o: vec3<f32>) -> mat4x4<f32> {
//     let d = -dot(view[3].xyz - o, n);
//     let reflect = mat4x4<f32>(
//         vec4<f32>(1.0 - 2.0*n.x*n.x, -2.0*n.x*n.y, -2.0*n.x*n.z, 0.0),
//         vec4<f32>(-2.0*n.y*n.x, 1.0 - 2.0*n.y*n.y, -2.0*n.y*n.z, 0.0),
//         vec4<f32>(-2.0*n.z*n.x, -2.0*n.z*n.y, 1.0 - 2.0*n.z*n.z, 0.0),
//         vec4<f32>(2.0*d*n.x, 2.0*d*n.y, 2.0*d*n.z, 1.0),
//     );
//     return view * reflect;
// }
