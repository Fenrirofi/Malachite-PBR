//! GPU vertex type shared by all geometry in the scene.

use bytemuck::{Pod, Zeroable};

/// A single mesh vertex carrying position and surface normal.
///
/// Both attributes are three-component `f32` vectors and are tightly packed
/// (no padding), matching the WGSL `VertexInput` struct in the scene shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// World-space (or object-space) position.
    pub position: [f32; 3],
    /// Outward-facing surface normal (should be unit length).
    pub normal: [f32; 3],
}

impl Vertex {
    /// Returns the wgpu vertex buffer layout descriptor for this type.
    ///
    /// Matches shader locations:
    /// - `@location(0)` → `position`
    /// - `@location(1)` → `normal`
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position at shader location 0
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // normal at shader location 1
                wgpu::VertexAttribute {
                    offset:          mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
