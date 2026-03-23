//! GPU vertex type shared by all geometry in the scene.

use bytemuck::{Pod, Zeroable};

/// A single mesh vertex carrying position, normal, UV, tangent and bitangent.
///
/// Layout in memory (tightly packed, no padding):
/// ```text
/// offset  0 — position  [f32; 3]  — 12 bytes
/// offset 12 — normal    [f32; 3]  — 12 bytes
/// offset 24 — uv        [f32; 2]  —  8 bytes
/// offset 32 — tangent   [f32; 3]  — 12 bytes
/// offset 44 — bitangent [f32; 3]  — 12 bytes
///                               total: 56 bytes
/// ```
///
/// Matches the WGSL `VertexInput` struct:
/// ```wgsl
/// struct VertexInput {
///     @location(0) position  : vec3<f32>,
///     @location(1) normal    : vec3<f32>,
///     @location(2) uv        : vec2<f32>,
///     @location(3) tangent   : vec3<f32>,
///     @location(4) bitangent : vec3<f32>,
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Outward-facing surface normal (unit length).
    pub normal: [f32; 3],
    /// Texture coordinates (0..1 range, V increases downward).
    pub uv: [f32; 2],
    /// Tangent vector — lies in the surface plane, points in the +U direction.
    /// Used together with `bitangent` and `normal` to build the TBN matrix
    /// for normal mapping.
    pub tangent: [f32; 3],
    /// Bitangent vector — lies in the surface plane, points in the +V direction.
    /// Derived from `normal × tangent` then re-orthogonalised via Gram-Schmidt.
    pub bitangent: [f32; 3],
}

impl Vertex {
    /// Returns the wgpu vertex buffer layout descriptor for this type.
    ///
    /// Shader locations:
    /// - `@location(0)` → `position`
    /// - `@location(1)` → `normal`
    /// - `@location(2)` → `uv`
    /// - `@location(3)` → `tangent`
    /// - `@location(4)` → `bitangent`
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 44,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}