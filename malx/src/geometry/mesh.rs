//! CPU-side mesh data and its GPU-resident counterpart.

use super::Vertex;
use wgpu::util::DeviceExt;

// ── CPU mesh ──────────────────────────────────────────────────────────────────

/// A triangle mesh stored in CPU memory.
///
/// Use [`Mesh::upload`] to transfer it to the GPU once and then draw it every
/// frame via the returned [`GpuMesh`].
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    /// Triangle indices (every three consecutive values form one triangle).
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Number of indices, i.e. the value passed to `draw_indexed` calls.
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Uploads vertex and index data to GPU buffers, returning a [`GpuMesh`]
    /// that is ready to be drawn.
    pub fn upload(&self, device: &wgpu::Device) -> GpuMesh {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("index_buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: self.index_count(),
        }
    }
}

// ── GPU mesh ──────────────────────────────────────────────────────────────────

/// GPU-resident mesh ready to be drawn with `draw_indexed`.
///
/// Created by [`Mesh::upload`]; the CPU-side data can be discarded afterwards.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  wgpu::Buffer,
    /// Cached index count (passed directly to `draw_indexed`).
    pub index_count:   u32,
}
