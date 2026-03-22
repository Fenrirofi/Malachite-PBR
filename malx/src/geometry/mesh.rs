use super::Vertex;
use wgpu::util::DeviceExt;

/// CPU-side mesh data.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices:  Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Upload to GPU, returning a [`GpuMesh`].
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

/// GPU-resident mesh — vertex + index buffers ready to draw.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  wgpu::Buffer,
    pub index_count:   u32,
}
