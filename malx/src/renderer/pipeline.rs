//! The main scene render pipeline: Blinn-Phong shading with depth testing.
//!
//! Bind group layout:
//! - Group 0, binding 0 — `CameraUniform` (vertex + fragment stages)
//! - Group 1, binding 0 — `ModelUniform`  (vertex stage only)
//!
//! The pipeline uses a 32-bit depth buffer, counter-clockwise winding, and
//! back-face culling.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::geometry::Vertex;
use crate::scene::camera::CameraUniform;

// ── WGSL shader source ────────────────────────────────────────────────────────

const SHADER: &str = include_str!("../../shaders/pbr_shader.wgsl");

// ── Per-object model uniform ──────────────────────────────────────────────────

/// Per-object GPU uniform: model matrix + its inverse-transpose for normals.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniform {
    /// Object-to-world (model) matrix.
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`, used to correctly transform normals when
    /// the model matrix contains non-uniform scaling.
    pub normal_mat: [[f32; 4]; 4],
}

// ── ScenePipeline ─────────────────────────────────────────────────────────────

/// Owns the wgpu render pipeline, bind groups, camera buffer, and depth texture
/// for the main scene rendering pass.
pub struct ScenePipeline {
    pub pipeline:                 wgpu::RenderPipeline,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub model_bind_group_layout:  wgpu::BindGroupLayout,
    pub camera_buffer:            wgpu::Buffer,
    pub camera_bind_group:        wgpu::BindGroup,
    pub depth_texture:            wgpu::Texture,
    pub depth_view:               wgpu::TextureView,
    pub depth_format:             wgpu::TextureFormat,
}

impl ScenePipeline {
    /// Creates all GPU resources: shader, bind groups, buffers, depth texture,
    /// and the render pipeline itself.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("pbr_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // ── Bind group layouts ────────────────────────────────────────────────
        // Group 0: camera (visible to both vertex and fragment stages).
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                }],
            });

        // Group 1: per-object model transform (vertex stage only).
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("model_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                }],
            });

        // ── Camera uniform buffer ─────────────────────────────────────────────
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("camera_buffer"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("camera_bg"),
            layout:  &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // ── Depth texture ─────────────────────────────────────────────────────
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let (depth_texture, depth_view) =
            Self::create_depth_texture(device, width, height, depth_format);

        // ── Render pipeline ───────────────────────────────────────────────────
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("scene_pipeline_layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&model_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("scene_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers:     &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    // Opaque: new colour replaces old (no alpha blend needed).
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:           wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face:         wgpu::FrontFace::Ccw, // counter-clockwise = front
                cull_mode:          Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              depth_format,
                depth_write_enabled: Some(true),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        Self {
            pipeline,
            camera_bind_group_layout,
            model_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            depth_view,
            depth_format,
        }
    }

    /// Recreates the depth texture after a window resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (tex, view) =
            Self::create_depth_texture(device, width, height, self.depth_format);
        self.depth_texture = tex;
        self.depth_view    = view;
    }

    /// Creates a depth texture and its default view at the given size.
    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("depth_texture"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Creates a per-frame, per-object uniform buffer + bind group for the
    /// given model transform.
    ///
    /// The buffer is allocated fresh each call; this is acceptable for a small
    /// number of objects.  For many objects, a single large buffer with dynamic
    /// offsets would be more efficient.
    pub fn create_model_bind_group(
        &self,
        device: &wgpu::Device,
        uniform: &ModelUniform,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("model_buffer"),
            contents: bytemuck::bytes_of(uniform),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("model_bg"),
            layout:  &self.model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: buffer.as_entire_binding(),
            }],
        });
        (buffer, bind_group)
    }
}
