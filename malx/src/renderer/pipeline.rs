use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::geometry::Vertex;
use crate::scene::camera::CameraUniform;

const SHADER: &str = r#"
// ── Uniforms ──────────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj : mat4x4<f32>,
    eye       : vec3<f32>,
    _pad      : f32,
}

struct ModelUniform {
    model      : mat4x4<f32>,
    normal_mat : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera : CameraUniform;
@group(1) @binding(0) var<uniform> model  : ModelUniform;

// ── Vertex ────────────────────────────────────────────────────────────────────

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos   : vec4<f32>,
    @location(0)       world_pos  : vec3<f32>,
    @location(1)       world_norm : vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos  = model.model * vec4<f32>(in.position, 1.0);
    let world_norm = normalize((model.normal_mat * vec4<f32>(in.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_pos   = camera.view_proj * world_pos;
    out.world_pos  = world_pos.xyz;
    out.world_norm = world_norm;
    return out;
}

// ── Fragment — Blinn-Phong ────────────────────────────────────────────────────

const LIGHT_DIR   : vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
const LIGHT_COLOR : vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const OBJECT_COLOR: vec3<f32> = vec3<f32>(0.4, 0.7, 1.0);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let norm     = normalize(in.world_norm);
    let light_d  = normalize(LIGHT_DIR);
    let view_d   = normalize(camera.eye - in.world_pos);
    let half_d   = normalize(light_d + view_d);

    let ambient  = 0.15 * OBJECT_COLOR;
    let diffuse  = max(dot(norm, light_d), 0.0) * OBJECT_COLOR * LIGHT_COLOR;
    let specular = pow(max(dot(norm, half_d), 0.0), 64.0) * LIGHT_COLOR * 0.5;

    return vec4<f32>(ambient + diffuse + specular, 1.0);
}
"#;

// ── Model push-constant uniform (per object) ─────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniform {
    pub model:      [[f32; 4]; 4],
    pub normal_mat: [[f32; 4]; 4],
}

// ── RenderPipeline + bind groups ─────────────────────────────────────────────

pub struct ScenePipeline {
    pub pipeline:            wgpu::RenderPipeline,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub model_bind_group_layout:  wgpu::BindGroupLayout,
    pub camera_buffer:       wgpu::Buffer,
    pub camera_bind_group:   wgpu::BindGroup,
    pub depth_texture:       wgpu::Texture,
    pub depth_view:          wgpu::TextureView,
    pub depth_format:        wgpu::TextureFormat,
}

impl ScenePipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("scene_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // ── Bind group layouts ────────────────────────────────────────────────
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

        // ── Camera buffer ─────────────────────────────────────────────────────
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

        // ── Pipeline ──────────────────────────────────────────────────────────
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               Some("scene_pipeline_layout"),
            bind_group_layouts:  &[Some(&camera_bind_group_layout), Some(&model_bind_group_layout)],
            immediate_size:      0,
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
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:           wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face:         wgpu::FrontFace::Ccw,
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

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (tex, view) = Self::create_depth_texture(device, width, height, self.depth_format);
        self.depth_texture = tex;
        self.depth_view    = view;
    }

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

    /// Creates a per-object bind group + uniform buffer for the given model matrix.
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
