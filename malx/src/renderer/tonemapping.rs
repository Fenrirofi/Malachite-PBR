//! Tonemapping pass: reads an HDR `Rgba16Float` texture and writes
//! tone-mapped + gamma-corrected colour to the swapchain surface.
//!
//! Uses the **ACES filmic** curve (Stephen Hill's fitted approximation).
//! The pass draws a single full-screen triangle — no vertex buffer needed.

use wgpu::util::DeviceExt;

// ── WGSL ─────────────────────────────────────────────────────────────────────

const TONEMAP_SHADER: &str = r#"
// ── Sampler + HDR texture ─────────────────────────────────────────────────────
@group(0) @binding(0) var hdr_texture : texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler : sampler;

// ── Exposure uniform ──────────────────────────────────────────────────────────
struct TonemapParams {
    exposure : f32,
    _pad0    : f32,
    _pad1    : f32,
    _pad2    : f32,
}
@group(0) @binding(2) var<uniform> params : TonemapParams;

// ── Full-screen triangle (no VBO) ─────────────────────────────────────────────
// Three hardcoded NDC positions that cover the entire clip space.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

// ── ACES filmic tone mapping (Stephen Hill fit) ───────────────────────────────
//
//   Approximates the ACES RRT+ODT transform with two 3×3 colour matrices and
//   a simple rational function.  Much faster than the full ACES pipeline while
//   giving visually equivalent results.
//
//   Reference: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    // RRT (Reference Rendering Transform) input matrix.
    let m1 = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    // ODT (Output Display Transform) output matrix.
    let m2 = mat3x3<f32>(
        vec3<f32>( 1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108,  1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605,  1.07602),
    );
    let v  = m1 * x;
    let a  = v * (v + 0.0245786) - 0.000090537;
    let b  = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(m2 * (a / b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Fragment ──────────────────────────────────────────────────────────────────
@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert fragment coord → UV (flip Y so top-left = (0,0))
    let tex_size = vec2<f32>(textureDimensions(hdr_texture));
    let uv       = frag_coord.xy / tex_size;

    // Sample HDR buffer.
    var hdr = textureSample(hdr_texture, hdr_sampler, uv).rgb;

    // Apply exposure before tone-mapping.
    hdr *= params.exposure;

    // ACES filmic tone mapping → LDR [0,1].
    var ldr = aces_filmic(hdr);

    // Gamma correction: linear → sRGB (γ = 2.2).
    ldr = pow(ldr, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(ldr, 1.0);
}
"#;

// ── TonemapParams (CPU side) ──────────────────────────────────────────────────

/// GPU uniform for the tonemapping pass.  Keep aligned to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TonemapParams {
    pub exposure: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self { exposure: 1.0, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0 }
    }
}

// ── TonemapPipeline ───────────────────────────────────────────────────────────

/// Owns the fullscreen-triangle pipeline, bind group, and the HDR texture.
///
/// Call `resize` whenever the window changes size, then `draw` once per frame
/// after the scene pass has written to `hdr_view`.
pub struct TonemapPipeline {
    pub pipeline:      wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group:    wgpu::BindGroup,
    pub params_buffer: wgpu::Buffer,

    /// The HDR offscreen texture rendered into by the scene pass.
    pub hdr_texture:   wgpu::Texture,
    /// Default view of `hdr_texture` (used as render attachment in scene pass).
    pub hdr_view:      wgpu::TextureView,

    pub hdr_format:    wgpu::TextureFormat,
}

impl TonemapPipeline {
    /// `surface_format` — the swapchain format (the *output* of tonemapping).
    /// `width / height` — initial framebuffer size.
    pub fn new(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width:          u32,
        height:         u32,
    ) -> Self {
        let hdr_format = wgpu::TextureFormat::Rgba16Float;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("tonemap_shader"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_SHADER.into()),
        });

        // ── Bind group layout: texture + sampler + params uniform ─────────────
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("tonemap_bgl"),
                entries: &[
                    // binding 0 — HDR texture
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // binding 1 — sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // binding 2 — params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding:    2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   None,
                        },
                        count: None,
                    },
                ],
            });

        // ── HDR texture + params buffer ───────────────────────────────────────
        let (hdr_texture, hdr_view) =
            Self::create_hdr_texture(device, width, height, hdr_format);

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("tonemap_params"),
            contents: bytemuck::bytes_of(&TonemapParams::default()),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("hdr_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = Self::create_bind_group(
            device, &bind_group_layout, &hdr_view, &sampler, &params_buffer,
        );

        // ── Pipeline ──────────────────────────────────────────────────────────
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("tonemap_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size:     0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("tonemap_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers:     &[], // full-screen triangle, no VBO
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
            primitive:      wgpu::PrimitiveState::default(),
            depth_stencil:  None, // tonemapping reads from texture, no depth needed
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // Store the sampler in the pipeline struct to keep it alive.
        // We don't need it after bind-group creation, but dropping it would
        // invalidate the bind group on some backends.  We store it in params_buffer
        // indirectly — actually we need to keep it alive.  The bind group holds
        // a reference so it's fine to drop the local `sampler` here.
        // (wgpu ref-counts all GPU objects.)

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            hdr_texture,
            hdr_view,
            hdr_format,
        }
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    /// Recreates the HDR texture and bind group after a window resize.
    /// The caller must also pass the existing `sampler` — or we recreate it too.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width:  u32,
        height: u32,
    ) {
        let (tex, view) =
            Self::create_hdr_texture(device, width, height, self.hdr_format);
        self.hdr_texture = tex;
        self.hdr_view    = view;

        // Recreate the sampler (cheap) so we don't have to store it separately.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("hdr_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.hdr_view,
            &sampler,
            &self.params_buffer,
        );
    }

    // ── Draw ──────────────────────────────────────────────────────────────────

    /// Records the tonemapping pass into `encoder`, writing to `surface_view`.
    ///
    /// The HDR scene must already have been rendered into `self.hdr_view`.
    pub fn draw(
        &self,
        encoder:      &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
        queue:        &wgpu::Queue,
        params:       &TonemapParams,
    ) {
        // Upload exposure each frame (typically unchanged, but cheap).
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           surface_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1); // full-screen triangle
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn create_hdr_texture(
        device: &wgpu::Device,
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("hdr_texture"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format,
            // TEXTURE_BINDING — read in tonemap shader
            // RENDER_ATTACHMENT — written in scene pass
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_bind_group(
        device:   &wgpu::Device,
        layout:   &wgpu::BindGroupLayout,
        hdr_view: &wgpu::TextureView,
        sampler:  &wgpu::Sampler,
        params:   &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("tonemap_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }
}
