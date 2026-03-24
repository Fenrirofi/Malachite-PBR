//! GPU render context: wgpu surface, device, queue, and the three render passes.
//!
//! # Render passes (HDR pipeline)
//!
//! Every frame consists of three passes:
//!
//! 1. **Scene pass** (WITH depth, target = HDR `Rgba16Float`) — draws all scene
//!    objects using Cook-Torrance PBR.  Also overlays the 3-D brush rings.
//!
//! 2. **Brush 2-D overlay pass** (NO depth, target = HDR `Rgba16Float`) — drawn
//!    only when the cursor misses the model; writes the 2-D fallback ring.
//!
//! 3. **Tonemap pass** (NO depth, target = swapchain) — reads the HDR texture,
//!    applies ACES filmic + exposure, writes LDR γ-corrected output.

use bytemuck::bytes_of;
use wgpu::{
    Color, CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, LoadOp, MemoryHints, Operations, PresentMode, Queue,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RequestAdapterOptions, StoreOp, Surface, SurfaceConfiguration, TextureUsages,
    TextureViewDescriptor,
};
use winit::window::Window;

use crate::scene::Scene;
use crate::scene::camera::CameraUniform;
use super::pipeline::{ModelUniform, ScenePipeline};
use super::tonemapping::{TonemapPipeline, TonemapParams};
use crate::painter::{BrushPipeline, BrushUniform};

/// Owns all wgpu resources and orchestrates frame rendering.
pub struct RenderContext<'window> {
    surface:        Surface<'window>,
    device:         Device,
    queue:          Queue,
    config:         SurfaceConfiguration,
    /// PBR scene pipeline — renders to HDR offscreen texture.
    pipeline:       Option<ScenePipeline>,
    /// Brush cursor pipeline (3-D rings + 2-D fallback).
    brush_pipeline: Option<BrushPipeline>,
    /// ACES tonemapping pipeline — reads HDR texture, writes to swapchain.
    tonemap:        Option<TonemapPipeline>,
    /// Background clear colour (dark near-black by default).
    pub clear_color: Color,
    /// Exposure multiplier fed to the ACES pass (default 1.0).
    pub exposure:   f32,
}

impl<'window> RenderContext<'window> {
    /// Initialises wgpu: creates the instance, surface, adapter, device,
    /// queue, swap chain, and all three render pipelines.
    pub async fn new(window: &'window Window) -> Self {
        let size = window.inner_size();

        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let surface  = instance.create_surface(window).expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::default(),
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label:                 Some("malx_device"),
                required_features:     Features::empty(),
                required_limits:       Limits::default(),
                memory_hints:          MemoryHints::default(),
                trace:                 wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("Failed to create device");

        // Prefer non-sRGB swapchain — we apply gamma ourselves in the tonemap pass.
        let caps          = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter().copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage:                         TextureUsages::RENDER_ATTACHMENT,
            format:                        surface_format,
            width:                         size.width.max(1),
            height:                        size.height.max(1),
            present_mode:                  PresentMode::Fifo,
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let hdr_format   = wgpu::TextureFormat::Rgba16Float;
        let depth_format = wgpu::TextureFormat::Depth32Float;

        // Scene + brush write to HDR offscreen; tonemap blit to swapchain.
        let pipeline       = ScenePipeline::new(&device, hdr_format, config.width, config.height);
        let brush_pipeline = BrushPipeline::new(&device, hdr_format, depth_format);
        let tonemap        = TonemapPipeline::new(&device, surface_format, config.width, config.height);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline:       Some(pipeline),
            brush_pipeline: Some(brush_pipeline),
            tonemap:        Some(tonemap),
            clear_color:    Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
            exposure:       1.0,
        }
    }

    /// Exposes the wgpu device so callers can upload meshes at startup.
    pub fn device(&self) -> &Device { &self.device }
    /// Exposes the wgpu queue (rarely needed outside this module).
    pub fn queue(&self)  -> &Queue  { &self.queue  }

    /// Handles a window resize: reconfigures the swap chain and recreates
    /// the depth texture + HDR offscreen texture to match the new dimensions.
    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        if let Some(p) = &mut self.pipeline {
            p.resize(&self.device, w, h);
        }
        if let Some(t) = &mut self.tonemap {
            t.resize(&self.device, w, h);
        }
    }

    /// Renders one frame (3-pass HDR pipeline).
    ///
    /// Pass 1: PBR scene + 3-D brush rings  →  HDR `Rgba16Float` texture
    /// Pass 2: 2-D brush overlay (if cursor off model)  →  same HDR texture
    /// Pass 3: ACES tonemap + γ  →  swapchain surface
    ///
    /// Returns `false` if the frame was skipped.
    pub fn render(&mut self, scene: Option<&Scene>, brush: Option<&BrushUniform>) -> bool {
        if self.config.width == 0 || self.config.height == 0 { return false; }

        // Acquire swapchain texture.
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                drop(t);
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t2) => t2,
                    _ => return false,
                }
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("[malx] Validation error on surface texture acquisition");
                return false;
            }
        };

        let surface_view = surface_texture.texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("malx_frame_encoder") },
        );

        // Upload camera uniform.
        if let (Some(scene), Some(pipeline)) = (scene, &self.pipeline) {
            let aspect      = self.config.width as f32 / self.config.height as f32;
            let cam_uniform = CameraUniform::from_camera(&scene.camera, aspect);
            self.queue.write_buffer(&pipeline.camera_buffer, 0, bytes_of(&cam_uniform));
        }

        // ── Pass 1: scene geometry + 3-D brush rings → HDR texture ──────────
        {
            // Use the HDR texture view owned by TonemapPipeline as the render target.
            let hdr_view = self.tonemap.as_ref().map(|t| &t.hdr_view);

            let depth_attachment = self.pipeline.as_ref().map(|p| {
                RenderPassDepthStencilAttachment {
                    view: &p.depth_view,
                    depth_ops: Some(Operations {
                        load:  LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            });

            // HDR clear colour — linear, no clamping needed (Rgba16Float).
            let hdr_clear = Color {
                r: self.clear_color.r,
                g: self.clear_color.g,
                b: self.clear_color.b,
                a: 1.0,
            };

            if let Some(hdr_view) = hdr_view {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("malx_scene_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view:           hdr_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: Operations {
                            load:  LoadOp::Clear(hdr_clear),
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: depth_attachment,
                    timestamp_writes:    None,
                    occlusion_query_set: None,
                    multiview_mask:      None,
                });

                if let (Some(scene), Some(pipeline)) = (scene, &self.pipeline) {
                    pass.set_pipeline(&pipeline.pipeline);
                    pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);

                    for obj in &scene.objects {
                        let normal_mat    = obj.transform.inverse().transpose();
                        let model_uniform = ModelUniform {
                            model:      obj.transform.to_cols_array_2d(),
                            normal_mat: normal_mat.to_cols_array_2d(),
                        };
                        let (_buf, model_bg) =
                            pipeline.create_model_bind_group(&self.device, &model_uniform);
                        pass.set_bind_group(1, &model_bg, &[]);
                        pass.set_vertex_buffer(0, obj.mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            obj.mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..obj.mesh.index_count, 0, 0..1);
                    }
                }

                // 3-D brush rings drawn on top of geometry (same HDR target).
                if let (Some(b), Some(bp)) = (brush, &self.brush_pipeline) {
                    bp.draw_3d(&mut pass, &self.device, &self.queue, b);
                }
            }
        }

        // ── Pass 2: 2-D brush fallback overlay → HDR texture ────────────────
        let needs_2d = brush.map(|b| b.hit.is_none()).unwrap_or(false);
        if needs_2d {
            let hdr_view = self.tonemap.as_ref().map(|t| &t.hdr_view);
            if let (Some(b), Some(bp), Some(hdr_view)) =
                (brush, &self.brush_pipeline, hdr_view)
            {
                let mut pass2 = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("malx_brush_2d_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view:           hdr_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: Operations {
                            load:  LoadOp::Load,   // preserve scene colour
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes:    None,
                    occlusion_query_set: None,
                    multiview_mask:      None,
                });
                bp.draw_2d(&mut pass2, &self.queue, b);
            }
        }

        // ── Pass 3: ACES tonemap + gamma → swapchain ─────────────────────────
        if let Some(tonemap) = &self.tonemap {
            let params = TonemapParams {
                exposure: self.exposure,
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            };
            tonemap.draw(&mut encoder, &surface_view, &self.queue, &params);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        true
    }
}