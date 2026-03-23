//! GPU render context: wgpu surface, device, queue, and the two render passes.
//!
//! # Render passes
//!
//! Every frame consists of two passes:
//!
//! 1. **Scene pass** (WITH depth) — draws all scene objects using Blinn-Phong
//!    shading, then overlays the 3-D brush rings on top of the model surface.
//!
//! 2. **Brush 2-D overlay pass** (NO depth) — drawn only when the cursor misses
//!    the model, showing a screen-space fallback ring at the cursor position.

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
use crate::painter::{BrushPipeline, BrushUniform};

/// Owns all wgpu resources and orchestrates frame rendering.
pub struct RenderContext<'window> {
    surface:        Surface<'window>,
    device:         Device,
    queue:          Queue,
    config:         SurfaceConfiguration,
    /// The main Blinn-Phong scene pipeline; `None` only during teardown.
    pipeline:       Option<ScenePipeline>,
    /// The brush cursor pipeline (3-D rings + 2-D fallback); `None` only during teardown.
    brush_pipeline: Option<BrushPipeline>,
    /// Background clear colour (dark near-black by default).
    pub clear_color: Color,
}

impl<'window> RenderContext<'window> {
    /// Initialises wgpu: creates the instance, surface, adapter, device,
    /// queue, swap chain, and both render pipelines.
    pub async fn new(window: &'window Window) -> Self {
        let size = window.inner_size();

        // Create the wgpu instance (backend is auto-selected).
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let surface  = instance.create_surface(window).expect("Failed to create surface");

        // Request any GPU adapter compatible with our surface.
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::default(),
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        // Request a logical device with default feature/limit requirements.
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

        // Choose an sRGB surface format if available (better colour accuracy).
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage:                         TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                         size.width.max(1),
            height:                        size.height.max(1),
            present_mode:                  PresentMode::Fifo, // vsync
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Build both pipelines with the same surface format and depth format.
        let depth_format   = wgpu::TextureFormat::Depth32Float;
        let pipeline       = ScenePipeline::new(&device, format, config.width, config.height);
        let brush_pipeline = BrushPipeline::new(&device, format, depth_format);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline:       Some(pipeline),
            brush_pipeline: Some(brush_pipeline),
            clear_color: Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
        }
    }

    /// Exposes the wgpu device so callers can upload meshes at startup.
    pub fn device(&self) -> &Device { &self.device }
    /// Exposes the wgpu queue (rarely needed outside this module).
    pub fn queue(&self)  -> &Queue  { &self.queue  }

    /// Handles a window resize: reconfigures the swap chain and recreates the
    /// depth texture to match the new dimensions.
    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        if let Some(p) = &mut self.pipeline {
            p.resize(&self.device, w, h);
        }
    }

    /// Renders one frame.
    ///
    /// Returns `false` if the frame was skipped (e.g. zero-size window or a
    /// recoverable surface error).
    pub fn render(&mut self, scene: Option<&Scene>, brush: Option<&BrushUniform>) -> bool {
        // Skip rendering for zero-size windows (common during minimisation).
        if self.config.width == 0 || self.config.height == 0 {
            return false;
        }

        // Acquire the next swap-chain texture, handling transient errors.
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,

            // Suboptimal (e.g. window was resized): reconfigure and retry once.
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                drop(t);
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t2) => t2,
                    _ => return false,
                }
            }

            // Surface lost or outdated: reconfigure and skip this frame.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return false;
            }

            // Timeout or occluded: skip this frame silently.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }

            // Validation error: log and skip.
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("[malx] Validation error on surface texture acquisition");
                return false;
            }
        };

        let view = surface_texture.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("malx_frame_encoder"),
        });

        // Upload the camera uniform before the first pass.
        if let (Some(scene), Some(pipeline)) = (scene, &self.pipeline) {
            let aspect      = self.config.width as f32 / self.config.height as f32;
            let cam_uniform = CameraUniform::from_camera(&scene.camera, aspect);
            self.queue.write_buffer(&pipeline.camera_buffer, 0, bytes_of(&cam_uniform));
        }

        // ── Pass 1: scene geometry + 3-D brush rings (WITH depth) ────────────
        {
            let depth_attachment = self.pipeline.as_ref().map(|p| {
                RenderPassDepthStencilAttachment {
                    view: &p.depth_view,
                    depth_ops: Some(Operations {
                        load:  LoadOp::Clear(1.0), // clear depth to maximum
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            });

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("malx_scene_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: Operations {
                        load:  LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes:    None,
                occlusion_query_set: None,
                multiview_mask:      None,
            });

            // Draw each scene object with its own model transform.
            if let (Some(scene), Some(pipeline)) = (scene, &self.pipeline) {
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);

                for obj in &scene.objects {
                    // Compute the inverse-transpose for correct normal transformation.
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

            // Draw the 3-D brush rings on top of the model (cursor on surface).
            if let (Some(b), Some(bp)) = (brush, &self.brush_pipeline) {
                bp.draw_3d(&mut pass, &self.device, &self.queue, b);
            }
        }

        // ── Pass 2: 2-D brush fallback overlay (NO depth, cursor off model) ──
        let needs_2d = brush.map(|b| b.hit.is_none()).unwrap_or(false);
        if needs_2d {
            if let (Some(b), Some(bp)) = (brush, &self.brush_pipeline) {
                let mut pass2 = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("malx_brush_2d_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view:           &view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: Operations {
                            load:  LoadOp::Load,   // preserve the scene colour
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None, // no depth needed for 2-D overlay
                    timestamp_writes:    None,
                    occlusion_query_set: None,
                    multiview_mask:      None,
                });
                bp.draw_2d(&mut pass2, &self.queue, b);
            }
        }

        // Submit all recorded commands and present the frame.
        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        true
    }
}
