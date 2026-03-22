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

pub struct RenderContext<'window> {
    surface:  Surface<'window>,
    device:   Device,
    queue:    Queue,
    config:   SurfaceConfiguration,
    pipeline: Option<ScenePipeline>,
    pub clear_color: Color,
}

impl<'window> RenderContext<'window> {
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
                label:                  Some("malx_device"),
                required_features:      Features::empty(),
                required_limits:        Limits::default(),
                memory_hints:           MemoryHints::default(),
                trace:                  wgpu::Trace::Off,
                experimental_features:  wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("Failed to create device");

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage:                          TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                          size.width.max(1),
            height:                         size.height.max(1),
            present_mode:                   PresentMode::Fifo,
            alpha_mode:                     caps.alpha_modes[0],
            view_formats:                   vec![],
            desired_maximum_frame_latency:  2,
        };
        surface.configure(&device, &config);

        // Build the scene pipeline immediately
        let pipeline = ScenePipeline::new(&device, format, config.width, config.height);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline: Some(pipeline),
            clear_color: Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
        }
    }

    // ── Accessors for app code ─────────────────────────────────────────────

    pub fn device(&self) -> &Device  { &self.device }
    pub fn queue(&self)  -> &Queue   { &self.queue  }

    // ── Resize ────────────────────────────────────────────────────────────

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        if let Some(p) = &mut self.pipeline {
            p.resize(&self.device, w, h);
        }
    }

    // ── Render ────────────────────────────────────────────────────────────

    /// Render the scene. Pass `None` to just clear the screen.
    pub fn render(&mut self, scene: Option<&Scene>) -> bool {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)    => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface.configure(&self.device, &self.config);
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("[malx] Validation error");
                return false;
            }
        };

        let view = surface_texture.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("malx_frame_encoder"),
        });

        // Upload camera uniform if we have a scene + pipeline
        if let (Some(scene), Some(pipeline)) = (scene, &self.pipeline) {
            let aspect = self.config.width as f32 / self.config.height as f32;
            let cam_uniform = CameraUniform::from_camera(&scene.camera, aspect);
            self.queue.write_buffer(&pipeline.camera_buffer, 0, bytes_of(&cam_uniform));
        }

        {
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

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("malx_main_pass"),
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
                timestamp_writes:         None,
                occlusion_query_set:      None,
                multiview_mask:           None,
            });

            // Draw scene objects
            if let (Some(scene), Some(pipeline)) = (scene, &self.pipeline) {
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);

                for obj in &scene.objects {
                    // Build per-object model uniform
                    let normal_mat = obj.transform.inverse().transpose();
                    let model_uniform = ModelUniform {
                        model:      obj.transform.to_cols_array_2d(),
                        normal_mat: normal_mat.to_cols_array_2d(),
                    };

                    // Create bind group on the fly (simple; for a real app cache these)
                    let (_buf, model_bg) = pipeline.create_model_bind_group(&self.device, &model_uniform);

                    pass.set_bind_group(1, &model_bg, &[]);
                    pass.set_vertex_buffer(0, obj.mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(obj.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..obj.mesh.index_count, 0, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        true
    }
}
