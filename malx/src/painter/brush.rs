//! Brush cursor renderer — draws two rings lying ON the model surface.
//!
//! # Visual design
//!
//! Each ring is a thin triangle-strip ribbon in world space, rendered inside
//! the 3‑D scene pass so perspective foreshortening and depth testing work
//! correctly.  The rings live in the tangent plane at the ray-hit point:
//!
//! - **Inner ring** — brush size (solid white)
//! - **Outer ring** — hardness boundary (dashed yellow, 24 dashes)
//! - **Centre dot** — small filled disk at the exact hit point
//!
//! When the cursor misses the model the 3‑D rings are simply not drawn.
//!
//! # Screen-space fallback (2‑D overlay)
//!
//! A separate screen-space pass is drawn when the cursor is off the model.
//! It renders the same two rings as pixel‑radius circles so the user always
//! has some visual feedback.  The radii in pixels are **constant** (no attempt
//! to match world size), because there is no surface to project onto.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use std::mem;
use wgpu::util::DeviceExt;

// ── Vertex types ──────────────────────────────────────────────────────────────

/// Vertex used by both the inner ring, outer ring, and centre dot geometry.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RingVertex {
    /// World-space position of this vertex.
    position: [f32; 3],
    /// Geometry kind — controls colour in the fragment shader:
    /// - `0` = inner ring / centre dot (solid white)
    /// - `2` = outer ring (dashed yellow)
    kind: f32,
    /// Arc parameter `[0, 1)` around the ring circumference.
    /// Used by the fragment shader to create dashes on the outer ring.
    arc_t: f32,
    /// Explicit padding to maintain 32‑byte alignment (3 × f32).
    _pad: [f32; 3],
}

impl RingVertex {
    /// wgpu vertex buffer layout for this struct.
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<RingVertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32 },
                wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32 },
            ],
        }
    }
}

// ── GPU uniforms ──────────────────────────────────────────────────────────────

/// Camera uniform for the 3‑D brush ring shader (view‑projection only).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BrushCameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

/// All per‑frame brush data passed from the application to the renderer.
pub struct BrushUniform {
    /// World‑space surface hit point; `None` when cursor misses the model.
    pub hit: Option<Vec3>,
    /// Surface normal at the hit point; used to build the tangent frame.
    pub normal: Option<Vec3>,
    /// Inner ring radius in world units.
    pub inner_r: f32,
    /// Outer ring radius in world units (>= inner_r).
    pub outer_r: f32,
    /// Cursor position in physical pixels (for the 2‑D fallback).
    pub cursor_px: [f32; 2],
    /// Viewport size in physical pixels.
    pub viewport: [f32; 2],
    /// World‑space camera eye position (for screen‑space ring sizing).
    pub eye: Vec3,
    /// The same view‑projection matrix used for the scene.
    pub view_proj: Mat4,
}

// ── WGSL: 3‑D ring shader ─────────────────────────────────────────────────────

const BRUSH_SHADER_3D: &str = r#"
struct Camera {
    view_proj : mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> cam : Camera;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) kind     : f32,
    @location(2) arc_t    : f32,
}

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       kind     : f32,
    @location(1)       arc_t    : f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = cam.view_proj * vec4<f32>(in.position, 1.0);
    out.kind     = in.kind;
    out.arc_t    = in.arc_t;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let k = in.kind;

    // kind <= 1.5 → inner ring or centre dot → solid white
    if k < 1.5 {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    // kind > 1.5 → outer ring → dashed orange‑yellow (24 dashes around the ring)
    let frac = fract(in.arc_t * 24.0);
    if frac > 0.55 {
        discard; // gap between dashes
    }
    return vec4<f32>(1.0, 0.55, 0.0, 1.0);
}
"#;

// ── WGSL: 2‑D fallback shader ─────────────────────────────────────────────────

const BRUSH_SHADER_2D: &str = r#"
struct Brush2D {
    cx_px    : f32,
    cy_px    : f32,
    inner_px : f32,
    outer_px : f32,
    line_w   : f32,
    _pad0    : f32,
    _pad1    : f32,
    _pad2    : f32,
}
@group(0) @binding(0) var<uniform> brush : Brush2D;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dx   = frag_coord.x - brush.cx_px;
    let dy   = frag_coord.y - brush.cy_px;
    let dist = sqrt(dx * dx + dy * dy);
    let rh   = brush.line_w;

    if dist < rh { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }

    if abs(dist - brush.inner_px) < rh {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0 - abs(dist - brush.inner_px) / rh);
    }

    if brush.outer_px > brush.inner_px + 1.0 {
        if abs(dist - brush.outer_px) < rh {
            let t = fract(atan2(dy, dx) / (2.0 * 3.14159265) * 24.0);
            if t < 0.55 {
                let orange = vec3<f32>(1.0, 0.55, 0.0);
                let alpha  = 1.0 - abs(dist - brush.outer_px) / rh;
                return vec4<f32>(orange, alpha);
            }
        }
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
"#;

// ── Geometry helpers ──────────────────────────────────────────────────────────

fn ring_vertices(
    centre:     Vec3,
    tangent_u:  Vec3,
    tangent_v:  Vec3,
    radius:     f32,
    half_w:     f32,
    kind:       f32,
    segments:   usize,
    vertices:   &mut Vec<RingVertex>,
    indices:    &mut Vec<u32>,
) {
    let base = vertices.len() as u32;

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let a = t * std::f32::consts::TAU;
        let c = a.cos();
        let s = a.sin();

        let pt     = centre + (tangent_u * c + tangent_v * s) * radius;
        let radial = (tangent_u * c + tangent_v * s).normalize();

        let p_inner = pt - radial * half_w;
        let p_outer = pt + radial * half_w;

        vertices.push(RingVertex { position: p_inner.to_array(), kind, arc_t: t, _pad: [0.0; 3] });
        vertices.push(RingVertex { position: p_outer.to_array(), kind, arc_t: t, _pad: [0.0; 3] });

        if i < segments {
            let b = base + (i * 2) as u32;
            indices.extend_from_slice(&[b, b+1, b+2, b+1, b+3, b+2]);
        }
    }
}

fn disk_vertices(
    centre:    Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    radius:    f32,
    kind:      f32,
    segments:  usize,
    vertices:  &mut Vec<RingVertex>,
    indices:   &mut Vec<u32>,
) {
    let base = vertices.len() as u32;

    vertices.push(RingVertex {
        position: centre.to_array(),
        kind,
        arc_t: 0.0,
        _pad: [0.0; 3],
    });

    for i in 0..=segments {
        let t  = i as f32 / segments as f32;
        let a  = t * std::f32::consts::TAU;
        let pt = centre + (tangent_u * a.cos() + tangent_v * a.sin()) * radius;

        vertices.push(RingVertex { position: pt.to_array(), kind, arc_t: t, _pad: [0.0; 3] });

        if i < segments {
            let rim_curr = base + 1 + i as u32;
            let rim_next = base + 1 + (i + 1) as u32;
            indices.extend_from_slice(&[base, rim_curr, rim_next]);
        }
    }
}

// ── BrushPipeline ─────────────────────────────────────────────────────────────

pub struct BrushPipeline {
    pipeline_3d: wgpu::RenderPipeline,
    cam_buf:     wgpu::Buffer,
    cam_bg:      wgpu::BindGroup,
    cam_bgl:     wgpu::BindGroupLayout,

    pipeline_2d: wgpu::RenderPipeline,
    uni2d_buf:   wgpu::Buffer,
    bg_2d:       wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct Uni2D {
    cx:     f32,
    cy:     f32,
    inner:  f32,
    outer:  f32,
    line_w: f32,
    _pad0:  f32,
    _pad1:  f32,
    _pad2:  f32,
}

impl BrushPipeline {
    pub fn new(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format:   wgpu::TextureFormat,
    ) -> Self {
        // ── 3‑D ring pipeline ─────────────────────────────────────────────────
        let shader_3d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("brush_3d_shader"),
            source: wgpu::ShaderSource::Wgsl(BRUSH_SHADER_3D.into()),
        });

        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush_cam_bgl"),
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

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("brush_cam_buf"),
            size:               mem::size_of::<BrushCameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("brush_cam_bg"),
            layout:  &cam_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: cam_buf.as_entire_binding(),
            }],
        });

        let layout_3d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("brush_3d_layout"),
            bind_group_layouts: &[Some(&cam_bgl)],
            immediate_size:     0,
        });

        let pipeline_3d = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("brush_3d_pipeline"),
            layout: Some(&layout_3d),
            vertex: wgpu::VertexState {
                module:      &shader_3d,
                entry_point: Some("vs_main"),
                buffers:     &[RingVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader_3d,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // FIX: depth_write disabled, LessEqual compare, no bias.
            // Geometry is physically offset along the normal in draw_3d()
            // so we don't need slope-scale bias here.
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              depth_format,
                depth_write_enabled: Some(false),
                depth_compare:       Some(wgpu::CompareFunction::LessEqual),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ── 2‑D fallback pipeline ─────────────────────────────────────────────
        let shader_2d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("brush_2d_shader"),
            source: wgpu::ShaderSource::Wgsl(BRUSH_SHADER_2D.into()),
        });

        let bgl2d = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush_2d_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let uni2d_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("brush_2d_uni"),
            size:               mem::size_of::<Uni2D>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_2d = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("brush_2d_bg"),
            layout:  &bgl2d,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uni2d_buf.as_entire_binding(),
            }],
        });

        let layout_2d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("brush_2d_layout"),
            bind_group_layouts: &[Some(&bgl2d)],
            immediate_size:     0,
        });

        let pipeline_2d = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("brush_2d_pipeline"),
            layout: Some(&layout_2d),
            vertex: wgpu::VertexState {
                module:      &shader_2d,
                entry_point: Some("vs_main"),
                buffers:     &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader_2d,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        Self {
            pipeline_3d, cam_buf, cam_bg, cam_bgl,
            pipeline_2d, uni2d_buf, bg_2d,
        }
    }

    pub fn draw_3d<'rpass>(
        &'rpass self,
        pass:   &mut wgpu::RenderPass<'rpass>,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        uni:    &BrushUniform,
    ) {
        let (hit, normal) = match (uni.hit, uni.normal) {
            (Some(h), Some(n)) => (h, n),
            _ => return,
        };

        // ── FIX 1: robust tangent frame ───────────────────────────────────────
        // Pick the world axis least parallel to the normal to avoid degeneracy.
        let ref_axis = if normal.z.abs() < 0.9 {
            Vec3::Z
        } else if normal.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let tang = (ref_axis - normal * normal.dot(ref_axis)).normalize();
        let bita = normal.cross(tang).normalize();

        // ── FIX 2: screen-space stroke width ─────────────────────────────────
        let target_px_width = 1.8;
        let px_per_unit = screen_pixels_per_world_unit(hit, uni.view_proj, uni.viewport);
        let half_w = if px_per_unit > 0.0 {
            (target_px_width / px_per_unit).clamp(0.001, uni.inner_r * 0.25)
        } else {
            0.001
        };

        // ── FIX 3: normal offset to avoid z-fighting ─────────────────────────
        // Push geometry away from the surface proportionally to camera distance.
        // This is more reliable than depth bias for coplanar geometry.
        let cam_dist = (uni.eye - hit).length();
        let normal_offset = (cam_dist * 0.0005).max(0.001);
        let hit_offset = hit + normal * normal_offset;

        let segs: usize = 64;

        let mut verts:   Vec<RingVertex> = Vec::new();
        let mut indices: Vec<u32>        = Vec::new();

        disk_vertices(hit_offset, tang, bita, 0.008, 0.0, 16, &mut verts, &mut indices);
        ring_vertices(hit_offset, tang, bita, uni.inner_r, half_w, 0.0, segs, &mut verts, &mut indices);

        if uni.outer_r > uni.inner_r + 0.001 {
            ring_vertices(hit_offset, tang, bita, uni.outer_r, half_w, 2.0, segs, &mut verts, &mut indices);
        }

        if verts.is_empty() { return; }

        queue.write_buffer(
            &self.cam_buf,
            0,
            bytemuck::bytes_of(&BrushCameraUniform {
                view_proj: uni.view_proj.to_cols_array_2d(),
            }),
        );

        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("brush_ring_vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("brush_ring_ibuf"),
            contents: bytemuck::cast_slice(&indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        pass.set_pipeline(&self.pipeline_3d);
        pass.set_bind_group(0, &self.cam_bg, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }

    pub fn draw_2d<'rpass>(
        &'rpass self,
        pass:  &mut wgpu::RenderPass<'rpass>,
        queue: &wgpu::Queue,
        uni:   &BrushUniform,
    ) {
        if uni.hit.is_some() {
            return;
        }

        const FALLBACK_INNER_PX: f32 = 100.0;
        const FALLBACK_OUTER_PX: f32 = 120.0;
        const LINE_WIDTH_PX: f32     = 1.8;

        let u = Uni2D {
            cx:     uni.cursor_px[0],
            cy:     uni.cursor_px[1],
            inner:  FALLBACK_INNER_PX,
            outer:  FALLBACK_OUTER_PX,
            line_w: LINE_WIDTH_PX,
            ..Default::default()
        };
        queue.write_buffer(&self.uni2d_buf, 0, bytemuck::bytes_of(&u));

        pass.set_pipeline(&self.pipeline_2d);
        pass.set_bind_group(0, &self.bg_2d, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ── Helper: pixels per world unit at a given point ───────────────────────────

/// Returns how many physical pixels correspond to one world unit at `pos`.
///
/// Uses all three world axes and takes the maximum, which avoids the
/// near-zero result that occurred when the original code used only Vec3::X
/// and the camera happened to look along X.
fn screen_pixels_per_world_unit(pos: Vec3, view_proj: Mat4, viewport: [f32; 2]) -> f32 {
    let p = view_proj.project_point3(pos);

    // X axis contribution (measured in horizontal pixels).
    let px_x = {
        let q = view_proj.project_point3(pos + Vec3::X);
        ((q.x - p.x) * viewport[0] * 0.5).abs()
    };

    // Y axis contribution (measured in vertical pixels).
    let px_y = {
        let q = view_proj.project_point3(pos + Vec3::Y);
        ((q.y - p.y) * viewport[1] * 0.5).abs()
    };

    // Z axis contribution (2-D screen distance).
    let px_z = {
        let q  = view_proj.project_point3(pos + Vec3::Z);
        let dx = (q.x - p.x) * viewport[0] * 0.5;
        let dy = (q.y - p.y) * viewport[1] * 0.5;
        (dx * dx + dy * dy).sqrt()
    };

    px_x.max(px_y).max(px_z)
}