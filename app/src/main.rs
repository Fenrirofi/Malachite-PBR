/// Application entry point for the `malx` 3-D sculpting viewer.
///
/// Responsibilities:
/// - Creates the winit window and event loop.
/// - Owns all application state: camera tumble, brush parameters, timing.
/// - Translates OS events into scene and brush updates, then requests redraws.
/// - Calls into `malx` (the library crate) for GPU rendering.
///
/// Controls:
/// - Alt + Left-drag      → tumble / orbit camera
/// - Alt + Scroll         → zoom (change camera distance)
/// - F + Scroll           → change brush inner radius (size)
/// - Shift + F + Scroll   → change brush hardness gap (outer – inner)
/// - Left-click on model  → print hit point and normal to stdout
use glam::{Mat4, Vec2, Vec3, Vec4};
use malx::{BrushUniform, Camera, RenderContext, Scene, SceneObject, sphere};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// ── Camera tumble ─────────────────────────────────────────────────────────────

/// Tracks the orbit / tumble state of the camera around the scene origin.
///
/// Internally stores yaw and pitch angles in degrees, and converts them to a
/// world-space eye position on demand via [`Tumble::eye`].
struct Tumble {
    /// Whether the left mouse button is currently held while Alt is held.
    active: bool,
    /// Screen-space X of the previous cursor position (used to compute deltas).
    last_x: f64,
    /// Screen-space Y of the previous cursor position.
    last_y: f64,
    /// Horizontal rotation around the world Y-axis, in degrees.
    yaw: f32,
    /// Vertical rotation (elevation), in degrees. Clamped to +-89 deg.
    pitch: f32,
    /// Distance from the camera eye to the orbit target, in world units.
    /// Adjusted by Alt + Scroll; clamped to [0.5, 20.0].
    camera_dist: f32,
}

impl Tumble {
    fn new() -> Self {
        Self {
            active: false,
            last_x: 0.0,
            last_y: 0.0,
            yaw: 0.0,
            pitch: 20.0,
            camera_dist: 3.0,
        }
    }

    /// Returns the world-space eye position for a camera orbiting `target` at
    /// the given distance, using the current yaw/pitch angles.
    fn eye(&self, target: Vec3) -> Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        let d = self.camera_dist;
        target + Vec3::new(
            d * p.cos() * y.sin(),
            d * p.sin(),
            d * p.cos() * y.cos(),
        )
    }

    /// Zooms by changing the camera distance.
    /// `delta` is in scroll lines; positive = scroll up = zoom in (closer).
    fn zoom(&mut self, delta: f32) {
        self.camera_dist = (self.camera_dist - delta * 0.3).clamp(0.5, 20.0);
    }

    /// Returns a normalised "up" vector consistent with the current orientation.
    ///
    /// Derived analytically so the camera never flips past the poles.
    fn up(&self) -> Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        Vec3::new(
            -p.sin() * y.sin(),
            p.cos(),
            -p.sin() * y.cos(),
        )
        .normalize()
    }
}

// ── Brush state ───────────────────────────────────────────────────────────────

/// All brush-related parameters owned by the application.
///
/// The GPU-side representation is built each frame from these values via
/// [`App::make_brush_uniform`].
struct BrushState {
    /// Inner ring radius in world units (defines the "hard" brush size).
    inner_r: f32,
    /// Gap between inner and outer ring in world units.
    /// The outer ring radius is `inner_r + gap_r`.
    gap_r: f32,
    /// World-unit change per scroll notch when resizing the brush.
    scroll_speed: f32,
    /// Cursor position in physical pixels (updated every CursorMoved event).
    cursor_px: Vec2,
    /// `true` once the cursor has entered the window; `false` after it leaves.
    cursor_visible: bool,
}

impl BrushState {
    fn new() -> Self {
        Self {
            inner_r:        0.3,   // roughly 30% of the unit-sphere radius
            gap_r:          0.2,   // outer ring at 0.5 world units
            scroll_speed:   0.02,
            cursor_px:      Vec2::ZERO,
            cursor_visible: false,
        }
    }

    /// Computed outer ring radius (inner + hardness gap).
    fn outer_r(&self) -> f32 {
        self.inner_r + self.gap_r
    }

    /// Adjust brush SIZE by `delta` scroll lines (F + scroll).
    /// The minimum inner radius is clamped to 0.01 so it never disappears.
    fn scroll_size(&mut self, delta: f32) {
        self.inner_r = (self.inner_r + delta * self.scroll_speed).max(0.01);
    }

    /// Adjust hardness GAP by `delta` scroll lines (Shift + F + scroll).
    /// The gap is clamped to >= 0 (inner and outer rings may coincide).
    fn scroll_hardness(&mut self, delta: f32) {
        self.gap_r = (self.gap_r + delta * self.scroll_speed).max(0.0);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Builds the combined view-projection matrix for the given scene and viewport.
fn build_view_proj(scene: &Scene, vp_w: f32, vp_h: f32) -> Mat4 {
    let cam  = &scene.camera;
    let view = Mat4::look_at_rh(cam.eye, cam.target, cam.up);
    let proj = Mat4::perspective_rh(
        cam.fov_y.to_radians(),
        vp_w / vp_h,
        cam.z_near,
        cam.z_far,
    );
    proj * view
}

/// Casts a ray from screen pixel `(px, py)` into the scene and tests it
/// against the unit sphere centred at the origin.
///
/// Returns `(hit_point, surface_normal)` in world space, or `None` if the ray
/// misses.
///
/// # Algorithm
/// 1. Convert the pixel to NDC.
/// 2. Unproject two clip-space points (z = -1 near, z = +1 far) using the
///    inverse view-projection matrix.
/// 3. Solve the ray-sphere intersection equation analytically.
/// 4. Only the nearest intersection in front of the camera (t > 0) is used.
fn raycast_sphere(
    px: f32,
    py: f32,
    vp_w: f32,
    vp_h: f32,
    vp: Mat4,
) -> Option<(Vec3, Vec3)> {
    // Convert pixel coordinates to Normalised Device Coordinates (NDC).
    let ndc_x =  (px / vp_w) * 2.0 - 1.0;
    let ndc_y = -(py / vp_h) * 2.0 + 1.0;

    let inv  = vp.inverse();

    // Unproject the near and far clip-space points to world space.
    let near = inv * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let far  = inv * Vec4::new(ndc_x, ndc_y,  1.0, 1.0);

    // Perspective-divide to get true world positions.
    let o = Vec3::from(near.truncate()) / near.w;
    let d = (Vec3::from(far.truncate()) / far.w - o).normalize();

    // Solve |o + t*d|^2 = 1  =>  t^2 + 2(o.d)t + (|o|^2 - 1) = 0
    let b    = 2.0 * o.dot(d);
    let c    = o.dot(o) - 1.0;
    let disc = b * b - 4.0 * c;

    if disc < 0.0 {
        return None; // ray misses the sphere entirely
    }

    let t = (-b - disc.sqrt()) * 0.5;
    if t < 0.0 {
        return None; // intersection is behind the camera
    }

    let hit  = o + d * t;
    let norm = hit.normalize(); // on a unit sphere the normal equals the position
    Some((hit, norm))
}

// ── App ───────────────────────────────────────────────────────────────────────

/// Top-level application state managed by winit.
///
/// All fields are `Option` so the struct can be initialised before the OS
/// window is ready (winit delivers the window in the `resumed` callback).
struct App {
    /// The OS window; `None` until `resumed` is first called.
    window: Option<&'static Window>,
    /// GPU render context (wgpu surface + device + queue + pipelines).
    ctx: Option<RenderContext<'static>>,
    /// The scene (camera + list of objects to render).
    scene: Option<Scene>,
    /// Camera orbit controller.
    tumble: Tumble,

    // ── Modifier key state ────────────────────────────────────────────────────
    /// Whether either Alt key is currently pressed.
    alt_held: bool,
    /// Whether either Shift key is currently pressed.
    shift_held: bool,
    /// Whether the F key is currently pressed (activates brush scroll mode).
    f_held: bool,

    /// Brush size / hardness parameters and cursor position.
    brush: BrushState,

    // ── FPS counter ───────────────────────────────────────────────────────────
    fps_last_time:   Instant,
    fps_frame_count: u32,
    /// Smoothed frames-per-second value, updated once per second.
    fps: f64,
}

impl App {
    fn new() -> Self {
        Self {
            window:          None,
            ctx:             None,
            scene:           None,
            tumble:          Tumble::new(),
            alt_held:        false,
            shift_held:      false,
            f_held:          false,
            brush:           BrushState::new(),
            fps_last_time:   Instant::now(),
            fps_frame_count: 0,
            fps:             0.0,
        }
    }

    /// Pushes the current tumble angles into the scene camera.
    ///
    /// Call this whenever yaw / pitch changes, or after initial setup.
    fn update_camera(&mut self) {
        if let Some(s) = &mut self.scene {
            s.camera.eye    = self.tumble.eye(Vec3::ZERO);
            s.camera.target = Vec3::ZERO;
            s.camera.up     = self.tumble.up();
        }
    }

    /// Returns the window's inner size in physical pixels.
    ///
    /// Falls back to `(1, 1)` before the window exists to avoid divide-by-zero
    /// in the ray-cast helpers.
    fn vp_size(&self) -> (f32, f32) {
        self.window
            .map(|w| {
                let s = w.inner_size();
                (s.width as f32, s.height as f32)
            })
            .unwrap_or((1.0, 1.0))
    }

    /// Asks the OS to issue a `RedrawRequested` event as soon as possible.
    fn request_redraw(&self) {
        if let Some(w) = self.window {
            w.request_redraw();
        }
    }

    /// Builds the [`BrushUniform`] that the renderer needs each frame.
    ///
    /// Ray-casts from the current cursor position to find the surface hit.
    /// Returns `None` while the cursor is outside the window.
    fn make_brush_uniform(&self) -> Option<BrushUniform> {
        if !self.brush.cursor_visible {
            return None;
        }

        let (vp_w, vp_h) = self.vp_size();
        let vp = self.scene
            .as_ref()
            .map(|s| build_view_proj(s, vp_w, vp_h))
            .unwrap_or(Mat4::IDENTITY);

        // Determine where the brush cursor lands on the model surface.
        let (hit, normal) = match raycast_sphere(
            self.brush.cursor_px.x,
            self.brush.cursor_px.y,
            vp_w,
            vp_h,
            vp,
        ) {
            Some(r) => (Some(r.0), Some(r.1)),
            None    => (None, None),
        };

        Some(BrushUniform {
            hit,
            normal,
            inner_r:   self.brush.inner_r,
            outer_r:   self.brush.outer_r(),
            cursor_px: [self.brush.cursor_px.x, self.brush.cursor_px.y],
            viewport:  [vp_w, vp_h],
            view_proj: vp,
        })
    }
}

impl ApplicationHandler for App {
    /// Called by winit when the application is (re-)activated.
    ///
    /// On first invocation: creates the window, initialises the GPU context,
    /// and builds the initial scene (one unit sphere at the origin).
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // already initialised on a previous resume
        }

        let attrs = Window::default_attributes()
            .with_title("malx")
            .with_inner_size(PhysicalSize::new(1280, 720));

        // SAFETY: We leak the Window so RenderContext can hold a 'static
        // reference to it.  The window lives for the whole program lifetime.
        let window: &'static Window =
            Box::leak(Box::new(event_loop.create_window(attrs).unwrap()));

        // Hide the OS cursor; we draw our own 3-D brush ring instead.
        window.set_cursor_visible(false);

        let ctx = pollster::block_on(RenderContext::new(window));

        let mut scene = Scene::new(Camera::default());
        let mesh = sphere(1.0, 32, 16).upload(ctx.device());
        scene.add(SceneObject::new(mesh, Mat4::IDENTITY));

        self.ctx    = Some(ctx);
        self.scene  = Some(scene);
        self.window = Some(window);
        self.update_camera();
    }

    /// Dispatches OS window events to the appropriate handler logic.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            // ── Window lifecycle ──────────────────────────────────────────────
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            // ── Keyboard: track modifier keys and F ───────────────────────────
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                let pressed = key_event.state == ElementState::Pressed;
                match key_event.physical_key {
                    PhysicalKey::Code(KeyCode::AltLeft | KeyCode::AltRight) => {
                        self.alt_held = pressed;
                        // Releasing Alt always cancels an active tumble drag.
                        if !pressed {
                            self.tumble.active = false;
                        }
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        self.shift_held = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyF) => {
                        self.f_held = pressed;
                    }
                    _ => {}
                }
            }

            // ── Scroll wheel ──────────────────────────────────────────────────
            // Priority: Alt+Scroll = zoom, F+Scroll = brush resize.
            // Both normalise the delta the same way so they feel consistent.
            WindowEvent::MouseWheel { delta, .. } => {
                // Normalise line-based and pixel-based deltas to a signed
                // line count so behaviour is consistent across input devices.
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y)  => y,
                    MouseScrollDelta::PixelDelta(pos)  => (pos.y / 20.0) as f32,
                };

                if self.alt_held {
                    // Alt + Scroll → zoom camera in/out.
                    self.tumble.zoom(lines);
                    self.update_camera();
                } else if self.f_held {
                    // F + Scroll → brush size; Shift+F + Scroll → hardness gap.
                    if self.shift_held {
                        self.brush.scroll_hardness(lines);
                    } else {
                        self.brush.scroll_size(lines);
                    }
                }
                self.request_redraw();
            }

            // ── Left mouse button press ───────────────────────────────────────
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.alt_held {
                    // Alt + LMB begins camera orbit.
                    self.tumble.active = true;
                } else {
                    // Plain click: ray-cast and log the surface hit.
                    let (vp_w, vp_h) = self.vp_size();
                    let vp = self.scene
                        .as_ref()
                        .map(|s| build_view_proj(s, vp_w, vp_h))
                        .unwrap_or(Mat4::IDENTITY);

                    let px = self.brush.cursor_px.x;
                    let py = self.brush.cursor_px.y;

                    match raycast_sphere(px, py, vp_w, vp_h, vp) {
                        Some((hit, n)) => println!(
                            "[brush] hit ({:.3},{:.3},{:.3}) \
                             normal ({:.3},{:.3},{:.3}) \
                             inner={:.3} outer={:.3}",
                            hit.x, hit.y, hit.z,
                            n.x,   n.y,   n.z,
                            self.brush.inner_r,
                            self.brush.outer_r(),
                        ),
                        None => println!("[brush] miss"),
                    }
                }
            }

            // ── Left mouse button release ─────────────────────────────────────
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.tumble.active = false;
            }

            // ── Cursor movement ───────────────────────────────────────────────
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);

                self.brush.cursor_px      = Vec2::new(x as f32, y as f32);
                self.brush.cursor_visible = true;

                // Update camera angles while a tumble drag is active.
                if self.tumble.active {
                    let dx = (x - self.tumble.last_x) as f32;
                    let dy = (y - self.tumble.last_y) as f32;

                    self.tumble.yaw   -= dx * 0.5;
                    self.tumble.pitch  = (self.tumble.pitch + dy * 0.3).clamp(-89.0, 89.0);

                    self.update_camera();
                }

                // Always update last_x/y so the first drag frame does not jump.
                self.tumble.last_x = x;
                self.tumble.last_y = y;

                self.request_redraw();
            }

            // ── Cursor enters / leaves the window ────────────────────────────
            WindowEvent::CursorLeft { .. } => {
                self.brush.cursor_visible = false;
                self.request_redraw();
            }
            WindowEvent::CursorEntered { .. } => {
                self.brush.cursor_visible = true;
                self.request_redraw();
            }

            // ── Window resize ─────────────────────────────────────────────────
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.resize(size.width, size.height);
                }
                self.request_redraw();
            }

            // ── Render ────────────────────────────────────────────────────────
            WindowEvent::RedrawRequested => {
                let brush_uniform = self.make_brush_uniform();

                if let Some(ctx) = &mut self.ctx {
                    ctx.render(self.scene.as_ref(), brush_uniform.as_ref());
                }

                // FPS counter: accumulate frames and update title bar once per second.
                self.fps_frame_count += 1;
                let elapsed = self.fps_last_time.elapsed().as_secs_f64();

                if elapsed >= 1.0 {
                    self.fps             = self.fps_frame_count as f64 / elapsed;
                    self.fps_frame_count = 0;
                    self.fps_last_time   = Instant::now();

                    if let Some(w) = self.window {
                        // Show a context hint while brush-resize keys are held.
                        let hint = if self.f_held {
                            if self.shift_held { "  Shift+F+scroll=hardness" }
                            else               { "  F+scroll=size" }
                        } else {
                            ""
                        };

                        w.set_title(&format!(
                            "malx — {:.0} fps  |  size {:.3}  gap {:.3}{}",
                            self.fps, self.brush.inner_r, self.brush.gap_r, hint,
                        ));
                    }
                }

                // Re-queue immediately (ControlFlow::Poll => render every tick).
                self.request_redraw();
            }

            _ => {}
        }
    }
}

fn main() {
    // Initialise env_logger so wgpu validation messages appear in the terminal
    // when RUST_LOG is set (e.g. `RUST_LOG=wgpu=warn cargo run`).
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    // Poll mode: the event loop spins continuously for maximum frame rate.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
