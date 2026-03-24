/// Application entry point for the `malx` 3-D sculpting viewer.
///
/// Kontrola kamery (styl Blender / Substance Painter):
///
/// | Akcja                    | Skrót                        |
/// |--------------------------|------------------------------|
/// | Orbit (obrót)            | Alt + LMB drag               |
/// | Pan (przesunięcie)       | Alt + MMB drag                |
/// | Zoom (dolly)             | Scroll                        |
/// | Focus / Frame            | F (natychmiastowy)            |
/// | Malowanie                | LMB drag (bez Alt)            |
/// | Widok front              | Numpad 1                     |
/// | Widok tył                | Numpad 9                     |
/// | Widok prawo              | Numpad 3                     |
/// | Widok lewo               | Numpad 7 (obrót 180°)        |
/// | Widok góra               | Numpad 7                     |
/// | Obrót o 15° (strzałki)   | Numpad 4/6/8/2               |
/// | Rozmiar pędzla           | F + Scroll                   |
/// | Twardość pędzla          | Shift + F + Scroll           |
use glam::{Mat4, Vec2, Vec3, Vec4};
use malx::{BrushUniform, Camera, RenderContext, Scene, SceneObject};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod loader;

// ── Stałe ─────────────────────────────────────────────────────────────────────

const ORBIT_SENSITIVITY: f32 = 0.35;   // stopni/piksel
const PAN_SENSITIVITY:   f32 = 0.0013; // świat/piksel (skalowane przez dist)
const ZOOM_SENSITIVITY:  f32 = 0.12;   // ułamek dist na notch
const INERTIA_DECAY:     f32 = 7.0;    // 1/s — wyżej = szybszy wygasz
const INERTIA_MIN:       f32 = 0.008;  // próg zatrzymania
const DIST_MIN:          f32 = 0.05;
const DIST_MAX:          f32 = 50.0;

// ── OrbitalCamera ─────────────────────────────────────────────────────────────

/// Profesjonalna kamera orbitalna z inercją, panem i presety numerycznymi.
struct OrbitalCamera {
    /// Punkt który kamera okrąża (przesuwany przez pan).
    target: Vec3,
    /// Kąt poziomy wokół osi Y świata [stopnie].
    yaw:    f32,
    /// Elewacja (pionowy) [stopnie], zaciskana do ±89°.
    pitch:  f32,
    /// Odległość oko → target.
    dist:   f32,

    // ── Stan dragu ────────────────────────────────────────────────────────────
    orbit_active: bool,
    pan_active:   bool,

    // ── Inercja ───────────────────────────────────────────────────────────────
    /// Prędkość orbitu [yaw_deg/s, pitch_deg/s].
    orbit_vel: Vec2,
    /// Prędkość panu [world/s].
    pan_vel:   Vec3,
    /// Znacznik czasu ostatniej klatki (do całkowania).
    last_tick: Instant,
}

impl OrbitalCamera {
    fn new() -> Self {
        Self {
            target:       Vec3::ZERO,
            yaw:          30.0,
            pitch:        20.0,
            dist:         3.0,
            orbit_active: false,
            pan_active:   false,
            orbit_vel:    Vec2::ZERO,
            pan_vel:      Vec3::ZERO,
            last_tick:    Instant::now(),
        }
    }

    // ── Wektory pochodne ──────────────────────────────────────────────────────

    fn eye(&self) -> Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        self.target + Vec3::new(
            self.dist * p.cos() * y.sin(),
            self.dist * p.sin(),
            self.dist * p.cos() * y.cos(),
        )
    }

    fn up(&self) -> Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        Vec3::new(-p.sin() * y.sin(), p.cos(), -p.sin() * y.cos()).normalize()
    }

    fn right(&self) -> Vec3 {
        (self.target - self.eye()).normalize().cross(self.up()).normalize()
    }

    // ── Orbit ─────────────────────────────────────────────────────────────────

    fn begin_orbit(&mut self) {
        self.orbit_active = true;
        self.orbit_vel    = Vec2::ZERO;
    }

    fn end_orbit(&mut self) {
        self.orbit_active = false;
    }

    /// Przesuń orbit o (dx,dy) pikseli. Zwraca `true` jeśli kamera się ruszyła.
    fn orbit_by(&mut self, dx: f32, dy: f32, dt: f32) -> bool {
        if !self.orbit_active { return false; }
        let dyaw   = -dx * ORBIT_SENSITIVITY;
        let dpitch =  dy * ORBIT_SENSITIVITY;
        self.yaw   += dyaw;
        self.pitch  = (self.pitch + dpitch).clamp(-89.0, 89.0);
        // EMA prędkości dla inercji.
        let vel = Vec2::new(dyaw / dt.max(1e-4), dpitch / dt.max(1e-4));
        self.orbit_vel = self.orbit_vel * 0.55 + vel * 0.45;
        true
    }

    // ── Pan ───────────────────────────────────────────────────────────────────

    fn begin_pan(&mut self) {
        self.pan_active = true;
        self.pan_vel    = Vec3::ZERO;
    }

    fn end_pan(&mut self) {
        self.pan_active = false;
    }

    fn pan_by(&mut self, dx: f32, dy: f32, dt: f32) -> bool {
        if !self.pan_active { return false; }
        let scale = self.dist * PAN_SENSITIVITY;
        let delta = self.right() * (-dx * scale) + self.up() * (dy * scale);
        self.target  += delta;
        let vel = delta / dt.max(1e-4);
        self.pan_vel = self.pan_vel * 0.55 + vel * 0.45;
        true
    }

    // ── Zoom ──────────────────────────────────────────────────────────────────

    fn zoom(&mut self, lines: f32) {
        self.dist = (self.dist * (1.0 - lines * ZOOM_SENSITIVITY))
            .clamp(DIST_MIN, DIST_MAX);
    }

    // ── Inercja ───────────────────────────────────────────────────────────────

    /// Całkuj inercję jedną klatkę. Zwraca `true` jeśli kamera wciąż się porusza.
    fn tick_inertia(&mut self) -> bool {
        let dt = self.last_tick.elapsed().as_secs_f32();
        self.last_tick = Instant::now();

        // Podczas aktywnego dragu użytkownik sam napędza ruch — nie dodawaj.
        if self.orbit_active || self.pan_active { return false; }

        let decay = (-INERTIA_DECAY * dt).exp();
        let mut moving = false;

        if self.orbit_vel.length() > INERTIA_MIN {
            let d = self.orbit_vel * dt;
            self.yaw   += d.x;
            self.pitch  = (self.pitch + d.y).clamp(-89.0, 89.0);
            self.orbit_vel *= decay;
            moving = true;
        } else {
            self.orbit_vel = Vec2::ZERO;
        }

        if self.pan_vel.length() > INERTIA_MIN {
            self.target  += self.pan_vel * dt;
            self.pan_vel *= decay;
            moving = true;
        } else {
            self.pan_vel = Vec3::ZERO;
        }

        moving
    }

    // ── Presety / focus ───────────────────────────────────────────────────────

    fn stop_inertia(&mut self) {
        self.orbit_vel = Vec2::ZERO;
        self.pan_vel   = Vec3::ZERO;
    }

    fn set_angles(&mut self, yaw: f32, pitch: f32) {
        self.yaw   = yaw;
        self.pitch  = pitch;
        self.stop_inertia();
    }

    fn nudge(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw   += dyaw;
        self.pitch  = (self.pitch + dpitch).clamp(-89.0, 89.0);
        self.stop_inertia();
    }

    /// Ustaw kamerę tak, by widziała sferę o promieniu `radius` wokół `center`.
    fn frame(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.dist   = (radius * 2.5).clamp(DIST_MIN, DIST_MAX);
        self.stop_inertia();
    }
}

// ── BrushState ────────────────────────────────────────────────────────────────

struct BrushState {
    inner_r:        f32,
    gap_r:          f32,
    scroll_speed:   f32,
    cursor_px:      Vec2,
    cursor_visible: bool,
}
impl BrushState {
    fn new() -> Self {
        Self { inner_r: 0.3, gap_r: 0.2, scroll_speed: 0.02,
               cursor_px: Vec2::ZERO, cursor_visible: false }
    }
    fn outer_r(&self)              -> f32 { self.inner_r + self.gap_r }
    fn scroll_size(&mut self, d: f32)     { self.inner_r = (self.inner_r + d * self.scroll_speed).max(0.01); }
    fn scroll_hardness(&mut self, d: f32) { self.gap_r   = (self.gap_r   + d * self.scroll_speed).max(0.0); }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_view_proj(scene: &Scene, vp_w: f32, vp_h: f32) -> Mat4 {
    let cam  = &scene.camera;
    let view = Mat4::look_at_rh(cam.eye, cam.target, cam.up);
    let proj = Mat4::perspective_rh(cam.fov_y.to_radians(), vp_w / vp_h, cam.z_near, cam.z_far);
    proj * view
}

fn raycast_sphere(px: f32, py: f32, vp_w: f32, vp_h: f32, vp: Mat4) -> Option<(Vec3, Vec3)> {
    let ndc_x =  (px / vp_w) * 2.0 - 1.0;
    let ndc_y = -(py / vp_h) * 2.0 + 1.0;
    let inv   = vp.inverse();
    let near  = inv * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let far_  = inv * Vec4::new(ndc_x, ndc_y,  1.0, 1.0);
    let o     = Vec3::from(near.truncate()) / near.w;
    let d     = (Vec3::from(far_.truncate()) / far_.w - o).normalize();
    let b = 2.0 * o.dot(d);
    let c = o.dot(o) - 1.0;
    let disc: f32 = b * b - 4.0 * c;
    if disc < 0.0 { return None; }
    let t = (-b - disc.sqrt()) * 0.5;
    if t < 0.0    { return None; }
    let hit = o + d * t;
    Some((hit, hit.normalize()))
}

// ── App ───────────────────────────────────────────────────────────────────────

struct App {
    window:  Option<&'static Window>,
    ctx:     Option<RenderContext<'static>>,
    scene:   Option<Scene>,
    cam:     OrbitalCamera,

    // Modifiers
    alt_held:   bool,
    shift_held: bool,
    f_held:     bool,

    /// True while LMB is held WITHOUT Alt — paint mode.
    painting_active: bool,

    // Cursor tracking (shared między kamerą i pędzlem)
    cursor_x: f64,
    cursor_y: f64,
    last_cursor_x: f64,
    last_cursor_y: f64,

    brush: BrushState,

    fps_last_time:   Instant,
    fps_frame_count: u32,
    fps:             f64,
}

impl App {
    fn new() -> Self {
        Self {
            window:         None,
            ctx:            None,
            scene:          None,
            cam:            OrbitalCamera::new(),
            alt_held:       false,
            shift_held:     false,
            f_held:         false,
            painting_active: false,
            cursor_x:       0.0,
            cursor_y:       0.0,
            last_cursor_x:  0.0,
            last_cursor_y:  0.0,
            brush:          BrushState::new(),
            fps_last_time:  Instant::now(),
            fps_frame_count:0,
            fps:            0.0,
        }
    }

    fn sync_camera(&mut self) {
        if let Some(s) = &mut self.scene {
            s.camera.eye    = self.cam.eye();
            s.camera.target = self.cam.target;
            s.camera.up     = self.cam.up();
        }
    }

    fn vp_size(&self) -> (f32, f32) {
        self.window
            .map(|w| { let s = w.inner_size(); (s.width as f32, s.height as f32) })
            .unwrap_or((1.0, 1.0))
    }

    fn request_redraw(&self) {
        if let Some(w) = self.window { w.request_redraw(); }
    }

    fn make_brush_uniform(&self) -> Option<BrushUniform> {
        if !self.brush.cursor_visible { return None; }
        let (vp_w, vp_h) = self.vp_size();
        let vp = self.scene
            .as_ref()
            .map(|s| build_view_proj(s, vp_w, vp_h))
            .unwrap_or(Mat4::IDENTITY);
        let (hit, normal) = match raycast_sphere(
            self.brush.cursor_px.x, self.brush.cursor_px.y, vp_w, vp_h, vp,
        ) {
            Some(r) => (Some(r.0), Some(r.1)),
            None    => (None, None),
        };
        Some(BrushUniform {
            hit, normal,
            inner_r:   self.brush.inner_r,
            outer_r:   self.brush.outer_r(),
            cursor_px: [self.brush.cursor_px.x, self.brush.cursor_px.y],
            viewport:  [vp_w, vp_h],
            eye:       self.cam.eye(),
            view_proj: vp,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("malx")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window: &'static Window =
            Box::leak(Box::new(event_loop.create_window(attrs).unwrap()));
        window.set_cursor_visible(false);

        let ctx   = pollster::block_on(RenderContext::new(window));
        let mut scene = Scene::new(Camera::default());

        let obj_path = std::path::Path::new("model/Koltuk.obj");
        let meshes = loader::load(obj_path).expect("nie można załadować OBJ");
        for mesh in meshes {
            let gpu = mesh.upload(ctx.device());
            scene.add(SceneObject::new(gpu, Mat4::IDENTITY));
        }

        self.ctx    = Some(ctx);
        self.scene  = Some(scene);
        self.window = Some(window);
        self.cam.frame(Vec3::ZERO, 1.0);
        self.sync_camera();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            // ── Klawiatura ────────────────────────────────────────────────────
            WindowEvent::KeyboardInput { event: ke, .. } => {
                let pressed = ke.state == ElementState::Pressed;
                match ke.physical_key {
                    PhysicalKey::Code(KeyCode::AltLeft | KeyCode::AltRight) => {
                        self.alt_held = pressed;
                        if !pressed {
                            // Alt released mid-drag: stop camera, don't start painting.
                            self.cam.end_orbit();
                            self.cam.end_pan();
                        }
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        self.shift_held = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyF) => {
                        if pressed && !self.f_held {
                            // Pierwsze naciśnięcie F → natychmiastowy focus.
                            self.cam.frame(Vec3::ZERO, 1.0);
                            self.sync_camera();
                            self.request_redraw();
                        }
                        self.f_held = pressed;
                    }
                    // Numpad – presety widoku
                    PhysicalKey::Code(KeyCode::Numpad1) if pressed => {
                        self.cam.set_angles(0.0,   0.0);
                        self.sync_camera(); self.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::Numpad9) if pressed => {
                        self.cam.set_angles(180.0, 0.0);
                        self.sync_camera(); self.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::Numpad3) if pressed => {
                        self.cam.set_angles(90.0,  0.0);
                        self.sync_camera(); self.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::Numpad7) if pressed => {
                        // Top: Numpad7
                        if self.alt_held {
                            self.cam.set_angles(0.0, -89.0); // bottom gdy Alt
                        } else {
                            self.cam.set_angles(0.0,  89.0);
                        }
                        self.sync_camera(); self.request_redraw();
                    }
                    // Strzałki – obrót o 15°
                    PhysicalKey::Code(KeyCode::Numpad4) if pressed => {
                        self.cam.nudge(-15.0,  0.0); self.sync_camera(); self.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::Numpad6) if pressed => {
                        self.cam.nudge( 15.0,  0.0); self.sync_camera(); self.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::Numpad8) if pressed => {
                        self.cam.nudge(0.0, -10.0); self.sync_camera(); self.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::Numpad2) if pressed => {
                        self.cam.nudge(0.0,  10.0); self.sync_camera(); self.request_redraw();
                    }
                    _ => {}
                }
            }

            // ── Scroll ────────────────────────────────────────────────────────
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => (p.y / 20.0) as f32,
                };
                if self.f_held {
                    if self.shift_held { self.brush.scroll_hardness(lines); }
                    else               { self.brush.scroll_size(lines); }
                } else {
                    self.cam.zoom(lines);
                    self.sync_camera();
                }
                self.request_redraw();
            }

            // ── Przyciski myszy ───────────────────────────────────────────────
            //
            //  Alt + LMB  → orbit
            //  Alt + MMB  → pan
            //  LMB alone  → paint (painting_active = true, kamera spoczywa)
            //
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed {
                            if self.alt_held {
                                // Alt+LMB → orbit
                                self.painting_active = false;
                                self.cam.end_pan();
                                self.cam.begin_orbit();
                            } else {
                                // LMB bez Alt → malowanie
                                self.cam.end_orbit();
                                self.cam.end_pan();
                                self.painting_active = true;
                            }
                        } else {
                            self.cam.end_orbit();
                            self.painting_active = false;
                        }
                    }
                    MouseButton::Middle => {
                        if pressed {
                            if self.alt_held {
                                // Alt+MMB → pan
                                self.cam.end_orbit();
                                self.cam.begin_pan();
                            }
                        } else {
                            self.cam.end_pan();
                        }
                    }
                    _ => {}
                }
            }

            // ── Ruch kursora ──────────────────────────────────────────────────
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                self.brush.cursor_px      = Vec2::new(x as f32, y as f32);
                self.brush.cursor_visible = true;

                let dx = (x - self.last_cursor_x) as f32;
                let dy = (y - self.last_cursor_y) as f32;
                self.last_cursor_x = x;
                self.last_cursor_y = y;

                let dt = self.cam.last_tick.elapsed().as_secs_f32();

                let moved = self.cam.orbit_by(dx, dy, dt)
                          | self.cam.pan_by(dx, dy, dt);

                if moved { self.sync_camera(); }
                self.request_redraw();
            }

            WindowEvent::CursorLeft { .. } => {
                self.brush.cursor_visible = false;
                self.cam.end_orbit();
                self.cam.end_pan();
                self.painting_active = false;
                self.request_redraw();
            }
            WindowEvent::CursorEntered { .. } => {
                self.brush.cursor_visible = true;
                self.request_redraw();
            }

            // ── Resize ────────────────────────────────────────────────────────
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx { ctx.resize(size.width, size.height); }
                self.request_redraw();
            }

            // ── Render ────────────────────────────────────────────────────────
            WindowEvent::RedrawRequested => {
                // Inercja — napędza redraw dopóki kamera coś kosztuje.
                if self.cam.tick_inertia() {
                    self.sync_camera();
                }

                let brush_uniform = self.make_brush_uniform();
                if let Some(ctx) = &mut self.ctx {
                    ctx.render(self.scene.as_ref(), brush_uniform.as_ref());
                }

                self.fps_frame_count += 1;
                let elapsed = self.fps_last_time.elapsed().as_secs_f64();
                if elapsed >= 1.0 {
                    self.fps             = self.fps_frame_count as f64 / elapsed;
                    self.fps_frame_count = 0;
                    self.fps_last_time   = Instant::now();
                    if let Some(w) = self.window {
                        let mode = if self.cam.pan_active    { "PAN" }
                              else if self.cam.orbit_active  { "ORBIT" }
                              else if self.painting_active   { "PAINT" }
                              else                           { "" };
                        let hint = if self.f_held {
                            if self.shift_held { "  Shift+F+scroll=hardness" }
                            else { "  F+scroll=size" }
                        } else { "" };
                        w.set_title(&format!(
                            "malx — {:.0} fps  |  sz {:.3}  gap {:.3}  dist {:.2}  {}{}",
                            self.fps, self.brush.inner_r, self.brush.gap_r,
                            self.cam.dist, mode, hint,
                        ));
                    }
                }
                self.request_redraw();
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}