use glam::{Mat4, Vec3};
use malx::{Camera, RenderContext, Scene, SceneObject, sphere};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

struct Tumble {
    active: bool,
    last_x: f64,
    last_y: f64,
    yaw:    f32, 
    pitch:  f32, 
}

impl Tumble {
    fn new() -> Self {
        Self { active: false, last_x: 0.0, last_y: 0.0, yaw: 0.0, pitch: 20.0 }
    }

    fn eye(&self, target: Vec3, distance: f32) -> Vec3 {
        let yaw   = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();

        // Pozycja kamery na sferze wokół target
        let x = distance * pitch.cos() * yaw.sin();
        let y = distance * pitch.sin();
        let z = distance * pitch.cos() * yaw.cos();

        target + Vec3::new(x, y, z)
    }

    fn up(&self) -> Vec3 {
        
        let pitch = self.pitch.to_radians();
        let yaw   = self.yaw.to_radians();

        let x = -pitch.sin() * yaw.sin();
        let y =  pitch.cos();
        let z = -pitch.sin() * yaw.cos();
        Vec3::new(x, y, z).normalize()
    }
}

struct App {
    window:          Option<&'static Window>,
    ctx:             Option<RenderContext<'static>>,
    scene:           Option<Scene>,
    tumble:          Tumble,
    alt_held:        bool,
    fps_last_time:   Instant,
    fps_frame_count: u32,
    fps:             f64,
}

impl App {
    fn new() -> Self {
        Self {
            window:          None,
            ctx:             None,
            scene:           None,
            tumble:          Tumble::new(),
            alt_held:        false,
            fps_last_time:   Instant::now(),
            fps_frame_count: 0,
            fps:             0.0,
        }
    }

    fn update_camera(&mut self) {
        if let Some(scene) = &mut self.scene {
            let target = Vec3::ZERO;
            scene.camera.eye    = self.tumble.eye(target, 3.0);
            scene.camera.target = target;
            scene.camera.up     = self.tumble.up();
        }
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

        let ctx = pollster::block_on(RenderContext::new(window));

        let mut scene = Scene::new(Camera::default());
        let sphere_mesh = sphere(1.0, 32, 16).upload(ctx.device());
        scene.add(SceneObject::new(sphere_mesh, Mat4::IDENTITY));

        self.ctx   = Some(ctx);
        self.scene = Some(scene);
        self.window = Some(window);
        self.update_camera();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            // Track Alt key
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                let pressed = key_event.state == ElementState::Pressed;
                if let PhysicalKey::Code(KeyCode::AltLeft | KeyCode::AltRight) = key_event.physical_key {
                    self.alt_held = pressed;
                    // Release tumble if alt released mid-drag
                    if !pressed {
                        self.tumble.active = false;
                    }
                }
            }

            // Alt + LMB press → start tumble
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                if self.alt_held {
                    self.tumble.active = state == ElementState::Pressed;
                }
            }

            // Mouse moved → rotate if tumbling
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);

                if self.tumble.active {
                    let dx = (x - self.tumble.last_x) as f32;
                    let dy = (y - self.tumble.last_y) as f32;

                    self.tumble.yaw   -= dx * 0.5;
                    self.tumble.pitch  = (self.tumble.pitch + dy * 0.3).clamp(-89.0, 89.0);

                    self.update_camera();
                    if let Some(w) = self.window { w.request_redraw(); }
                }

                self.tumble.last_x = x;
                self.tumble.last_y = y;
            }

            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.resize(size.width, size.height);
                }
                if let Some(w) = self.window { w.request_redraw(); }
            }

            WindowEvent::RedrawRequested => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.render(self.scene.as_ref());
                }

                self.fps_frame_count += 1;
                let elapsed = self.fps_last_time.elapsed().as_secs_f64();
                if elapsed >= 1.0 {
                    self.fps             = self.fps_frame_count as f64 / elapsed;
                    self.fps_frame_count = 0;
                    self.fps_last_time   = Instant::now();
                    if let Some(w) = self.window {
                        w.set_title(&format!("malx — {:.0} fps", self.fps));
                    }
                }

                if let Some(w) = self.window { w.request_redraw(); }
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