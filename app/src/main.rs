use malx::RenderContext;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<&'static Window>,
    ctx: Option<RenderContext<'static>>,
    fps_last_time: Instant,
    fps_frame_count: u32,
    fps: f64,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            ctx: None,
            fps_last_time: Instant::now(),
            fps_frame_count: 0,
            fps: 0.0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes().with_title("Malachite");
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        let window: &'static Window = Box::leak(Box::new(window));
        let ctx = pollster::block_on(RenderContext::new(window));

        self.window = Some(window);
        self.ctx = Some(ctx);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.resize(size.width, size.height);
                }
                if let Some(w) = self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.render();
                }

                // --- FPS counter ---
                self.fps_frame_count += 1;
                let elapsed = self.fps_last_time.elapsed().as_secs_f64();
                if elapsed >= 1.0 {
                    self.fps = self.fps_frame_count as f64 / elapsed;
                    self.fps_frame_count = 0;
                    self.fps_last_time = Instant::now();

                    if let Some(w) = self.window {
                        w.set_title(&format!("malx — {:.0} fps", self.fps));
                    }
                }

                if let Some(w) = self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}