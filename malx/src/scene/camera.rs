use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Camera uniform uploaded to the GPU every frame.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub eye:       [f32; 3],
    pub _pad:      f32,
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera, aspect: f32) -> Self {
        let view = Mat4::look_at_rh(camera.eye, camera.target, camera.up);
        let proj = Mat4::perspective_rh(
            camera.fov_y.to_radians(),
            aspect,
            camera.z_near,
            camera.z_far,
        );
        Self {
            view_proj: (proj * view).to_cols_array_2d(),
            eye:       camera.eye.to_array(),
            _pad:      0.0,
        }
    }
}

/// Camera parameters (CPU side).
pub struct Camera {
    pub eye:    Vec3,
    pub target: Vec3,
    pub up:     Vec3,
    pub fov_y:  f32, // degrees
    pub z_near: f32,
    pub z_far:  f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye:    Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up:     Vec3::Y,
            fov_y:  45.0,
            z_near: 0.1,
            z_far:  100.0,
        }
    }
}
