//! Camera definitions: CPU-side parameters and the GPU uniform they produce.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

// ── GPU uniform ───────────────────────────────────────────────────────────────

/// Camera data uploaded to the GPU every frame.
///
/// The layout mirrors the `CameraUniform` struct in the scene WGSL shader.
/// The four-byte `_pad` field satisfies the 16-byte alignment rule for the
/// `eye` vector that follows `view_proj`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined view × projection matrix (column-major, ready for the shader).
    pub view_proj: [[f32; 4]; 4],
    /// World-space camera eye position (used in the Blinn-Phong specular term).
    pub eye: [f32; 3],
    /// Padding to reach the next 16-byte alignment boundary.
    pub _pad: f32,
}

impl CameraUniform {
    /// Builds a [`CameraUniform`] from a [`Camera`] and the viewport aspect ratio.
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

// ── CPU camera ────────────────────────────────────────────────────────────────

/// CPU-side camera parameters.
///
/// These are updated every frame by the application's tumble controller and
/// converted into a [`CameraUniform`] before being uploaded to the GPU.
pub struct Camera {
    /// World-space eye (camera) position.
    pub eye: Vec3,
    /// Point the camera looks at.
    pub target: Vec3,
    /// "Up" direction (must not be parallel to the view direction).
    pub up: Vec3,
    /// Vertical field of view in **degrees**.
    pub fov_y: f32,
    /// Near clip-plane distance.
    pub z_near: f32,
    /// Far clip-plane distance.
    pub z_far: f32,
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
