use glam::Mat4;
use crate::geometry::GpuMesh;

/// A single renderable object in the scene.
pub struct SceneObject {
    pub mesh:      GpuMesh,
    /// Model-to-world transform.
    pub transform: Mat4,
}

impl SceneObject {
    pub fn new(mesh: GpuMesh, transform: Mat4) -> Self {
        Self { mesh, transform }
    }
}
