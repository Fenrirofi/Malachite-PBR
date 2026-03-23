//! A single renderable object placed in the scene.

use glam::Mat4;
use crate::geometry::GpuMesh;

/// A renderable scene object: a GPU mesh plus a model-to-world transform.
///
/// Multiple objects may share the same mesh data if needed in the future.
pub struct SceneObject {
    /// GPU-resident vertex and index data.
    pub mesh: GpuMesh,
    /// Model-to-world transformation matrix.
    ///
    /// Applied in the vertex shader alongside the camera view-projection.
    /// Use [`Mat4::IDENTITY`] to place the mesh directly at the world origin.
    pub transform: Mat4,
}

impl SceneObject {
    pub fn new(mesh: GpuMesh, transform: Mat4) -> Self {
        Self { mesh, transform }
    }
}
