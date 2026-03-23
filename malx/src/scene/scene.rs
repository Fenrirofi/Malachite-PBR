//! The scene: a collection of objects plus an active camera.

use super::SceneObject;
use super::camera::Camera;

/// Holds all renderable objects and the camera used to view them.
///
/// For now this is a simple flat list; a spatial acceleration structure
/// (e.g. BVH) can be added here when the object count grows.
pub struct Scene {
    pub camera:  Camera,
    pub objects: Vec<SceneObject>,
}

impl Scene {
    /// Creates a new empty scene with the given camera.
    pub fn new(camera: Camera) -> Self {
        Self { camera, objects: Vec::new() }
    }

    /// Appends an object to the scene.
    pub fn add(&mut self, object: SceneObject) {
        self.objects.push(object);
    }
}
