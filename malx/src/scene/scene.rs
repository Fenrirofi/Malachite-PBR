use super::SceneObject;
use super::camera::Camera;

/// Holds all objects and the active camera.
pub struct Scene {
    pub camera:  Camera,
    pub objects: Vec<SceneObject>,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self { camera, objects: Vec::new() }
    }

    pub fn add(&mut self, object: SceneObject) {
        self.objects.push(object);
    }
}
