//! Asset loaders — belongs in the app, not in the `malx` engine library.
//!
//! The engine (`malx`) deals with GPU representation of a [`Mesh`]; it doesn't
//! care where the data came from.  Loading is an application-level concern:
//! file paths, supported formats, error reporting to the user.
//!
//! # Usage
//! ```no_run
//! let meshes = loader::load("assets/helmet.glb").unwrap();
//! for mesh in meshes {
//!     let gpu = mesh.upload(&ctx.device);
//!     scene.add(SceneObject::new(gpu, Mat4::IDENTITY));
//! }
//! ```

pub mod gltf;
pub mod obj;

use std::{fmt, path::Path};
use malx::Mesh;

/// Unified error type for all loaders.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Obj(String),
    Gltf(String),
    UnsupportedFormat(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e)                  => write!(f, "I/O error: {e}"),
            Self::Obj(m)                 => write!(f, "OBJ error: {m}"),
            Self::Gltf(m)                => write!(f, "glTF error: {m}"),
            Self::UnsupportedFormat(ext) => write!(f, "unsupported format: .{ext}"),
        }
    }
}

impl std::error::Error for LoadError {}
impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

/// Loads meshes from a file, dispatching on extension (`.obj`, `.gltf`, `.glb`).
pub fn load(path: impl AsRef<Path>) -> Result<Vec<Mesh>, LoadError> {
    let path = path.as_ref();
    let ext  = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "obj"          => obj::load(path),
        "gltf" | "glb" => gltf::load(path),
        other          => Err(LoadError::UnsupportedFormat(other.to_owned())),
    }
}