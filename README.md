# Malachite

A 3D renderer written in Rust using [wgpu](https://github.com/gfx-rs/wgpu).

![Screenshot](screenshots/23.04.2026.png)

## Features

- **wgpu rendering backend** — works on Vulkan, Metal, DX12 and WebGPU
- **PBR-ready vertex format** — every mesh vertex carries position, normal, UV, tangent and bitangent (56 bytes, tightly packed)
- **Normal mapping support** — TBN matrix is computed per-vertex in the shader, ready to sample a normal map
- **Per-object model transform** — each object has its own model matrix + inverse-transpose normal matrix uploaded as a uniform
- **Perspective camera** — view/projection matrix and eye position sent to the GPU every frame
- **Procedural geometry** — UV sphere generator with correct tangent frames and seam-safe UV wrapping
- **Depth testing** — 32-bit depth buffer, CCW winding, back-face culling
- **Blinn-Phong shading** — ambient + diffuse lighting in the fragment shader (PBR textures ready to plug in)

## Project structure

```
malx/          # rendering library
  src/
    geometry/  # Vertex, Mesh, procedural primitives (sphere, ...)
    renderer/  # wgpu context, ScenePipeline, bind groups
    painter/   # high-level draw API
    scene/     # Camera, Object, Scene
  shaders/
    pbr_shader.wgsl
app/           # example application using malx
```

## Building

```bash
cargo run
```

Requires Rust stable and a GPU with Vulkan / Metal / DX12 support.

---

## Changelog

### 2026-03-23
- Dodano tangenty i bitangenty do vertex formatu (`Vertex`: position, normal, uv, tangent, bitangent — 56 bajtów)
- Shader (`pbr_shader.wgsl`) przepisany od nowa: poprawne `vs_main` i `fs_main`, bind groupy dla `CameraUniform` i `ModelUniform`
- TBN matrix przekazywana z vertex do fragment shadera — gotowość pod normal mapping
- Per-object `ModelUniform` z macierzą modelu i inverse-transpose dla normalnych