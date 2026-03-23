//! Procedural mesh generators for basic geometric shapes.

use std::f32::consts::PI;
use super::{Mesh, Vertex};

/// Generates a UV-sphere centred at the origin.
///
/// # Parameters
/// - `radius`  — radius in world units
/// - `sectors` — number of longitude subdivisions (clamped to ≥ 3, recommended 32)
/// - `stacks`  — number of latitude subdivisions  (clamped to ≥ 2, recommended 16)
///
/// # Winding
/// Counter-clockwise from the outside (matches the scene pipeline's CCW front
/// face setting with back-face culling enabled).
pub fn sphere(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let sectors = sectors.max(3);
    let stacks  = stacks.max(2);

    let mut vertices = Vec::new();
    let mut indices  = Vec::new();

    let sector_step = 2.0 * PI / sectors as f32; // longitude step (radians)
    let stack_step  = PI / stacks as f32;         // latitude step  (radians)

    // Generate vertices row by row from top (+π/2) to bottom (-π/2).
    for i in 0..=stacks {
        let stack_angle = PI / 2.0 - i as f32 * stack_step; // +π/2 → -π/2
        let xy = radius * stack_angle.cos(); // radius of this latitude circle
        let z  = radius * stack_angle.sin(); // height at this latitude

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;

            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            // For a sphere, the outward normal is simply the normalised position.
            vertices.push(Vertex {
                position: [x, y, z],
                normal:   [x / radius, y / radius, z / radius],
            });
        }
    }

    // Build indices: two triangles per quad, skipping degenerate polars.
    for i in 0..stacks {
        for j in 0..sectors {
            // Indices of the four corners of this quad.
            let k1 = i * (sectors + 1) + j;       // top-left
            let k2 = k1 + sectors + 1;             // bottom-left

            // Upper triangle (skipped at the top pole where it degenerates).
            if i != 0 {
                indices.push(k1);
                indices.push(k2);
                indices.push(k1 + 1);
            }
            // Lower triangle (skipped at the bottom pole where it degenerates).
            if i != stacks - 1 {
                indices.push(k1 + 1);
                indices.push(k2);
                indices.push(k2 + 1);
            }
        }
    }

    Mesh::new(vertices, indices)
}
