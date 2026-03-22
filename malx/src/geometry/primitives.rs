use std::f32::consts::PI;
use super::{Mesh, Vertex};

/// UV-sphere centred at origin.
///
/// * `radius`  — radius in world units
/// * `sectors` — longitude subdivisions (≥ 3, recommended 32)
/// * `stacks`  — latitude  subdivisions (≥ 2, recommended 16)
pub fn sphere(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let sectors = sectors.max(3);
    let stacks  = stacks.max(2);

    let mut vertices = Vec::new();
    let mut indices  = Vec::new();

    let sector_step = 2.0 * PI / sectors as f32;
    let stack_step  = PI / stacks as f32;

    for i in 0..=stacks {
        let stack_angle = PI / 2.0 - i as f32 * stack_step; // +π/2 → -π/2
        let xy = radius * stack_angle.cos();
        let z  = radius * stack_angle.sin();

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;

            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            // Normal = position / radius (unit sphere)
            vertices.push(Vertex {
                position: [x, y, z],
                normal:   [x / radius, y / radius, z / radius],
            });
        }
    }

    // Indices — two triangles per quad
    for i in 0..stacks {
        for j in 0..sectors {
            let k1 = i * (sectors + 1) + j;
            let k2 = k1 + sectors + 1;

            // upper triangle (skip degenerate at top pole)
            if i != 0 {
                indices.push(k1);
                indices.push(k2);
                indices.push(k1 + 1);
            }
            // lower triangle (skip degenerate at bottom pole)
            if i != stacks - 1 {
                indices.push(k1 + 1);
                indices.push(k2);
                indices.push(k2 + 1);
            }
        }
    }

    Mesh::new(vertices, indices)
}
