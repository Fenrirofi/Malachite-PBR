//! Procedural mesh generators for basic geometric shapes.

use std::f32::consts::PI;
use super::{Mesh, Vertex};

/// Generates a UV-sphere centred at the origin with full PBR vertex data.
///
/// Every vertex carries position, normal, UV coordinates, tangent, and
/// bitangent — everything needed for normal mapping and PBR texturing.
///
/// # Parameters
/// - `radius`  — radius in world units
/// - `sectors` — longitude subdivisions (≥ 3, recommended 32)
/// - `stacks`  — latitude subdivisions  (≥ 2, recommended 16)
///
/// # UV layout
/// U wraps around the equator (0 at –Z, 0.5 at +Z, 1 back at –Z).
/// V goes from 0 at the north pole to 1 at the south pole.
///
/// # Tangent frame
/// The tangent points in the +U direction (along longitude lines).
/// The bitangent is re-orthogonalised via Gram-Schmidt to stay perpendicular
/// to the normal, which avoids pole artifacts where the analytic tangent
/// degenerates.
///
/// # Winding
/// Counter-clockwise from the outside (matches CCW front-face + back-cull).
pub fn sphere(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let sectors = sectors.max(3);
    let stacks  = stacks.max(2);

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices:  Vec<u32>    = Vec::new();

    let sector_step = 2.0 * PI / sectors as f32;
    let stack_step  = PI / stacks as f32;

    // ── Vertex generation ─────────────────────────────────────────────────────
    //
    // We iterate stacks top-to-bottom (north pole first).
    // Each stack produces (sectors + 1) vertices — the +1 duplicates the seam
    // vertex so the UV wraps cleanly from u=0 to u=1.

    for i in 0..=stacks {
        let stack_angle = PI / 2.0 - i as f32 * stack_step; // +π/2 → -π/2
        let cos_stack   = stack_angle.cos();
        let sin_stack   = stack_angle.sin();

        let xy = radius * cos_stack; // projected radius at this latitude
        let z  = radius * sin_stack; // height at this latitude

        // V increases from 0 (north) to 1 (south).
        let v = i as f32 / stacks as f32;

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;
            let cos_sec = sector_angle.cos();
            let sin_sec = sector_angle.sin();

            let x = xy * cos_sec;
            let y = xy * sin_sec;

            // ── Position & normal ─────────────────────────────────────────────
            // On a unit sphere the outward normal equals the normalised position.
            let nx = x / radius;
            let ny = y / radius;
            let nz = z / radius;

            // ── UV ────────────────────────────────────────────────────────────
            // U increases eastward (in the direction of increasing sector_angle).
            let u = j as f32 / sectors as f32;

            // ── Analytic tangent (∂position/∂u, normalised) ───────────────────
            // Differentiating position w.r.t. sector_angle:
            //   ∂x/∂θ = -xy * sin θ
            //   ∂y/∂θ =  xy * cos θ
            //   ∂z/∂θ = 0
            // This degenerates at the poles (xy → 0), so we re-orthogonalise
            // below via Gram-Schmidt.
            let mut tx = -sin_sec;
            let mut ty =  cos_sec;
            let     tz =  0.0_f32;

            // ── Gram-Schmidt re-orthogonalisation ─────────────────────────────
            // t_ortho = normalize(t - dot(t, n) * n)
            // Removes any component of t that leaks along n, which is
            // especially important near the poles where cos_stack ≈ 0.
            let dot = tx * nx + ty * ny + tz * nz;
            tx -= dot * nx;
            ty -= dot * ny;
            // tz -= dot * nz  (tz is 0, stays 0 after subtraction)

            let t_len = (tx * tx + ty * ty + tz * tz).sqrt();
            let (tx, ty, tz) = if t_len > 1e-6 {
                (tx / t_len, ty / t_len, tz / t_len)
            } else {
                // Exact pole: tangent is undefined — use a stable fallback.
                // Choosing world X works unless the pole is exactly on X,
                // which can't happen because the poles are on the Z axis.
                gram_schmidt_fallback([nx, ny, nz])
            };

            // ── Bitangent = cross(normal, tangent) ────────────────────────────
            // n × t gives a right-handed TBN frame.
            let bx = ny * tz - nz * ty;
            let by = nz * tx - nx * tz;
            let bz = nx * ty - ny * tx;

            vertices.push(Vertex {
                position:  [x, y, z],
                normal:    [nx, ny, nz],
                uv:        [u, v],
                tangent:   [tx, ty, tz],
                bitangent: [bx, by, bz],
            });
        }
    }

    // ── Index generation ──────────────────────────────────────────────────────
    //
    // Two triangles per quad, with degenerate triangles skipped at the poles.

    for i in 0..stacks {
        for j in 0..sectors {
            let k1 = i * (sectors + 1) + j;   // top-left of quad
            let k2 = k1 + sectors + 1;         // bottom-left of quad

            // Upper triangle — skipped at north pole (would degenerate).
            if i != 0 {
                indices.push(k1);
                indices.push(k2);
                indices.push(k1 + 1);
            }
            // Lower triangle — skipped at south pole (would degenerate).
            if i != stacks - 1 {
                indices.push(k1 + 1);
                indices.push(k2);
                indices.push(k2 + 1);
            }
        }
    }

    Mesh::new(vertices, indices)
}

/// Picks a tangent that is guaranteed to be non-parallel to `normal`,
/// then projects out the normal component to get an orthogonal tangent.
///
/// Used only at the exact poles of a sphere where the analytic tangent
/// degenerates (length ≈ 0).
fn gram_schmidt_fallback(n: [f32; 3]) -> (f32, f32, f32) {
    // Try world X first; fall back to world Y if the normal is along X.
    let candidate = if n[0].abs() < 0.9 {
        [1.0_f32, 0.0, 0.0]
    } else {
        [0.0_f32, 1.0, 0.0]
    };

    let dot = candidate[0] * n[0] + candidate[1] * n[1] + candidate[2] * n[2];
    let tx  = candidate[0] - dot * n[0];
    let ty  = candidate[1] - dot * n[1];
    let tz  = candidate[2] - dot * n[2];

    let len = (tx * tx + ty * ty + tz * tz).sqrt();
    (tx / len, ty / len, tz / len)
}