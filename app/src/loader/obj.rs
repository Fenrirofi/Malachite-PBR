//! Wavefront OBJ loader.
//!
//! # What is parsed
//! - `v`  — positions
//! - `vn` — normals
//! - `vt` — texture coordinates (UV)
//! - `f`  — faces (triangulated; quads and n-gons are fan-triangulated)
//! - `o` / `g` — object / group boundaries (each becomes a separate [`Mesh`])
//! - `mtllib` / `usemtl` — silently ignored (material data is not loaded)
//!
//! # Tangent frame
//! Tangents and bitangents are computed from UV deltas using the standard
//! Lengyel method (per-triangle contributions accumulated then normalised).
//! A Gram-Schmidt fallback handles degenerate UV maps (seams, poles).

use std::{collections::HashMap, fs, path::Path};
use malx::{Mesh, Vertex};
use super::LoadError;

pub fn load(path: &Path) -> Result<Vec<Mesh>, LoadError> {
    let source = fs::read_to_string(path).map_err(LoadError::Io)?;
    parse(&source)
}

struct RawData {
    positions: Vec<[f32; 3]>,
    normals:   Vec<[f32; 3]>,
    uvs:       Vec<[f32; 2]>,
}

struct SubMesh {
    name:  String,
    faces: Vec<[FaceVertex; 3]>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct FaceVertex { pos: u32, uv: u32, nor: u32 }

fn parse(source: &str) -> Result<Vec<Mesh>, LoadError> {
    let mut raw = RawData { positions: vec![], normals: vec![], uvs: vec![] };
    let mut meshes: Vec<SubMesh> = vec![];
    let mut current = SubMesh { name: "default".into(), faces: vec![] };

    for (line_no, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        let mut parts = line.splitn(2, char::is_whitespace);
        let keyword   = parts.next().unwrap_or("");
        let rest      = parts.next().unwrap_or("").trim();

        match keyword {
            "v"       => raw.positions.push(parse_vec3(rest, line_no)?),
            "vn"      => raw.normals.push(parse_vec3(rest, line_no)?),
            "vt"      => raw.uvs.push(parse_vec2(rest, line_no)?),
            "f"       => current.faces.extend(parse_face(rest, line_no)?),
            "o" | "g" => {
                if !current.faces.is_empty() { meshes.push(current); }
                current = SubMesh { name: rest.to_owned(), faces: vec![] };
            }
            _ => {}
        }
    }

    if !current.faces.is_empty() { meshes.push(current); }
    if meshes.is_empty() {
        return Err(LoadError::Obj("file contains no geometry".into()));
    }

    meshes.iter().map(|s| build_mesh(s, &raw)).collect()
}

fn build_mesh(sub: &SubMesh, raw: &RawData) -> Result<Mesh, LoadError> {
    let mut vertex_map: HashMap<u64, u32> = HashMap::new();
    let mut vertices:   Vec<Vertex>       = vec![];
    let mut indices:    Vec<u32>          = vec![];
    let mut tan_acc:    Vec<[f32; 3]>     = vec![];
    let mut btan_acc:   Vec<[f32; 3]>     = vec![];

    for tri in &sub.faces {
        let vi: [u32; 3] = std::array::from_fn(|c| {
            let fv  = tri[c];
            let key = (fv.pos as u64) | ((fv.uv as u64) << 20) | ((fv.nor as u64) << 40);
            *vertex_map.entry(key).or_insert_with(|| {
                let pos = raw.positions.get(fv.pos as usize).copied().unwrap_or([0.0; 3]);
                let nor = if fv.nor > 0 { raw.normals.get((fv.nor-1) as usize).copied().unwrap_or([0.0,1.0,0.0]) } else { [0.0,1.0,0.0] };
                let uv  = if fv.uv  > 0 { raw.uvs.get((fv.uv-1) as usize).copied().unwrap_or([0.0; 2]) } else { [0.0; 2] };
                let idx = vertices.len() as u32;
                vertices.push(Vertex { position: pos, normal: nor, uv, tangent: [1.0,0.0,0.0], bitangent: [0.0,1.0,0.0] });
                tan_acc.push([0.0; 3]);
                btan_acc.push([0.0; 3]);
                idx
            })
        });

        indices.extend_from_slice(&vi);

        // Lengyel tangent accumulation
        let (p0,p1,p2) = (vertices[vi[0] as usize].position, vertices[vi[1] as usize].position, vertices[vi[2] as usize].position);
        let (u0,u1,u2) = (vertices[vi[0] as usize].uv,       vertices[vi[1] as usize].uv,       vertices[vi[2] as usize].uv);
        let e1 = sub3(p1, p0); let e2 = sub3(p2, p0);
        let du1 = u1[0]-u0[0]; let dv1 = u1[1]-u0[1];
        let du2 = u2[0]-u0[0]; let dv2 = u2[1]-u0[1];
        let det = du1*dv2 - du2*dv1;
        if det.abs() > 1e-8 {
            let r = 1.0/det;
            let t = scale3(add3(scale3(e1, dv2), scale3(e2, -dv1)), r);
            let b = scale3(add3(scale3(e2, du1), scale3(e1, -du2)), r);
            for &i in &vi {
                tan_acc[i as usize]  = add3(tan_acc[i as usize],  t);
                btan_acc[i as usize] = add3(btan_acc[i as usize], b);
            }
        }
    }

    for (i, v) in vertices.iter_mut().enumerate() {
        let n     = v.normal;
        let t     = tan_acc[i];
        let t_orth = normalize3(sub3(t, scale3(n, dot3(t, n))));
        let cross  = cross3(n, t_orth);
        let sign   = if dot3(cross, btan_acc[i]) < 0.0 { -1.0 } else { 1.0 };
        v.tangent   = t_orth;
        v.bitangent = scale3(cross, sign);
    }

    Ok(Mesh::new(vertices, indices))
}

// ─── Face parsing ─────────────────────────────────────────────────────────────

fn parse_face(rest: &str, line_no: usize) -> Result<Vec<[FaceVertex; 3]>, LoadError> {
    let corners: Vec<FaceVertex> = rest.split_whitespace()
        .map(|tok| parse_face_vertex(tok, line_no))
        .collect::<Result<_, _>>()?;
    if corners.len() < 3 {
        return Err(LoadError::Obj(format!("line {}: face needs ≥3 vertices", line_no+1)));
    }
    Ok((1..corners.len()-1).map(|i| [corners[0], corners[i], corners[i+1]]).collect())
}

fn parse_face_vertex(tok: &str, line_no: usize) -> Result<FaceVertex, LoadError> {
    let err = || LoadError::Obj(format!("line {}: bad face token '{tok}'", line_no+1));
    let mut it = tok.split('/');
    let pos: i32 = it.next().ok_or_else(err)?.parse().map_err(|_| err())?;
    let uv_s     = it.next().unwrap_or("");
    let nor_s    = it.next().unwrap_or("");
    let uv:  u32 = if uv_s.is_empty()  { 0 } else { uv_s.parse().map_err(|_| err())? };
    let nor: u32 = if nor_s.is_empty() { 0 } else { nor_s.parse().map_err(|_| err())? };
    if pos <= 0 { return Err(LoadError::Obj(format!("line {}: non-positive index", line_no+1))); }
    Ok(FaceVertex { pos: (pos-1) as u32, uv, nor })
}

// ─── Primitive parsers ────────────────────────────────────────────────────────

fn parse_vec3(rest: &str, ln: usize) -> Result<[f32; 3], LoadError> {
    let e = || LoadError::Obj(format!("line {}: bad vec3", ln+1));
    let mut it = rest.split_whitespace();
    Ok([it.next().and_then(|s| s.parse().ok()).ok_or_else(e)?,
        it.next().and_then(|s| s.parse().ok()).ok_or_else(e)?,
        it.next().and_then(|s| s.parse().ok()).ok_or_else(e)?])
}

fn parse_vec2(rest: &str, ln: usize) -> Result<[f32; 2], LoadError> {
    let e = || LoadError::Obj(format!("line {}: bad vec2", ln+1));
    let mut it = rest.split_whitespace();
    Ok([it.next().and_then(|s| s.parse().ok()).ok_or_else(e)?,
        it.next().and_then(|s| s.parse().ok()).ok_or_else(e)?])
}

// ─── Vector helpers ───────────────────────────────────────────────────────────

fn add3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
fn sub3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn scale3(a: [f32;3], s: f32)   -> [f32;3] { [a[0]*s, a[1]*s, a[2]*s] }
fn dot3(a: [f32;3], b: [f32;3]) -> f32     { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn normalize3(v: [f32;3]) -> [f32;3] {
    let l = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if l > 1e-8 { scale3(v, 1.0/l) } else { [1.0, 0.0, 0.0] }
}