//! glTF 2.0 / GLB loader.
//!
//! # Attribute mapping
//! | glTF semantic  | [`Vertex`] field                                    |
//! |----------------|-----------------------------------------------------|
//! | `POSITION`     | `position`                                          |
//! | `NORMAL`       | `normal`                                            |
//! | `TEXCOORD_0`   | `uv`                                                |
//! | `TANGENT`      | `tangent` + `bitangent` (reconstructed from vec4 w) |
//!
//! If `TANGENT` is absent, tangents are computed from UV deltas (Lengyel).

use std::path::Path;
use ::gltf::{accessor::DataType, mesh::Mode, Semantic};
use malx::{Mesh, Vertex};
use super::LoadError;

pub fn load(path: &Path) -> Result<Vec<Mesh>, LoadError> {
    let (doc, buffers, _images) = ::gltf::import(path)
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

    let mut meshes = vec![];
    for gltf_mesh in doc.meshes() {
        for prim in gltf_mesh.primitives() {
            if prim.mode() != Mode::Triangles { continue; }
            meshes.push(convert_primitive(&prim, &buffers)?);
        }
    }

    if meshes.is_empty() {
        return Err(LoadError::Gltf("file contains no triangle primitives".into()));
    }
    Ok(meshes)
}

fn convert_primitive(
    prim: &::gltf::Primitive<'_>,
    buffers: &[::gltf::buffer::Data],
) -> Result<Mesh, LoadError> {
    let positions = read_vec3(prim, &Semantic::Positions, buffers)?
        .ok_or_else(|| LoadError::Gltf("primitive has no POSITION".into()))?;
    let n = positions.len();

    let normals      = read_vec3(prim, &Semantic::Normals,      buffers)?.unwrap_or_else(|| vec![[0.0,1.0,0.0]; n]);
    let uvs          = read_vec2(prim, &Semantic::TexCoords(0), buffers)?.unwrap_or_else(|| vec![[0.0;2]; n]);
    let raw_tangents = read_vec4(prim, &Semantic::Tangents,     buffers)?;
    let indices      = read_indices(prim, buffers)?.unwrap_or_else(|| (0..n as u32).collect());

    let mut vertices: Vec<Vertex> = (0..n).map(|i| Vertex {
        position:  positions[i],
        normal:    normals[i],
        uv:        uvs[i],
        tangent:   [1.0, 0.0, 0.0],
        bitangent: [0.0, 1.0, 0.0],
    }).collect();

    if let Some(ref tans) = raw_tangents {
        for (i, v) in vertices.iter_mut().enumerate() {
            let t  = [tans[i][0], tans[i][1], tans[i][2]];
            let hw = tans[i][3];
            let bt = cross3(v.normal, t);
            v.tangent   = t;
            v.bitangent = scale3(bt, hw);
        }
    } else {
        compute_tangents(&mut vertices, &indices);
    }

    Ok(Mesh::new(vertices, indices))
}

fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut tan_acc:  Vec<[f32;3]> = vec![[0.0;3]; vertices.len()];
    let mut btan_acc: Vec<[f32;3]> = vec![[0.0;3]; vertices.len()];

    for tri in indices.chunks_exact(3) {
        let (i0,i1,i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (p0,p1,p2) = (vertices[i0].position, vertices[i1].position, vertices[i2].position);
        let (u0,u1,u2) = (vertices[i0].uv,       vertices[i1].uv,       vertices[i2].uv);
        let e1 = sub3(p1,p0); let e2 = sub3(p2,p0);
        let du1=u1[0]-u0[0]; let dv1=u1[1]-u0[1];
        let du2=u2[0]-u0[0]; let dv2=u2[1]-u0[1];
        let det = du1*dv2 - du2*dv1;
        if det.abs() < 1e-8 { continue; }
        let r = 1.0/det;
        let t = scale3(add3(scale3(e1,dv2), scale3(e2,-dv1)), r);
        let b = scale3(add3(scale3(e2,du1), scale3(e1,-du2)), r);
        for &vi in &[i0,i1,i2] {
            tan_acc[vi]  = add3(tan_acc[vi],  t);
            btan_acc[vi] = add3(btan_acc[vi], b);
        }
    }

    for (i, v) in vertices.iter_mut().enumerate() {
        let n   = v.normal;
        let t   = tan_acc[i];
        let t_o = normalize3(sub3(t, scale3(n, dot3(t,n))));
        let cr  = cross3(n, t_o);
        let sgn = if dot3(cr, btan_acc[i]) < 0.0 { -1.0 } else { 1.0 };
        v.tangent   = t_o;
        v.bitangent = scale3(cr, sgn);
    }
}

// ─── Accessor helpers ─────────────────────────────────────────────────────────

fn read_vec3(prim: &::gltf::Primitive<'_>, sem: &Semantic, bufs: &[::gltf::buffer::Data]) -> Result<Option<Vec<[f32;3]>>, LoadError> {
    let acc = match prim.get(sem) { Some(a) => a, None => return Ok(None) };
    let view = acc.view().ok_or_else(|| LoadError::Gltf("sparse accessor not supported".into()))?;
    let data = &bufs[view.buffer().index()];
    let base = view.offset() + acc.offset();
    let stride = view.stride().unwrap_or(12);
    Ok(Some((0..acc.count()).map(|i| {
        let o = base + i*stride;
        [rf32(data,o,acc.data_type()), rf32(data,o+4,acc.data_type()), rf32(data,o+8,acc.data_type())]
    }).collect()))
}

fn read_vec2(prim: &::gltf::Primitive<'_>, sem: &Semantic, bufs: &[::gltf::buffer::Data]) -> Result<Option<Vec<[f32;2]>>, LoadError> {
    let acc = match prim.get(sem) { Some(a) => a, None => return Ok(None) };
    let view = acc.view().ok_or_else(|| LoadError::Gltf("sparse accessor not supported".into()))?;
    let data = &bufs[view.buffer().index()];
    let base = view.offset() + acc.offset();
    let stride = view.stride().unwrap_or(8);
    Ok(Some((0..acc.count()).map(|i| {
        let o = base + i*stride;
        [rf32(data,o,acc.data_type()), rf32(data,o+4,acc.data_type())]
    }).collect()))
}

fn read_vec4(prim: &::gltf::Primitive<'_>, sem: &Semantic, bufs: &[::gltf::buffer::Data]) -> Result<Option<Vec<[f32;4]>>, LoadError> {
    let acc = match prim.get(sem) { Some(a) => a, None => return Ok(None) };
    let view = acc.view().ok_or_else(|| LoadError::Gltf("sparse accessor not supported".into()))?;
    let data = &bufs[view.buffer().index()];
    let base = view.offset() + acc.offset();
    let stride = view.stride().unwrap_or(16);
    Ok(Some((0..acc.count()).map(|i| {
        let o = base + i*stride;
        [rf32(data,o,acc.data_type()), rf32(data,o+4,acc.data_type()), rf32(data,o+8,acc.data_type()), rf32(data,o+12,acc.data_type())]
    }).collect()))
}

fn read_indices(prim: &::gltf::Primitive<'_>, bufs: &[::gltf::buffer::Data]) -> Result<Option<Vec<u32>>, LoadError> {
    let acc = match prim.indices() { Some(a) => a, None => return Ok(None) };
    let view = acc.view().ok_or_else(|| LoadError::Gltf("sparse index accessor not supported".into()))?;
    let data = &bufs[view.buffer().index()];
    let base = view.offset() + acc.offset();
    let esz = match acc.data_type() { DataType::U8 => 1, DataType::U16 => 2, DataType::U32 => 4, o => return Err(LoadError::Gltf(format!("unsupported index type {o:?}"))) };
    let stride = view.stride().unwrap_or(esz);
    Ok(Some((0..acc.count()).map(|i| {
        let o = base + i*stride;
        match acc.data_type() {
            DataType::U8  => data[o] as u32,
            DataType::U16 => u16::from_le_bytes([data[o], data[o+1]]) as u32,
            DataType::U32 => u32::from_le_bytes([data[o], data[o+1], data[o+2], data[o+3]]),
            _ => unreachable!(),
        }
    }).collect()))
}

fn rf32(data: &[u8], o: usize, dtype: DataType) -> f32 {
    match dtype {
        DataType::F32 => f32::from_le_bytes([data[o],data[o+1],data[o+2],data[o+3]]),
        DataType::I8  => data[o] as i8 as f32 / 127.0,
        DataType::U8  => data[o] as f32 / 255.0,
        DataType::I16 => i16::from_le_bytes([data[o],data[o+1]]) as f32 / 32767.0,
        DataType::U16 => u16::from_le_bytes([data[o],data[o+1]]) as f32 / 65535.0,
        DataType::U32 => u32::from_le_bytes([data[o],data[o+1],data[o+2],data[o+3]]) as f32,
    }
}

// ─── Vector helpers ───────────────────────────────────────────────────────────

fn add3(a:[f32;3],b:[f32;3])->[f32;3]     { [a[0]+b[0],a[1]+b[1],a[2]+b[2]] }
fn sub3(a:[f32;3],b:[f32;3])->[f32;3]     { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
fn scale3(a:[f32;3],s:f32)  ->[f32;3]     { [a[0]*s,a[1]*s,a[2]*s] }
fn dot3(a:[f32;3],b:[f32;3])->f32         { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn cross3(a:[f32;3],b:[f32;3])->[f32;3]  { [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]] }
fn normalize3(v:[f32;3])->[f32;3]         { let l=(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt(); if l>1e-8{scale3(v,1.0/l)}else{[1.0,0.0,0.0]} }