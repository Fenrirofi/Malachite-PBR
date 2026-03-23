struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct ModelUniform {
    model:      mat4x4<f32>,
    normal_mat: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> model:  ModelUniform;

struct VertexInput {
    @location(0) position  : vec3<f32>,
    @location(1) normal    : vec3<f32>,
    @location(2) uv        : vec2<f32>,
    @location(3) tangent   : vec3<f32>,
    @location(4) bitangent : vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos   : vec4<f32>,
    @location(0)       world_pos  : vec3<f32>,
    @location(1)       world_norm : vec3<f32>,
    @location(2)       uv         : vec2<f32>,
    @location(3)       world_tan  : vec3<f32>,
    @location(4)       world_bitan: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos   = model.model      * vec4<f32>(in.position,  1.0);
    let world_norm  = normalize((model.normal_mat * vec4<f32>(in.normal,    0.0)).xyz);
    let world_tan   = normalize((model.normal_mat * vec4<f32>(in.tangent,   0.0)).xyz);
    let world_bitan = normalize((model.normal_mat * vec4<f32>(in.bitangent, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_pos    = camera.view_proj * world_pos;
    out.world_pos   = world_pos.xyz;
    out.world_norm  = world_norm;
    out.uv          = in.uv;
    out.world_tan   = world_tan;
    out.world_bitan = world_bitan;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Prosta Blinn-Phong / ambient + diffuse
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.0));
    let diffuse   = max(dot(in.world_norm, light_dir), 0.0);
    let ambient   = 0.1;
    let color     = vec3<f32>(0.6, 0.7, 0.9); // placeholder kolor obiektu

    return vec4<f32>(color * (ambient + diffuse), 1.0);
}