// ── Uniforms ──────────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj : mat4x4<f32>,
    eye       : vec3<f32>,
    _pad      : f32,
}

struct ModelUniform {
    model      : mat4x4<f32>,
    normal_mat : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera : CameraUniform;
@group(1) @binding(0) var<uniform> model  : ModelUniform;

// ── Vertex I/O ────────────────────────────────────────────────────────────────

struct VertexInput {
    @location(0) position  : vec3<f32>,
    @location(1) normal    : vec3<f32>,
    @location(2) uv        : vec2<f32>,
    @location(3) tangent   : vec3<f32>,
    @location(4) bitangent : vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos    : vec4<f32>,
    @location(0)       world_pos   : vec3<f32>,
    @location(1)       world_norm  : vec3<f32>,
    @location(2)       uv          : vec2<f32>,
    @location(3)       world_tan   : vec3<f32>,
    @location(4)       world_bitan : vec3<f32>,
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

// ── GGX / Cook-Torrance BRDF ─────────────────────────────────────────────────
//
// We use the standard microfacet model:
//
//   f(l, v) = (D * G * F) / (4 * dot(N,L) * dot(N,V))
//
// where:
//   D — GGX normal distribution function (NDF)
//   G — Smith height-correlated masking-shadowing term
//   F — Schlick Fresnel approximation
//
// Everything is computed in linear colour space.  Output is raw HDR radiance
// written into an Rgba16Float offscreen texture.  ACES tonemapping and gamma
// correction are applied in a separate fullscreen pass (tonemapping.rs).

const PI : f32 = 3.14159265358979;

// ── Placeholder material ──────────────────────────────────────────────────────
// Replace these constants with texture samples once PBR maps are wired up.
const BASE_COLOR : vec3<f32> = vec3<f32>(0.6, 0.7, 0.9);   // albedo (linear sRGB)
const METALLIC   : f32       = 0.0;                          // 0 = dielectric, 1 = metal
const ROUGHNESS  : f32       = 0.4;                          // perceptual roughness
const AO         : f32       = 1.0;                          // ambient occlusion factor

// ── Scene lights ──────────────────────────────────────────────────────────────
// Directional key-light + cooler fill-light. Values are in lux (HDR).
const LIGHT_DIR_0   : vec3<f32> = vec3<f32>( 1.0,  2.0,  1.0);
const LIGHT_COLOR_0 : vec3<f32> = vec3<f32>(23.47, 21.31, 20.79); // warm key
const LIGHT_DIR_1   : vec3<f32> = vec3<f32>(-1.0, -0.5, -0.8);
const LIGHT_COLOR_1 : vec3<f32> = vec3<f32>( 2.0,  2.2,  2.5);   // cool fill

// ── GGX Normal Distribution Function (Trowbridge-Reitz) ──────────────────────
// D(h) = α² / (π · (dot(N,H)² · (α² − 1) + 1)²)
// Gives the fraction of microfacets aligned with the half-vector H.
fn D_GGX(N_dot_H: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness; // Disney remapping: perceptual → linear α
    let a2 = a * a;
    let d  = N_dot_H * N_dot_H * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// ── Smith G₁ (Schlick-GGX) ───────────────────────────────────────────────────
// Approximates the probability that a microfacet visible from direction X is
// not shadowed or masked. Applied separately for V and L then multiplied.
fn G1_SchlickGGX(N_dot_X: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;   // Disney k for direct lighting
    return N_dot_X / (N_dot_X * (1.0 - k) + k);
}

// ── Smith height-correlated masking-shadowing ─────────────────────────────────
fn G_Smith(N_dot_V: f32, N_dot_L: f32, roughness: f32) -> f32 {
    return G1_SchlickGGX(N_dot_V, roughness)
         * G1_SchlickGGX(N_dot_L, roughness);
}

// ── Schlick Fresnel ───────────────────────────────────────────────────────────
// Approximates how reflectance increases as the viewing angle grazes the surface.
// F(v,h) = F₀ + (1 − F₀) · (1 − dot(V,H))⁵
fn F_Schlick(V_dot_H: f32, F0: vec3<f32>) -> vec3<f32> {
    let x  = clamp(1.0 - V_dot_H, 0.0, 1.0);
    let x5 = x * x * x * x * x;
    return F0 + (1.0 - F0) * x5;
}

// ── Full Cook-Torrance BRDF for a single punctual light ───────────────────────
fn cook_torrance(
    N        : vec3<f32>,  // surface normal (unit)
    V        : vec3<f32>,  // direction to camera (unit)
    L        : vec3<f32>,  // direction to light (unit)
    albedo   : vec3<f32>,
    metallic : f32,
    roughness: f32,
    F0       : vec3<f32>,  // reflectance at normal incidence
) -> vec3<f32> {
    let H = normalize(V + L);

    let N_dot_L = max(dot(N, L), 0.0);
    let N_dot_V = max(dot(N, V), 1e-4);
    let N_dot_H = max(dot(N, H), 0.0);
    let V_dot_H = max(dot(V, H), 0.0);

    // Early-out: light is behind the surface.
    if N_dot_L <= 0.0 { return vec3<f32>(0.0); }

    let D = D_GGX(N_dot_H, roughness);
    let G = G_Smith(N_dot_V, N_dot_L, roughness);
    let F = F_Schlick(V_dot_H, F0);

    // Specular lobe: (D·G·F) / (4·N·V·N·L)
    let specular = (D * G * F) / max(4.0 * N_dot_V * N_dot_L, 1e-4);

    // Diffuse lobe: Lambertian, energy-conserving.
    // k_d is the refracted (non-reflected) fraction; metals absorb it entirely.
    let k_d     = (1.0 - F) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;

    return (diffuse + specular) * N_dot_L;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
//
// Outputs raw HDR radiance into an Rgba16Float render target.
// Tone-mapping and gamma correction are applied in a separate fullscreen pass.

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_norm);
    let V = normalize(camera.eye - in.world_pos);

    let roughness = clamp(ROUGHNESS, 0.04, 1.0);

    // Base reflectance at normal incidence.
    // Dielectrics use a constant ~0.04; metals tint it with the albedo.
    let F0 = mix(vec3<f32>(0.04), BASE_COLOR, METALLIC);

    // Accumulate outgoing radiance from all punctual lights.
    var Lo = vec3<f32>(0.0);

    Lo += cook_torrance(N, V, normalize(LIGHT_DIR_0), BASE_COLOR, METALLIC, roughness, F0)
        * LIGHT_COLOR_0;

    Lo += cook_torrance(N, V, normalize(LIGHT_DIR_1), BASE_COLOR, METALLIC, roughness, F0)
        * LIGHT_COLOR_1;

    // Ambient term: a simple IBL stand-in until a real environment map is added.
    let ambient = vec3<f32>(0.03) * BASE_COLOR * AO;

    // Raw HDR radiance — no tonemapping, no gamma.
    // A separate fullscreen ACES pass reads this Rgba16Float texture.
    let color = ambient + Lo;

    return vec4<f32>(color, 1.0);
}
