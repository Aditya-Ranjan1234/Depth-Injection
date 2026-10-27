// WGSL Shader — runs 100% on the GPU.
// The vertex shader draws one giant triangle over the full screen.
// The fragment shader runs for every pixel and computes the lit color.

export const shadersWGSL = /* wgsl */`

// ─── Uniform Data (CPU sends these values to the GPU each frame) ───────────
// WGSL aligns vec3f to 16 bytes, so each vec3f field actually occupies 16 bytes.
// Layout (offsets in bytes):
//   0:  cameraFocalLength  (fx, fy, unused, pad)
//   16: cameraPrincipalPoint (cx, cy, unused, pad)
//   32: lightPosition (x, y, z, pad)
struct SceneUniforms {
  cameraFocalLength    : vec3f,  // bytes 0–11  (padded to 16)
  cameraPrincipalPoint : vec3f,  // bytes 16–27 (padded to 16)
  lightPosition        : vec3f,  // bytes 32–43 (padded to 16)
};

@group(0) @binding(0) var<uniform> scene        : SceneUniforms;
@group(0) @binding(1) var          cameraTexture : texture_2d<f32>;
@group(0) @binding(2) var          depthTexture  : texture_2d<f32>;

// ─── Reconstruct 3D position from pixel coord + depth ──────────────────────
// Camera intrinsics (focal length, principal point) let us "un-project" a 2D
// pixel back into 3D space, just like reversing a camera projection.
fn reconstructPosition(coord: vec2f, rawDepth: f32) -> vec3f {
  let depth = max(rawDepth, 0.001); // never divide by zero
  let fx = scene.cameraFocalLength.x;
  let fy = scene.cameraFocalLength.y;
  let cx = scene.cameraPrincipalPoint.x;
  let cy = scene.cameraPrincipalPoint.y;
  let x = (coord.x - cx) * depth / fx;
  let y = (coord.y - cy) * depth / fy;
  return vec3f(x, y, depth);
}

fn projectPosition(position: vec3f) -> vec2f {
  let fx = scene.cameraFocalLength.x;
  let fy = scene.cameraFocalLength.y;
  let cx = scene.cameraPrincipalPoint.x;
  let cy = scene.cameraPrincipalPoint.y;
  let z = max(position.z, 0.001);
  return vec2f(position.x * fx / z + cx, position.y * fy / z + cy);
}

// ─── Estimate surface normal via cross product of neighbour positions ───────
fn calculateNormal(
  coord     : vec2f,
  centerD   : f32,
  leftD     : f32,
  rightD    : f32,
  upD       : f32,
  downD     : f32
) -> vec3f {
  let pL = reconstructPosition(vec2f(coord.x - 1.0, coord.y), leftD);
  let pR = reconstructPosition(vec2f(coord.x + 1.0, coord.y), rightD);
  let pU = reconstructPosition(vec2f(coord.x, coord.y - 1.0), upD);
  let pD = reconstructPosition(vec2f(coord.x, coord.y + 1.0), downD);
  let dx = pR - pL;
  let dy = pD - pU;
  let n  = cross(dx, dy);
  if (length(n) < 0.00001) { return vec3f(0.0, 0.0, 1.0); }
  return normalize(n);
}

// ─── Diffuse lighting (Lambert's cosine law) ────────────────────────────────
fn calculateLighting(position: vec3f, normal: vec3f) -> f32 {
  let lightVec = scene.lightPosition - position;
  let dist2 = max(dot(lightVec, lightVec), 0.08);
  let L = normalize(lightVec);
  let diffuse = max(dot(normal, L), 0.0);
  let viewDir = normalize(vec3f(0.0, 0.0, 1.5) - position);
  let halfVec = normalize(L + viewDir);
  let specular = pow(max(dot(normal, halfVec), 0.0), 18.0) * 0.25;
  let attenuation = 0.35 / dist2;
  return diffuse * attenuation + specular * attenuation;
}

// ─── Vertex Shader ──────────────────────────────────────────────────────────
// We draw exactly 3 vertices (a big triangle that covers the whole screen).
// No vertex buffer needed — positions are computed from vertex_index.
@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
  // Two oversized triangles trick: just one triangle with coords outside [-1,1]
  // covers the entire NDC square.
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
  );
  return vec4f(pos[vIdx], 0.0, 1.0);
}

// ─── Fragment Shader ─────────────────────────────────────────────────────────
// Runs once per pixel. Reads from cameraTexture + depthTexture and outputs
// the final lit colour.
@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let ic = vec2<i32>(fragCoord.xy);

  // Sample the depth at this pixel and its 4 neighbours
  let d  = textureLoad(depthTexture, ic,                        0).r;
  let dL = textureLoad(depthTexture, ic + vec2<i32>(-1,  0),    0).r;
  let dR = textureLoad(depthTexture, ic + vec2<i32>( 1,  0),    0).r;
  let dU = textureLoad(depthTexture, ic + vec2<i32>( 0, -1),    0).r;
  let dD = textureLoad(depthTexture, ic + vec2<i32>( 0,  1),    0).r;

  // Sample original camera colour
  let color = textureLoad(cameraTexture, ic, 0).rgb;

  // Reconstruct 3D geometry
  let pos3d  = reconstructPosition(fragCoord.xy, d);
  let normal = calculateNormal(fragCoord.xy, d, dL, dR, dU, dD);

  // Lighting: diffuse + ambient so we can always see something
  let diffuse = calculateLighting(pos3d, normal);
  let ambient = 0.52;
  let lit     = clamp(color * (ambient + diffuse), vec3f(0.0), vec3f(1.0));
  let glow = clamp(vec3f(diffuse * 0.45), vec3f(0.0), vec3f(1.0));
  let lightPixel = projectPosition(scene.lightPosition);
  let lightDist = length(fragCoord.xy - lightPixel);
  let radius = 42.0;
  let core = smoothstep(radius, 0.0, lightDist);
  let halo = smoothstep(radius * 2.8, radius, lightDist) * 0.22;
  let lightColor = vec3f(1.0, 0.96, 0.82) * (core * 0.55 + halo);
  let shaded = mix(lit, glow, 0.08);

  return vec4f(clamp(shaded + lightColor, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;
