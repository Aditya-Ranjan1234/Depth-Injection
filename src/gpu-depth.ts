export const depthComputeWGSL = /* wgsl */`
@group(0) @binding(0) var cameraTex : texture_2d<f32>;
@group(0) @binding(1) var depthTex : texture_storage_2d<r32float, write>;

fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(depthTex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let p = vec2<i32>(gid.xy);
  var sum = 0.0;
  var weightSum = 0.0;
  let centerUv = vec2f(gid.xy) / vec2f(dims);
  let centerBias = clamp(1.0 - distance(centerUv, vec2f(0.5, 0.42)) * 1.8, 0.0, 1.0);

  for (var oy = -2; oy <= 2; oy = oy + 1) {
    for (var ox = -2; ox <= 2; ox = ox + 1) {
      let q = clamp(p + vec2<i32>(ox, oy), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1), i32(dims.y - 1)));
      let sample = textureLoad(cameraTex, q, 0).rgb;
      let lum = luma(sample);
      let dist = f32(abs(ox) + abs(oy));
      let weight = 1.0 / (1.0 + dist);
      sum = sum + lum * weight;
      weightSum = weightSum + weight;
    }
  }

  let brightness = sum / weightSum;
  let depth = clamp(0.20 + brightness * 0.45 + centerBias * 0.10, 0.10, 0.85);

  textureStore(depthTex, p, vec4f(depth, 0.0, 0.0, 1.0));
}
`;
