# The Magic Flashlight: Understanding Depth-Aware Lighting!

Welcome! Have you ever wanted to shine a flashlight into your webcam and see the light bounce off your face? That's what we are building! We are going to use an **AI Brain** (a Machine Learning model) to guess how far away things are, and a **Graphics Card** (GPU) to draw the light.

Let's look at our code line by line and understand exactly what it does, like we are reading a storybook!

---

## 1. Waking up the AI (The `ml-real.ts` file)

This file is responsible for loading our AI brain from our local computer (not the internet!) and asking it to look at our camera.

```typescript
import { pipeline, env } from '@xenova/transformers';
```
> **What it does:** We are importing our AI tools. `pipeline` is like a magical factory line where images go in and data comes out. `env` holds the rules for our factory.

```typescript
env.allowRemoteModels = false;
```
> **What it does:** This tells our factory, "Do NOT go to the internet to find AI brains!" We want to use the brain we already downloaded locally so it works fast and offline.

```typescript
env.localModelPath = '/models/';
```
> **What it does:** This tells the factory exactly *where* to look for the brain. It looks in our `public/models/` folder.

```typescript
let depthEstimator: any = null;
```
> **What it does:** We create an empty box called `depthEstimator`. Soon, we will put our loaded AI brain inside this box.

```typescript
export async function initDepthModel() {
```
> **What it does:** This is a recipe to wake up our AI. `async` means this recipe takes some time (like waiting for a cake to bake), so the computer knows to wait for it.

```typescript
  depthEstimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf', { 
    device: 'webgpu' 
  });
}
```
> **What it does:** This is the heavy lifting! We ask the factory to build a `depth-estimation` brain (a brain that guesses distance). `device: 'webgpu'` tells the brain to use the super-fast Graphics Card (GPU) instead of the slow normal computer chip (CPU).

```typescript
export async function estimateDepth(imageSource: HTMLVideoElement, depthTexture: GPUTexture, device: GPUDevice) {
```
> **What it does:** This recipe asks the AI to guess distances. It takes three things: 
> 1. `imageSource`: The video from our webcam.
> 2. `depthTexture`: A blank piece of digital paper on the Graphics Card where we will write the answers.
> 3. `device`: The Graphics Card itself.

```typescript
  const result = await depthEstimator(imageSource);
```
> **What it does:** We tell the AI, "Look at this video frame!" The AI thinks for a split second and hands back a `result` containing a bunch of numbers representing distances.

```typescript
  const depthData = result.depth.data;
```
> **What it does:** We open the result and grab just the raw list of numbers (`depth.data`). Bright numbers mean "close", dark numbers mean "far away".

```typescript
  device.queue.writeTexture(
    { texture: depthTexture },
    depthData,
    { bytesPerRow: depthTexture.width * 4, rowsPerImage: depthTexture.height },
    { width: depthTexture.width, height: depthTexture.height, depthOrArrayLayers: 1 }
  );
}
```
> **What it does:** We take our list of numbers (which are currently sitting in the normal computer memory) and shove them straight into the Graphics Card's memory (`depthTexture`). The long math at the end just tells the GPU how big the digital paper is so it doesn't color outside the lines!

---

## 2. Setting the Stage (The `shaders.ts` file)

Shaders are tiny, super-fast programs that run on the Graphics Card. They decide what color every single pixel on your screen should be.

```typescript
import tgpu from 'typegpu';
import { vec3f, f32, vec2f, i32 } from 'typegpu/data';
```
> **What it does:** We import our drawing tools. `vec3f` means a box that holds 3 numbers (like X, Y, Z for 3D space).

```typescript
export const SceneUniforms = tgpu.struct({
  cameraFocalLength: vec3f,
  cameraPrincipalPoint: vec3f,
  lightPosition: vec3f,
});
```
> **What it does:** We create a blueprint for the "rules of our world". It holds the camera settings and where our virtual flashlight (`lightPosition`) is floating in space!

```typescript
export const reconstructPosition = tgpu['~unstable'].fn([vec2f, f32], vec3f).impl(`
    (coord: vec2f, depth: f32) -> vec3f {
      let intrinsics = sceneUniformsBuffer.cameraFocalLength;
      let center = sceneUniformsBuffer.cameraPrincipalPoint;
      
      let x = (coord.x - center.x) * depth / intrinsics.x;
      let y = (coord.y - center.y) * depth / intrinsics.y;
      
      return vec3f(x, y, depth);
    }
`);
```
> **What it does:** This is real math! The AI only gave us a flat picture with distances (like a topographic map). This shader takes a pixel's 2D position (X, Y) and its distance (Z), and uses triangles to figure out exactly where that pixel exists in a real 3D room. It turns a flat picture into a 3D sculpture!

```typescript
export const calculateNormal = tgpu['~unstable'].fn([vec2f, f32, f32, f32, f32, f32], vec3f).impl(`
    (coord: vec2f, centerDepth: f32, leftDepth: f32, rightDepth: f32, upDepth: f32, downDepth: f32) -> vec3f {
      // ... looks up the 3D positions of the pixel and its neighbors ...
      let dx = pR - pL;
      let dy = pD - pU;
      return normalize(cross(dx, dy));
    }
`);
```
> **What it does:** To know how light bounces off a surface, we need to know which way the surface is facing. A "Normal" is a little arrow pointing straight out from a surface. This code looks at a pixel, looks at the pixels next to it, and calculates which way the "hill" is sloping.

```typescript
export const calculateLighting = tgpu['~unstable'].fn([vec3f, vec3f], f32).impl(`
    (position: vec3f, normal: vec3f) -> f32 {
      let L = normalize(sceneUniformsBuffer.lightPosition - position);
      let diffuse = max(dot(normal, L), 0.0);
      return diffuse * 1.0;
    }
`);
```
> **What it does:** The Magic Flashlight! 
> 1. `L` calculates a line pointing from the pixel exactly toward our flashlight.
> 2. `dot(normal, L)` checks if the surface is facing the flashlight. If they point at each other, the math equals 1 (bright!). If they face away, it equals 0 (dark!).

---

## 3. The Movie Director (The `main.ts` file)

This file controls the whole show. It turns on the webcam, moves the light around, and tells the AI and the Graphics card when to do their jobs.

```typescript
const video = document.createElement('video');
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
video.srcObject = stream;
```
> **What it does:** It creates an invisible TV (`video`), asks your browser for permission to use the webcam, and plugs the webcam into the TV.

```typescript
const cameraTexture = device.createTexture({ ... });
const depthTexture = device.createTexture({ ... });
```
> **What it does:** It creates two digital canvases on the Graphics Card. `cameraTexture` will hold the actual colors of your face. `depthTexture` will hold the black-and-white distance map from the AI.

```typescript
await initDepthModel();
```
> **What it does:** Wait patiently for the AI brain to load before we start the movie.

```typescript
function frame() {
```
> **What it does:** This is a loop that runs 60 times every second to create smooth video!

```typescript
  tgpu.write(sceneUniformsBuffer, {
    lightPosition: tgpu.vec3f(Math.sin(time) * 300, Math.cos(time) * 300, 2.0),
  });
```
> **What it does:** We change the `lightPosition` slightly every frame using sine and cosine (circle math). This makes the virtual flashlight orbit around your head automatically!

```typescript
  device.queue.copyExternalImageToTexture({ source: video }, { texture: cameraTexture });
```
> **What it does:** Every frame, we quickly peel the picture off the invisible TV and stick it onto the Graphics Card's canvas (`cameraTexture`).

```typescript
  if (!isPredicting) {
    estimateDepth(video, depthTexture, device).then(...)
  }
```
> **What it does:** We ask the AI to guess the depth. But AI is sometimes a little slow! If we waited for it, the video would freeze and look choppy. So we tell the AI, "Take your time, let me know when you're done!" and we keep playing the video smoothly.

```typescript
  passEncoder.draw(3); 
```
> **What it does:** Finally, we tell the Graphics Card to draw a giant triangle that covers the whole screen. While it colors in this triangle, our Shaders run on every single pixel, mashing the video colors, the AI depth, and the spinning flashlight together to create the final magical image!

---
**And that is how we turn a flat webcam into a glowing 3D world, right inside the web browser!**
