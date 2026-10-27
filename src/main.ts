import { initWebGPU } from './typegpu-setup';
import { createRenderPipeline } from './pipeline';
import { depthComputeWGSL } from './gpu-depth';

async function main() {
  const status = document.getElementById('status');
  const canvas = document.getElementById('render-canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('Missing <canvas id="render-canvas"> element.');

  const setStatus = (message: string) => {
    if (status) status.textContent = message;
    console.log(message);
  };
  
  // Set up a hidden video element to capture webcam
  const video = document.createElement('video');
  video.autoplay = true;
  video.style.display = 'none';
  document.body.appendChild(video);

  // Ask for webcam permission
  setStatus('Requesting camera access...');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: 'user',
    },
    audio: false,
  });
  video.srcObject = stream;
  // Wait for the video to actually have dimensions
  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(null);
    };
  });
  await video.play();
  setStatus('Camera ready. Initializing GPU...');

  // Get our WebGPU ready
  const { device, context, presentationFormat } = await initWebGPU(canvas);
  const renderPipeline = createRenderPipeline(device, presentationFormat);

  // Set canvas size to match video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const width = canvas.width;
  const height = canvas.height;

  // Create our textures
  const cameraTexture = device.createTexture({
    size: [width, height, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const depthTexture = device.createTexture({
    size: [width, height, 1],
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Create Uniform Buffer
  // 3 vec3f fields: cameraFocalLength, cameraPrincipalPoint, lightPosition
  // Each vec3f takes 16 bytes due to alignment (4 floats * 4 bytes). Total: 48 bytes.
  const uniformBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformData = new Float32Array(12); // 3 vec4s essentially

  // Create Bind Group
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: cameraTexture.createView() },
      { binding: 2, resource: depthTexture.createView() },
    ]
  });

  const depthComputePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: depthComputeWGSL }),
      entryPoint: 'cs_main',
    },
  });

  const depthComputeBindGroup = device.createBindGroup({
    layout: depthComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: cameraTexture.createView() },
      { binding: 1, resource: depthTexture.createView() },
    ],
  });

  const light = {
    x: 0.2,
    y: -0.1,
    z: 0.8,
  };
  let draggingLight = false;
  const pointerToLight = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((clientY - rect.top) / rect.height) * 2 - 1;
    light.x = -nx * 0.8;
    light.y = -ny * 0.8;
  };

  canvas.addEventListener('pointerdown', (event) => {
    draggingLight = true;
    canvas.setPointerCapture(event.pointerId);
    pointerToLight(event.clientX, event.clientY);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!draggingLight) return;
    pointerToLight(event.clientX, event.clientY);
  });
  canvas.addEventListener('pointerup', (event) => {
    draggingLight = false;
    canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
  }, { passive: false });

  canvas.style.cursor = 'grab';
  canvas.addEventListener('pointerdown', () => {
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointerup', () => {
    canvas.style.cursor = 'grab';
  });

  setStatus('Real-time GPU depth approximation running. Drag on the canvas to move the circular light.');

  function frame() {
    // Update Uniforms
    // cameraFocalLength (x, y, z, pad)
    uniformData[0] = 500.0; uniformData[1] = 500.0; uniformData[2] = 0.0;  uniformData[3] = 0.0;
    // cameraPrincipalPoint (x, y, z, pad)
    uniformData[4] = width / 2; uniformData[5] = height / 2; uniformData[6] = 0.0; uniformData[7] = 0.0;
    // Keep the light depth fixed so it behaves like a handheld circular lamp.
    uniformData[8] = light.x;
    uniformData[9] = light.y;
    uniformData[10] = light.z;
    uniformData[11] = 0.0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Copy the current video frame to the GPU texture if ready
    if (video.readyState >= 2 && video.videoWidth > 0) {
      device.queue.copyExternalImageToTexture(
        { source: video },
        { texture: cameraTexture },
        [width, height]
      );
    }

    const commandEncoder = device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(depthComputePipeline);
    computePass.setBindGroup(0, depthComputeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    computePass.end();

    const textureView = context.getCurrentTexture().createView();
    
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setBindGroup(0, renderBindGroup);
    passEncoder.draw(3); 
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  setStatus('Rendering. Drag on the canvas to move the light. Hold Shift and drag to move it in depth.');
  requestAnimationFrame(frame);
}

main().catch((error) => {
  console.error(error);
  const status = document.getElementById('status');
  if (status) {
    status.textContent = `Startup failed: ${error instanceof Error ? error.message : String(error)}`;
  }
});
