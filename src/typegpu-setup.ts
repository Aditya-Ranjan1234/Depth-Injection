// Pure WebGPU setup — no TypeGPU wrapping needed for this setup step.
export async function initWebGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported in this browser. Use Chrome or Edge.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found.');

  const device = await adapter.requestDevice();

  // Listen for GPU errors so they show in console
  device.lost.then((info) => {
    console.error('GPU device lost:', info.message);
  });

  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  // NOTE: We call configure() AFTER canvas width/height is already set.
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'opaque', // 'opaque' is safest — no transparency issues
  });

  return { device, context, presentationFormat };
}