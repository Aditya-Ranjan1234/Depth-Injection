import { pipeline, env } from '@xenova/transformers';
import fs from 'fs';
import path from 'path';

// We want to save the models in the "public" folder so our web app can serve them locally!
const modelDir = path.join(process.cwd(), 'public', 'models');
if (!fs.existsSync(modelDir)) fs.mkdirSync(modelDir, { recursive: true });

env.cacheDir = modelDir;

console.log('Starting local model download from Hugging Face...');
try {
    // This will fetch the ONNX model files and save them locally in public/models
    await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf', { quantized: true });
    console.log('✅ Model successfully downloaded locally!');
} catch (e) {
    console.error('Error downloading model:', e);
}
