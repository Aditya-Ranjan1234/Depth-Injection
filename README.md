# Depth Injection

An interactive browser demo that uses a webcam and WebGPU shaders to create a depth-aware light effect. Drag the light across the camera view to explore the result.

## Requirements

- Chrome or Edge with WebGPU enabled
- A webcam, with permission granted to the site
- HTTPS in production (GitHub Pages provides this automatically)

## Run locally

```powershell
npm install
npm run dev
```

Open the local URL printed by Vite, allow camera access, and drag on the canvas.

## Production build

```powershell
npm run build
npm run preview
```

The deployable output is written to `dist/`. The Vite configuration uses relative asset paths, so the build works from the GitHub Pages project path `/Depth-Injection/`.

## GitHub Pages deployment

In GitHub, open **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/ (root)`, then save. Because GitHub Pages serves the repository root, copy the contents of `dist/` to a deployment branch or configure a Pages workflow to publish `dist/`.

For the simplest setup, use the included build output with a GitHub Actions Pages workflow, or publish `dist/` through the Pages settings if your repository configuration supports it.

## Project layout

```text
src/       TypeScript application and WebGPU shaders
public/    Static assets and local model files
dist/      Generated production build (ignored by Git)
```
