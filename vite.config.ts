import { defineConfig } from 'vite';

// Relative asset URLs keep the production build working at a GitHub Pages
// project URL as well as at the domain root.
export default defineConfig({
  base: './',
});
