import { defineConfig } from "vite";

export default defineConfig({
  root: "web",
  publicDir: false,
  build: { outDir: "../dist", emptyOutDir: true },
  server: { port: 8080, host: true },
  preview: { port: 8080, host: true },
});
