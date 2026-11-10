import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
  test: { include: ["test/**/*.test.ts"], exclude: ["worker/**", "node_modules/**"], testTimeout: 120000 },
} as any);
