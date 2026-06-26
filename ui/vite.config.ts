import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 開発時は api/server.py (FastAPI, :8000) を /api と /ws にプロキシ。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
