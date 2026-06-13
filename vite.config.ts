import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_PAGES === "true" ? "/tiny-real-time-world-model/" : "/",
  server: {
    host: "127.0.0.1",
    port: 5173
  }
});

