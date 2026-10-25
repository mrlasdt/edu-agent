import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the browser talks to the Vite server and /chat is proxied to the
// gateway, so no CORS is needed. Override the target with GATEWAY_URL.
const gateway = process.env.GATEWAY_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/chat": gateway,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test-setup.ts",
    exclude: ["e2e/**", "node_modules/**"],
  },
});
