import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: Number(process.env.PORT) || 8080,
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(
    Boolean
  ),
  // Strip console.log / debug / info in production minification; keep
  // console.warn and console.error for observability of real problems.
  esbuild: {
    pure:
      mode === "production"
        ? ["console.log", "console.debug", "console.info"]
        : [],
  },
  preview: {
    allowedHosts: ["lerobot-lelab.hf.space"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
