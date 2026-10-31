import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// DASHBOARD_BASE is the URL path the site is served from: "/" for Vercel or a custom domain,
// "/<repo>/" for a GitHub Pages project site (see docs/deploy-dashboard.md).
export default defineConfig({
  base: process.env["DASHBOARD_BASE"] ?? "/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Fully static: the dashboard reads JSON from public/data, so no API server is needed.
  server: { port: 5173 },
  preview: { port: 4173 },
});
