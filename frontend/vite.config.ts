import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// In development `/api/*` is proxied to the FastAPI backend with the `/api`
// prefix stripped, so the browser talks to the same origin and the backend's
// loopback-only default auth keeps working. Production builds read
// VITE_API_BASE (see src/settings.ts) instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.API_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: 5173,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
