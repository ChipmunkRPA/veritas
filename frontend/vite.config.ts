import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

/** `/api` goes to the dev API on :8160 unless VERITAS_API_TARGET names another one (the fake-detector API of `make e2e-fake`).
 *  `vite preview` serves the built UI through the same proxy. */
const apiProxy = { '/api': { target: process.env.VERITAS_API_TARGET ?? 'http://127.0.0.1:8160', changeOrigin: true } }

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5230, proxy: apiProxy },
  preview: { proxy: apiProxy },
  // assetsInlineLimit 0: small fonts are not inlined as data: URIs, so the PL-04 policy `default-src 'self'` covers them
  build: { outDir: 'dist', sourcemap: false, assetsInlineLimit: 0 },
})
