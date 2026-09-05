import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build lands inside the Python package, and that output IS committed.
// The README's promise is `pip install -e . && revenew serve` with no other
// toolchain, and a judge cloning this repo should not need node to see the
// console. Source stays here in frontend/ so the build is reproducible; the
// artifact ships alongside it, the same way the LLM cassettes do.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../revenew/api/static',
    emptyOutDir: true,
    // One chunk. The console is four routes over a single JSON payload --
    // code-splitting it would trade a trivial size win for extra requests on
    // a machine that is often serving this over localhost during a demo.
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  server: {
    // `npm run dev` proxies the API to a `revenew serve` on the default port,
    // so the frontend can hot-reload against the real database.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
