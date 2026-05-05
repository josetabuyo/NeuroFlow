import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    watch: {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8510',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://localhost:8510',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
