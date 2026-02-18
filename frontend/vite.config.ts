import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8501',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://localhost:8501',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
