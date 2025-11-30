import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Proxy is disabled when using remote backend (Cloud Run)
    // Uncomment below to use local backend instead
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:3001',
    //     changeOrigin: true
    //   }
    // }
  }
});

