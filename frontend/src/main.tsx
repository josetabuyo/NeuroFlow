import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'

// Minimal global reset
const style = document.createElement('style');
style.textContent = `
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  html, body, #root {
    height: 100%;
    width: 100%;
    overflow: hidden;
  }
  body {
    background: #0a0a0f;
    color: #e0e0ff;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  ::-webkit-scrollbar {
    width: 6px;
  }
  ::-webkit-scrollbar-track {
    background: #0a0a0f;
  }
  ::-webkit-scrollbar-thumb {
    background: #2a2a3e;
    border-radius: 3px;
  }
  @keyframes neuro-spin {
    to { transform: rotate(360deg); }
  }
  .neuro-spinner {
    width: 36px;
    height: 36px;
    margin: 0 auto;
    border: 3px solid #2a2a3e;
    border-top-color: #4cc9f0;
    border-radius: 50%;
    animation: neuro-spin 0.8s linear infinite;
  }
  .neuro-spinner-sm {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid #555;
    border-top-color: #e0e0ff;
    border-radius: 50%;
    animation: neuro-spin 0.8s linear infinite;
  }
`;
document.head.appendChild(style);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
