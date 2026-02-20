import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: "html",
  timeout: 60_000,

  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
  },

  webServer: [
    {
      command:
        "cd ../backend && source ../venv/bin/activate && python -m uvicorn main:app --port 8501",
      port: 8501,
      reuseExistingServer: !process.env.CI,
      timeout: 15_000,
    },
    {
      command: "npm run dev",
      port: 5173,
      reuseExistingServer: !process.env.CI,
      timeout: 15_000,
    },
  ],
});
