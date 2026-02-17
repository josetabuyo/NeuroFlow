import { test, expect } from "@playwright/test";
import {
  waitForConnection,
  startExperiment,
  getActiveCount,
  getGeneration,
  clickCanvasCenter,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForConnection(page);
});

// ---------------------------------------------------------------------------
// 1. Carga inicial
// ---------------------------------------------------------------------------
test("1. Carga inicial", async ({ page }) => {
  // Title
  await expect(page.locator("h1")).toHaveText("NeuroFlow");

  // Two experiment buttons in the sidebar
  await expect(
    page.getByRole("button", { name: /Von Neumann/i })
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Kohonen/i })
  ).toBeVisible();

  // "Iniciar Experimento" button
  await expect(
    page.getByRole("button", { name: "Iniciar Experimento" })
  ).toBeVisible();

  // State shows "ready" (not "disconnected")
  await expect(page.getByText("ready")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 2. Iniciar Von Neumann
// ---------------------------------------------------------------------------
test("2. Iniciar Von Neumann", async ({ page }) => {
  await startExperiment(page, "von_neumann");

  // Canvas is visible
  await expect(page.locator("canvas")).toBeVisible();

  // Brush palette: 5 brush buttons + ON toggle
  await expect(page.getByRole("button", { name: "Punto" })).toBeVisible();
  await expect(page.getByRole("button", { name: "3×3" })).toBeVisible();
  await expect(page.getByRole("button", { name: "5×5" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Cruz" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Diamante" })).toBeVisible();
  await expect(page.getByRole("button", { name: "ON", exact: true })).toBeVisible();

  // Generation shows "0/50"
  const gen = await getGeneration(page);
  expect(gen).toBe("0/50");

  // Active cells = 0
  const active = await getActiveCount(page);
  expect(active).toBe(0);

  // Controls are enabled
  await expect(
    page.getByRole("button", { name: "Play" })
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Step" })
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Reset" })
  ).toBeEnabled();
});

// ---------------------------------------------------------------------------
// 3. Iniciar Kohonen
// ---------------------------------------------------------------------------
test("3. Iniciar Kohonen", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // Canvas visible
  await expect(page.locator("canvas")).toBeVisible();

  // Generation "0/30"
  const gen = await getGeneration(page);
  expect(gen).toBe("0/30");

  // Active cells > 0 (Kohonen starts with random neurons)
  const active = await getActiveCount(page);
  expect(active).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// 4. Paleta de pinceles — selección de pincel
// ---------------------------------------------------------------------------
test("4. Paleta de pinceles — selección de pincel", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // 5 brush buttons visible
  const punto = page.getByRole("button", { name: "Punto" });
  const tres = page.getByRole("button", { name: "3×3" });
  const cinco = page.getByRole("button", { name: "5×5" });
  const cruz = page.getByRole("button", { name: "Cruz" });
  const diamante = page.getByRole("button", { name: "Diamante" });

  await expect(punto).toBeVisible();
  await expect(tres).toBeVisible();
  await expect(cinco).toBeVisible();
  await expect(cruz).toBeVisible();
  await expect(diamante).toBeVisible();

  // Punto selected by default (cyan border)
  await expect(punto).toHaveCSS("border", "2px solid rgb(76, 201, 240)");

  // Click 3×3 → selected
  await tres.click();
  await expect(tres).toHaveCSS("border", "2px solid rgb(76, 201, 240)");
  // Punto deselected
  await expect(punto).not.toHaveCSS("border", "2px solid rgb(76, 201, 240)");

  // Click 5×5 → selected
  await cinco.click();
  await expect(cinco).toHaveCSS("border", "2px solid rgb(76, 201, 240)");
  await expect(tres).not.toHaveCSS("border", "2px solid rgb(76, 201, 240)");
});

// ---------------------------------------------------------------------------
// 5. Paleta de pinceles — toggle ON/OFF
// ---------------------------------------------------------------------------
test("5. Paleta de pinceles — toggle ON/OFF", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // Toggle shows "ON" by default
  const onBtn = page.getByRole("button", { name: "ON", exact: true });
  await expect(onBtn).toBeVisible();

  // Click → changes to "OFF"
  await onBtn.click();
  const offBtn = page.getByRole("button", { name: "OFF", exact: true });
  await expect(offBtn).toBeVisible();

  // Click again → back to "ON"
  await offBtn.click();
  await expect(
    page.getByRole("button", { name: "ON", exact: true })
  ).toBeVisible();
});

// ---------------------------------------------------------------------------
// 6. Paint con pincel — verificar vía WebSocket
// ---------------------------------------------------------------------------
test("6. Paint con pincel — verificar vía WebSocket", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // Get initial active count
  const initialActive = await getActiveCount(page);
  expect(initialActive).toBeGreaterThan(0);

  // Switch to OFF mode (deactivate)
  await page.getByRole("button", { name: "ON", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "OFF", exact: true })
  ).toBeVisible();

  // Select 5×5 brush (large)
  await page.getByRole("button", { name: "5×5" }).click();

  // Click center of canvas
  await clickCanvasCenter(page);

  // Wait for the frame to update
  await page.waitForTimeout(500);

  // Active count should have decreased
  const newActive = await getActiveCount(page);
  expect(newActive).toBeLessThan(initialActive);
});

// ---------------------------------------------------------------------------
// 7. Step avanza la generación
// ---------------------------------------------------------------------------
test("7. Step avanza la generación", async ({ page }) => {
  await startExperiment(page, "von_neumann");

  // Gen starts at "0/50"
  expect(await getGeneration(page)).toBe("0/50");

  // Click Step
  await page.getByRole("button", { name: "Step" }).click();
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("1/50", {
    timeout: 3_000,
  });

  // Click Step again
  await page.getByRole("button", { name: "Step" }).click();
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("2/50", {
    timeout: 3_000,
  });
});

// ---------------------------------------------------------------------------
// 8. Play / Pause
// ---------------------------------------------------------------------------
test("8. Play / Pause", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // Click Play
  await page.getByRole("button", { name: "Play" }).click();

  // State changes to "running"
  await expect(page.getByText("running")).toBeVisible({ timeout: 3_000 });

  // Wait for some generations to pass
  await page.waitForTimeout(600);

  // Generation should be > 0
  const gen = await getGeneration(page);
  const genNum = parseInt(gen.split("/")[0], 10);
  expect(genNum).toBeGreaterThan(0);

  // Click Pause
  await page.getByRole("button", { name: "Pause" }).click();

  // State changes to "paused"
  await expect(page.getByText("paused")).toBeVisible({ timeout: 3_000 });

  // Save generation and verify it stops
  const genAfterPause = await getGeneration(page);
  await page.waitForTimeout(400);
  const genAfterWait = await getGeneration(page);
  expect(genAfterWait).toBe(genAfterPause);
});

// ---------------------------------------------------------------------------
// 9. Reset vuelve al inicio
// ---------------------------------------------------------------------------
test("9. Reset vuelve al inicio", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // Click Step 3 times
  const stepBtn = page.getByRole("button", { name: "Step" });
  await stepBtn.click();
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("1/30", {
    timeout: 3_000,
  });
  await stepBtn.click();
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("2/30", {
    timeout: 3_000,
  });
  await stepBtn.click();
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("3/30", {
    timeout: 3_000,
  });

  // Click Reset
  await page.getByRole("button", { name: "Reset" }).click();

  // Gen returns to "0/30"
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("0/30", {
    timeout: 3_000,
  });

  // State is "ready"
  await expect(page.getByText("ready")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 10. Inspeccionar desactiva la paleta
// ---------------------------------------------------------------------------
test("10. Inspeccionar desactiva la paleta", async ({ page }) => {
  await startExperiment(page, "kohonen");

  // The brush palette container — the absolutely-positioned div that wraps all brush buttons
  // It's the innermost div that directly contains the Punto button and the ON/OFF toggle
  const palette = page
    .getByRole("button", { name: "Punto" })
    .locator("..");

  // Palette is enabled (opacity 1)
  await expect(palette).toHaveCSS("opacity", "1");

  // Click "Inspeccionar"
  await page.getByRole("button", { name: "Inspeccionar" }).click();

  // Palette is disabled (opacity 0.3)
  await expect(palette).toHaveCSS("opacity", "0.3");
  await expect(palette).toHaveCSS("pointer-events", "none");

  // Click "✕ Inspeccionar" to toggle off
  await page.getByRole("button", { name: /Inspeccionar/ }).click();

  // Palette returns to enabled
  await expect(palette).toHaveCSS("opacity", "1");
  await expect(palette).toHaveCSS("pointer-events", "auto");
});

// ---------------------------------------------------------------------------
// 11. Cambiar de experimento
// ---------------------------------------------------------------------------
test("11. Cambiar de experimento", async ({ page }) => {
  // Start Von Neumann
  await startExperiment(page, "von_neumann");
  await expect(page.locator("canvas")).toBeVisible();

  // Click Kohonen in the sidebar
  await page.getByRole("button", { name: /Kohonen/i }).click();

  // Click "Iniciar Experimento"
  await page.getByRole("button", { name: "Iniciar Experimento" }).click();

  // Wait for canvas to appear
  await expect(page.locator("canvas")).toBeVisible({ timeout: 5_000 });

  // Gen = "0/30" (new experiment)
  await expect(page.locator("text=Gen:").locator("strong")).toHaveText("0/30", {
    timeout: 3_000,
  });

  // Active > 0 (Kohonen has random neurons)
  const active = await getActiveCount(page);
  expect(active).toBeGreaterThan(0);
});
