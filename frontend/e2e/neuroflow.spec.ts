import { test, expect } from "@playwright/test";
import {
  waitForConnection,
  startExperiment,
  getStepCount,
  getActiveCount,
  clickCanvasCenter,
  getBrushPalette,
  getBrushSizeLabel,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForConnection(page);
});

// ---------------------------------------------------------------------------
// 1. Carga inicial
// ---------------------------------------------------------------------------
test("1. Carga inicial", async ({ page }) => {
  await expect(page.locator("h1")).toHaveText("NeuroFlow");

  await expect(
    page.getByRole("button", { name: "Deamons Lab" })
  ).toBeVisible();

  await expect(
    page.getByRole("button", { name: "Iniciar Experimento" })
  ).toBeVisible();

  await expect(page.getByText("ready")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 2. Iniciar Deamons Lab
// ---------------------------------------------------------------------------
test("2. Iniciar Deamons Lab", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  await expect(page.locator("main canvas")).toBeVisible();

  const palette = getBrushPalette(page);
  await expect(palette).toBeVisible();
  await expect(
    palette.getByRole("button", { name: "Aumentar pincel" })
  ).toBeVisible();
  await expect(
    palette.getByRole("button", { name: "Reducir pincel" })
  ).toBeVisible();
  await expect(
    palette.getByRole("button", { name: "ON", exact: true })
  ).toBeVisible();

  expect(await getBrushSizeLabel(page)).toBe("1×1");

  await expect(page.getByRole("button", { name: "Play" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Step" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Reset" })).toBeEnabled();
});

// ---------------------------------------------------------------------------
// 3. Iniciar y verificar estado inicial
// ---------------------------------------------------------------------------
test("3. Iniciar y verificar estado inicial", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  await expect(page.locator("main canvas")).toBeVisible();

  const steps = await getStepCount(page);
  expect(steps).toBe(0);

  const active = await getActiveCount(page);
  expect(active).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// 4. Paleta de pinceles — ajuste de tamaño
// ---------------------------------------------------------------------------
test("4. Paleta de pinceles — ajuste de tamaño", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const palette = getBrushPalette(page);
  const increaseBtn = palette.getByRole("button", { name: "Aumentar pincel" });
  const decreaseBtn = palette.getByRole("button", { name: "Reducir pincel" });

  expect(await getBrushSizeLabel(page)).toBe("1×1");

  await increaseBtn.click();
  expect(await getBrushSizeLabel(page)).toBe("3×3");

  await increaseBtn.click();
  expect(await getBrushSizeLabel(page)).toBe("5×5");

  await decreaseBtn.click();
  expect(await getBrushSizeLabel(page)).toBe("3×3");

  await decreaseBtn.click();
  expect(await getBrushSizeLabel(page)).toBe("1×1");
});

// ---------------------------------------------------------------------------
// 5. Paleta de pinceles — toggle ON/OFF
// ---------------------------------------------------------------------------
test("5. Paleta de pinceles — toggle ON/OFF", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const onBtn = page.getByRole("button", { name: "ON", exact: true });
  await expect(onBtn).toBeVisible();

  await onBtn.click();
  const offBtn = page.getByRole("button", { name: "OFF", exact: true });
  await expect(offBtn).toBeVisible();

  await offBtn.click();
  await expect(
    page.getByRole("button", { name: "ON", exact: true })
  ).toBeVisible();
});

// ---------------------------------------------------------------------------
// 6. Paint con pincel — verificar vía WebSocket
// ---------------------------------------------------------------------------
test("6. Paint con pincel — verificar vía WebSocket", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const initialActive = await getActiveCount(page);
  expect(initialActive).toBeGreaterThan(0);

  // Switch to OFF mode (deactivate)
  await page.getByRole("button", { name: "ON", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "OFF", exact: true })
  ).toBeVisible();

  // Increase brush to 5×5
  const palette = getBrushPalette(page);
  const increaseBtn = palette.getByRole("button", { name: "Aumentar pincel" });
  await increaseBtn.click(); // 3×3
  await increaseBtn.click(); // 5×5

  await clickCanvasCenter(page);
  await page.waitForTimeout(500);

  const newActive = await getActiveCount(page);
  expect(newActive).toBeLessThan(initialActive);
});

// ---------------------------------------------------------------------------
// 7. Step avanza la generación
// ---------------------------------------------------------------------------
test("7. Step avanza la generación", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  expect(await getStepCount(page)).toBe(0);

  await page.getByRole("button", { name: "Step" }).click();
  await expect(async () => {
    expect(await getStepCount(page)).toBe(1);
  }).toPass({ timeout: 3_000 });

  await page.getByRole("button", { name: "Step" }).click();
  await expect(async () => {
    expect(await getStepCount(page)).toBe(2);
  }).toPass({ timeout: 3_000 });
});

// ---------------------------------------------------------------------------
// 8. Play / Pause
// ---------------------------------------------------------------------------
test("8. Play / Pause", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  await page.getByRole("button", { name: "Play" }).click();
  await expect(page.getByText("running")).toBeVisible({ timeout: 3_000 });

  await page.waitForTimeout(600);

  const steps = await getStepCount(page);
  expect(steps).toBeGreaterThan(0);

  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByText("paused")).toBeVisible({ timeout: 3_000 });

  const stepsAfterPause = await getStepCount(page);
  await page.waitForTimeout(400);
  const stepsAfterWait = await getStepCount(page);
  expect(stepsAfterWait).toBe(stepsAfterPause);
});

// ---------------------------------------------------------------------------
// 9. Reset vuelve al inicio
// ---------------------------------------------------------------------------
test("9. Reset vuelve al inicio", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const stepBtn = page.getByRole("button", { name: "Step" });
  await stepBtn.click();
  await expect(async () => {
    expect(await getStepCount(page)).toBe(1);
  }).toPass({ timeout: 3_000 });
  await stepBtn.click();
  await expect(async () => {
    expect(await getStepCount(page)).toBe(2);
  }).toPass({ timeout: 3_000 });

  await page.getByRole("button", { name: "Reset" }).click();

  await expect(async () => {
    expect(await getStepCount(page)).toBe(0);
  }).toPass({ timeout: 5_000 });
  await expect(page.getByText("ready")).toBeVisible({ timeout: 5_000 });
});

// ---------------------------------------------------------------------------
// 10. Inspeccionar desactiva los controles de pincel
// ---------------------------------------------------------------------------
test("10. Inspeccionar desactiva los controles de pincel", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const brushControls = page.getByTestId("brush-controls");
  await expect(brushControls).toHaveCSS("opacity", "1");

  await page.getByRole("button", { name: "Herramienta inspeccionar" }).click();

  await expect(brushControls).toHaveCSS("opacity", "0.3");
  await expect(brushControls).toHaveCSS("pointer-events", "none");

  await page.getByRole("button", { name: "Herramienta pincel" }).click();

  await expect(brushControls).toHaveCSS("opacity", "1");
  await expect(brushControls).toHaveCSS("pointer-events", "auto");
});

// ---------------------------------------------------------------------------
// 11. Steps per tick
// ---------------------------------------------------------------------------
test("11. Steps per tick", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const select = page.locator("main select");
  await expect(select).toHaveValue("1");

  await select.selectOption("10");
  await expect(select).toHaveValue("10");

  await page.getByRole("button", { name: "Step" }).click();
  await expect(async () => {
    expect(await getStepCount(page)).toBe(10);
  }).toPass({ timeout: 3_000 });
});

// ---------------------------------------------------------------------------
// 12. Límites del tamaño de pincel
// ---------------------------------------------------------------------------
test("12. Límites del tamaño de pincel", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  const palette = getBrushPalette(page);
  const decreaseBtn = palette.getByRole("button", { name: "Reducir pincel" });
  const increaseBtn = palette.getByRole("button", { name: "Aumentar pincel" });

  // Min size (1×1): decrease disabled
  expect(await getBrushSizeLabel(page)).toBe("1×1");
  await expect(decreaseBtn).toBeDisabled();

  // Increase to max (15×15): 7 clicks (1→3→5→7→9→11→13→15)
  for (let i = 0; i < 7; i++) {
    await increaseBtn.click();
  }
  expect(await getBrushSizeLabel(page)).toBe("15×15");
  await expect(increaseBtn).toBeDisabled();
});

// ---------------------------------------------------------------------------
// 13. Deamons Lab — seleccionar máscara de conexionado
// ---------------------------------------------------------------------------
test("13. Deamons Lab — seleccionar máscara", async ({ page }) => {
  // Deamons Lab is selected by default; mask selector ("Conexionado") is visible
  await expect(page.getByText("Conexionado")).toBeVisible();

  const maskSelect = page.locator("aside select").first();
  await expect(maskSelect).toBeVisible();

  const initialValue = await maskSelect.inputValue();
  const options = maskSelect.locator("option");
  const optionCount = await options.count();
  expect(optionCount).toBeGreaterThan(1);

  const secondOption = await options.nth(1).getAttribute("value");
  expect(secondOption).toBeTruthy();
  await maskSelect.selectOption(secondOption!);
  expect(await maskSelect.inputValue()).not.toBe(initialValue);

  await page.getByRole("button", { name: "Iniciar Experimento" }).click();
  await expect(page.locator("main canvas")).toBeVisible({ timeout: 45_000 });
});

// ---------------------------------------------------------------------------
// 14. Re-iniciar experimento durante Play detiene la red
// ---------------------------------------------------------------------------
test("14. Re-iniciar experimento durante Play detiene la red", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  await page.getByRole("button", { name: "Play" }).click();
  await expect(page.getByText("running")).toBeVisible({ timeout: 3_000 });

  await page.waitForTimeout(400);
  const stepsBefore = await getStepCount(page);
  expect(stepsBefore).toBeGreaterThan(0);

  await startExperiment(page, "deamons_lab");

  await expect(page.getByText("ready")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByRole("button", { name: "Play" })).toBeEnabled();

  const stepsAfterInit = await getStepCount(page);
  await page.waitForTimeout(600);
  const stepsAfterWait = await getStepCount(page);
  expect(stepsAfterWait).toBe(stepsAfterInit);
});

// ---------------------------------------------------------------------------
// 15. Inspeccionar muestra mapa de conexiones
// ---------------------------------------------------------------------------
test("15. Inspeccionar muestra mapa de conexiones", async ({ page }) => {
  await startExperiment(page, "deamons_lab");

  await page.getByRole("button", { name: "Herramienta inspeccionar" }).click();

  await clickCanvasCenter(page);

  // Legend switches to connection-map colors
  await expect(page.getByText("Excitatorio (+1)")).toBeVisible({
    timeout: 3_000,
  });
  await expect(page.getByText("Inhibitorio (-1)")).toBeVisible();
});
