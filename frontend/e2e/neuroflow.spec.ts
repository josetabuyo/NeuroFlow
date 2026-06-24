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
// 1. Initial load
// ---------------------------------------------------------------------------
test("1. Initial load", async ({ page }) => {
  await expect(page.locator("h1")).toHaveText("NeuroFlow");

  await expect(
    page.getByText("Config Templates")
  ).toBeVisible();

  await expect(
    page.getByRole("button", { name: "Start Experiment" })
  ).toBeVisible();

  await expect(page.getByText("ready")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 2. Start experiment
// ---------------------------------------------------------------------------
test("2. Start experiment", async ({ page }) => {
  await startExperiment(page);

  await expect(page.locator("main canvas").first()).toBeVisible();

  const palette = getBrushPalette(page);
  await expect(palette).toBeVisible();
  await expect(
    palette.getByRole("button", { name: "Increase brush" })
  ).toBeVisible();
  await expect(
    palette.getByRole("button", { name: "Decrease brush" })
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
// 3. Start and verify initial state
// ---------------------------------------------------------------------------
test("3. Start and verify initial state", async ({ page }) => {
  await startExperiment(page);

  await expect(page.locator("main canvas").first()).toBeVisible();

  const steps = await getStepCount(page);
  expect(steps).toBe(0);

  const active = await getActiveCount(page);
  expect(active).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// 4. Brush palette — size adjustment
// ---------------------------------------------------------------------------
test("4. Brush palette — size adjustment", async ({ page }) => {
  await startExperiment(page);

  const palette = getBrushPalette(page);
  const increaseBtn = palette.getByRole("button", { name: "Increase brush" });
  const decreaseBtn = palette.getByRole("button", { name: "Decrease brush" });

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
// 5. Brush palette — toggle ON/OFF
// ---------------------------------------------------------------------------
test("5. Brush palette — toggle ON/OFF", async ({ page }) => {
  await startExperiment(page);

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
// 6. Paint with brush — verify via WebSocket
// ---------------------------------------------------------------------------
test("6. Paint with brush — verify via WebSocket", async ({ page }) => {
  await startExperiment(page);

  const initialActive = await getActiveCount(page);
  expect(initialActive).toBeGreaterThan(0);

  // Switch to OFF mode (deactivate)
  await page.getByRole("button", { name: "ON", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "OFF", exact: true })
  ).toBeVisible();

  // Increase brush to 5×5
  const palette = getBrushPalette(page);
  const increaseBtn = palette.getByRole("button", { name: "Increase brush" });
  await increaseBtn.click(); // 3×3
  await increaseBtn.click(); // 5×5

  await clickCanvasCenter(page);
  await page.waitForTimeout(500);

  const newActive = await getActiveCount(page);
  expect(newActive).toBeLessThan(initialActive);
});

// ---------------------------------------------------------------------------
// 7. Step advances the generation
// ---------------------------------------------------------------------------
test("7. Step advances the generation", async ({ page }) => {
  await startExperiment(page);

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
  await startExperiment(page);

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
// 9. Reset returns to initial state
// ---------------------------------------------------------------------------
test("9. Reset returns to initial state", async ({ page }) => {
  await startExperiment(page);

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
// 10. Inspect disables brush controls
// ---------------------------------------------------------------------------
test("10. Inspect disables brush controls", async ({ page }) => {
  await startExperiment(page);

  const brushControls = page.getByTestId("brush-controls");
  await expect(brushControls).toHaveCSS("opacity", "1");

  await page.getByRole("button", { name: "Inspect tool" }).click();

  await expect(brushControls).toHaveCSS("opacity", "0.3");
  await expect(brushControls).toHaveCSS("pointer-events", "none");

  await page.getByRole("button", { name: "Brush tool" }).click();

  await expect(brushControls).toHaveCSS("opacity", "1");
  await expect(brushControls).toHaveCSS("pointer-events", "auto");
});

// ---------------------------------------------------------------------------
// 11. Steps per tick
// ---------------------------------------------------------------------------
test("11. Steps per tick", async ({ page }) => {
  await startExperiment(page);

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
// 12. Brush size limits
// ---------------------------------------------------------------------------
test("12. Brush size limits", async ({ page }) => {
  await startExperiment(page);

  const palette = getBrushPalette(page);
  const decreaseBtn = palette.getByRole("button", { name: "Decrease brush" });
  const increaseBtn = palette.getByRole("button", { name: "Increase brush" });

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
// 13. Template selector changes config
// ---------------------------------------------------------------------------
test("13. Template selector changes config", async ({ page }) => {
  await expect(page.getByText("Config Templates")).toBeVisible();

  const templateSelect = page.locator("aside select").first();
  await expect(templateSelect).toBeVisible();

  const options = templateSelect.locator("option");
  const optionCount = await options.count();
  expect(optionCount).toBeGreaterThan(1);

  const secondOption = await options.nth(1).getAttribute("value");
  expect(secondOption).toBeTruthy();
  await templateSelect.selectOption(secondOption!);

  await page.getByRole("button", { name: "Start Experiment" }).click();
  await expect(page.locator("main canvas").first()).toBeVisible({ timeout: 45_000 });
});

// ---------------------------------------------------------------------------
// 14. Restarting experiment during Play stops the network
// ---------------------------------------------------------------------------
test("14. Restarting experiment during Play stops the network", async ({ page }) => {
  await startExperiment(page);

  await page.getByRole("button", { name: "Play" }).click();
  await expect(page.getByText("running")).toBeVisible({ timeout: 3_000 });

  await page.waitForTimeout(400);
  const stepsBefore = await getStepCount(page);
  expect(stepsBefore).toBeGreaterThan(0);

  // Refresh the experiment
  await page.getByRole("button", { name: "Refresh Experiment" }).click();

  await expect(page.getByText("ready")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByRole("button", { name: "Play" })).toBeEnabled();

  const stepsAfterInit = await getStepCount(page);
  await page.waitForTimeout(600);
  const stepsAfterWait = await getStepCount(page);
  expect(stepsAfterWait).toBe(stepsAfterInit);
});

// ---------------------------------------------------------------------------
// 15. Inspect shows connection map
// ---------------------------------------------------------------------------
test("15. Inspect shows connection map", async ({ page }) => {
  await startExperiment(page);

  await page.getByRole("button", { name: "Inspect tool" }).click();

  await clickCanvasCenter(page);

  await expect(page.getByText("Excitatory (+1)")).toBeVisible({
    timeout: 3_000,
  });
  await expect(page.getByText("Inhibitory (-1)")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 16. Paint mode: mousedown paints 1s, mouseup clears to 0s (transient signal)
// ---------------------------------------------------------------------------
test("16. Paint mode: mousedown injects 1s, mouseup clears them", async ({ page }) => {
  await startExperiment(page);

  // No continuous mode button should exist
  await expect(page.getByTitle("Continuous (signal follows cursor)")).not.toBeVisible();
  await expect(page.getByTitle("Pointer (click/drag to paint)")).not.toBeVisible();

  // Increase brush to 5×5 for easier detection
  const increaseBtn = getBrushPalette(page).getByRole("button", { name: "Increase brush" });
  await increaseBtn.click();
  await increaseBtn.click();
  expect(await getBrushSizeLabel(page)).toBe("5×5");

  const canvas = page.locator("main canvas").first();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Canvas not found");
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  // Mousedown — cells should light up
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.waitForTimeout(150);
  const activeWhilePressed = await getActiveCount(page);
  expect(activeWhilePressed).toBeGreaterThan(0);

  // Mouseup — painted cells should be cleared back to 0
  await page.mouse.up();
  await page.waitForTimeout(300);
  const activeAfterRelease = await getActiveCount(page);
  expect(activeAfterRelease).toBeLessThan(activeWhilePressed);
});
