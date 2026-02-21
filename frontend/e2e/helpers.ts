import { Page, expect } from "@playwright/test";

/** Wait for the WebSocket to connect (state shows "ready"). */
export async function waitForConnection(page: Page): Promise<void> {
  await expect(page.getByText("ready")).toBeVisible({ timeout: 10_000 });
}

/** Start an experiment and wait for canvas + controls to be ready. */
export async function startExperiment(
  page: Page,
  experiment: "deamons_lab"
): Promise<void> {
  await page.getByRole("button", { name: "Deamons Lab" }).click();
  await page.getByRole("button", { name: "Start Experiment" }).click();
  // Use "main canvas" to avoid matching the MaskPreview canvas in the sidebar.
  // deamons_lab with large masks can take 30+ seconds to build on the backend.
  await expect(page.locator("main canvas")).toBeVisible({ timeout: 45_000 });
  await expect(
    page.getByRole("button", { name: "Step" })
  ).toBeEnabled({ timeout: 10_000 });
}

/**
 * Get the step count from the stats display.
 * Handles formatted numbers ("0", "1.0k", "1.5M") and optional " / total".
 */
export async function getStepCount(page: Page): Promise<number> {
  const el = page.locator("text=Steps:").locator("strong");
  const text = await el.textContent();
  if (!text) return 0;
  const raw = text.trim().split(/\s/)[0];
  if (raw.endsWith("k")) return Math.round(parseFloat(raw) * 1_000);
  if (raw.endsWith("M")) return Math.round(parseFloat(raw) * 1_000_000);
  return parseInt(raw, 10);
}

/** Get the active cells count from the stats display. */
export async function getActiveCount(page: Page): Promise<number> {
  const el = page.locator("text=Active:").locator("strong");
  const text = await el.textContent();
  return parseInt(text ?? "0", 10);
}

/** Click in the center of the main experiment canvas. */
export async function clickCanvasCenter(page: Page): Promise<void> {
  const canvas = page.locator("main canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Canvas not found");
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
}

/** Get the brush palette container element. */
export function getBrushPalette(page: Page) {
  return page.getByTestId("brush-palette");
}

/** Get the brush size label text (e.g. "1×1"). */
export async function getBrushSizeLabel(page: Page): Promise<string> {
  return (await page.getByTestId("brush-size-label").textContent()) ?? "";
}
