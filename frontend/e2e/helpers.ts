import { Page, expect } from "@playwright/test";

/** Wait for the WebSocket to connect (state stops being "disconnected"). */
export async function waitForConnection(page: Page): Promise<void> {
  await expect(page.getByText("ready")).toBeVisible({ timeout: 10_000 });
}

/** Start an experiment and wait for the canvas to appear. */
export async function startExperiment(
  page: Page,
  experiment: "von_neumann" | "kohonen" | "kohonen_lab"
): Promise<void> {
  if (experiment === "kohonen") {
    await page
      .getByRole("button", { name: "Kohonen (Competencia Lateral 2D)" })
      .click();
  } else if (experiment === "kohonen_lab") {
    await page
      .getByRole("button", { name: "Kohonen Lab" })
      .click();
  }
  await page.getByRole("button", { name: "Iniciar Experimento" }).click();
  await expect(page.locator("canvas")).toBeVisible({ timeout: 5_000 });
}

/** Get the active cells count from the stats display. */
export async function getActiveCount(page: Page): Promise<number> {
  const el = page.locator("text=Activas:").locator("strong");
  const text = await el.textContent();
  return parseInt(text ?? "0", 10);
}

/** Get the generation text (e.g. "0/50"). */
export async function getGeneration(page: Page): Promise<string> {
  const el = page.locator("text=Gen:").locator("strong");
  return (await el.textContent()) ?? "";
}

/** Click in the center of the canvas. */
export async function clickCanvasCenter(page: Page): Promise<void> {
  const canvas = page.locator("canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Canvas not found");
  await page.mouse.click(
    box.x + box.width / 2,
    box.y + box.height / 2
  );
}

/** Get the brush palette container element. */
export function getBrushPalette(page: Page) {
  return page.locator("canvas").locator("..").locator("div").filter({
    has: page.getByRole("button", { name: "Punto" }),
  });
}
