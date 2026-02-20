/** Dynamic square brush generation for the brush palette. */

/**
 * Generate offsets for an NxN square brush centered at (0,0).
 * `size` must be an odd positive integer (1, 3, 5, 7, …).
 */
export function generateSquareBrush(size: number): [number, number][] {
  const r = Math.floor(size / 2);
  const offsets: [number, number][] = [];
  for (let dy = -r; dy <= r; dy++)
    for (let dx = -r; dx <= r; dx++) offsets.push([dx, dy]);
  return offsets;
}

/** Minimum brush side length. */
export const MIN_BRUSH_SIZE = 1;
/** Maximum brush side length. */
export const MAX_BRUSH_SIZE = 15;

/** Step to next valid brush size (always odd). */
export function nextBrushSize(current: number): number {
  const next = current + 2;
  return next <= MAX_BRUSH_SIZE ? next : current;
}

/** Step to previous valid brush size (always odd). */
export function prevBrushSize(current: number): number {
  const prev = current - 2;
  return prev >= MIN_BRUSH_SIZE ? prev : current;
}
