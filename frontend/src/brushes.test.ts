import { describe, it, expect } from "vitest";
import { generateCircleBrush, generateSquareBrush, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE } from "./brushes";

describe("generateCircleBrush", () => {
  it("size 1 produces only the center cell", () => {
    const offsets = generateCircleBrush(1);
    expect(offsets).toHaveLength(1);
    const [dx, dy] = offsets[0];
    expect(dx).toBe(0);
    expect(dy).toBe(0);
  });

  it("size 3 produces a plus/cross (5 cells, no corners)", () => {
    const offsets = generateCircleBrush(3);
    const set = new Set(offsets.map(([dx, dy]) => `${dx},${dy}`));
    expect(set.has("0,0")).toBe(true);
    expect(set.has("1,0")).toBe(true);
    expect(set.has("-1,0")).toBe(true);
    expect(set.has("0,1")).toBe(true);
    expect(set.has("0,-1")).toBe(true);
    // corners excluded
    expect(set.has("1,1")).toBe(false);
    expect(set.has("-1,-1")).toBe(false);
    expect(offsets).toHaveLength(5);
  });

  it("size 5 excludes the four diagonal corners", () => {
    const offsets = generateCircleBrush(5);
    const set = new Set(offsets.map(([dx, dy]) => `${dx},${dy}`));
    expect(set.has("2,2")).toBe(false);
    expect(set.has("2,-2")).toBe(false);
    expect(set.has("-2,2")).toBe(false);
    expect(set.has("-2,-2")).toBe(false);
    // center and axis cells are included
    expect(set.has("0,0")).toBe(true);
    expect(set.has("2,0")).toBe(true);
    expect(set.has("0,2")).toBe(true);
  });

  it("is symmetric on all four axes", () => {
    for (const size of [1, 3, 5, 7, 9]) {
      const offsets = generateCircleBrush(size);
      const set = new Set(offsets.map(([dx, dy]) => `${dx},${dy}`));
      for (const [dx, dy] of offsets) {
        expect(set.has(`${-dx},${dy}`), `missing [-${dx},${dy}] for size ${size}`).toBe(true);
        expect(set.has(`${dx},${-dy}`), `missing [${dx},-${dy}] for size ${size}`).toBe(true);
        expect(set.has(`${-dx},${-dy}`), `missing [-${dx},-${dy}] for size ${size}`).toBe(true);
      }
    }
  });

  it("always includes the center cell", () => {
    for (const size of [1, 3, 5, 7, 9, 11, 13, 15]) {
      const offsets = generateCircleBrush(size);
      const set = new Set(offsets.map(([dx, dy]) => `${dx},${dy}`));
      expect(set.has("0,0"), `center missing for size ${size}`).toBe(true);
    }
  });

  it("circle is strictly smaller than the bounding square for size > 1", () => {
    for (const size of [3, 5, 7, 9]) {
      const circle = generateCircleBrush(size);
      expect(circle.length).toBeLessThan(size * size);
    }
  });
});

describe("generateSquareBrush (deprecated alias)", () => {
  it("is identical to generateCircleBrush", () => {
    expect(generateSquareBrush(1)).toEqual(generateCircleBrush(1));
    expect(generateSquareBrush(3)).toEqual(generateCircleBrush(3));
    expect(generateSquareBrush(5)).toEqual(generateCircleBrush(5));
  });
});

describe("brush constants", () => {
  it("MIN_BRUSH_SIZE is 1", () => expect(MIN_BRUSH_SIZE).toBe(1));
  it("MAX_BRUSH_SIZE is 15", () => expect(MAX_BRUSH_SIZE).toBe(15));
});
