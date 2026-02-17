/** Brush shape definitions for the brush palette. */

import type { BrushShape } from "./types";

export const BRUSHES: BrushShape[] = [
  {
    id: "1x1",
    name: "Punto",
    offsets: [[0, 0]],
  },
  {
    id: "3x3",
    name: "3×3",
    offsets: [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0],  [0, 0],  [1, 0],
      [-1, 1],  [0, 1],  [1, 1],
    ],
  },
  {
    id: "5x5",
    name: "5×5",
    offsets: (() => {
      const o: [number, number][] = [];
      for (let dy = -2; dy <= 2; dy++)
        for (let dx = -2; dx <= 2; dx++)
          o.push([dx, dy]);
      return o;
    })(),
  },
  {
    id: "cross",
    name: "Cruz",
    offsets: [
      [0, -2],
      [0, -1],
      [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
      [0, 1],
      [0, 2],
    ],
  },
  {
    id: "diamond",
    name: "Diamante",
    offsets: [
      [0, -2],
      [-1, -1], [0, -1], [1, -1],
      [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
      [-1, 1], [0, 1], [1, 1],
      [0, 2],
    ],
  },
];
