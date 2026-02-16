/** HTML5 Canvas that renders the neuron grid and handles clicks. */

import { useRef, useEffect, useCallback } from "react";

interface PixelCanvasProps {
  grid: number[][];
  width: number;
  height: number;
  inputRow?: number;
  outputRow?: number;
  connectionMap?: (number | null)[][] | null;
  inspectedCell?: { x: number; y: number } | null;
  onCellClick: (x: number, y: number) => void;
}

const COLORS = {
  inactive: "#0a0a0a",
  active: "#ffffff",
  inputInactive: "#0d1b2a",
  inputActive: "#4cc9f0",
  outputInactive: "#1a0a0a",
  outputActive: "#f72585",
  gridLine: "#1a1a2e",
};

function weightToColor(weight: number | null): string {
  if (weight === null) return "#111111";
  if (weight === 999) return "#ffff00";

  const w = Math.max(-1, Math.min(1, weight));

  if (w > 0) {
    const g = Math.round(w * 255);
    return `rgb(0, ${g}, 0)`;
  } else if (w < 0) {
    const abs = Math.abs(w);
    const r = Math.round(abs * 139);
    const b = Math.round(abs * 255);
    return `rgb(${r}, 0, ${b})`;
  }
  return "#000000";
}

export function PixelCanvas({
  grid,
  width,
  height,
  inputRow,
  outputRow,
  connectionMap,
  inspectedCell,
  onCellClick,
}: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const getCellSize = useCallback(() => {
    if (!containerRef.current) return 10;
    const containerWidth = containerRef.current.clientWidth;
    const containerHeight = containerRef.current.clientHeight;
    const cellW = Math.floor(containerWidth / width);
    const cellH = Math.floor(containerHeight / height);
    return Math.max(2, Math.min(cellW, cellH));
  }, [width, height]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellSize = getCellSize();
    const canvasWidth = cellSize * width;
    const canvasHeight = cellSize * height;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Clear
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    if (grid.length === 0) return;

    const hasConnectionMap = connectionMap != null && connectionMap.length > 0;

    // Draw cells
    for (let row = 0; row < Math.min(height, grid.length); row++) {
      for (let col = 0; col < Math.min(width, grid[row].length); col++) {
        if (hasConnectionMap && connectionMap[row] && connectionMap[row][col] !== undefined) {
          ctx.fillStyle = weightToColor(connectionMap[row][col]);
        } else if (!hasConnectionMap) {
          const active = grid[row][col] > 0;
          const isInput = inputRow != null && row === inputRow;
          const isOutput = outputRow != null && row === outputRow;

          if (isInput) {
            ctx.fillStyle = active ? COLORS.inputActive : COLORS.inputInactive;
          } else if (isOutput) {
            ctx.fillStyle = active ? COLORS.outputActive : COLORS.outputInactive;
          } else {
            ctx.fillStyle = active ? COLORS.active : COLORS.inactive;
          }
        } else {
          ctx.fillStyle = "#111111";
        }

        ctx.fillRect(
          col * cellSize,
          row * cellSize,
          cellSize - 1,
          cellSize - 1
        );

        // Draw yellow border for inspected cell
        if (
          hasConnectionMap &&
          inspectedCell &&
          col === inspectedCell.x &&
          row === inspectedCell.y
        ) {
          ctx.strokeStyle = "#ffff00";
          ctx.lineWidth = 2;
          ctx.strokeRect(
            col * cellSize + 1,
            row * cellSize + 1,
            cellSize - 3,
            cellSize - 3
          );
        }
      }
    }
  }, [grid, width, height, inputRow, outputRow, connectionMap, inspectedCell, getCellSize]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on resize
  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const cellSize = getCellSize();
      const x = Math.floor((e.clientX - rect.left) / cellSize);
      const y = Math.floor((e.clientY - rect.top) / cellSize);

      if (x >= 0 && x < width && y >= 0 && y < height) {
        onCellClick(x, y);
      }
    },
    [width, height, getCellSize, onCellClick]
  );

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        aspectRatio: `${width} / ${height}`,
        maxHeight: "70vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{
          cursor: "crosshair",
          imageRendering: "pixelated",
          borderRadius: "4px",
          border: "1px solid #2a2a3e",
        }}
      />
    </div>
  );
}
