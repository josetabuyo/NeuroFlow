/** HTML5 Canvas that renders the neuron grid and handles clicks. */

import { useRef, useEffect, useCallback } from "react";

interface PixelCanvasProps {
  grid: number[][];
  width: number;
  height: number;
  inputRow: number;
  outputRow: number;
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

export function PixelCanvas({
  grid,
  width,
  height,
  inputRow,
  outputRow,
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

    // Draw cells
    for (let row = 0; row < Math.min(height, grid.length); row++) {
      for (let col = 0; col < Math.min(width, grid[row].length); col++) {
        const active = grid[row][col] > 0;
        const isInput = row === inputRow;
        const isOutput = row === outputRow;

        if (isInput) {
          ctx.fillStyle = active ? COLORS.inputActive : COLORS.inputInactive;
        } else if (isOutput) {
          ctx.fillStyle = active ? COLORS.outputActive : COLORS.outputInactive;
        } else {
          ctx.fillStyle = active ? COLORS.active : COLORS.inactive;
        }

        ctx.fillRect(
          col * cellSize,
          row * cellSize,
          cellSize - 1,
          cellSize - 1
        );
      }
    }
  }, [grid, width, height, inputRow, outputRow, getCellSize]);

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
