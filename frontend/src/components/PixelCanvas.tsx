/** HTML5 Canvas that renders the neuron grid and handles clicks. */

import { useRef, useEffect, useCallback, useState } from "react";

interface PixelCanvasProps {
  grid: number[][];
  tensionGrid?: number[][] | null;
  tensionMode?: boolean;
  width: number;
  height: number;
  inputRow?: number;
  outputRow?: number;
  connectionMap?: (number | null)[][] | null;
  inspectedCell?: { x: number; y: number } | null;
  onCellClick: (x: number, y: number) => void;
  onCellDrag?: (x: number, y: number) => void;
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

function tensionToColor(tension: number): string {
  const t = Math.max(-1, Math.min(1, tension));
  if (t > 0) {
    const r = Math.round(255 * Math.min(1, t * 2));
    const g = Math.round(140 * Math.min(1, t * 1.5));
    return `rgb(${r}, ${g}, 0)`;
  } else if (t < 0) {
    const abs = Math.abs(t);
    const b = Math.round(255 * Math.min(1, abs * 2));
    const r = Math.round(80 * Math.min(1, abs * 1.5));
    return `rgb(${r}, 0, ${b})`;
  }
  return "#0a0a0a";
}

export function PixelCanvas({
  grid,
  tensionGrid,
  tensionMode,
  width,
  height,
  inputRow,
  outputRow,
  connectionMap,
  inspectedCell,
  onCellClick,
  onCellDrag,
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
    const showTension = tensionMode && tensionGrid != null && tensionGrid.length > 0;

    // Draw main grid: tension heatmap or normal activation
    for (let row = 0; row < Math.min(height, grid.length); row++) {
      for (let col = 0; col < Math.min(width, grid[row].length); col++) {
        if (showTension && tensionGrid[row] && tensionGrid[row][col] !== undefined) {
          ctx.fillStyle = tensionToColor(tensionGrid[row][col]);
        } else {
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
        }

        ctx.fillRect(
          col * cellSize,
          row * cellSize,
          cellSize - 1,
          cellSize - 1
        );
      }
    }

    // Draw connection map as semi-transparent overlay
    if (hasConnectionMap) {
      ctx.globalAlpha = 0.6;
      for (let row = 0; row < Math.min(height, grid.length); row++) {
        for (let col = 0; col < Math.min(width, grid[row].length); col++) {
          if (connectionMap[row] && connectionMap[row][col] !== undefined) {
            ctx.fillStyle = weightToColor(connectionMap[row][col]);
            ctx.fillRect(
              col * cellSize,
              row * cellSize,
              cellSize - 1,
              cellSize - 1
            );
          }
        }
      }
      ctx.globalAlpha = 1.0;

      // Draw yellow border for inspected cell (fully opaque)
      if (inspectedCell) {
        ctx.strokeStyle = "#ffff00";
        ctx.lineWidth = 2;
        ctx.strokeRect(
          inspectedCell.x * cellSize + 1,
          inspectedCell.y * cellSize + 1,
          cellSize - 3,
          cellSize - 3
        );
      }
    }
  }, [grid, tensionGrid, tensionMode, width, height, inputRow, outputRow, connectionMap, inspectedCell, getCellSize]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on resize
  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  const [isDragging, setIsDragging] = useState(false);
  const lastCellRef = useRef<string | null>(null);

  const getCellFromEvent = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const cellSize = getCellSize();
      const x = Math.floor((e.clientX - rect.left) / cellSize);
      const y = Math.floor((e.clientY - rect.top) / cellSize);
      if (x >= 0 && x < width && y >= 0 && y < height) return { x, y };
      return null;
    },
    [width, height, getCellSize]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const cell = getCellFromEvent(e);
      if (!cell) return;
      setIsDragging(true);
      lastCellRef.current = `${cell.x},${cell.y}`;
      onCellClick(cell.x, cell.y);
    },
    [getCellFromEvent, onCellClick]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDragging) return;
      const cell = getCellFromEvent(e);
      if (!cell) return;
      const key = `${cell.x},${cell.y}`;
      if (key === lastCellRef.current) return;
      lastCellRef.current = key;
      onCellDrag?.(cell.x, cell.y);
    },
    [isDragging, getCellFromEvent, onCellDrag]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    lastCellRef.current = null;
  }, []);

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      setIsDragging(false);
      lastCellRef.current = null;
    };
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, []);

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
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
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
