/** HTML5 Canvas that renders the neuron grid and handles clicks. */

import { useRef, useEffect, useCallback, useState } from "react";

interface PixelCanvasProps {
  grid: number[][];
  tensionGrid?: number[][] | null;
  tensionMode?: boolean;
  width: number;
  height: number;
  /**
   * When provided, renders this as the full primary display using weight colors
   * (green = excitatory, purple = inhibitory, yellow = self).
   * The normal activation grid is ignored.
   */
  weightGrid?: (number | null)[][] | null;
  inspectedCell?: { x: number; y: number } | null;
  onCellClick: (x: number, y: number) => void;
  onCellDrag?: (x: number, y: number) => void;
}

const COLORS = {
  inactive: "#0a0a0a",
  active: "#ffffff",
  gridLine: "#1a1a2e",
};

export function weightToColor(weight: number | null): string {
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
  weightGrid,
  inspectedCell,
  onCellClick,
  onCellDrag,
}: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const getCellSize = useCallback(() => {
    if (!containerRef.current) return 10;
    const cellW = Math.floor(containerRef.current.clientWidth / width);
    const cellH = Math.floor(containerRef.current.clientHeight / height);
    return Math.max(2, Math.min(cellW, cellH));
  }, [width, height]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellSize = getCellSize();
    canvas.width  = cellSize * width;
    canvas.height = cellSize * height;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // ── Weight-only mode (inspect popup) ──
    if (weightGrid && weightGrid.length > 0) {
      for (let row = 0; row < Math.min(height, weightGrid.length); row++) {
        for (let col = 0; col < Math.min(width, weightGrid[row]?.length ?? 0); col++) {
          ctx.fillStyle = weightToColor(weightGrid[row][col]);
          ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
        }
      }
      return;
    }

    if (grid.length === 0) return;

    // ── Normal activation / tension mode ──
    const showTension = tensionMode && tensionGrid != null && tensionGrid.length > 0;

    for (let row = 0; row < Math.min(height, grid.length); row++) {
      for (let col = 0; col < Math.min(width, grid[row].length); col++) {
        if (showTension && tensionGrid![row]?.[col] !== undefined) {
          ctx.fillStyle = tensionToColor(tensionGrid![row][col]);
        } else {
          ctx.fillStyle = grid[row][col] > 0 ? COLORS.active : COLORS.inactive;
        }
        ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
      }
    }

    // ── Inspected cell yellow border ──
    if (inspectedCell) {
      ctx.strokeStyle = "#ffff00";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        inspectedCell.x * cellSize + 1,
        inspectedCell.y * cellSize + 1,
        cellSize - 3,
        cellSize - 3,
      );
    }
  }, [grid, tensionGrid, tensionMode, width, height, weightGrid, inspectedCell, getCellSize]);

  useEffect(() => { draw(); }, [draw]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  const [isDragging, setIsDragging] = useState(false);
  const lastCellRef = useRef<string | null>(null);

  const getCellFromEvent = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const cellSize = getCellSize();
      const x = Math.floor((e.clientX - rect.left) / cellSize);
      const y = Math.floor((e.clientY - rect.top) / cellSize);
      if (x >= 0 && x < width && y >= 0 && y < height) return { x, y };
      return null;
    },
    [width, height, getCellSize],
  );

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = getCellFromEvent(e);
    if (!cell) return;
    setIsDragging(true);
    lastCellRef.current = `${cell.x},${cell.y}`;
    onCellClick(cell.x, cell.y);
  }, [getCellFromEvent, onCellClick]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    const cell = getCellFromEvent(e);
    if (!cell) return;
    const key = `${cell.x},${cell.y}`;
    if (key === lastCellRef.current) return;
    lastCellRef.current = key;
    onCellDrag?.(cell.x, cell.y);
  }, [isDragging, getCellFromEvent, onCellDrag]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    lastCellRef.current = null;
  }, []);

  useEffect(() => {
    const up = () => { setIsDragging(false); lastCellRef.current = null; };
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", display: "flex", justifyContent: "center", alignItems: "center" }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{ cursor: "crosshair", imageRendering: "pixelated", borderRadius: "4px", border: "1px solid #2a2a3e" }}
      />
    </div>
  );
}
