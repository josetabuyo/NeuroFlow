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
  nerveCircles?: Array<{ cx: number; cy: number; radius: number }> | null;
  overlayGrid?: (number | null)[][] | null;
  neuronLabels?: (string | null)[][] | null;
  inspectedCell?: { x: number; y: number } | null;
  onCellClick: (x: number, y: number) => void;
  onCellDrag?: (x: number, y: number) => void;
  onDragEnd?: (cells: { x: number; y: number }[]) => void;
  brushSize?: number;
  brushMode?: "activate" | "deactivate";
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
  nerveCircles,
  overlayGrid,
  neuronLabels,
  inspectedCell,
  onCellClick,
  onCellDrag,
  onDragEnd,
  brushSize,
  brushMode,
}: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [hoverCell, setHoverCell] = useState<{ x: number; y: number } | null>(null);
  const lastCellRef = useRef<string | null>(null);
  const paintedCellsRef = useRef<{ x: number; y: number }[]>([]);
  const paintedKeysRef = useRef<Set<string>>(new Set());

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
      if (overlayGrid && overlayGrid.length > 0) {
        for (let row = 0; row < Math.min(height, overlayGrid.length); row++) {
          for (let col = 0; col < Math.min(width, overlayGrid[row]?.length ?? 0); col++) {
            const v = overlayGrid[row][col];
            if (v === null || v === 0) continue;
            const alpha = Math.min(0.8, Math.abs(v) * 0.75);
            ctx.fillStyle = `rgba(255, 140, 0, ${alpha})`;
            ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
          }
        }
      }
      if (nerveCircles && nerveCircles.length > 0) {
        ctx.strokeStyle = "rgba(0, 220, 255, 0.6)";
        ctx.lineWidth = 1.5;
        for (const c of nerveCircles) {
          const px = c.cx * cellSize + cellSize / 2;
          const py = c.cy * cellSize + cellSize / 2;
          ctx.beginPath();
          ctx.arc(px, py, c.radius * cellSize, 0, 2 * Math.PI);
          ctx.stroke();
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

    // ── Neuron labels overlay ──
    if (neuronLabels && neuronLabels.length > 0) {
      const fontSize = Math.max(8, Math.round(cellSize * 0.55));
      ctx.font = `bold ${fontSize}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let row = 0; row < Math.min(height, neuronLabels.length); row++) {
        for (let col = 0; col < Math.min(width, (neuronLabels[row]?.length ?? 0)); col++) {
          const lbl = neuronLabels[row]?.[col];
          if (!lbl) continue;
          const cx = col * cellSize + cellSize / 2;
          const cy = row * cellSize + cellSize / 2;
          ctx.fillStyle = "rgba(0,0,0,0.55)";
          const tw = ctx.measureText(lbl).width;
          ctx.fillRect(cx - tw / 2 - 2, cy - fontSize / 2 - 1, tw + 4, fontSize + 2);
          ctx.fillStyle = "#ffff00";
          ctx.fillText(lbl, cx, cy);
        }
      }
    }

    // ── Brush cursor overlay (only while mouse is pressed) ──
    if (hoverCell && brushSize && isDragging) {
      const r = Math.floor(brushSize / 2);
      ctx.fillStyle = brushMode === "deactivate"
        ? "rgba(80,80,80,0.55)"
        : "rgba(255,255,255,0.25)";
      ctx.strokeStyle = brushMode === "deactivate"
        ? "rgba(120,120,120,0.8)"
        : "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1;
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy > r * r + r * 0.5) continue;
          const cx = hoverCell.x + dx;
          const cy = hoverCell.y + dy;
          if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
            ctx.fillRect(cx * cellSize, cy * cellSize, cellSize - 1, cellSize - 1);
            ctx.strokeRect(cx * cellSize + 0.5, cy * cellSize + 0.5, cellSize - 2, cellSize - 2);
          }
        }
      }
    }
  }, [grid, tensionGrid, tensionMode, width, height, weightGrid, nerveCircles, overlayGrid, neuronLabels, inspectedCell, getCellSize, hoverCell, brushSize, brushMode, isDragging]);

  useEffect(() => { draw(); }, [draw]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

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

  const trackCell = (cell: { x: number; y: number }) => {
    const key = `${cell.x},${cell.y}`;
    if (!paintedKeysRef.current.has(key)) {
      paintedKeysRef.current.add(key);
      paintedCellsRef.current.push(cell);
    }
  };

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = getCellFromEvent(e);
    if (!cell) return;
    paintedCellsRef.current = [];
    paintedKeysRef.current = new Set();
    trackCell(cell);
    setIsDragging(true);
    setHoverCell(cell);
    lastCellRef.current = `${cell.x},${cell.y}`;
    onCellClick(cell.x, cell.y);
  }, [getCellFromEvent, onCellClick]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = getCellFromEvent(e);
    if (!cell) return;
    setHoverCell(cell);

    if (!isDragging) return;
    const key = `${cell.x},${cell.y}`;
    if (key === lastCellRef.current) return;
    lastCellRef.current = key;
    trackCell(cell);
    onCellDrag?.(cell.x, cell.y);
  }, [isDragging, getCellFromEvent, onCellDrag]);

  const handleMouseUp = useCallback(() => {
    if (paintedCellsRef.current.length > 0) {
      onDragEnd?.(paintedCellsRef.current);
      paintedCellsRef.current = [];
      paintedKeysRef.current = new Set();
    }
    setIsDragging(false);
    lastCellRef.current = null;
  }, [onDragEnd]);

  const handleMouseLeave = useCallback(() => {
    setHoverCell(null);
    lastCellRef.current = null;
  }, []);

  const onDragEndRef = useRef(onDragEnd);
  onDragEndRef.current = onDragEnd;

  useEffect(() => {
    const up = () => {
      if (paintedCellsRef.current.length > 0) {
        onDragEndRef.current?.(paintedCellsRef.current);
        paintedCellsRef.current = [];
        paintedKeysRef.current = new Set();
      }
      setIsDragging(false);
      lastCellRef.current = null;
    };
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
        onMouseLeave={handleMouseLeave}
        style={{ cursor: "crosshair", imageRendering: "pixelated", borderRadius: "4px", border: "1px solid #2a2a3e" }}
      />
    </div>
  );
}
