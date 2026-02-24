/** Compact canvas for visualizing small grids (input frame, weight maps, etc.) */

import { useRef, useEffect, useCallback } from "react";

interface MiniGridProps {
  label: string;
  grid: number[][];
  width: number;
  height: number;
  subtitle?: string;
  colorMode?: "activation" | "weight";
}

function weightColor(w: number): string {
  const clamped = Math.max(0, Math.min(1, w));
  const g = Math.round(clamped * 255);
  return `rgb(0, ${g}, 0)`;
}

export function MiniGrid({
  label,
  grid,
  width,
  height,
  subtitle,
  colorMode = "activation",
}: MiniGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || grid.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const containerSize = canvas.parentElement?.clientWidth ?? 120;
    const cellSize = Math.max(2, Math.floor(containerSize / Math.max(width, height)));
    const canvasW = cellSize * width;
    const canvasH = cellSize * height;

    canvas.width = canvasW;
    canvas.height = canvasH;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvasW, canvasH);

    for (let row = 0; row < Math.min(height, grid.length); row++) {
      for (let col = 0; col < Math.min(width, (grid[row]?.length ?? 0)); col++) {
        const val = grid[row][col];
        if (colorMode === "weight") {
          ctx.fillStyle = weightColor(val);
        } else {
          ctx.fillStyle = val > 0.5 ? "#4cc9f0" : "#0d1b2a";
        }
        ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
      }
    }
  }, [grid, width, height, colorMode]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div style={{ textAlign: "center" }}>
      <div style={{
        fontSize: "0.65rem",
        color: "#888",
        textTransform: "uppercase",
        letterSpacing: "0.05em",
        marginBottom: "4px",
      }}>
        {label}
        <span style={{ color: "#555", marginLeft: "4px" }}>
          {width}×{height}
        </span>
      </div>
      <div style={{
        border: "1px solid #2a2a3e",
        borderRadius: "4px",
        overflow: "hidden",
        display: "inline-block",
      }}>
        <canvas
          ref={canvasRef}
          style={{ imageRendering: "pixelated", display: "block" }}
        />
      </div>
      {subtitle && (
        <div style={{
          fontSize: "0.6rem",
          color: "#666",
          marginTop: "3px",
        }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}
