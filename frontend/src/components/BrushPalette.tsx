/** Brush palette — floating vertical toolbar for selecting brush shapes. */

import type { BrushShape } from "../types";

interface BrushPaletteProps {
  brushes: BrushShape[];
  selectedBrush: string;
  brushMode: "activate" | "deactivate";
  disabled: boolean;
  onSelectBrush: (id: string) => void;
  onToggleMode: () => void;
}

function renderBrushPreview(offsets: [number, number][]): React.ReactNode {
  const minX = Math.min(...offsets.map(([dx]) => dx));
  const maxX = Math.max(...offsets.map(([dx]) => dx));
  const minY = Math.min(...offsets.map(([, dy]) => dy));
  const maxY = Math.max(...offsets.map(([, dy]) => dy));
  const w = maxX - minX + 1;
  const h = maxY - minY + 1;

  const pixelSize = Math.max(2, Math.floor(28 / Math.max(w, h)));

  const set = new Set(offsets.map(([dx, dy]) => `${dx},${dy}`));

  const pixels: React.ReactNode[] = [];
  for (let dy = minY; dy <= maxY; dy++) {
    for (let dx = minX; dx <= maxX; dx++) {
      if (set.has(`${dx},${dy}`)) {
        pixels.push(
          <div
            key={`${dx},${dy}`}
            style={{
              position: "absolute",
              left: (dx - minX) * pixelSize,
              top: (dy - minY) * pixelSize,
              width: pixelSize,
              height: pixelSize,
              background: "#e0e0ff",
              borderRadius: 1,
            }}
          />
        );
      }
    }
  }

  return (
    <div
      style={{
        position: "relative",
        width: w * pixelSize,
        height: h * pixelSize,
      }}
    >
      {pixels}
    </div>
  );
}

export function BrushPalette({
  brushes,
  selectedBrush,
  brushMode,
  disabled,
  onSelectBrush,
  onToggleMode,
}: BrushPaletteProps) {
  const isActivate = brushMode === "activate";

  return (
    <div
      style={{
        position: "absolute",
        right: 8,
        top: "50%",
        transform: "translateY(-50%)",
        display: "flex",
        flexDirection: "column",
        gap: 4,
        background: "#0d0d14cc",
        border: "1px solid #2a2a3e",
        borderRadius: 8,
        padding: 6,
        opacity: disabled ? 0.3 : 1,
        pointerEvents: disabled ? "none" : "auto",
        zIndex: 10,
      }}
    >
      {brushes.map((brush) => (
        <button
          key={brush.id}
          onClick={() => onSelectBrush(brush.id)}
          title={brush.name}
          style={{
            width: 36,
            height: 36,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#1a1a2e",
            border:
              selectedBrush === brush.id
                ? "2px solid #4cc9f0"
                : "1px solid #2a2a3e",
            borderRadius: 6,
            cursor: "pointer",
            padding: 0,
          }}
        >
          {renderBrushPreview(brush.offsets)}
        </button>
      ))}

      <button
        onClick={onToggleMode}
        title={isActivate ? "Activar (ON)" : "Desactivar (OFF)"}
        style={{
          width: 36,
          height: 24,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: isActivate ? "#4cc9f0" : "#f72585",
          color: isActivate ? "#000" : "#fff",
          border: "none",
          borderRadius: 6,
          cursor: "pointer",
          fontSize: 11,
          fontWeight: 700,
          padding: 0,
          marginTop: 2,
        }}
      >
        {isActivate ? "ON" : "OFF"}
      </button>
    </div>
  );
}
