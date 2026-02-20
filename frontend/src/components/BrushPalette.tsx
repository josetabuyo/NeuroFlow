/** Brush palette — floating vertical toolbar with brush size controls. */

import { generateSquareBrush, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE } from "../brushes";

interface BrushPaletteProps {
  brushSize: number;
  brushMode: "activate" | "deactivate";
  disabled: boolean;
  onIncrease: () => void;
  onDecrease: () => void;
  onToggleMode: () => void;
}

function renderBrushPreview(size: number): React.ReactNode {
  const offsets = generateSquareBrush(size);
  const r = Math.floor(size / 2);
  const pixelSize = Math.max(2, Math.floor(28 / Math.max(size, 1)));

  const pixels: React.ReactNode[] = [];
  for (const [dx, dy] of offsets) {
    pixels.push(
      <div
        key={`${dx},${dy}`}
        style={{
          position: "absolute",
          left: (dx + r) * pixelSize,
          top: (dy + r) * pixelSize,
          width: pixelSize,
          height: pixelSize,
          background: "#e0e0ff",
          borderRadius: 1,
        }}
      />
    );
  }

  return (
    <div
      style={{
        position: "relative",
        width: size * pixelSize,
        height: size * pixelSize,
      }}
    >
      {pixels}
    </div>
  );
}

export function BrushPalette({
  brushSize,
  brushMode,
  disabled,
  onIncrease,
  onDecrease,
  onToggleMode,
}: BrushPaletteProps) {
  const isActivate = brushMode === "activate";
  const pixelCount = brushSize * brushSize;
  const canIncrease = brushSize < MAX_BRUSH_SIZE;
  const canDecrease = brushSize > MIN_BRUSH_SIZE;

  return (
    <div
      data-testid="brush-palette"
      style={{
        position: "absolute",
        right: 8,
        top: "50%",
        transform: "translateY(-50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
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
      {/* Increase size */}
      <button
        onClick={onIncrease}
        disabled={!canIncrease}
        title="Aumentar tamaño"
        aria-label="Aumentar pincel"
        style={{
          width: 36,
          height: 24,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#1a1a2e",
          border: "1px solid #2a2a3e",
          borderRadius: 6,
          cursor: canIncrease ? "pointer" : "default",
          color: canIncrease ? "#e0e0ff" : "#444",
          fontSize: 16,
          fontWeight: 700,
          padding: 0,
        }}
      >
        +
      </button>

      {/* Brush preview */}
      <div
        title={`Pincel ${brushSize}×${brushSize} (${pixelCount} px)`}
        style={{
          width: 36,
          height: 36,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#1a1a2e",
          border: "2px solid #4cc9f0",
          borderRadius: 6,
        }}
      >
        {renderBrushPreview(brushSize)}
      </div>

      {/* Size label */}
      <span
        data-testid="brush-size-label"
        style={{
          fontSize: 10,
          color: "#8888aa",
          textAlign: "center",
          lineHeight: 1,
          userSelect: "none",
        }}
      >
        {brushSize}×{brushSize}
      </span>

      {/* Decrease size */}
      <button
        onClick={onDecrease}
        disabled={!canDecrease}
        title="Reducir tamaño"
        aria-label="Reducir pincel"
        style={{
          width: 36,
          height: 24,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#1a1a2e",
          border: "1px solid #2a2a3e",
          borderRadius: 6,
          cursor: canDecrease ? "pointer" : "default",
          color: canDecrease ? "#e0e0ff" : "#444",
          fontSize: 16,
          fontWeight: 700,
          padding: 0,
        }}
      >
        −
      </button>

      {/* ON / OFF toggle */}
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
