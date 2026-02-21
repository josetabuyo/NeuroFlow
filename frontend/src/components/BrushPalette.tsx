/** Toolbar — floating vertical toolbar with tool selection + brush controls. */

import { generateSquareBrush, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE } from "../brushes";

interface BrushPaletteProps {
  brushSize: number;
  brushMode: "activate" | "deactivate";
  inspectMode: boolean;
  tensionMode: boolean;
  canInspect: boolean;
  onIncrease: () => void;
  onDecrease: () => void;
  onToggleMode: () => void;
  onToggleInspect: () => void;
  onToggleTension: () => void;
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

const toolBtnStyle = (
  active: boolean,
  color: string,
): React.CSSProperties => ({
  width: 36,
  height: 30,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: active ? color : "#1a1a2e",
  color: active ? "#0a0a0f" : "#666",
  border: active ? `2px solid ${color}` : "1px solid #2a2a3e",
  borderRadius: 6,
  cursor: "pointer",
  fontSize: 15,
  fontWeight: 700,
  padding: 0,
  transition: "all 0.15s",
});

export function BrushPalette({
  brushSize,
  brushMode,
  inspectMode,
  tensionMode,
  canInspect,
  onIncrease,
  onDecrease,
  onToggleMode,
  onToggleInspect,
  onToggleTension,
}: BrushPaletteProps) {
  const isActivate = brushMode === "activate";
  const pixelCount = brushSize * brushSize;
  const canIncrease = !inspectMode && brushSize < MAX_BRUSH_SIZE;
  const canDecrease = !inspectMode && brushSize > MIN_BRUSH_SIZE;
  const brushActive = !inspectMode;

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
        zIndex: 10,
      }}
    >
      <button
        onClick={onToggleTension}
        title={tensionMode ? "Hide tensions" : "View surface tensions"}
        aria-label="View tensions"
        style={{
          ...toolBtnStyle(tensionMode, "#ff6b35"),
          width: "100%",
          cursor: "pointer",
        }}
      >
        ≋
      </button>

      <button
        onClick={onToggleInspect}
        disabled={!canInspect && !inspectMode}
        title="Inspect connections"
        aria-label={inspectMode ? "Brush tool" : "Inspect tool"}
        style={{
          ...toolBtnStyle(inspectMode, "#ffff00"),
          width: "100%",
          cursor: canInspect || inspectMode ? "pointer" : "not-allowed",
        }}
      >
        ⊙
      </button>

      <div
        style={{
          width: "100%",
          height: 1,
          background: "#2a2a3e",
        }}
      />

      <div
        data-testid="brush-controls"
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 4,
          opacity: inspectMode ? 0.3 : 1,
          pointerEvents: inspectMode ? "none" : "auto",
          transition: "opacity 0.15s",
        }}
      >
        <button
          onClick={onIncrease}
          disabled={!canIncrease}
          title="Increase size"
          aria-label="Increase brush"
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

        <div
          title={`Brush ${brushSize}×${brushSize} (${pixelCount} px)`}
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

        <button
          onClick={onDecrease}
          disabled={!canDecrease}
          title="Decrease size"
          aria-label="Decrease brush"
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

        <button
          onClick={onToggleMode}
          title={isActivate ? "Activate (ON)" : "Deactivate (OFF)"}
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
    </div>
  );
}
