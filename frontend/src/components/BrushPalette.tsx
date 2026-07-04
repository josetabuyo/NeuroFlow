/** Draw palette — brush size controls and mode toggle, expanded from the draw button. */

import { generateCircleBrush, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE } from "../brushes";

interface BrushPaletteProps {
  brushSize: number;
  brushMode: "activate" | "deactivate";
  dimmed?: boolean;
  onIncrease: () => void;
  onDecrease: () => void;
  onToggleMode: () => void;
  drawNoise?: number;
  onDrawNoiseChange?: (v: number) => void;
}

function renderBrushPreview(size: number): React.ReactNode {
  const offsets = generateCircleBrush(size);
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
  dimmed = false,
  onIncrease,
  onDecrease,
  onToggleMode,
  drawNoise,
  onDrawNoiseChange,
}: BrushPaletteProps) {
  const isActivate = brushMode === "activate";
  const pixelCount = generateCircleBrush(brushSize).length;
  const canIncrease = !dimmed && brushSize < MAX_BRUSH_SIZE;
  const canDecrease = !dimmed && brushSize > MIN_BRUSH_SIZE;

  return (
    <div
      data-testid="brush-palette"
      style={{
        boxSizing: "border-box",
        width: 32,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
        background: "#0d0d14cc",
        border: "1px solid #2a2a3e",
        borderRadius: 8,
        padding: 3,
        opacity: dimmed ? 0.3 : 1,
        pointerEvents: dimmed ? "none" : "auto",
        transition: "opacity 0.15s",
      }}
    >
      <div
        data-testid="brush-controls"
        style={{
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 4,
        }}
      >
        <button
          onClick={onIncrease}
          disabled={!canIncrease}
          title="Increase size"
          aria-label="Increase brush"
          style={{
            width: "100%",
            height: 22,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#1a1a2e",
            border: "1px solid #2a2a3e",
            borderRadius: 6,
            cursor: canIncrease ? "pointer" : "default",
            color: canIncrease ? "#e0e0ff" : "#444",
            fontSize: 14,
            fontWeight: 700,
            padding: 0,
          }}
        >
          +
        </button>

        <div
          title={`Brush ${brushSize}×${brushSize} (${pixelCount} px)`}
          style={{
            width: "100%",
            aspectRatio: "1 / 1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#1a1a2e",
            border: "2px solid #4cc9f0",
            borderRadius: 6,
            overflow: "hidden",
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
            width: "100%",
            height: 22,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#1a1a2e",
            border: "1px solid #2a2a3e",
            borderRadius: 6,
            cursor: canDecrease ? "pointer" : "default",
            color: canDecrease ? "#e0e0ff" : "#444",
            fontSize: 14,
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
            width: "100%",
            height: 22,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: isActivate ? "#4cc9f0" : "#f72585",
            color: isActivate ? "#000" : "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontSize: 10,
            fontWeight: 700,
            padding: 0,
            marginTop: 2,
          }}
        >
          {isActivate ? "ON" : "OFF"}
        </button>
      </div>

      {drawNoise !== undefined && onDrawNoiseChange && (
        <>
          <div style={{ width: "100%", height: 1, background: "#2a2a3e" }} />
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 3,
            }}
          >
            <span style={{ fontSize: 9, color: "#8888aa", userSelect: "none" }}>noise</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={drawNoise}
              onChange={(e) => onDrawNoiseChange(parseFloat(e.target.value))}
              title={`Background noise: ${drawNoise.toFixed(2)}`}
              style={{
                writingMode: "vertical-lr",
                direction: "rtl",
                width: 20,
                height: 60,
                cursor: "pointer",
                accentColor: "#a78bfa",
              }}
            />
            <span style={{ fontSize: 9, color: "#a78bfa", userSelect: "none" }}>
              {drawNoise.toFixed(2)}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
