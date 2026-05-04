/** Scene — manages the canvas area with draggable/resizable layer boxes. */

import { useRef, useState, useEffect, useCallback } from "react";
import { LayerBox, type BoxLayout } from "./LayerBox";
import { PixelCanvas } from "./PixelCanvas";
import { BrushPalette } from "./BrushPalette";

// ── Layout constants ────────────────────────────────────────────────────────
const TARGET_PX = 360;   // target box side for auto-sizing
const GAP       = 44;    // gap between boxes
const MARGIN    = 28;    // min distance from scene edge
const UNIT_PX   = 8;    // pixel size in 1:1 mode

/** Compute pixel-per-neuron size so the grid fits nicely in ~TARGET_PX. */
function autoPixelSize(gw: number, gh: number): number {
  return Math.max(4, Math.floor(TARGET_PX / Math.max(gw, gh)));
}

type Boxes = Record<string, BoxLayout>;

// ── Props ───────────────────────────────────────────────────────────────────
interface SceneProps {
  // Hidden layer
  grid: number[][];
  hiddenW: number;
  hiddenH: number;
  tensionGrid?: number[][] | null;
  tensionMode?: boolean;
  connectionMap?: (number | null)[][] | null;
  inspectedCell?: { x: number; y: number } | null;
  inspectInfo?: {
    activation: number;
    tension: number;
    total_dendritas: number;
    total_sinapsis: number;
  } | null;

  // Input layer
  inputFrame?: number[][] | null;
  inputW?: number;
  inputH?: number;
  /** Learned input weights — shown as overlay on input layer when inspecting. */
  inputWeightGrid?: number[][] | null;

  // Interaction
  onCellClick: (x: number, y: number) => void;
  onCellDrag: (x: number, y: number) => void;

  // BrushPalette props
  brushSize: number;
  brushMode: "activate" | "deactivate";
  inspectMode: boolean;
  canInspect: boolean;
  onIncrease: () => void;
  onDecrease: () => void;
  onToggleMode: () => void;
  onToggleInspect: () => void;
  onToggleTension: () => void;

  isInitializing?: boolean;
}

// ── Component ───────────────────────────────────────────────────────────────
export function Scene({
  grid, hiddenW, hiddenH,
  tensionGrid, tensionMode,
  connectionMap, inspectedCell, inspectInfo,
  inputFrame, inputW = 20, inputH = 20, inputWeightGrid,
  onCellClick, onCellDrag,
  brushSize, brushMode, inspectMode, canInspect,
  onIncrease, onDecrease, onToggleMode, onToggleInspect, onToggleTension,
  isInitializing,
}: SceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [oneToOne, setOneToOne] = useState(false);
  const [boxes, setBoxes] = useState<Boxes>({});
  const [layoutKey, setLayoutKey] = useState(0);

  const hasInput = !!(inputFrame && inputFrame.length > 0);
  const isInspecting = connectionMap != null;

  // ── Derive actual input dims from the frame ────────────────────────────
  const realInputW = hasInput ? (inputFrame![0]?.length ?? inputW) : inputW;
  const realInputH = hasInput ? inputFrame!.length : inputH;

  // ── Initialize layout ──────────────────────────────────────────────────
  useEffect(() => {
    if (grid.length === 0 || !containerRef.current) return;
    const { clientWidth: cw, clientHeight: ch } = containerRef.current;

    const hps = autoPixelSize(hiddenW, hiddenH);
    const hw = hps * hiddenW;
    const hh = hps * hiddenH;

    const ips = autoPixelSize(realInputW, realInputH);
    const iw = ips * realInputW;
    const ih = ips * realInputH;

    const totalW = hw + (hasInput ? GAP + iw : 0);
    const startX = Math.max(MARGIN, Math.floor((cw - totalW) / 2));

    const newBoxes: Boxes = {
      hidden: {
        x: startX,
        y: Math.max(MARGIN, Math.floor((ch - hh) / 2)),
        w: hw,
        h: hh,
      },
    };

    if (hasInput) {
      newBoxes.input = {
        x: startX + hw + GAP,
        y: Math.max(MARGIN, Math.floor((ch - ih) / 2)),
        w: iw,
        h: ih,
      };
    }

    setBoxes(newBoxes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutKey, grid.length > 0]);

  // When input layer appears for the first time and hidden box is already placed
  useEffect(() => {
    if (!hasInput || !containerRef.current) return;
    setBoxes(prev => {
      if (prev.input) return prev; // already positioned
      const hidden = prev.hidden;
      if (!hidden) return prev;
      const { clientHeight: ch } = containerRef.current!;
      const ips = autoPixelSize(realInputW, realInputH);
      const iw = ips * realInputW;
      const ih = ips * realInputH;
      return {
        ...prev,
        input: {
          x: hidden.x + hidden.w + GAP,
          y: Math.max(MARGIN, Math.floor((ch - ih) / 2)),
          w: iw,
          h: ih,
        },
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasInput]);

  // ── 1:1 mode — resize boxes so each neuron = UNIT_PX px ───────────────
  const toggleOneToOne = useCallback(() => {
    setOneToOne(prev => {
      const next = !prev;
      if (next) {
        setBoxes(b => {
          const n = { ...b };
          if (n.hidden) n.hidden = { ...n.hidden, w: UNIT_PX * hiddenW, h: UNIT_PX * hiddenH };
          if (n.input)  n.input  = { ...n.input,  w: UNIT_PX * realInputW, h: UNIT_PX * realInputH };
          return n;
        });
      }
      return next;
    });
  }, [hiddenW, hiddenH, realInputW, realInputH]);

  // ── Reset layout ───────────────────────────────────────────────────────
  const resetLayout = useCallback(() => {
    setOneToOne(false);
    setBoxes({});
    setLayoutKey(k => k + 1);
  }, []);

  // ── Box update (move or resize) ────────────────────────────────────────
  const updateBox = useCallback((id: string, layout: BoxLayout) => {
    setBoxes(prev => ({ ...prev, [id]: layout }));
    setOneToOne(false);
  }, []);

  const hiddenBox = boxes.hidden;
  const inputBox  = boxes.input;

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}
    >
      {/* ── Scene toolbar ── */}
      <div style={{
        position: "absolute", top: 8, left: 8, display: "flex", gap: 6, zIndex: 30,
      }}>
        <button
          onClick={toggleOneToOne}
          title="Toggle 1:1 scale — all neurons the same pixel size"
          style={{
            padding: "3px 9px",
            background: oneToOne ? "#4cc9f0" : "#14142a",
            border: `1px solid ${oneToOne ? "#4cc9f0" : "#2a2a3e"}`,
            borderRadius: 4,
            color: oneToOne ? "#0a0a0a" : "#555",
            fontSize: "0.62rem",
            fontFamily: "monospace",
            fontWeight: 600,
            cursor: "pointer",
            letterSpacing: "0.05em",
            transition: "all 0.15s",
          }}
        >
          1:1
        </button>
        <button
          onClick={resetLayout}
          title="Reset layout"
          style={{
            padding: "3px 8px",
            background: "#14142a",
            border: "1px solid #2a2a3e",
            borderRadius: 4,
            color: "#444",
            fontSize: "0.75rem",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          ↺
        </button>
      </div>

      {/* ── Hidden layer box ── */}
      {hiddenBox && (
        <LayerBox id="hidden" label="Hidden Layer" layout={hiddenBox} onUpdate={updateBox}>
          <PixelCanvas
            grid={grid}
            tensionGrid={tensionGrid}
            tensionMode={tensionMode}
            width={hiddenW}
            height={hiddenH}
            connectionMap={connectionMap}
            inspectedCell={inspectedCell}
            onCellClick={onCellClick}
            onCellDrag={onCellDrag}
          />
          {/* Inspect info overlay */}
          {inspectInfo && inspectedCell && (
            <div style={{
              position: "absolute", top: 8, right: 8,
              background: "rgba(10,10,20,0.92)",
              border: "1px solid #2a2a3e",
              borderRadius: 6,
              padding: "8px 12px",
              fontSize: "0.7rem",
              fontFamily: "monospace",
              color: "#ccc",
              display: "flex", flexDirection: "column", gap: 3,
              pointerEvents: "none",
              zIndex: 10,
            }}>
              <div style={{ color: "#ffff00", fontWeight: 600, fontSize: "0.75rem", marginBottom: 2 }}>
                Neuron ({inspectedCell.x}, {inspectedCell.y})
              </div>
              <div>
                <span style={{ color: "#888" }}>Activation: </span>
                <span style={{ color: inspectInfo.activation > 0.5 ? "#fff" : "#555" }}>
                  {inspectInfo.activation.toFixed(2)}
                </span>
              </div>
              <div>
                <span style={{ color: "#888" }}>Tension: </span>
                <span style={{ color: inspectInfo.tension > 0 ? "#ff8c00" : inspectInfo.tension < 0 ? "#5000ff" : "#555" }}>
                  {inspectInfo.tension >= 0 ? "+" : ""}{inspectInfo.tension.toFixed(4)}
                </span>
              </div>
              <div style={{ borderTop: "1px solid #1a1a2e", paddingTop: 3, marginTop: 2, color: "#666", fontSize: "0.6rem" }}>
                {inspectInfo.total_dendritas} dendrites / {inspectInfo.total_sinapsis} synapses
              </div>
            </div>
          )}
        </LayerBox>
      )}

      {/* ── Input layer box ── */}
      {inputBox && hasInput && (
        <LayerBox
          id="input"
          label="Input Layer"
          layout={inputBox}
          onUpdate={updateBox}
          highlighted={isInspecting}
        >
          {/*
            weightOverlay shows the learned input weights for the inspected neuron
            as a green overlay — same visual language as the connection map on the hidden layer.
            Only shown during inspection (isInspecting). Not a duplicate: this is the
            input-region counterpart of the hidden-layer connectionMap overlay.
          */}
          <PixelCanvas
            grid={inputFrame!}
            width={realInputW}
            height={realInputH}
            weightOverlay={isInspecting ? (inputWeightGrid ?? null) : null}
            onCellClick={() => {}}
          />
        </LayerBox>
      )}

      {/* ── BrushPalette ── */}
      <div style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", zIndex: 30 }}>
        <BrushPalette
          brushSize={brushSize}
          brushMode={brushMode}
          inspectMode={inspectMode}
          tensionMode={tensionMode ?? false}
          canInspect={canInspect}
          onIncrease={onIncrease}
          onDecrease={onDecrease}
          onToggleMode={onToggleMode}
          onToggleInspect={onToggleInspect}
          onToggleTension={onToggleTension}
        />
      </div>

      {/* ── Initializing overlay ── */}
      {isInitializing && (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(10,10,15,0.75)",
          zIndex: 40,
        }}>
          <div style={{ textAlign: "center", color: "#888" }}>
            <div className="neuro-spinner" />
            <p style={{ marginTop: 12, fontSize: "0.85rem" }}>Building network...</p>
          </div>
        </div>
      )}
    </div>
  );
}
