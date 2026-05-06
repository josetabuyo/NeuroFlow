/** Scene — manages the canvas area with draggable/resizable region boxes. */

import { useRef, useState, useEffect, useCallback } from "react";
import { LayerBox, type BoxLayout } from "./LayerBox";
import { PixelCanvas } from "./PixelCanvas";
import { BrushPalette } from "./BrushPalette";

// ── Layout constants ────────────────────────────────────────────────────────
const TARGET_PX = 360;
const GAP       = 44;
const MARGIN    = 28;
const UNIT_PX   = 8;

const MAX_PS = 48;  // cap so tiny grids (1×2) don't get enormous cells

function autoPixelSize(gw: number, gh: number): number {
  return Math.min(MAX_PS, Math.max(4, Math.floor(TARGET_PX / Math.max(gw, gh))));
}

type Boxes = Record<string, BoxLayout>;

interface SceneProps {
  /** All region grids keyed by region id, in display order. */
  regions: Record<string, number[][]>;
  /** Tension grids keyed by region id — shown when tensionMode is on. */
  tensionRegions?: Record<string, number[][]>;
  /** Which region is the interactive tissue (brush / inspect). */
  tissueId: string;
  /** Which region shows the learned weight overlay during inspection. */
  inputId?: string | null;
  /** Label grids keyed by region id (for output regions with a label source). */
  labels?: Record<string, number[][]>;

  tensionMode?: boolean;
  connectionMap?: (number | null)[][] | null;
  inspectedCell?: { x: number; y: number } | null;
  inspectedRegionId?: string | null;
  inspectInfo?: {
    activation: number;
    tension: number;
    total_dendritas: number;
    total_sinapsis: number;
  } | null;
  inputWeightGrid?: number[][] | null;

  // Interaction
  onCellClick: (x: number, y: number, regionId?: string) => void;
  onCellDrag: (x: number, y: number) => void;

  // BrushPalette
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

export function Scene({
  regions,
  tensionRegions = {},
  tissueId,
  inputId,
  labels = {},
  tensionGrid: _tensionGrid, tensionMode,
  connectionMap, inspectedCell, inspectedRegionId, inspectInfo,
  inputWeightGrid,
  onCellClick, onCellDrag,
  brushSize, brushMode, inspectMode, canInspect,
  onIncrease, onDecrease, onToggleMode, onToggleInspect, onToggleTension,
  isInitializing,
}: SceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [oneToOne, setOneToOne] = useState(false);
  const [boxes, setBoxes] = useState<Boxes>({});
  const [layoutKey, setLayoutKey] = useState(0);

  const isInspecting = connectionMap != null;

  // ── Initialize / re-layout when regions change or layoutKey bumps ────────
  useEffect(() => {
    const ids = Object.keys(regions);
    if (ids.length === 0 || !containerRef.current) return;
    const { clientWidth: cw, clientHeight: ch } = containerRef.current;

    // Compute natural size per region
    const sizes = ids.map(id => {
      const grid = regions[id];
      const gh = grid.length;
      const gw = grid[0]?.length ?? 1;
      const ps = autoPixelSize(gw, gh);
      return { id, w: ps * gw, h: ps * gh };
    });

    const totalH = sizes.reduce((s, r, i) => s + r.h + (i > 0 ? GAP : 0), 0);
    let y = Math.max(MARGIN, Math.floor((ch - totalH) / 2));

    const newBoxes: Boxes = {};
    for (const { id, w, h } of sizes) {
      newBoxes[id] = {
        x: Math.max(MARGIN, Math.floor((cw - w) / 2)),
        y,
        w,
        h,
      };
      y += h + GAP;
    }
    setBoxes(newBoxes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutKey, Object.keys(regions).join(",")]);

  // Position any new region that appears after initial layout
  useEffect(() => {
    const ids = Object.keys(regions);
    if (ids.length === 0 || !containerRef.current) return;
    const { clientHeight: ch } = containerRef.current;

    setBoxes(prev => {
      const newEntries = ids.filter(id => !prev[id]);
      if (newEntries.length === 0) return prev;

      const updated = { ...prev };
      // Place new regions below the bottommost existing box
      const bottommost = Object.values(prev).reduce(
        (mx, b) => Math.max(mx, b.y + b.h),
        MARGIN,
      );
      let y = bottommost + GAP;
      for (const id of newEntries) {
        const grid = regions[id];
        const gh = grid.length;
        const gw = grid[0]?.length ?? 1;
        const ps = autoPixelSize(gw, gh);
        const w = ps * gw;
        const h = ps * gh;
        const { clientWidth: cw } = containerRef.current!;
        updated[id] = {
          x: Math.max(MARGIN, Math.floor((cw - w) / 2)),
          y,
          w,
          h,
        };
        y += h + GAP;
      }
      return updated;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.keys(regions).join(",")]);

  // ── 1:1 mode ──────────────────────────────────────────────────────────────
  const toggleOneToOne = useCallback(() => {
    setOneToOne(prev => {
      const next = !prev;
      if (next) {
        setBoxes(b => {
          const n = { ...b };
          for (const id of Object.keys(n)) {
            const grid = regions[id];
            if (!grid) continue;
            const gh = grid.length;
            const gw = grid[0]?.length ?? 1;
            n[id] = { ...n[id], w: UNIT_PX * gw, h: UNIT_PX * gh };
          }
          return n;
        });
      }
      return next;
    });
  }, [regions]);

  const resetLayout = useCallback(() => {
    setOneToOne(false);
    setBoxes({});
    setLayoutKey(k => k + 1);
  }, []);

  const updateBox = useCallback((id: string, layout: BoxLayout) => {
    setBoxes(prev => ({ ...prev, [id]: layout }));
    setOneToOne(false);
  }, []);

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
          title="Toggle 1:1 scale"
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

      {/* ── Region boxes ── */}
      {Object.entries(regions).map(([id, grid]) => {
        const box = boxes[id];
        if (!box) return null;

        const isTissue = id === tissueId;
        const isInputRegion = id === inputId;
        const isInspectedRegion = id === inspectedRegionId;
        const gh = grid.length;
        const gw = grid[0]?.length ?? 1;
        const labelGrid = labels[id];

        return (
          <LayerBox
            key={id}
            id={id}
            label={id}
            layout={box}
            onUpdate={updateBox}
            highlighted={isInputRegion && inputWeightGrid != null}
          >
            <PixelCanvas
              grid={grid}
              tensionGrid={tensionRegions[id] ?? undefined}
              tensionMode={tensionMode}
              width={gw}
              height={gh}
              connectionMap={isTissue ? connectionMap : undefined}
              inspectedCell={isInspectedRegion ? inspectedCell : undefined}
              weightOverlay={isInputRegion ? (inputWeightGrid ?? null) : null}
              onCellClick={(cx, cy) => onCellClick(cx, cy, id)}
              onCellDrag={isTissue ? onCellDrag : undefined}
            />

            {/* Inspect info overlay on the inspected region's box */}
            {isInspectedRegion && inspectInfo && inspectedCell && (
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

            {/* Label column overlay for output regions */}
            {labelGrid && labelGrid.length > 0 && (
              <div style={{
                position: "absolute", top: 0, right: -48,
                display: "flex", flexDirection: "column",
                gap: 2, zIndex: 5,
              }}>
                {labelGrid.map((row, ri) =>
                  row.map((v, ci) => (
                    <div key={`lbl-${ri}-${ci}`} style={{
                      width: 40, height: box.h / (gh || 1) - 2,
                      minHeight: 16,
                      background: v > 0.5 ? "#4cc9f080" : "transparent",
                      border: `1px solid ${v > 0.5 ? "#4cc9f0" : "#2a2a3e"}`,
                      borderRadius: 2,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: "0.5rem", color: v > 0.5 ? "#4cc9f0" : "#333",
                      fontFamily: "monospace",
                    }}>
                      {v > 0.5 ? "▶" : ""}
                    </div>
                  ))
                )}
                <div style={{ fontSize: "0.45rem", color: "#444", textAlign: "center", marginTop: 2 }}>label</div>
              </div>
            )}
          </LayerBox>
        );
      })}

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
