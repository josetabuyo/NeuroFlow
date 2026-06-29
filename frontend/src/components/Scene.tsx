/** Scene — manages the canvas area with draggable/resizable region boxes. */

import { useRef, useState, useEffect, useCallback } from "react";
import { LayerBox, HEADER_H, type BoxLayout } from "./LayerBox";
import { PixelCanvas } from "./PixelCanvas";
import { BrushPalette } from "./BrushPalette";
import type { RegionOverlay } from "../types";

const TARGET_PX = 360;
const GAP       = 44;
const MARGIN    = 28;
const UNIT_PX   = 8;
const INFO_W    = 180;

function autoPixelSize(gw: number, gh: number): number {
  const MAX_PS = 48;
  return Math.min(MAX_PS, Math.max(4, Math.floor(TARGET_PX / Math.max(gw, gh))));
}

type Boxes = Record<string, BoxLayout>;

interface SceneProps {
  regions: Record<string, number[][]>;
  tensionRegions?: Record<string, number[][]>;
  tissueId: string;
  inputId?: string | null;
  labels?: Record<string, number[][]>;
  neuronLabelMap?: Record<string, string[][]>;

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
  sourceWeightGrids?: Record<string, { grid: (number | null)[][], width: number, height: number }>;
  regionOverlays?: Record<string, RegionOverlay>;
  nerveCircles?: Array<{ cx: number; cy: number; radius: number }> | null;

  tensionMode?: boolean;

  onCellClick: (x: number, y: number, regionId?: string) => void;
  onCellDrag: (x: number, y: number, regionId?: string) => void;
  onCellDragEnd?: (cells: { x: number; y: number }[], regionId?: string) => void;

  brushSize: number;
  brushMode: "activate" | "deactivate";
  inspectMode: boolean;
  canInspect: boolean;
  onIncrease: () => void;
  onDecrease: () => void;
  onToggleMode: () => void;
  onToggleInspect: () => void;
  onToggleTension: () => void;

  drawNoise?: number;
  onDrawNoiseChange?: (v: number) => void;

  isInitializing?: boolean;
}

export function Scene({
  regions,
  tensionRegions = {},
  tissueId,
  inputId,
  labels = {},
  neuronLabelMap = {},
  connectionMap,
  inspectedCell,
  inspectedRegionId,
  inspectInfo,
  inputWeightGrid,
  sourceWeightGrids,
  tensionMode,
  regionOverlays,
  nerveCircles,
  onCellClick, onCellDrag, onCellDragEnd,
  brushSize, brushMode, inspectMode, canInspect,
  onIncrease, onDecrease, onToggleMode, onToggleInspect, onToggleTension,
  drawNoise, onDrawNoiseChange,
  isInitializing,
}: SceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [oneToOne, setOneToOne] = useState(false);
  const [boxes, setBoxes] = useState<Boxes>({});
  const [layoutKey, setLayoutKey] = useState(0);

  // Info panel position (draggable, independent of boxes state)
  const [infoPanelPos, setInfoPanelPos] = useState<{ x: number; y: number } | null>(null);
  const infoDragRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  const lastInspectedKeyRef = useRef<string | null>(null);

  const isInspecting = connectionMap != null;
  const tissueInspectKey = `inspect:${tissueId}`;
  const inputInspectKey = inputId ? `inspect:${inputId}` : null;
  // Only show the tissue weight popup if there are actual non-null weights
  const hasAnyTissueWeights = !!connectionMap?.some(row => row.some(v => v !== null));

  // ── Initialize layout ───────────────────────────────────────────────
  useEffect(() => {
    const ids = Object.keys(regions);
    if (ids.length === 0 || !containerRef.current) return;
    const { clientWidth: cw, clientHeight: ch } = containerRef.current;

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

    setBoxes(prev => {
      const newEntries = ids.filter(id => !prev[id]);
      if (newEntries.length === 0) return prev;

      const updated = { ...prev };
      const bottommost = Object.values(prev).reduce((mx, b) => Math.max(mx, b.y + b.h), MARGIN);
      let y = bottommost + GAP;
      for (const id of newEntries) {
        const grid = regions[id];
        const gh = grid.length;
        const gw = grid[0]?.length ?? 1;
        const ps = autoPixelSize(gw, gh);
        const w = ps * gw;
        const h = ps * gh;
        const { clientWidth: cw } = containerRef.current!;
        updated[id] = { x: Math.max(MARGIN, Math.floor((cw - w) / 2)), y, w, h };
        y += h + GAP;
      }
      return updated;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.keys(regions).join(",")]);

  // ── Inspect popup boxes ─────────────────────────────────────────────
  // Input neurons (NeuronaEntrada) have no connections — suppress the weight popup for them.
  const isInputNeuron = !!inputId && inspectedRegionId === inputId;

  useEffect(() => {
    if (!isInspecting || isInputNeuron) {
      setBoxes(prev => {
        if (!prev[tissueInspectKey]) return prev;
        const without: Boxes = { ...prev };
        delete without[tissueInspectKey];
        if (inputInspectKey) delete without[inputInspectKey];
        return without;
      });
      if (!isInspecting) {
        setInfoPanelPos(null);
        lastInspectedKeyRef.current = null;
      }
      return;
    }
    // Place tissue inspect popup (once per region change, next to tissue box)
    setBoxes(prev => {
      if (prev[tissueInspectKey]) return prev;
      const tb = prev[tissueId];
      if (!tb) return prev;
      return {
        ...prev,
        [tissueInspectKey]: { x: tb.x + tb.w + GAP, y: tb.y, w: tb.w, h: tb.h },
      };
    });
  }, [isInspecting, isInputNeuron, tissueId, inspectedRegionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Show/hide inspect:input box based on inputWeightGrid availability
  useEffect(() => {
    if (!isInspecting) return;
    setBoxes(prev => {
      if (inputWeightGrid && inputInspectKey) {
        if (prev[inputInspectKey]) return prev;
        const ib = inputId ? prev[inputId] : null;
        if (!ib) return prev;
        return { ...prev, [inputInspectKey]: { x: ib.x + ib.w + GAP, y: ib.y, w: ib.w, h: ib.h } };
      } else {
        if (!inputInspectKey || !prev[inputInspectKey]) return prev;
        const without = { ...prev };
        delete without[inputInspectKey];
        return without;
      }
    });
  }, [inputWeightGrid !== null, isInspecting, inputId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Show/hide inspect:{id} boxes based on sourceWeightGrids content
  useEffect(() => {
    if (!isInspecting) return;
    const entries = Object.entries(sourceWeightGrids ?? {});
    setBoxes(prev => {
      let next = { ...prev };
      // Remove stale inspect boxes
      const activeKeys = new Set(entries.map(([id]) => `inspect:${id}`));
      for (const key of Object.keys(next)) {
        if (key.startsWith("inspect:") && key !== tissueInspectKey && key !== inputInspectKey && !activeKeys.has(key)) {
          const { [key]: _, ...rest } = next;
          next = rest;
        }
      }
      // Add new inspect boxes
      let offset = 0;
      for (const [id, { width, height }] of entries) {
        const boxKey = `inspect:${id}`;
        if (next[boxKey]) continue;
        const anchor = next[id] ?? (inspectedRegionId ? next[inspectedRegionId] : null);
        if (!anchor) continue;
        const ps = autoPixelSize(width, height);
        const bw = width * ps + 24;
        const bh = height * ps + HEADER_H + 8;
        next = { ...next, [boxKey]: { x: anchor.x + anchor.w + GAP + offset, y: anchor.y, w: bw, h: bh } };
        offset += GAP;
      }
      return next;
    });
  }, [JSON.stringify(Object.keys(sourceWeightGrids ?? {})), isInspecting, inspectedRegionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Position info panel when inspecting a new cell
  useEffect(() => {
    if (!inspectedCell || !inspectedRegionId) return;
    const key = `${inspectedRegionId}:${inspectedCell.x}:${inspectedCell.y}`;
    if (key === lastInspectedKeyRef.current) return;
    lastInspectedKeyRef.current = key;

    const regionBox = boxes[inspectedRegionId];
    if (!regionBox) return;
    // Place panel to the left of the region box, outside any neuron grid
    const panelX = Math.max(MARGIN, regionBox.x - INFO_W - GAP);
    const panelY = regionBox.y + HEADER_H;
    setInfoPanelPos({ x: panelX, y: panelY });
  }, [inspectedCell?.x, inspectedCell?.y, inspectedRegionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 1:1 mode ───────────────────────────────────────────────────────
  const toggleOneToOne = useCallback(() => {
    setOneToOne(prev => {
      const next = !prev;
      if (next) {
        setBoxes(b => {
          const n = { ...b };
          for (const id of Object.keys(regions)) {
            if (!n[id]) continue;
            const grid = regions[id];
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

  // ── SVG helper: box center ──────────────────────────────────────────
  const boxCenter = (b: BoxLayout) => ({ x: b.x + b.w / 2, y: b.y + HEADER_H + b.h / 2 });

  // Cell position in scene coords
  const cellPos = (regionId: string, cx: number, cy: number) => {
    const b = boxes[regionId];
    if (!b) return null;
    const grid = regions[regionId];
    const gw = grid?.[0]?.length ?? 1;
    const gh = grid?.length ?? 1;
    return { x: b.x + (cx + 0.5) * (b.w / gw), y: b.y + HEADER_H + (cy + 0.5) * (b.h / gh) };
  };

  const tCell = (inspectedCell && inspectedRegionId) ? cellPos(inspectedRegionId, inspectedCell.x, inspectedCell.y) : null;

  // ── Nerve circle center in tissue-inspect popup scene coords ────────
  // Used as the origin of the dashed connector line (center→center).
  const nerveLineStart = (() => {
    if (!nerveCircles || nerveCircles.length === 0) return null;
    const pb = boxes[tissueInspectKey];
    if (!pb) return null;
    const tissueGrid = regions[tissueId];
    const tw = tissueGrid?.[0]?.length ?? 1;
    const th = tissueGrid?.length ?? 1;
    const c = nerveCircles[0];
    return {
      x: pb.x + (c.cx + 0.5) * (pb.w / tw),
      y: pb.y + HEADER_H + (c.cy + 0.5) * (pb.h / th),
    };
  })();

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}
    >
      {/* ── SVG connector layer ── */}
      <svg
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 15 }}
      >
        {/* tissue inspect popup → inspected neuron (center-to-center via nerve circle if available) */}
        {boxes[tissueInspectKey] && hasAnyTissueWeights && tCell && (() => {
          const start = nerveLineStart ?? boxCenter(boxes[tissueInspectKey]);
          return (
            <line
              x1={start.x} y1={start.y}
              x2={tCell.x} y2={tCell.y}
              stroke="#22c55e" strokeWidth={1.5} strokeDasharray="5 4" opacity={0.5}
            />
          );
        })()}

        {/* input inspect popup → inspected neuron */}
        {inputInspectKey && boxes[inputInspectKey] && tCell && (
          <line
            x1={boxCenter(boxes[inputInspectKey]).x} y1={boxCenter(boxes[inputInspectKey]).y}
            x2={tCell.x} y2={tCell.y}
            stroke="#22c55e" strokeWidth={1.5} strokeDasharray="5 4" opacity={0.5}
          />
        )}

        {/* inspect:output → inspected neuron */}
        {boxes["inspect:output"] && tCell && (
          <line
            x1={boxCenter(boxes["inspect:output"]).x} y1={boxCenter(boxes["inspect:output"]).y}
            x2={tCell.x} y2={tCell.y}
            stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="5 4" opacity={0.5}
          />
        )}

        {/* inspect:nociceptor → inspected neuron */}
        {boxes["inspect:nociceptor"] && tCell && (
          <line
            x1={boxCenter(boxes["inspect:nociceptor"]).x} y1={boxCenter(boxes["inspect:nociceptor"]).y}
            x2={tCell.x} y2={tCell.y}
            stroke="#f43f5e" strokeWidth={1.5} strokeDasharray="5 4" opacity={0.5}
          />
        )}

        {/* info panel → inspected cell */}
        {infoPanelPos && tCell && (
          <line
            x1={infoPanelPos.x + INFO_W / 2} y1={infoPanelPos.y + 16}
            x2={tCell.x} y2={tCell.y}
            stroke="#ffff00" strokeWidth={1.5} strokeDasharray="4 4" opacity={0.6}
          />
        )}
      </svg>

      {/* ── Scene toolbar ── */}
      <div style={{ position: "absolute", top: 8, left: 8, display: "flex", gap: 6, zIndex: 30 }}>
        <button onClick={toggleOneToOne} title="Toggle 1:1 scale" style={{
          padding: "3px 9px", background: oneToOne ? "#4cc9f0" : "#14142a",
          border: `1px solid ${oneToOne ? "#4cc9f0" : "#2a2a3e"}`, borderRadius: 4,
          color: oneToOne ? "#0a0a0a" : "#555", fontSize: "0.62rem", fontFamily: "monospace",
          fontWeight: 600, cursor: "pointer", letterSpacing: "0.05em", transition: "all 0.15s",
        }}>1:1</button>
        <button onClick={resetLayout} title="Reset layout" style={{
          padding: "3px 8px", background: "#14142a", border: "1px solid #2a2a3e",
          borderRadius: 4, color: "#444", fontSize: "0.75rem", cursor: "pointer", transition: "all 0.15s",
        }}>↺</button>
      </div>

      {/* ── Region boxes ── */}
      {Object.entries(regions).map(([id, grid]) => {
        const box = boxes[id];
        if (!box) return null;
        const isTissue = id === tissueId;
        const gh = grid.length;
        const gw = grid[0]?.length ?? 1;
        const labelGrid = labels[id];

        return (
          <LayerBox key={id} id={id} label={id} layout={box} onUpdate={updateBox}>
            <PixelCanvas
              grid={grid}
              tensionGrid={tensionRegions[id]}
              tensionMode={tensionMode}
              width={gw}
              height={gh}
              neuronLabels={neuronLabelMap[id] ?? null}
              inspectedCell={id === inspectedRegionId ? inspectedCell : undefined}
              onCellClick={(cx, cy) => onCellClick(cx, cy, id)}
              onCellDrag={(cx, cy) => onCellDrag(cx, cy, id)}
              onDragEnd={(cells) => onCellDragEnd?.(cells, id)}
              brushSize={brushSize}
              brushMode={brushMode}
            />

            {/* Label column beside output regions */}
            {labelGrid && labelGrid.length > 0 && (
              <div style={{
                position: "absolute", top: 0, right: -48,
                display: "flex", flexDirection: "column", gap: 2, zIndex: 5,
              }}>
                {labelGrid.map((row, ri) =>
                  row.map((v, ci) => (
                    <div key={`lbl-${ri}-${ci}`} style={{
                      width: 40,
                      height: box.h / (gh || 1) - 2,
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

      {/* ── Inspect popup: tissue weights ── */}
      {boxes[tissueInspectKey] && connectionMap && hasAnyTissueWeights && (() => {
        const grid = regions[tissueId];
        const gw = grid?.[0]?.length ?? 1;
        const gh = grid?.length ?? 1;
        return (
          <LayerBox
            id={tissueInspectKey}
            label={`weights ← ${inspectedRegionId ?? tissueId}`}
            layout={boxes[tissueInspectKey]}
            onUpdate={updateBox}
            highlighted
          >
            <PixelCanvas
              grid={[]}
              width={gw}
              height={gh}
              weightGrid={connectionMap}
              nerveCircles={nerveCircles}
              overlayGrids={Object.values(regionOverlays ?? {})
                .sort((a, b) => b.density - a.density)
                .map(o => o.grid)}
              onCellClick={() => {}}
            />
          </LayerBox>
        );
      })()}

      {/* ── Inspect popup: input weights ── */}
      {inputInspectKey && boxes[inputInspectKey] && inputWeightGrid && inputId && (() => {
        const grid = regions[inputId];
        const gw = grid?.[0]?.length ?? 1;
        const gh = grid?.length ?? 1;
        return (
          <LayerBox
            id={inputInspectKey}
            label={`weights ← ${inputId}`}
            layout={boxes[inputInspectKey]}
            onUpdate={updateBox}
            highlighted
          >
            <PixelCanvas
              grid={[]}
              width={gw}
              height={gh}
              weightGrid={inputWeightGrid as (number | null)[][]}
              onCellClick={() => {}}
            />
          </LayerBox>
        );
      })()}

      {/* ── Inspect popups: per-source-region weight grids (output_L/T, nociceptor_L/T, …) ── */}
      {Object.entries(sourceWeightGrids ?? {}).map(([id, { grid, width, height }]) => {
        const boxKey = `inspect:${id}`;
        if (!boxes[boxKey]) return null;
        return (
          <LayerBox
            key={boxKey}
            id={boxKey}
            label={`weights ← ${id}`}
            layout={boxes[boxKey]}
            onUpdate={updateBox}
            highlighted
          >
            <PixelCanvas
              grid={[]}
              width={width}
              height={height}
              weightGrid={grid}
              onCellClick={() => {}}
            />
          </LayerBox>
        );
      })}

      {/* ── Floating neuron info panel ── */}
      {infoPanelPos && isInspecting && inspectInfo && inspectedCell && inspectedRegionId && (
        <div
          onPointerDown={e => {
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
            infoDragRef.current = { sx: e.clientX, sy: e.clientY, ox: infoPanelPos.x, oy: infoPanelPos.y };
          }}
          onPointerMove={e => {
            if (!infoDragRef.current) return;
            const { sx, sy, ox, oy } = infoDragRef.current;
            setInfoPanelPos({ x: ox + e.clientX - sx, y: oy + e.clientY - sy });
          }}
          onPointerUp={() => { infoDragRef.current = null; }}
          style={{
            position: "absolute",
            left: infoPanelPos.x,
            top: infoPanelPos.y,
            width: INFO_W,
            zIndex: 25,
            background: "rgba(10,10,20,0.96)",
            border: "1px solid #ffff00",
            borderRadius: 6,
            padding: "8px 12px",
            fontSize: "0.7rem",
            fontFamily: "monospace",
            color: "#ccc",
            display: "flex",
            flexDirection: "column",
            gap: 4,
            cursor: "grab",
            userSelect: "none",
          }}
        >
          {(() => {
            const lbl = inspectedRegionId
              ? (neuronLabelMap[inspectedRegionId]?.[inspectedCell.y]?.[inspectedCell.x] ?? null)
              : null;
            return lbl ? (
              <div style={{
                color: "#000", background: "#ffff00", fontWeight: 700,
                fontSize: "1.1rem", textAlign: "center", borderRadius: 3,
                padding: "2px 0", letterSpacing: 2, marginBottom: 4,
              }}>
                {lbl}
              </div>
            ) : null;
          })()}
          <div style={{ color: "#ffff00", fontWeight: 600, fontSize: "0.72rem", marginBottom: 1 }}>
            {inspectedRegionId} ({inspectedCell.x}, {inspectedCell.y})
          </div>
          <div>
            <span style={{ color: "#888" }}>act: </span>
            <span style={{ color: inspectInfo.activation > 0.5 ? "#fff" : "#555" }}>
              {inspectInfo.activation.toFixed(3)}
            </span>
          </div>
          <div>
            <span style={{ color: "#888" }}>tension: </span>
            <span style={{ color: inspectInfo.tension > 0 ? "#ff8c00" : inspectInfo.tension < 0 ? "#8b5cf6" : "#555" }}>
              {inspectInfo.tension >= 0 ? "+" : ""}{inspectInfo.tension.toFixed(4)}
            </span>
          </div>
          <div style={{ borderTop: "1px solid #1a1a2e", paddingTop: 3, color: "#555", fontSize: "0.6rem" }}>
            {inspectInfo.total_dendritas}d / {inspectInfo.total_sinapsis}s
          </div>
        </div>
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
          drawNoise={drawNoise}
          onDrawNoiseChange={onDrawNoiseChange}
        />
      </div>

      {/* ── Initializing overlay ── */}
      {isInitializing && (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(10,10,15,0.75)", zIndex: 40,
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
