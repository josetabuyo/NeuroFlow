/** Sidebar — template selector, JSON config editor, previews, and start button. */

import { useEffect, useRef, useMemo, useState, useCallback } from "react";
import type { ConfigTemplate, DendriteInfo, ExperimentConfig, ExperimentState, ExperimentStats, Metadata } from "../types";
import { JsonConfigEditor } from "./JsonConfigEditor";

function weightToColor(weight: number | null): string {
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

const PREVIEW_DISPLAY_PX = 350;

function MaskPreview({
  grid,
  dendrites,
}: {
  grid: (number | null)[][];
  dendrites?: DendriteInfo[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selected, setSelected] = useState<number | null>(null);

  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  const cellPx = Math.max(2, Math.floor(PREVIEW_DISPLAY_PX / Math.max(rows, cols)));

  // Reset selection when grid changes (new wiring loaded)
  useEffect(() => { setSelected(null); }, [grid]);

  // Build highlighted cell set when a dendrite is selected
  const highlightSet = useMemo(() => {
    if (selected === null || !dendrites?.[selected]) return null;
    return new Set(dendrites[selected].cells.map(([c, r]) => `${c},${r}`));
  }, [selected, dendrites]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = cellPx * cols;
    canvas.height = cellPx * rows;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellFill = cellPx > 2 ? cellPx - 1 : cellPx;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const val = grid[row][col];
        if (highlightSet) {
          ctx.fillStyle = highlightSet.has(`${col},${row}`) ? weightToColor(val) : "#1c1c26";
        } else {
          ctx.fillStyle = weightToColor(val);
        }
        ctx.fillRect(col * cellPx, row * cellPx, cellFill, cellFill);
      }
    }
  }, [grid, cellPx, rows, cols, highlightSet]);

  const handleDotClick = useCallback((i: number) => {
    setSelected(prev => prev === i ? null : i);
  }, []);

  const canvasPx = cellPx * cols;

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          aspectRatio: `${cols} / ${rows || 1}`,
          display: "block",
          imageRendering: "pixelated",
          borderRadius: "4px",
          border: "1px solid #2a2a3e",
        }}
      />
      {/* Dendrite dots overlay */}
      {dendrites && canvasPx > 0 && dendrites.map((d, i) => {
        const pct_x = (d.centroid[0] / cols) * 100;
        const pct_y = (d.centroid[1] / rows) * 100;
        const isSelected = selected === i;
        const color = weightToColor(d.avg_effective);
        return (
          <div
            key={i}
            title={`Dendrita ${i + 1} · avg: ${d.avg_effective > 0 ? "+" : ""}${d.avg_effective.toFixed(3)} · ${d.cells.length} sin.`}
            onClick={() => handleDotClick(i)}
            style={{
              position: "absolute",
              left: `${pct_x}%`,
              top: `${pct_y}%`,
              width: isSelected ? 10 : 7,
              height: isSelected ? 10 : 7,
              borderRadius: "50%",
              background: color,
              border: isSelected ? "2px solid #fff" : "1px solid rgba(255,255,255,0.4)",
              cursor: "pointer",
              transform: "translate(-50%, -50%)",
              zIndex: 2,
              boxShadow: isSelected ? `0 0 6px ${color}` : "none",
              transition: "all 0.1s",
            }}
          />
        );
      })}
      {selected !== null && dendrites?.[selected] && (
        <div style={{
          fontSize: "0.6rem",
          color: "#888",
          fontFamily: "monospace",
          marginTop: 3,
          textAlign: "center",
        }}>
          Dendrita {selected + 1} · {dendrites[selected].cells.length} sin. · avg {dendrites[selected].avg_effective > 0 ? "+" : ""}{dendrites[selected].avg_effective.toFixed(3)}
          {" · "}
          <span
            style={{ color: "#aaa", cursor: "pointer", textDecoration: "underline" }}
            onClick={() => setSelected(null)}
          >
            deselect
          </span>
        </div>
      )}
    </div>
  );
}


interface SidebarProps {
  templates: ConfigTemplate[];
  selectedTemplate: string;
  config: ExperimentConfig;
  metadata?: Metadata;
  state: ExperimentState;
  stats: ExperimentStats | null;
  onSelectTemplate: (id: string) => void;
  onConfigChange: (config: ExperimentConfig) => void;
  onStart: () => void;
  onRefresh?: () => void;
  connected: boolean;
  experimentActive?: boolean;
  width?: number;
  onPrevRun?: () => void;
  onNextRun?: () => void;
  canGoPrev?: boolean;
  canGoNext?: boolean;
  runPosition?: number;
  runTotal?: number;
}

export function Sidebar({
  templates,
  selectedTemplate,
  config,
  metadata,
  state,
  stats,
  onSelectTemplate,
  onConfigChange,
  onStart,
  onRefresh,
  connected,
  experimentActive,
  width = 380,
  onPrevRun,
  onNextRun,
  canGoPrev = false,
  canGoNext = false,
  runPosition = 0,
  runTotal = 0,
}: SidebarProps) {
  const isInitializing = state === "initializing";

  const masks = metadata?.masks ?? [];
  const activeMask = masks.length > 0
    ? (masks.find((m) => m.id === config.wiring?.mask) ?? null)
    : null;

  const hasMasks = masks.length > 0;

  const [deamonPreview, setDeamonPreview] = useState<{ preview_grid: (number | null)[][], mask_stats: Record<string, unknown>, dendrites?: DendriteInfo[] } | null>(null);

  const deamonWiring = (config.wiring as Record<string, unknown> | undefined)?.deamon;
  useEffect(() => {
    if (!deamonWiring) { setDeamonPreview(null); return; }
    fetch("/api/preview-wiring", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(deamonWiring),
    })
      .then((r) => r.json())
      .then(setDeamonPreview)
      .catch(() => setDeamonPreview(null));
  }, [JSON.stringify(deamonWiring)]);

  const previewGrid = deamonWiring
    ? (deamonPreview?.preview_grid ?? null)
    : (activeMask?.preview_grid ?? null);

  const activeMaskStats = deamonWiring
    ? (deamonPreview?.mask_stats ?? null)
    : (activeMask?.mask_stats ?? null);

  const activeDendrites: DendriteInfo[] | undefined = deamonWiring
    ? (deamonPreview?.dendrites ?? undefined)
    : (activeMask?.dendrites ?? undefined);

  const selectedTpl = useMemo(
    () => templates.find((t) => t.id === selectedTemplate),
    [templates, selectedTemplate],
  );

  return (
    <aside
      style={{
        width: `${width}px`,
        minWidth: `${width}px`,
        background: "#12121a",
        borderRight: "none",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        overflowY: "auto",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", alignItems: "baseline", gap: "8px" }}>
        <h1
          style={{
            fontSize: "1.3rem",
            fontWeight: 700,
            color: "#e0e0ff",
            margin: 0,
            letterSpacing: "0.05em",
          }}
        >
          NeuroFlow
        </h1>
        <span
          style={{
            fontSize: "0.7rem",
            color: "#666",
            fontFamily: "monospace",
          }}
        >
          v7.0.0
        </span>
      </div>

      {/* Start / Refresh button */}
      <button
        onClick={experimentActive && onRefresh ? onRefresh : onStart}
        disabled={!connected || isInitializing}
        style={{
          padding: "10px",
          background: isInitializing ? "#2a2a3e" : connected ? (experimentActive ? "#06d6a0" : "#4cc9f0") : "#333",
          color: isInitializing ? "#888" : connected ? "#0a0a0a" : "#666",
          border: "none",
          borderRadius: "6px",
          fontSize: "0.9rem",
          fontWeight: 600,
          cursor: !connected || isInitializing ? "not-allowed" : "pointer",
          transition: "all 0.15s",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "8px",
        }}
      >
        {isInitializing && <span className="neuro-spinner-sm" />}
        {isInitializing ? "Initializing..." : !connected ? "Connecting..." : experimentActive ? "Refresh Experiment" : "Start Experiment"}
      </button>

      {/* Template selector */}
      <div>
        <h3
          style={{
            fontSize: "0.75rem",
            textTransform: "uppercase",
            color: "#888",
            marginBottom: "10px",
            letterSpacing: "0.1em",
          }}
        >
          Config Templates
        </h3>
        <select
          value={selectedTemplate}
          onChange={(e) => onSelectTemplate(e.target.value)}
          style={{
            width: "100%",
            padding: "8px 12px",
            background: "#0d0d14",
            border: "1px solid #2a2a3e",
            borderRadius: "6px",
            color: "#e0e0ff",
            fontSize: "0.85rem",
            cursor: "pointer",
          }}
        >
          {templates.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
        {/* Run history navigation */}
        {runTotal > 0 && (
          <div style={{ display: "flex", gap: "4px", alignItems: "center", marginTop: "8px" }}>
            <button
              onClick={onPrevRun}
              disabled={!canGoPrev}
              title="Previous executed config"
              style={{
                padding: "6px 10px",
                background: canGoPrev ? "#1e1e3a" : "#0d0d14",
                border: `1px solid ${canGoPrev ? "#4cc9f0" : "#2a2a3e"}`,
                borderRadius: "4px",
                color: canGoPrev ? "#e0e0ff" : "#444",
                fontSize: "0.8rem",
                cursor: canGoPrev ? "pointer" : "not-allowed",
                transition: "all 0.15s",
              }}
            >
              &#9664;
            </button>
            <span
              style={{
                flex: 1,
                textAlign: "center",
                fontSize: "0.7rem",
                color: "#888",
                fontFamily: "monospace",
                userSelect: "none",
              }}
            >
              Run {runPosition} / {runTotal}
            </span>
            <button
              onClick={onNextRun}
              disabled={!canGoNext}
              title="Next executed config"
              style={{
                padding: "6px 10px",
                background: canGoNext ? "#1e1e3a" : "#0d0d14",
                border: `1px solid ${canGoNext ? "#4cc9f0" : "#2a2a3e"}`,
                borderRadius: "4px",
                color: canGoNext ? "#e0e0ff" : "#444",
                fontSize: "0.8rem",
                cursor: canGoNext ? "pointer" : "not-allowed",
                transition: "all 0.15s",
              }}
            >
              &#9654;
            </button>
          </div>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {/* JSON Config Editor */}
        <JsonConfigEditor config={config} onChange={onConfigChange} metadata={metadata} />

        {/* Mask preview */}
        {hasMasks && previewGrid && (
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: "8px" }}>
              <label
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  color: "#888",
                  letterSpacing: "0.1em",
                }}
              >
                Neuron Preview
              </label>
              {activeMaskStats && (
                <span
                  style={{
                    fontSize: "0.65rem",
                    color: "#555",
                    fontFamily: "monospace",
                  }}
                >
                  <span style={{ color: "#4ade80" }}>{(activeMaskStats as Record<string, unknown>).exc_dendrites as number}</span>
                  {" + "}
                  <span style={{ color: "#8b00ff" }}>{(activeMaskStats as Record<string, unknown>).inh_dendrites as number}</span>
                  {" = "}
                  <span style={{ color: "#e0e0ff" }}>{(activeMaskStats as Record<string, unknown>).total_dendrites as number}</span>
                  {" dendrites"}
                </span>
              )}
            </div>
            <MaskPreview grid={previewGrid} dendrites={activeDendrites} />
          </div>
        )}

        {/* Synapse stats */}
        {activeMaskStats && (
          <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
            <label
              style={{
                fontSize: "0.75rem",
                textTransform: "uppercase",
                color: "#888",
                letterSpacing: "0.1em",
                marginBottom: "2px",
              }}
            >
              Synapses
            </label>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "4px 12px",
                fontSize: "0.7rem",
                color: "#666",
                fontFamily: "monospace",
                padding: "6px 8px",
                background: "#0d0d14",
                borderRadius: "4px",
                border: "1px solid #1a1a2e",
              }}
            >
              {(() => {
                const s = activeMaskStats as Record<string, unknown>;
                return (<>
                  <span>Exc: <strong style={{ color: "#4ade80" }}>{s.excitatory_synapses as number}</strong></span>
                  <span>Inh: <strong style={{ color: "#8b00ff" }}>{s.inhibitory_synapses as number}</strong></span>
                  <span>Ratio: <strong style={{ color: "#888" }}>{s.ratio_exc_inh as number}</strong></span>
                  <span>R.exc: <strong style={{ color: "#4ade80" }}>{s.excitation_radius as number}</strong>{" "}R.inh: <strong style={{ color: "#8b00ff" }}>{s.inhibition_radius as number}</strong></span>
                </>);
              })()}
            </div>
          </div>
        )}

        {/* Current char indicator — tiny, text only */}
        {stats?.current_char && experimentActive && (
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "1.1rem",
              color: "#4cc9f0",
              letterSpacing: "0.05em",
              userSelect: "none",
              alignSelf: "flex-start",
            }}
            title={`Processing "${stats.current_char}" — frame ${(stats.frame_in_char ?? 0) + 1} / ${stats.frames_per_char ?? "?"}`}
          >
            {stats.current_char}
          </span>
        )}

      </div>

      <div style={{ marginTop: "auto", fontSize: "0.7rem", color: "#444" }}>
        <p>Click on any cell to activate/deactivate neurons.</p>
        <p style={{ marginTop: "4px" }}>
          Use Play to see automatic propagation.
        </p>
      </div>
    </aside>
  );
}
