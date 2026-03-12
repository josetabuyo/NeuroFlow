/** Sidebar — template selector, JSON config editor, previews, and start button. */

import { useEffect, useRef, useMemo } from "react";
import type { ConfigTemplate, ExperimentConfig, ExperimentState, ExperimentStats, Metadata } from "../types";
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

function MaskPreview({ grid }: { grid: (number | null)[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rows = grid.length;
    const cols = rows > 0 ? grid[0].length : 0;
    if (rows === 0 || cols === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellPx = Math.max(2, Math.floor(PREVIEW_DISPLAY_PX / Math.max(rows, cols)));
    canvas.width = cellPx * cols;
    canvas.height = cellPx * rows;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellFill = cellPx > 2 ? cellPx - 1 : cellPx;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        ctx.fillStyle = weightToColor(grid[row][col]);
        ctx.fillRect(col * cellPx, row * cellPx, cellFill, cellFill);
      }
    }
  }, [grid]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        aspectRatio: `${grid[0]?.length ?? 1} / ${grid.length || 1}`,
        display: "block",
        imageRendering: "pixelated",
        borderRadius: "4px",
        border: "1px solid #2a2a3e",
      }}
    />
  );
}

const INPUT_PREVIEW_PX = 120;

function InputPreview({ grid }: { grid: number[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rows = grid.length;
    const cols = rows > 0 ? grid[0].length : 0;
    if (rows === 0 || cols === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellPx = Math.max(1, Math.floor(INPUT_PREVIEW_PX / Math.max(rows, cols)));
    canvas.width = cellPx * cols;
    canvas.height = cellPx * rows;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        ctx.fillStyle = grid[row][col] > 0.5 ? "#4cc9f0" : "#0a0a0a";
        ctx.fillRect(col * cellPx, row * cellPx, cellPx, cellPx);
      }
    }
  }, [grid]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: `${INPUT_PREVIEW_PX}px`,
        height: `${INPUT_PREVIEW_PX}px`,
        display: "block",
        imageRendering: "pixelated",
        borderRadius: "4px",
        border: "1px solid #2a2a3e",
      }}
    />
  );
}

interface SidebarProps {
  templates: ConfigTemplate[];
  selectedTemplate: string;
  config: ExperimentConfig;
  metadata?: Metadata;
  state: ExperimentState;
  stats: ExperimentStats | null;
  inputFrame: number[][] | null;
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
  inputFrame,
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
    ? (masks.find((m) => m.id === config.wiring?.mask) ?? masks[0])
    : null;

  const hasMasks = masks.length > 0;
  const previewGrid = activeMask?.preview_grid ?? null;

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
      <div>
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
          v0.2.0
        </span>
      </div>

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
        {selectedTpl && (
          <p
            style={{
              fontSize: "0.7rem",
              color: "#555",
              marginTop: "6px",
              lineHeight: "1.4",
            }}
          >
            {selectedTpl.description}
          </p>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {/* JSON Config Editor */}
        <JsonConfigEditor config={config} onChange={onConfigChange} metadata={metadata} />

        {/* Run history navigation */}
        {runTotal > 0 && (
          <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
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
              {activeMask?.mask_stats && (
                <span
                  style={{
                    fontSize: "0.65rem",
                    color: "#555",
                    fontFamily: "monospace",
                  }}
                >
                  <span style={{ color: "#4ade80" }}>{activeMask.mask_stats.exc_dendrites}</span>
                  {" + "}
                  <span style={{ color: "#8b00ff" }}>{activeMask.mask_stats.inh_dendrites}</span>
                  {" = "}
                  <span style={{ color: "#e0e0ff" }}>{activeMask.mask_stats.total_dendrites}</span>
                  {" dendrites"}
                </span>
              )}
            </div>
            <MaskPreview grid={previewGrid} />
          </div>
        )}

        {/* Synapse stats */}
        {activeMask?.mask_stats && (
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
              <span>
                Exc: <strong style={{ color: "#4ade80" }}>{activeMask.mask_stats.excitatory_synapses}</strong>
              </span>
              <span>
                Inh: <strong style={{ color: "#8b00ff" }}>{activeMask.mask_stats.inhibitory_synapses}</strong>
              </span>
              <span>
                Ratio: <strong style={{ color: "#888" }}>{activeMask.mask_stats.ratio_exc_inh}</strong>
              </span>
              <span>
                R.exc: <strong style={{ color: "#4ade80" }}>{activeMask.mask_stats.excitation_radius}</strong>{" "}
                R.inh: <strong style={{ color: "#8b00ff" }}>{activeMask.mask_stats.inhibition_radius}</strong>
              </span>
            </div>
          </div>
        )}

        {/* Input preview (live, only when experiment is running) */}
        {inputFrame && experimentActive && (
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <label
              style={{
                fontSize: "0.75rem",
                textTransform: "uppercase",
                color: "#888",
                letterSpacing: "0.1em",
              }}
            >
              Input Preview
              {stats?.current_char && (
                <span style={{ color: "#4cc9f0", marginLeft: "6px", textTransform: "none" }}>
                  &quot;{stats.current_char}&quot;
                  {stats.frame_in_char !== undefined && stats.frames_per_char !== undefined && (
                    <span style={{ color: "#555", marginLeft: "4px" }}>
                      ({stats.frame_in_char + 1}/{stats.frames_per_char})
                    </span>
                  )}
                </span>
              )}
            </label>
            <InputPreview grid={inputFrame} />
          </div>
        )}

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
