/** Sidebar — experiment selector, JSON config editor, previews, and start button. */

import { useEffect, useRef, useMemo } from "react";
import type { ExperimentInfo, ExperimentConfig, ExperimentState, ExperimentStats } from "../types";
import { JsonConfigEditor } from "./JsonConfigEditor";

function configMatches(a: ExperimentConfig, b: ExperimentConfig): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]) as Set<keyof ExperimentConfig>;
  for (const k of keys) {
    if (k === "description") continue;
    const va = a[k];
    const vb = b[k];
    if (JSON.stringify(va) !== JSON.stringify(vb)) return false;
  }
  return true;
}

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

function applyBalance(
  grid: (number | null)[][],
  balance: number,
): (number | null)[][] {
  return grid.map((row) =>
    row.map((cell) => {
      if (cell === null || cell === 999) return cell;
      if (balance > 0 && cell < 0) return cell * (1 - balance);
      if (balance < 0 && cell > 0) return cell * (1 + balance);
      return cell;
    }),
  );
}

const PREVIEW_DISPLAY_PX = 220;

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

    const cellPx = Math.max(1, Math.floor(PREVIEW_DISPLAY_PX / Math.max(rows, cols)));
    canvas.width = cellPx * cols;
    canvas.height = cellPx * rows;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        ctx.fillStyle = weightToColor(grid[row][col]);
        ctx.fillRect(col * cellPx, row * cellPx, cellPx, cellPx);
      }
    }
  }, [grid]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        aspectRatio: "1 / 1",
        display: "block",
        imageRendering: "pixelated",
        borderRadius: "4px",
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
  experiments: ExperimentInfo[];
  selectedExperiment: string;
  config: ExperimentConfig;
  state: ExperimentState;
  stats: ExperimentStats | null;
  inputFrame: number[][] | null;
  onSelectExperiment: (id: string) => void;
  onConfigChange: (config: ExperimentConfig) => void;
  onStart: () => void;
  onRefresh?: () => void;
  connected: boolean;
  experimentActive?: boolean;
  width?: number;
}

export function Sidebar({
  experiments,
  selectedExperiment,
  config,
  state,
  stats,
  inputFrame,
  onSelectExperiment,
  onConfigChange,
  onStart,
  onRefresh,
  connected,
  experimentActive,
  width = 380,
}: SidebarProps) {
  const selectedExp = experiments.find((e) => e.id === selectedExperiment);
  const isInitializing = state === "initializing";
  const configPresets = useMemo(
    () => selectedExp?.config_presets ?? [],
    [selectedExp?.config_presets]
  );

  // Derive selected preset from config (clears when user edits JSON)
  const selectedPresetId = useMemo(() => {
    if (!configPresets.length) return "";
    const match = configPresets.find((p) => configMatches(p.config, config));
    return match ? match.id : "";
  }, [config, configPresets]);

  const masks = selectedExp?.masks ?? [];
  const activeMask = masks.length > 0
    ? (masks.find((m) => m.id === config.mask) ?? masks[0])
    : null;

  const hasMasks = masks.length > 0;

  const balancedGrid = useMemo(() => {
    if (!activeMask?.preview_grid) return null;
    const balance = config.balance ?? 0;
    if (balance === 0 || config.balance_mode === "none") return activeMask.preview_grid;
    return applyBalance(activeMask.preview_grid, balance);
  }, [activeMask, config.balance, config.balance_mode]);

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
          v0.1.0
        </span>
      </div>

      {/* Experiment selector */}
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
          Experiments
        </h3>
        {experiments.map((exp) => (
          <button
            key={exp.id}
            onClick={() => onSelectExperiment(exp.id)}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              padding: "10px 12px",
              marginBottom: "4px",
              background:
                selectedExperiment === exp.id ? "#1e1e3a" : "transparent",
              border:
                selectedExperiment === exp.id
                  ? "1px solid #4cc9f0"
                  : "1px solid transparent",
              borderRadius: "6px",
              color: selectedExperiment === exp.id ? "#4cc9f0" : "#aaa",
              cursor: "pointer",
              fontSize: "0.85rem",
              transition: "all 0.15s",
            }}
          >
            {exp.name}
          </button>
        ))}
      </div>

      {selectedExp && (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {/* Config preset dropdown (Dynamic SOM only) */}
          {configPresets.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <label
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  color: "#888",
                  letterSpacing: "0.1em",
                }}
              >
                Configuration Preset
              </label>
              <select
                value={selectedPresetId}
                onChange={(e) => {
                  const id = e.target.value;
                  if (id) {
                    const preset = configPresets.find((p) => p.id === id);
                    if (preset) onConfigChange(preset.config);
                  }
                }}
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
                <option value="">-- Select preset --</option>
                {configPresets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* JSON Config Editor */}
          <JsonConfigEditor config={config} onChange={onConfigChange} experimentInfo={selectedExp} />

          {/* Mask preview (reactive to config.mask and config.balance) */}
          {hasMasks && balancedGrid && (
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  color: "#888",
                  display: "block",
                  letterSpacing: "0.1em",
                }}
              >
                Preview
              </label>
              <MaskPreview grid={balancedGrid} />
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
                    "{stats.current_char}"
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
      )}

      <div style={{ marginTop: "auto", fontSize: "0.7rem", color: "#444" }}>
        <p>
          {selectedExp?.rules
            ? "Click on the bottom row (blue) to activate input neurons."
            : "Click on any cell to activate/deactivate neurons."}
        </p>
        <p style={{ marginTop: "4px" }}>
          Use Play to see automatic propagation.
        </p>
      </div>
    </aside>
  );
}
