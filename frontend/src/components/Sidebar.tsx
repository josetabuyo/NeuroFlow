/** Sidebar — template selector, JSON config editor, previews, and start button. */

import React, { useEffect, useRef, useMemo, useState, useCallback } from "react";
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
  onLoadDefault?: () => void;
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
  onLoadDefault,
}: SidebarProps) {
  const isInitializing = state === "initializing";

  const masks = metadata?.masks ?? [];

  // Support both canonical (regions[]) and legacy (wiring at top level)
  const anyConfig = config as unknown as Record<string, unknown>;
  const legacyWiring: Record<string, unknown> | undefined = !Array.isArray(anyConfig.regions)
    ? config.wiring as Record<string, unknown> | undefined
    : undefined;
  const activeMask = masks.length > 0 && legacyWiring?.mask
    ? (masks.find((m) => m.id === legacyWiring.mask) ?? null)
    : null;
  const hasMasks = masks.length > 0;

  // Collect all regions with daemon wiring for the combo
  const configKey = JSON.stringify(Array.isArray(anyConfig.regions) ? anyConfig.regions : legacyWiring);
  const daemonRegions = useMemo((): Array<{ id: string; daemon: Record<string, unknown> }> => {
    const result: Array<{ id: string; daemon: Record<string, unknown> }> = [];
    if (Array.isArray(anyConfig.regions)) {
      for (const r of anyConfig.regions as Array<Record<string, unknown>>) {
        const w = r.wiring as Record<string, unknown> | undefined;
        if (w?.deamon) result.push({ id: (r.id as string) || 'region', daemon: w.deamon as Record<string, unknown> });
      }
    } else if (legacyWiring?.deamon) {
      result.push({ id: 'tissue', daemon: legacyWiring.deamon as Record<string, unknown> });
    }
    return result;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configKey]);

  const [selectedDaemonId, setSelectedDaemonId] = useState<string>('');

  useEffect(() => {
    if (daemonRegions.length === 0) { setSelectedDaemonId(''); return; }
    if (!daemonRegions.find(r => r.id === selectedDaemonId)) setSelectedDaemonId(daemonRegions[0].id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [daemonRegions]);

  const selectedDaemon = daemonRegions.find(r => r.id === selectedDaemonId) ?? daemonRegions[0] ?? null;

  const [daemonPreview, setDaemonPreview] = useState<{ preview_grid: (number | null)[][], mask_stats: Record<string, unknown>, dendrites?: DendriteInfo[] } | null>(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!selectedDaemon) { setDaemonPreview(null); return; }
    fetch("/api/preview-wiring", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selectedDaemon.daemon),
    })
      .then(r => r.json())
      .then(setDaemonPreview)
      .catch(() => setDaemonPreview(null));
  }, [JSON.stringify(selectedDaemon?.daemon)]); // eslint-disable-line react-hooks/exhaustive-deps

  const previewGrid = selectedDaemon
    ? (daemonPreview?.preview_grid ?? null)
    : (activeMask?.preview_grid ?? null);

  const activeMaskStats = selectedDaemon
    ? (daemonPreview?.mask_stats ?? null)
    : (activeMask?.mask_stats ?? null);

  const activeDendrites: DendriteInfo[] | undefined = selectedDaemon
    ? (daemonPreview?.dendrites ?? undefined)
    : (activeMask?.dendrites ?? undefined);


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
        {/* Load Default button */}
        <button
          onClick={onLoadDefault}
          title="Load the template's default config from file"
          style={{
            marginTop: "6px",
            padding: "5px 10px",
            background: "#1e1e3a",
            border: "1px solid #4a4a7a",
            borderRadius: "4px",
            color: "#a0a0cc",
            fontSize: "0.75rem",
            cursor: "pointer",
            width: "100%",
            transition: "all 0.15s",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.borderColor = "#4cc9f0";
            (e.currentTarget as HTMLButtonElement).style.color = "#e0e0ff";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.borderColor = "#4a4a7a";
            (e.currentTarget as HTMLButtonElement).style.color = "#a0a0cc";
          }}
        >
          Load Default
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {/* JSON Config Editor */}
        <JsonConfigEditor config={config} onChange={onConfigChange} metadata={metadata} />

        {/* Daemon Preview */}
        {(daemonRegions.length > 0 || (hasMasks && previewGrid)) && previewGrid && (
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
              <label
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  color: "#888",
                  letterSpacing: "0.1em",
                  flexShrink: 0,
                }}
              >
                Daemon Preview
              </label>
              {daemonRegions.length > 1 && (
                <select
                  value={selectedDaemonId}
                  onChange={e => setSelectedDaemonId(e.target.value)}
                  style={{
                    fontSize: "0.65rem",
                    background: "#0d0d14",
                    color: "#aaa",
                    border: "1px solid #2a2a3e",
                    borderRadius: "3px",
                    padding: "1px 4px",
                    cursor: "pointer",
                    outline: "none",
                  }}
                >
                  {daemonRegions.map(r => (
                    <option key={r.id} value={r.id}>{r.id}</option>
                  ))}
                </select>
              )}
              {daemonRegions.length === 1 && (
                <span style={{ fontSize: "0.65rem", color: "#555", fontFamily: "monospace" }}>
                  {selectedDaemonId}
                </span>
              )}
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

      <HelpPanel />

      <div style={{ marginTop: "auto", fontSize: "0.7rem", color: "#444" }}>
        <p>Click on any cell to activate/deactivate neurons.</p>
        <p style={{ marginTop: "4px" }}>
          Use Play to see automatic propagation.
        </p>
      </div>
    </aside>
  );
}

function HelpPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: "8px" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          padding: "5px 10px",
          background: "#12122a",
          border: "1px solid #2a2a4a",
          borderRadius: "4px",
          color: "#666",
          fontSize: "0.72rem",
          cursor: "pointer",
          textAlign: "left",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>Config reference</span>
        <span style={{ fontSize: "0.65rem" }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div
          style={{
            background: "#0d0d1a",
            border: "1px solid #2a2a4a",
            borderTop: "none",
            borderRadius: "0 0 4px 4px",
            padding: "10px",
            fontSize: "0.7rem",
            color: "#888",
            lineHeight: "1.6",
            maxHeight: "400px",
            overflowY: "auto",
          }}
        >
          <HelpContent />
        </div>
      )}
    </div>
  );
}

function HelpContent() {
  const S = ({ children }: { children: React.ReactNode }) => (
    <span style={{ color: "#4cc9f0", fontFamily: "monospace" }}>{children}</span>
  );
  const Soft = () => <span style={{ color: "#4ade80", fontSize: "0.65rem" }}>soft</span>;
  const Hard = () => <span style={{ color: "#f87171", fontSize: "0.65rem" }}>hard</span>;
  const Row = ({ name, type, desc }: { name: string; type: "soft" | "hard"; desc: string }) => (
    <div style={{ marginBottom: "4px" }}>
      <S>{name}</S> {type === "soft" ? <Soft /> : <Hard />}
      <span style={{ color: "#666", marginLeft: "4px" }}>{desc}</span>
    </div>
  );
  const Section = ({ title }: { title: string }) => (
    <div style={{ color: "#a0a0cc", fontWeight: 600, marginTop: "10px", marginBottom: "4px", textTransform: "uppercase", fontSize: "0.65rem", letterSpacing: "0.08em" }}>{title}</div>
  );
  return (
    <div>
      <div style={{ color: "#555", marginBottom: "8px", fontSize: "0.65rem" }}>
        <span style={{ color: "#4ade80" }}>soft</span> = applies live &nbsp;
        <span style={{ color: "#f87171" }}>hard</span> = requires Refresh
      </div>

      <Section title="Region › wiring" />
      <Row name="process_mode" type="soft" desc="min_vs_max · avg_vs_avg · avg_vs_avg_normalized · sum · group_avg" />
      <Row name="tension_function" type="soft" desc='{"x":1,"x_pow_2":3,"x_pow_3":20} — polynomial transform on tension' />
      <Row name="learning_rate" type="soft" desc="intra-region Hebbian rate" />
      <Row name="umbral" type="soft" desc="per-neuron firing threshold (default 0)" />
      <Row name="mask / deamon" type="hard" desc="wiring topology" />
      <Row name="dendrite_exc_weight" type="hard" desc="global excitatory dendrite weight" />
      <Row name="dendrite_inh_weight" type="hard" desc="global inhibitory dendrite weight (negative)" />

      <Section title="Region › source (ascii)" />
      <Row name="text" type="soft" desc='chars to cycle: "AB" or synthetics "HALF_TOP,HALF_BOT,BARS_H,BARS_V,DOT_TL,DOT_BR"' />
      <Row name="frames_per_char" type="soft" desc="steps per character" />
      <Row name="font / font_size" type="soft" desc="font id and size in px" />
      <Row name="noise.background" type="soft" desc="random pixel flip probability 0–1" />
      <Row name="noise.shift" type="soft" desc="random 1-px shift each frame" />
      <Row name="noise.inter_char" type="soft" desc="blank frame between characters" />
      <Row name="type" type="hard" desc="ascii · error_diff · label" />

      <Section title="Region › spiking" />
      <Row name="up_ticks" type="soft" desc="max consecutive active steps before rest" />
      <Row name="down_ticks" type="soft" desc="refractory period length" />

      <Section title="Region › grid" />
      <Row name="width / height" type="hard" desc="grid dimensions" />

      <Section title="Connections" />
      <Row name="learning.rate" type="soft" desc="Hebbian learning rate for this connection" />
      <Row name="learning.exclude_range" type="soft" desc="[lo, hi] — skip weights in this range" />
      <Row name="weight" type="hard" desc="initial dendrite weight" />
      <Row name="density" type="hard" desc="fraction of source neurons sampled" />
      <Row name="from / to / type" type="hard" desc="topology" />
    </div>
  );
}
