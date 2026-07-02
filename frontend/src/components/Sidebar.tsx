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


// ── Grip handle ──────────────────────────────────────────────────────────────

function GripHandle({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      onMouseDown={onMouseDown}
      title="Drag to reorder"
      style={{ cursor: 'grab', display: 'flex', alignItems: 'center', padding: '2px 3px', flexShrink: 0, opacity: 0.35 }}
    >
      <svg width={8} height={12} viewBox="0 0 8 12">
        {([[1,1],[5,1],[1,5],[5,5],[1,9],[5,9]] as [number,number][]).map(([x,y]) => (
          <circle key={`${x}-${y}`} cx={x} cy={y} r={1.3} fill="#aaa" />
        ))}
      </svg>
    </div>
  );
}

// ── Collapsible + draggable panel ─────────────────────────────────────────────

type PanelId = 'config' | 'daemon' | 'synapses' | 'output_fn';
const DEFAULT_PANEL_ORDER: PanelId[] = ['config', 'daemon', 'synapses', 'output_fn'];

function PanelSection({
  id,
  label,
  collapsed,
  onToggle,
  headerExtra,
  isDragging,
  isDragOver,
  onDragHandleMouseDown,
  children,
}: {
  id: PanelId;
  label: string;
  collapsed: boolean;
  onToggle: () => void;
  headerExtra?: React.ReactNode;
  isDragging: boolean;
  isDragOver: boolean;
  onDragHandleMouseDown: (e: React.MouseEvent) => void;
  children: React.ReactNode;
}) {
  return (
    <div
      data-panel-id={id}
      style={{
        opacity: isDragging ? 0.35 : 1,
        borderTop: isDragOver ? '2px solid #7c4dff' : '2px solid transparent',
        transition: 'border-color 0.12s, opacity 0.12s',
        paddingTop: isDragOver ? 0 : 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: collapsed ? 0 : '6px' }}>
        <GripHandle onMouseDown={onDragHandleMouseDown} />
        <span
          onClick={onToggle}
          style={{
            flex: 1,
            fontSize: '0.75rem',
            textTransform: 'uppercase',
            color: '#888',
            letterSpacing: '0.1em',
            cursor: 'pointer',
            userSelect: 'none',
          }}
        >
          {label}
        </span>
        {headerExtra}
        <span
          onClick={onToggle}
          style={{ color: '#444', fontSize: '0.65rem', cursor: 'pointer', userSelect: 'none', paddingLeft: '4px' }}
        >
          {collapsed ? '▸' : '▾'}
        </span>
      </div>
      {!collapsed && children}
    </div>
  );
}

// ── Tension function visualizer (headless — header lives in PanelSection) ────

function TensionFunctionViz({ fn, softMode }: { fn: Record<string, number>; softMode: boolean }) {
  const evalFn = (x: number): number => {
    let v = 0;
    for (const [key, coeff] of Object.entries(fn)) {
      if (key === 'x') v += coeff * x;
      else if (key === 'b') v += coeff;
      else if (key.startsWith('x_pow_')) v += coeff * Math.pow(x, parseInt(key.split('_pow_')[1]));
    }
    return Math.max(-1, Math.min(1, v));
  };

  const W = 330, H = 120;
  const PAD = { top: 10, right: 12, bottom: 18, left: 26 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const sx = (x: number) => PAD.left + ((x + 1) / 2) * plotW;
  const sy = (y: number) => PAD.top + ((1 - y) / 2) * plotH;

  const N = 300;
  const pts = Array.from({ length: N + 1 }, (_, i) => {
    const x = -1 + (2 * i) / N;
    return `${sx(x).toFixed(1)},${sy(evalFn(x)).toFixed(1)}`;
  }).join(' ');

  let crossX: number | null = null;
  for (let i = 0; i < N; i++) {
    const x0 = -1 + (2 * i) / N;
    const x1 = -1 + (2 * (i + 1)) / N;
    const y0 = evalFn(x0), y1 = evalFn(x1);
    if (y0 <= 0 && y1 > 0) { crossX = x0 + (x1 - x0) * (-y0) / (y1 - y0); break; }
  }

  const formulaParts = Object.entries(fn).map(([k, v]) => {
    const vStr = v > 0 ? `+${v}` : `${v}`;
    if (k === 'b') return `${vStr}`;
    if (k === 'x') return `${vStr}x`;
    return `${vStr}x^${k.split('_pow_')[1]}`;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <svg width={W} height={H} style={{ background: '#0d0d14', borderRadius: '4px', border: '1px solid #1a1a2e', display: 'block' }}>
        <line x1={sx(-1)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#1e1e30" strokeWidth={1} />
        <line x1={sx(0)} y1={sy(-1)} x2={sx(0)} y2={sy(1)} stroke="#1e1e30" strokeWidth={1} />
        <line x1={sx(-1)} y1={sy(1)} x2={sx(1)} y2={sy(1)} stroke="#1a1a28" strokeWidth={1} strokeDasharray="2,4" />
        <line x1={sx(-1)} y1={sy(-1)} x2={sx(1)} y2={sy(-1)} stroke="#1a1a28" strokeWidth={1} strokeDasharray="2,4" />
        {crossX !== null && (
          <rect x={sx(crossX)} y={sy(1)} width={sx(1) - sx(crossX)} height={plotH / 2} fill="rgba(74,222,128,0.04)" />
        )}
        {softMode && crossX !== null && (
          <line x1={sx(crossX)} y1={sy(-1)} x2={sx(crossX)} y2={sy(1)} stroke="#4ade80" strokeWidth={1} strokeDasharray="3,3" strokeOpacity={0.3} />
        )}
        <text x={PAD.left - 4} y={sy(1) + 3} fontSize={8} fill="#444" textAnchor="end">1</text>
        <text x={PAD.left - 4} y={sy(0) + 3} fontSize={8} fill="#444" textAnchor="end">0</text>
        <text x={PAD.left - 4} y={sy(-1) + 3} fontSize={8} fill="#444" textAnchor="end">-1</text>
        <text x={sx(-1)} y={H - 4} fontSize={8} fill="#444" textAnchor="middle">-1</text>
        <text x={sx(0)} y={H - 4} fontSize={8} fill="#444" textAnchor="middle">0</text>
        <text x={sx(1)} y={H - 4} fontSize={8} fill="#444" textAnchor="middle">1</text>
        <polyline points={pts} fill="none" stroke="#ff8c00" strokeWidth={1.5} strokeLinejoin="round" />
        {crossX !== null && (
          <>
            <circle cx={sx(crossX)} cy={sy(0)} r={3} fill="#ff8c00" />
            <text x={sx(crossX) + 4} y={sy(0) - 4} fontSize={8} fill="#ff8c00">{crossX.toFixed(3)}</text>
          </>
        )}
        {crossX === null && (
          <text x={W / 2} y={sy(0) - 6} fontSize={8} fill="#555" textAnchor="middle">no activation</text>
        )}
      </svg>
      <div style={{ fontSize: '0.6rem', color: '#444', fontFamily: 'monospace' }}>
        f(x) = {formulaParts.join(' ')}
      </div>
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
  onLoadSession?: () => void;
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
  onLoadSession,
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
  const configKey = JSON.stringify([
    Array.isArray(anyConfig.regions) ? anyConfig.regions : legacyWiring,
    Array.isArray(anyConfig.connections) ? anyConfig.connections : [],
  ]);
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
    // Also look in top-level connections[]
    if (Array.isArray(anyConfig.connections)) {
      for (const conn of anyConfig.connections as Array<Record<string, unknown>>) {
        if (conn.deamon) {
          const id = (conn.on as string) || 'connection';
          result.push({ id, daemon: conn.deamon as Record<string, unknown> });
        }
      }
    }
    return result;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configKey]);

  // ── Panel order + collapse state (persisted) ───────────────────────────────
  const [panelOrder, setPanelOrder] = useState<PanelId[]>(() => {
    try { const s = localStorage.getItem('nf_panel_order'); if (s) return JSON.parse(s); } catch {}
    return DEFAULT_PANEL_ORDER;
  });
  const [collapsed, setCollapsed] = useState<Set<PanelId>>(() => {
    try { const s = localStorage.getItem('nf_panel_collapsed'); if (s) return new Set(JSON.parse(s)); } catch {}
    return new Set<PanelId>();
  });
  const toggleCollapsed = (id: PanelId) => setCollapsed(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    try { localStorage.setItem('nf_panel_collapsed', JSON.stringify([...next])); } catch {}
    return next;
  });

  // ── Drag-to-reorder state ──────────────────────────────────────────────────
  const [draggingId, setDraggingId] = useState<PanelId | null>(null);
  const [dragOverId, setDragOverId] = useState<PanelId | null>(null);
  const dragStartY = useRef<number>(0);
  const didDrag = useRef(false);

  const startDrag = (id: PanelId, e: React.MouseEvent) => {
    e.preventDefault();
    didDrag.current = false;
    dragStartY.current = e.clientY;
    setDraggingId(id);

    const onMove = (mv: MouseEvent) => {
      if (Math.abs(mv.clientY - dragStartY.current) > 4) didDrag.current = true;
      // Hit-test panels by position
      const panels = document.querySelectorAll<HTMLElement>('[data-panel-id]');
      let overPanelId: PanelId | null = null;
      panels.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (mv.clientY >= rect.top && mv.clientY <= rect.bottom) {
          overPanelId = el.dataset.panelId as PanelId;
        }
      });
      setDragOverId(overPanelId);
    };
    const onUp = () => {
      setDraggingId(null);
      setDragOverId(prev => {
        if (prev && prev !== id) {
          setPanelOrder(order => {
            const next = [...order];
            const from = next.indexOf(id);
            const to = next.indexOf(prev);
            if (from !== -1 && to !== -1) {
              next.splice(from, 1);
              next.splice(to, 0, id);
              try { localStorage.setItem('nf_panel_order', JSON.stringify(next)); } catch {}
            }
            return next;
          });
        }
        return null;
      });
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  // ── Output function region selection ──────────────────────────────────────
  const fnRegions = useMemo(() => {
    const regions = Array.isArray(anyConfig.regions) ? (anyConfig.regions as Record<string, unknown>[]) : [];
    return regions.filter(r => {
      const t = r.tension as Record<string, unknown> | undefined;
      return t?.function && typeof t.function === 'object';
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configKey]);

  const [selectedFnId, setSelectedFnId] = useState<string>(() => (fnRegions[0]?.id as string) ?? '');
  useEffect(() => {
    if (fnRegions.length > 0 && !fnRegions.find(r => r.id === selectedFnId)) {
      setSelectedFnId(fnRegions[0].id as string);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(fnRegions.map(r => r.id))]);

  const selectedFnRegion = fnRegions.find(r => r.id === selectedFnId) ?? fnRegions[0] ?? null;
  const activeFn = selectedFnRegion
    ? (((selectedFnRegion.tension as Record<string, unknown>).function) as Record<string, number>)
    : null;
  const activeFnSoft = selectedFnRegion?.activation === 'soft';

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
        {/* Load Default / Load Session buttons */}
        <div style={{ marginTop: "6px", display: "flex", gap: "6px" }}>
          <button
            onClick={onLoadDefault}
            title="Load the template's default config from file"
            style={{
              flex: 1,
              padding: "5px 10px",
              background: "#1e1e3a",
              border: "1px solid #4a4a7a",
              borderRadius: "4px",
              color: "#a0a0cc",
              fontSize: "0.75rem",
              cursor: "pointer",
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
          <button
            onClick={onLoadSession}
            title="Load last session config from local file"
            style={{
              flex: 1,
              padding: "5px 10px",
              background: "#1e1e3a",
              border: "1px solid #4a4a7a",
              borderRadius: "4px",
              color: "#a0a0cc",
              fontSize: "0.75rem",
              cursor: "pointer",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "#7c4dff";
              (e.currentTarget as HTMLButtonElement).style.color = "#e0e0ff";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "#4a4a7a";
              (e.currentTarget as HTMLButtonElement).style.color = "#a0a0cc";
            }}
          >
            Load Session
          </button>
          <SessionFileButtons templateId={selectedTemplate} />
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {(() => {
          const selectStyle: React.CSSProperties = {
            fontSize: '0.65rem', background: '#0d0d14', color: '#aaa',
            border: '1px solid #2a2a3e', borderRadius: '3px', padding: '1px 4px',
            cursor: 'pointer', outline: 'none',
          };

          const hasDaemon = (daemonRegions.length > 0 || (hasMasks && !!previewGrid)) && !!previewGrid;

          const panelContent: Record<PanelId, { visible: boolean; label: string; headerExtra?: React.ReactNode; body: React.ReactNode }> = {
            config: {
              visible: true,
              label: 'Config',
              body: <JsonConfigEditor config={config} onChange={onConfigChange} metadata={metadata} />,
            },
            daemon: {
              visible: hasDaemon,
              label: 'Daemon Preview',
              headerExtra: (
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {daemonRegions.length > 1 && (
                    <select value={selectedDaemonId} onChange={e => setSelectedDaemonId(e.target.value)} style={selectStyle}>
                      {daemonRegions.map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
                    </select>
                  )}
                  {daemonRegions.length === 1 && (
                    <span style={{ fontSize: '0.65rem', color: '#555', fontFamily: 'monospace' }}>{selectedDaemonId}</span>
                  )}
                  {activeMaskStats && (
                    <span style={{ fontSize: '0.65rem', color: '#555', fontFamily: 'monospace' }}>
                      <span style={{ color: '#4ade80' }}>{(activeMaskStats as Record<string, unknown>).exc_dendrites as number}</span>
                      {' + '}
                      <span style={{ color: '#8b00ff' }}>{(activeMaskStats as Record<string, unknown>).inh_dendrites as number}</span>
                      {' dendrites'}
                    </span>
                  )}
                </span>
              ),
              body: previewGrid ? <MaskPreview grid={previewGrid} dendrites={activeDendrites} /> : null,
            },
            synapses: {
              visible: !!activeMaskStats,
              label: 'Synapses',
              body: (() => {
                if (!activeMaskStats) return null;
                const s = activeMaskStats as Record<string, unknown>;
                return (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', fontSize: '0.7rem', color: '#666', fontFamily: 'monospace', padding: '6px 8px', background: '#0d0d14', borderRadius: '4px', border: '1px solid #1a1a2e' }}>
                    <span>Exc: <strong style={{ color: '#4ade80' }}>{s.excitatory_synapses as number}</strong></span>
                    <span>Inh: <strong style={{ color: '#8b00ff' }}>{s.inhibitory_synapses as number}</strong></span>
                    <span>Ratio: <strong style={{ color: '#888' }}>{s.ratio_exc_inh as number}</strong></span>
                    <span>R.exc: <strong style={{ color: '#4ade80' }}>{s.excitation_radius as number}</strong>{' '}R.inh: <strong style={{ color: '#8b00ff' }}>{s.inhibition_radius as number}</strong></span>
                  </div>
                );
              })(),
            },
            output_fn: {
              visible: !!activeFn,
              label: 'Output Function',
              headerExtra: (
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {fnRegions.length > 1 && (
                    <select value={selectedFnId} onChange={e => setSelectedFnId(e.target.value)} style={selectStyle}>
                      {fnRegions.map(r => <option key={r.id as string} value={r.id as string}>{r.id as string}</option>)}
                    </select>
                  )}
                  {fnRegions.length === 1 && (
                    <span style={{ fontSize: '0.65rem', color: '#555', fontFamily: 'monospace' }}>{selectedFnId}</span>
                  )}
                  {activeFnSoft && <span style={{ fontSize: '0.6rem', color: '#4ade80', fontFamily: 'monospace' }}>soft</span>}
                </span>
              ),
              body: activeFn ? <TensionFunctionViz fn={activeFn} softMode={activeFnSoft} /> : null,
            },
          };

          return panelOrder.map(id => {
            const p = panelContent[id];
            if (!p.visible) return null;
            return (
              <PanelSection
                key={id}
                id={id}
                label={p.label}
                collapsed={collapsed.has(id)}
                onToggle={() => toggleCollapsed(id)}
                headerExtra={p.headerExtra}
                isDragging={draggingId === id}
                isDragOver={dragOverId === id && draggingId !== id}
                onDragHandleMouseDown={e => startDrag(id, e)}
              >
                {p.body}
              </PanelSection>
            );
          });
        })()}

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

function SessionFileButtons({ templateId }: { templateId: string }) {
  const [copied, setCopied] = useState(false);

  const copyPath = async () => {
    try {
      const res = await fetch(`/api/session/file-path/${templateId}`);
      const { path } = await res.json();
      await navigator.clipboard.writeText(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {/* ignore */}
  };

  const revealInFinder = () =>
    fetch(`/api/session/reveal/${templateId}`, { method: "POST" }).catch(() => {});

  const btnStyle: React.CSSProperties = {
    padding: "5px 7px",
    background: "#1e1e3a",
    border: "1px solid #4a4a7a",
    borderRadius: "4px",
    color: "#a0a0cc",
    fontSize: "0.85rem",
    cursor: "pointer",
    transition: "all 0.15s",
    lineHeight: 1,
  };

  return (
    <div style={{ display: "flex", gap: "4px" }}>
      <button
        onClick={copyPath}
        title="Copy session file path"
        style={{ ...btnStyle, color: copied ? "#06d6a0" : "#a0a0cc" }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#4cc9f0"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#4a4a7a"; }}
      >
        {copied ? "✓" : "⎘"}
      </button>
      <button
        onClick={revealInFinder}
        title="Reveal session file in Finder"
        style={btnStyle}
        onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#ffd166"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#4a4a7a"; }}
      >
        📂
      </button>
    </div>
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

      <Section title="Region (top-level)" />
      <Row name="process_mode" type="soft" desc="min_vs_max · avg_vs_avg · avg_vs_avg_normalized · sum · group_avg" />
      <Row name="tension.function" type="soft" desc='{"x":1,"x_pow_2":3,"x_pow_3":20,"b":-0.7} — polynomial + bias (b shifts firing point; threshold = 0 always)' />

      <Section title='Connection › on (intra-region wiring)' />
      <Row name="on" type="hard" desc='region id — e.g. "tissue". Marks this as an intra-region wiring entry' />
      <Row name="deamon" type="hard" desc='daemon wiring object. Contains mask or shape/excitatory/inhibitory' />
      <Row name="deamon.mask" type="hard" desc='mask string — e.g. "deamon_e3_g2_i12_de1_di1"' />
      <Row name="deamon.shape / centroid / excitatory / inhibitory" type="hard" desc="daemon wiring geometry" />
      <Row name="deamon.excitatory.weight" type="hard" desc="global excitatory dendrite weight" />
      <Row name="deamon.inhibitory.weight" type="hard" desc="global inhibitory dendrite weight (negative)" />
      <Row name="deamon.learning.rate" type="soft" desc="intra-region Hebbian rate" />

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

      <Section title="Connections (inter-region full)" />
      <Row name="full.learning.rate" type="soft" desc="Hebbian learning rate for this connection" />
      <Row name="full.learning.exclude_range" type="soft" desc="[lo, hi] — skip weights in this range" />
      <Row name="full.weight" type="hard" desc="initial dendrite weight" />
      <Row name="full.density" type="hard" desc="fraction of source neurons sampled" />
      <Row name="from / to" type="hard" desc="topology" />

      <Section title="Orchestrator" />
      <Row name='{"from":{"tick":0,"set":"..."},"to":{"tick":N,"set":"..."}}' type="hard" desc="gradient — interpolates scalar between two ticks" />
      <Row name='{"at":{"tick":N},"set":"..."}' type="hard" desc='one-shot — fires once at tick N. paths: connections[N]["full"]["weight"], regions[N]["text"]' />
      <Row name='{"at":{"tick":0},"inject":{...}}' type="hard" desc='inject — writes activations directly into a region after procesar(). tick-0 also applies at setup (visible in step 0)' />
      <Row name='{"at":{"tick":0,"tick_end":20},"inject":{...}}' type="hard" desc="inject sustained — repeats the pattern every tick in [tick, tick_end]" />
      <Row name="inject.region" type="hard" desc="region id to write into" />
      <Row name='inject.template: "noise"' type="hard" desc="uniform random [0,1] activations" />
      <Row name='inject.template: "image"' type="hard" desc='PNG/JPG mapped to activations. inject.src — path relative to backend/configs/. PNG with alpha: white=activate, black=silence, transparent=leave unchanged.' />
    </div>
  );
}
