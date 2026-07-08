/** Sidebar — template selector, JSON config editor, previews, and start button. */

import React, { useEffect, useRef, useMemo, useState, useCallback } from "react";
import type { Experiment, DendriteInfo, ExperimentConfig, ExperimentState, ExperimentStats, Metadata } from "../types";
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
  experiments: Experiment[];
  selectedExperimentId: number | null;
  config: ExperimentConfig;
  metadata?: Metadata;
  state: ExperimentState;
  stats: ExperimentStats | null;
  onSelectExperiment: (id: number) => void;
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
  onRevert?: () => void;
  onCreateExperiment?: () => void;
  onRenameExperiment?: (id: number, name: string) => void;
  onDeleteExperiment?: (id: number) => void;
  onReorderExperiment?: (id: number, direction: "up" | "down") => void;
}

export function Sidebar({
  experiments,
  selectedExperimentId,
  config,
  metadata,
  state,
  stats,
  onSelectExperiment,
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
  onRevert,
  onCreateExperiment,
  onRenameExperiment,
  onDeleteExperiment,
  onReorderExperiment,
}: SidebarProps) {
  const [renaming, setRenaming] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const selectedExperiment = experiments.find((e) => e.id === selectedExperimentId) ?? null;

  const startRenaming = useCallback(() => {
    if (!selectedExperiment) return;
    setRenameDraft(selectedExperiment.name);
    setRenaming(true);
  }, [selectedExperiment]);

  const commitRename = useCallback(() => {
    setRenaming(false);
    if (selectedExperimentId != null && renameDraft.trim()) {
      onRenameExperiment?.(selectedExperimentId, renameDraft.trim());
    }
  }, [selectedExperimentId, renameDraft, onRenameExperiment]);

  const handleDelete = useCallback(() => {
    if (selectedExperimentId == null) return;
    if (confirm(`Delete experiment "${selectedExperiment?.name}"? This also deletes its run history.`)) {
      onDeleteExperiment?.(selectedExperimentId);
    }
  }, [selectedExperimentId, selectedExperiment, onDeleteExperiment]);
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
        {renaming ? (
          <input
            autoFocus
            value={renameDraft}
            onChange={(e) => setRenameDraft(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") setRenaming(false);
            }}
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "#0d0d14",
              border: "1px solid #4cc9f0",
              borderRadius: "6px",
              color: "#e0e0ff",
              fontSize: "0.85rem",
            }}
          />
        ) : (
          <select
            value={selectedExperimentId ?? ""}
            onChange={(e) => onSelectExperiment(Number(e.target.value))}
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
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name}
              </option>
            ))}
          </select>
        )}
        {/* Experiment ABM controls */}
        <div style={{ marginTop: "6px", display: "flex", gap: "6px" }}>
          <button onClick={onCreateExperiment} title="Create a new experiment" style={sidebarButtonStyle}>
            + New
          </button>
          <button onClick={startRenaming} disabled={!selectedExperiment} title="Rename experiment" style={sidebarButtonStyle}>
            Rename
          </button>
          <button onClick={handleDelete} disabled={!selectedExperiment} title="Delete experiment" style={sidebarButtonStyle}>
            Delete
          </button>
          <button
            onClick={() => selectedExperimentId != null && onReorderExperiment?.(selectedExperimentId, "up")}
            disabled={!selectedExperiment}
            title="Move up"
            style={{ ...sidebarButtonStyle, flex: "0 0 auto", padding: "5px 8px" }}
          >
            &#9650;
          </button>
          <button
            onClick={() => selectedExperimentId != null && onReorderExperiment?.(selectedExperimentId, "down")}
            disabled={!selectedExperiment}
            title="Move down"
            style={{ ...sidebarButtonStyle, flex: "0 0 auto", padding: "5px 8px" }}
          >
            &#9660;
          </button>
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
              body: (
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "6px", alignItems: "center" }}>
                    <div />
                    {runTotal > 0 ? (
                      <div style={{ display: "flex", gap: "2px", alignItems: "center", justifySelf: "center" }}>
                        <button
                          onClick={onPrevRun}
                          disabled={!canGoPrev}
                          title="Previous executed config"
                          style={{
                            padding: "2px 6px",
                            background: "transparent",
                            border: `1px solid ${canGoPrev ? "#3a3a5a" : "#22222e"}`,
                            borderRadius: "3px",
                            color: canGoPrev ? "#777" : "#3a3a3a",
                            fontSize: "0.65rem",
                            cursor: canGoPrev ? "pointer" : "not-allowed",
                            transition: "all 0.15s",
                          }}
                        >
                          &#9664;
                        </button>
                        <span
                          style={{
                            fontSize: "0.65rem",
                            color: "#555",
                            fontFamily: "monospace",
                            userSelect: "none",
                            padding: "0 2px",
                          }}
                        >
                          Run {runPosition}/{runTotal}
                        </span>
                        <button
                          onClick={onNextRun}
                          disabled={!canGoNext}
                          title="Next executed config"
                          style={{
                            padding: "2px 6px",
                            background: "transparent",
                            border: `1px solid ${canGoNext ? "#3a3a5a" : "#22222e"}`,
                            borderRadius: "3px",
                            color: canGoNext ? "#777" : "#3a3a3a",
                            fontSize: "0.65rem",
                            cursor: canGoNext ? "pointer" : "not-allowed",
                            transition: "all 0.15s",
                          }}
                        >
                          &#9654;
                        </button>
                      </div>
                    ) : (
                      <div />
                    )}
                    <div style={{ display: "flex", justifyContent: "flex-end" }}>
                      {onRevert && (
                        <button
                          onClick={onRevert}
                          title="Discard unsaved edits and reload the experiment's saved config"
                          style={{
                            padding: "2px 6px",
                            background: "transparent",
                            border: "1px solid #3a3a5a",
                            borderRadius: "3px",
                            color: "#666",
                            fontSize: "0.65rem",
                            cursor: "pointer",
                            transition: "all 0.15s",
                          }}
                          onMouseEnter={(e) => {
                            (e.currentTarget as HTMLButtonElement).style.borderColor = "#7c4dff";
                            (e.currentTarget as HTMLButtonElement).style.color = "#e0e0ff";
                          }}
                          onMouseLeave={(e) => {
                            (e.currentTarget as HTMLButtonElement).style.borderColor = "#3a3a5a";
                            (e.currentTarget as HTMLButtonElement).style.color = "#666";
                          }}
                        >
                          Discard changes
                        </button>
                      )}
                    </div>
                  </div>
                  <JsonConfigEditor config={config} onChange={onConfigChange} metadata={metadata} />
                </div>
              ),
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
              body: activeFn ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    {fnRegions.length > 1 && (
                      <select value={selectedFnId} onChange={e => setSelectedFnId(e.target.value)} style={selectStyle}>
                        {fnRegions.map(r => <option key={r.id as string} value={r.id as string}>{r.id as string}</option>)}
                      </select>
                    )}
                    {fnRegions.length === 1 && (
                      <span style={{ fontSize: '0.65rem', color: '#555', fontFamily: 'monospace' }}>{selectedFnId}</span>
                    )}
                    {activeFnSoft && <span style={{ fontSize: '0.6rem', color: '#4ade80', fontFamily: 'monospace' }}>soft</span>}
                  </div>
                  <TensionFunctionViz fn={activeFn} softMode={activeFnSoft} />
                </div>
              ) : null,
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

const sidebarButtonStyle: React.CSSProperties = {
  flex: 1,
  padding: "5px 10px",
  background: "#1e1e3a",
  border: "1px solid #4a4a7a",
  borderRadius: "4px",
  color: "#a0a0cc",
  fontSize: "0.75rem",
  cursor: "pointer",
  transition: "all 0.15s",
};

type ConfigRow = { section: string; key: string; type: "soft" | "hard"; desc: string };

// `key` is the literal single JSON property name — what you'd actually type in a
// config. It's the unit the search/autocomplete/usage-lookup operate on. The same
// key can legitimately appear in more than one row (e.g. "weight" shows up under
// full, deamon.excitatory and nerve.from) — that's real, not a bug.
const CONFIG_ROWS: ConfigRow[] = [
  { section: "Region (top-level)", key: "process_mode", type: "soft", desc: "min_vs_max · avg_vs_avg · avg_vs_avg_normalized · sum · group_avg" },
  { section: "Region (top-level)", key: "function", type: "soft", desc: 'tension.function — {"x":1,"x_pow_2":3,"x_pow_3":20,"b":-0.7} polynomial + bias (b shifts firing point; threshold = 0 always)' },
  { section: "Region (top-level)", key: "threshold", type: "soft", desc: "firing threshold for the region (legacy alias: umbral). Default 0.0" },
  { section: "Region (top-level)", key: "activation", type: "soft", desc: '"soft" — region emits continuous tension in [0,1] instead of a binary spike' },
  { section: "Region (top-level)", key: "border", type: "hard", desc: 'grid.border — "connected" (toroidal wrap, default) or "cut" (edges truncated, fewer synapses)' },
  { section: "Region (top-level)", key: "neuron_labels", type: "hard", desc: 'per-neuron class-label grid, e.g. [["A","B",...],...] — used by label_mismatch source and handlers' },
  { section: "Region (top-level)", key: "handler", type: "hard", desc: "name of a Python handler class (experiments/handlers/*.py) that populates neuron_labels" },
  { section: "Region (top-level)", key: "threshold", type: "hard", desc: "daemon.threshold — cluster detection: activation cutoff to count a neuron as active. Default 0.5" },
  { section: "Region (top-level)", key: "min_size", type: "hard", desc: "daemon.min_size — cluster detection: minimum cluster size. Default 3" },

  { section: "Connection › on (intra-region wiring)", key: "on", type: "hard", desc: 'region id — e.g. "tissue". Marks this as an intra-region wiring entry' },
  { section: "Connection › on (intra-region wiring)", key: "deamon", type: "hard", desc: "daemon wiring object. Contains mask or shape/excitatory/inhibitory" },
  { section: "Connection › on (intra-region wiring)", key: "mask", type: "hard", desc: 'deamon.mask — mask string, e.g. "deamon_e3_g2_i12_de1_di1"' },
  { section: "Connection › on (intra-region wiring)", key: "shape", type: "hard", desc: "deamon.shape — daemon footprint shape" },
  { section: "Connection › on (intra-region wiring)", key: "centroid", type: "hard", desc: "deamon.centroid — daemon center placement" },
  { section: "Connection › on (intra-region wiring)", key: "weight", type: "hard", desc: "deamon.excitatory.weight — global excitatory dendrite weight" },
  { section: "Connection › on (intra-region wiring)", key: "weight", type: "hard", desc: "deamon.inhibitory.weight — global inhibitory dendrite weight (negative)" },
  { section: "Connection › on (intra-region wiring)", key: "rate", type: "soft", desc: "deamon.learning.rate — intra-region Hebbian rate" },
  { section: "Connection › on (intra-region wiring)", key: "exclude_range", type: "soft", desc: "deamon.learning.exclude_range — [lo, hi], skip weight updates for this dendrite group while the weight is in this range" },
  { section: "Connection › on (intra-region wiring)", key: "density", type: "hard", desc: "excitatory.density — excitatory dendrite sampling fraction" },
  { section: "Connection › on (intra-region wiring)", key: "noise", type: "hard", desc: "excitatory.noise — excitatory weight jitter" },
  { section: "Connection › on (intra-region wiring)", key: "multiplier", type: "hard", desc: "inhibitory.multiplier — inhibitory ring strength multiplier" },
  { section: "Connection › on (intra-region wiring)", key: "density", type: "hard", desc: "inhibitory.density — inhibitory dendrite sampling fraction" },
  { section: "Connection › on (intra-region wiring)", key: "noise", type: "hard", desc: "inhibitory.noise — inhibitory weight jitter" },
  { section: "Connection › on (intra-region wiring)", key: "sectors", type: "hard", desc: "inhibitory.sectors — number of angular sectors in the inhibitory ring" },
  { section: "Connection › on (intra-region wiring)", key: "shift", type: "hard", desc: "centroid.shift — [dx, dy] static offset of the daemon centroid. Stacks with centroid.twist if both are set" },
  { section: "Connection › on (intra-region wiring)", key: "twist", type: "hard", desc: "centroid.twist — {center:[cx,cy], direction:\"cw\"|\"ccw\", max_magnitude|fix_magnitude}. Rotational shift tangent to the circle around center. max_magnitude scales 0 at center -> max at the farthest grid corner; fix_magnitude is constant everywhere (takes precedence if both given). Setting this alone enables twist mode (shift not required); adds on top of shift if both are set" },
  { section: "Connection › on (intra-region wiring)", key: "random", type: "hard", desc: "centroid.random — per-neuron random jitter (±1) of the shift. Default false. Ignored when centroid.twist is set" },
  { section: "Connection › on (intra-region wiring)", key: "fixed", type: "hard", desc: "wiring.fixed — disables random weight jitter" },

  { section: "Region › source (ascii)", key: "text", type: "soft", desc: 'chars to cycle: "AB" or synthetics "HALF_TOP,HALF_BOT,BARS_H,BARS_V,DOT_TL,DOT_BR"' },
  { section: "Region › source (ascii)", key: "frames_per_char", type: "soft", desc: "steps per character" },
  { section: "Region › source (ascii)", key: "font", type: "soft", desc: "font id" },
  { section: "Region › source (ascii)", key: "font_size", type: "soft", desc: "font size in px" },
  { section: "Region › source (ascii)", key: "background", type: "soft", desc: "noise.background — random pixel flip probability 0–1" },
  { section: "Region › source (ascii)", key: "shift", type: "soft", desc: "noise.shift — random 1-px shift each frame" },
  { section: "Region › source (ascii)", key: "inter_char", type: "soft", desc: "noise.inter_char — blank frame between characters" },
  { section: "Region › source (ascii)", key: "type", type: "hard", desc: "ascii · error_diff · label · label_mismatch · draw" },

  { section: "Region › source (label_mismatch)", key: "char_region", type: "hard", desc: 'region providing the current char. Default "input"' },
  { section: "Region › source (label_mismatch)", key: "label_region", type: "hard", desc: "region whose neuron_labels define the expected class" },
  { section: "Region › source (label_mismatch)", key: "mode", type: "hard", desc: '"mismatch" (default) or "active_avg" — broadcasts mean label activation to all nociceptor neurons' },

  { section: "Region › source (error_diff)", key: "label_region", type: "hard", desc: "target region for prediction (alias: output_region)" },
  { section: "Region › source (error_diff)", key: "target", type: "hard", desc: '"input" (default) or "label"' },
  { section: "Region › source (error_diff)", key: "diff_mode", type: "hard", desc: '"abs" (default) or "relu"' },

  { section: "Region › source (draw)", key: "noise", type: "soft", desc: "background noise on the user-painted region — nested {background:n} or a flat scalar" },
  { section: "Region › source (draw)", key: "loop", type: "soft", desc: '{frames:N, brush:{radius}, points:[{x,y}|null,...]} — one point per step-phase, replayed on a loop even after releasing the cursor. Written live while painting; survives reset. A plain number N is also accepted and auto-upgraded to this shape.' },

  { section: "Region › spiking", key: "up_ticks", type: "soft", desc: "max consecutive active steps before rest" },
  { section: "Region › spiking", key: "down_ticks", type: "soft", desc: "refractory period length" },

  { section: "Region › grid", key: "width", type: "hard", desc: "grid width" },
  { section: "Region › grid", key: "height", type: "hard", desc: "grid height" },

  { section: "Connections (inter-region full)", key: "rate", type: "soft", desc: "full.learning.rate — Hebbian learning rate for this connection" },
  { section: "Connections (inter-region full)", key: "exclude_range", type: "soft", desc: "full.learning.exclude_range — [lo, hi], skip weight updates for this connection's synapses while the weight is in this range" },
  { section: "Connections (inter-region full)", key: "weight", type: "hard", desc: "full.weight — initial dendrite weight" },
  { section: "Connections (inter-region full)", key: "density", type: "hard", desc: "full.density — fraction of source neurons sampled" },
  { section: "Connections (inter-region full)", key: "from", type: "hard", desc: "source region id (topology)" },
  { section: "Connections (inter-region full)", key: "to", type: "hard", desc: "destination region id (topology)" },
  { section: "Connections (inter-region full)", key: "id", type: "hard", desc: "explicit connection id, for orchestrator addressing" },

  { section: "Connections › portion", key: "type", type: "hard", desc: '"portion" — divides the source region into a grid of blocks, wires each destination neuron to its block' },
  { section: "Connections › portion", key: "portion", type: "hard", desc: "[rows, cols] — block grid shape (legacy alias: input.portion)" },

  { section: "Connections › nerve", key: "nerve", type: "hard", desc: 'places a spatial nerve circle on a host region and wires axons in/out of it — {"on":"<region>","nerve":{...}}' },
  { section: "Connections › nerve", key: "insertion", type: "hard", desc: 'nerve.insertion — {"x":int,"y":int} exact placement, or "random" (default)' },
  { section: "Connections › nerve", key: "radius", type: "hard", desc: "nerve.radius — circle radius. Default 6" },
  { section: "Connections › nerve", key: "region", type: "hard", desc: "nerve.from.region / nerve.to.region — source/destination region ids" },
  { section: "Connections › nerve", key: "density", type: "hard", desc: "nerve.from.density / nerve.to.density — fraction of neurons sampled. Default 0.1" },
  { section: "Connections › nerve", key: "weight", type: "soft", desc: "nerve.from.weight / nerve.to.weight — dendrite weight. Default 0.5" },
  { section: "Connections › nerve", key: "rate", type: "soft", desc: "nerve.from.learning.rate / nerve.to.learning.rate — Hebbian rate for that side" },
  { section: "Connections › nerve", key: "exclude_range", type: "soft", desc: "nerve.from.learning.exclude_range / nerve.to.learning.exclude_range (or ...deamon.learning.exclude_range when deamon-shaped) — [lo, hi], skip weight updates for that side's synapses while the weight is in this range" },
  { section: "Connections › nerve", key: "deamon", type: "hard", desc: "nerve.to.deamon — daemon-shaped wiring centered on each nerve circle, instead of plain full/dense wiring" },

  { section: "Orchestrator", key: "gradient", type: "hard", desc: 'pattern — interpolates a scalar between two ticks: {"from":{"tick":0,"set":"..."},"to":{"tick":N,"set":"..."}}' },
  { section: "Orchestrator", key: "one_shot", type: "hard", desc: 'pattern — fires once at tick N: {"at":{"tick":N},"set":"..."}. paths: connections[N]["full"]["weight"], regions[N]["text"], connections[N]["nerve"]["from"|"to"]["weight"|"rate"] — also addressable by connection "id" or by nerve region name' },
  { section: "Orchestrator", key: "inject", type: "hard", desc: 'writes activations directly into a region after procesar(): {"at":{"tick":0},"inject":{...}}. tick-0 also applies at setup (visible in step 0)' },
  { section: "Orchestrator", key: "tick_end", type: "hard", desc: 'inject sustained — repeats the injected pattern every tick in [tick, tick_end]: {"at":{"tick":0,"tick_end":20},"inject":{...}}' },
  { section: "Orchestrator", key: "region", type: "hard", desc: "inject.region — region id to write into" },
  { section: "Orchestrator", key: "template", type: "hard", desc: 'inject.template — "noise" (uniform random [0,1] activations) or "image" (PNG/JPG mapped to activations)' },
  { section: "Orchestrator", key: "src", type: "hard", desc: 'inject.src — path to the PNG/JPG, relative to backend/configs/. Alpha channel: white=activate, black=silence, transparent=leave unchanged' },
];

const UNIQUE_KEYS = Array.from(new Set(CONFIG_ROWS.map((r) => r.key))).sort();

type UsageResult =
  | { found: true; key: string; snippet: unknown; experiment_name: string; timestamp: string; source: string }
  | { found: false; key: string };

function HelpPanel() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState<{ text: string; tone: "ok" | "warn" } | null>(null);
  const [usage, setUsage] = useState<UsageResult | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const q = query.trim().toLowerCase();
  const acceptedKey = UNIQUE_KEYS.find((k) => k.toLowerCase() === q);
  const suggestions = q && !acceptedKey ? UNIQUE_KEYS.filter((k) => k.toLowerCase().startsWith(q)) : [];
  const topSuggestion = suggestions[0];

  useEffect(() => {
    if (!acceptedKey) {
      setUsage(null);
      return;
    }
    let cancelled = false;
    setUsageLoading(true);
    fetch(`/api/config-reference/usage?key=${encodeURIComponent(acceptedKey)}`)
      .then((r) => r.json())
      .then((data: UsageResult) => {
        if (!cancelled) setUsage(data);
      })
      .catch(() => {
        if (!cancelled) setUsage(null);
      })
      .finally(() => {
        if (!cancelled) setUsageLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [acceptedKey]);

  const showNotice = useCallback((text: string, tone: "ok" | "warn") => {
    setNotice({ text, tone });
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(null), 2200);
  }, []);

  const copyText = useCallback(
    (value: string, okMessage: string) => {
      navigator.clipboard?.writeText(value).then(() => showNotice(okMessage, "ok"));
    },
    [showNotice]
  );

  const copyRealUsage = useCallback(() => {
    if (usageLoading) return;
    if (usage && usage.found) {
      copyText(JSON.stringify(usage.snippet, null, 2), `Copied ✓ real example (${usage.experiment_name})`);
    } else {
      showNotice("No usage recorded — add an experiment that uses it, or retire it from the code", "warn");
    }
  }, [usage, usageLoading, copyText, showNotice]);

  return (
    <div style={{ marginTop: "8px" }}>
      <div
        onClick={() => setOpen((o) => !o)}
        style={{ display: "flex", alignItems: "center", gap: "5px", marginBottom: open ? "6px" : 0, cursor: "pointer" }}
      >
        <div style={{ display: "flex", alignItems: "center", padding: "2px 3px", flexShrink: 0, opacity: 0.35 }}>
          <svg width={8} height={12} viewBox="0 0 8 12">
            {([[1,1],[5,1],[1,5],[5,5],[1,9],[5,9]] as [number,number][]).map(([x,y]) => (
              <circle key={`${x}-${y}`} cx={x} cy={y} r={1.3} fill="#aaa" />
            ))}
          </svg>
        </div>
        <span
          style={{
            flex: 1,
            fontSize: "0.75rem",
            textTransform: "uppercase",
            color: "#888",
            letterSpacing: "0.1em",
            userSelect: "none",
          }}
        >
          Config Reference
        </span>
        <span style={{ color: "#444", fontSize: "0.65rem", userSelect: "none", paddingLeft: "4px" }}>
          {open ? "▾" : "▸"}
        </span>
      </div>
      {open && (
        <div
          style={{
            background: "#0d0d1a",
            border: "1px solid #2a2a4a",
            borderRadius: "5px",
            padding: "10px",
            fontSize: "0.72rem",
            color: "#888",
            lineHeight: "1.6",
          }}
        >
          <div style={{ position: "relative", marginBottom: "6px" }}>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Tab") {
                  if (topSuggestion) {
                    e.preventDefault();
                    setQuery(topSuggestion);
                  }
                  return;
                }
                if (e.key === "Enter") {
                  e.preventDefault();
                  const target = acceptedKey ?? topSuggestion;
                  if (target) copyText(target, `Copied ✓ "${target}"`);
                  else showNotice(`No config named "${query}"`, "warn");
                  return;
                }
                const atEnd = e.currentTarget.selectionStart === query.length;
                if ((e.key === "ArrowRight" && atEnd && acceptedKey) || (e.key === " " && acceptedKey)) {
                  e.preventDefault();
                  copyRealUsage();
                }
              }}
              placeholder="Config to use… (Tab completes, Enter copies the name)"
              style={{
                width: "100%",
                boxSizing: "border-box",
                padding: "7px 8px",
                background: "#12122a",
                border: `1px solid ${acceptedKey ? "#4ade80" : "#3a3a6a"}`,
                borderRadius: "4px",
                color: "#e0e0ff",
                fontSize: "0.75rem",
                fontFamily: "monospace",
                outline: "none",
              }}
            />
            {notice && (
              <span
                style={{
                  position: "absolute",
                  right: "6px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  fontSize: "0.62rem",
                  color: notice.tone === "ok" ? "#4ade80" : "#fbbf24",
                  background: "#12122a",
                  padding: "0 4px",
                  maxWidth: "70%",
                  textAlign: "right",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                title={notice.text}
              >
                {notice.text}
              </span>
            )}
          </div>

          <div style={{ color: "#555", fontSize: "0.62rem", marginBottom: "8px" }}>
            {acceptedKey ? (
              <>
                <code style={{ color: "#4ade80" }}>{acceptedKey}</code> ready — Enter copies the name ·
                → or space copies the real-usage JSON
              </>
            ) : topSuggestion ? (
              <>
                Tab → <code style={{ color: "#8fb8ff" }}>{topSuggestion}</code>
              </>
            ) : (
              <>
                <span style={{ color: "#4ade80" }}>soft</span> = applies live &nbsp;
                <span style={{ color: "#f87171" }}>hard</span> = requires Refresh
              </>
            )}
          </div>

          <div
            style={{
              maxHeight: "360px",
              minHeight: "120px",
              height: "360px",
              overflowY: "auto",
              resize: "vertical",
              border: "1px solid #1a1a2e",
              borderRadius: "4px",
              padding: "8px",
            }}
          >
            <HelpContent query={query} acceptedKey={acceptedKey} usage={usage} usageLoading={usageLoading} onCopyKey={(k) => copyText(k, `Copied ✓ "${k}"`)} />
          </div>
        </div>
      )}
    </div>
  );
}

function HelpContent({
  query,
  acceptedKey,
  usage,
  usageLoading,
  onCopyKey,
}: {
  query: string;
  acceptedKey: string | undefined;
  usage: UsageResult | null;
  usageLoading: boolean;
  onCopyKey: (key: string) => void;
}) {
  const q = query.trim().toLowerCase();

  const filtered = acceptedKey
    ? CONFIG_ROWS.filter((row) => row.key === acceptedKey)
    : q
    ? CONFIG_ROWS.filter(
        (row) => row.key.toLowerCase().includes(q) || row.desc.toLowerCase().includes(q) || row.section.toLowerCase().includes(q)
      )
    : CONFIG_ROWS;

  const sections: string[] = [];
  for (const row of filtered) {
    if (!sections.includes(row.section)) sections.push(row.section);
  }

  return (
    <div>
      {filtered.length === 0 && <div style={{ color: "#666", fontStyle: "italic" }}>No configs match "{query}"</div>}

      {sections.map((section, sIdx) => (
        <div key={section} style={{ marginTop: sIdx === 0 ? 0 : "14px" }}>
          <div
            style={{
              color: "#8fb8ff",
              fontWeight: 700,
              marginBottom: "6px",
              paddingBottom: "3px",
              borderBottom: "1px solid #22223f",
              fontSize: "0.7rem",
              letterSpacing: "0.02em",
            }}
          >
            {section}
          </div>
          {filtered
            .filter((row) => row.section === section)
            .map((row, i) => (
              <div
                key={`${row.key}-${i}`}
                onClick={() => onCopyKey(row.key)}
                title="Click to copy the name"
                style={{
                  cursor: "pointer",
                  padding: "5px 0",
                  borderBottom: i === filtered.filter((r) => r.section === section).length - 1 ? "none" : "1px dashed #1c1c33",
                }}
              >
                <code
                  style={{
                    color: "#4cc9f0",
                    background: "#141428",
                    padding: "1px 6px",
                    borderRadius: "3px",
                    fontFamily: "monospace",
                    fontWeight: 600,
                  }}
                >
                  {row.key}
                </code>{" "}
                <span
                  style={{
                    fontSize: "0.58rem",
                    fontWeight: 700,
                    padding: "1px 5px",
                    borderRadius: "3px",
                    color: row.type === "soft" ? "#4ade80" : "#f87171",
                    background: row.type === "soft" ? "#132218" : "#2a1414",
                  }}
                >
                  {row.type}
                </span>
                <div style={{ color: "#999", marginTop: "2px" }}>{row.desc}</div>
              </div>
            ))}
        </div>
      ))}

      {acceptedKey && (
        <div style={{ marginTop: "12px" }}>
          <div
            style={{
              color: "#8fb8ff",
              fontWeight: 700,
              marginBottom: "6px",
              paddingBottom: "3px",
              borderBottom: "1px solid #22223f",
              fontSize: "0.7rem",
              letterSpacing: "0.02em",
            }}
          >
            REAL EXAMPLE (last usage)
          </div>
          {usageLoading && <div style={{ color: "#666", fontStyle: "italic" }}>Looking up last usage…</div>}
          {!usageLoading && usage && usage.found && (
            <div>
              <div style={{ color: "#666", marginBottom: "4px" }}>
                {usage.experiment_name} · {usage.timestamp}
              </div>
              <pre
                style={{
                  background: "#141428",
                  border: "1px solid #22223f",
                  borderRadius: "4px",
                  padding: "8px",
                  color: "#c0c0e0",
                  fontSize: "0.68rem",
                  overflowX: "auto",
                  margin: 0,
                }}
              >
                {JSON.stringify(usage.snippet, null, 2)}
              </pre>
            </div>
          )}
          {!usageLoading && usage && !usage.found && (
            <div style={{ color: "#fbbf24" }}>
              No usage recorded in any experiment. Add one that uses it — or if it isn't worth keeping, it's time to retire it.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
