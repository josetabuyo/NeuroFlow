/** NeuroFlow — Main Application Layout. */

import { useState, useEffect, useCallback, useRef, useMemo, type PointerEvent as ReactPointerEvent } from "react";
import { Sidebar } from "./components/Sidebar";
import { Controls } from "./components/Controls";
import { Scene } from "./components/Scene";
import { useExperiment } from "./hooks/useExperiment";
import { generateCircleBrush } from "./brushes";
import type { Experiment, ExperimentDetail, ExperimentConfig, Metadata } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "";

const DEFAULT_CONFIG: ExperimentConfig = {
  grid: { width: 50, height: 50 },
  wiring: {
    mask: "deamon_3_en_50",
    dendrite_exc_weight: 1,
    dendrite_inh_weight: -1,
    process_mode: "min_vs_max",
  },
};

function getConfigGrid(config: ExperimentConfig): { width: number; height: number } {
  const any = config as unknown as Record<string, unknown>;
  if (Array.isArray(any.regions)) {
    const mainRegion = (any.regions as Array<Record<string, unknown>>).find(r => r.wiring);
    const g = mainRegion?.grid as { width?: number; height?: number } | undefined;
    return { width: g?.width ?? 50, height: g?.height ?? 50 };
  }
  return { width: config.grid?.width ?? 50, height: config.grid?.height ?? 50 };
}

// Extract only the soft-updatable fields for change detection.
// Works for both flat legacy format and canonical regions[]/connections[] format.
function extractSoftFingerprint(cfg: ExperimentConfig): string {
  const any = cfg as unknown as Record<string, unknown>;
  if (Array.isArray(any.regions)) {
    return JSON.stringify({
      regions: (any.regions as Record<string, unknown>[]).map((r) => ({
        umbral: r.umbral,
        spiking: r.spiking,
        process_mode: r.process_mode,
        tension: r.tension,
        source: r.source,
      })),
      connections: ((any.connections as Record<string, unknown>[]) ?? []).map((c) => ({
        learning: (c.full as Record<string, unknown>)?.learning ?? c.learning,
      })),
    });
  }
  // flat format
  return JSON.stringify({
    learning: cfg.learning,
    noise: cfg.noise,
    spiking: cfg.spiking,
    process_mode: cfg.wiring?.process_mode,
    tension: cfg.wiring?.tension,
    input: cfg.input,
  });
}

const SIDEBAR_DEFAULT = 380;
const SIDEBAR_MIN = 280;
const SIDEBAR_MAX = 700;

function App() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [metadata, setMetadata] = useState<Metadata | undefined>(undefined);
  const [selectedExperimentId, setSelectedExperimentId] = useState<number | null>(null);
  const [config, setConfig] = useState<ExperimentConfig>(DEFAULT_CONFIG);
  const [stepsPerTick, setStepsPerTick] = useState(1);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);
  const [isResizing, setIsResizing] = useState(false);

  // ── Execution history ──
  const [runHistory, setRunHistory] = useState<ExperimentConfig[]>([]);
  const [runIndex, setRunIndex] = useState(-1);

  const canGoPrev = runIndex > 0;
  const canGoNext = runIndex >= 0 && runIndex < runHistory.length - 1;

  const selectedExperimentIdRef = useRef(selectedExperimentId);
  selectedExperimentIdRef.current = selectedExperimentId;

  const loadHistory = useCallback((experimentId: number, currentConfig?: ExperimentConfig) => {
    fetch(`${API_URL}/api/experiments/${experimentId}/runs`)
      .then((r) => r.json())
      .then((data: { history: { config: ExperimentConfig }[] }) => {
        if (selectedExperimentIdRef.current !== experimentId) return;
        const configs = data.history.map((h) => h.config);
        setRunHistory(configs);
        setRunIndex(configs.length > 0 ? configs.length - 1 : -1);
        if (currentConfig) setConfig(currentConfig);
      })
      .catch(() => {});
  }, []);

  const saveExecution = useCallback(
    (experimentId: number, cfg: ExperimentConfig) => {
      fetch(`${API_URL}/api/experiments/${experimentId}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      })
        .then((r) => r.json())
        .then((data: { id: number }) => {
          if (data.id !== -1) {
            setRunHistory((prev) => [...prev, cfg]);
            setRunIndex((prev) => prev + 1);
          }
        })
        .catch(() => {});
    },
    [],
  );

  const goPrev = useCallback(() => {
    setRunIndex((i) => {
      if (i <= 0) return i;
      const next = i - 1;
      setConfig(runHistory[next]);
      return next;
    });
  }, [runHistory]);

  const goNext = useCallback(() => {
    setRunIndex((i) => {
      if (i >= runHistory.length - 1) return i;
      const next = i + 1;
      setConfig(runHistory[next]);
      return next;
    });
  }, [runHistory]);

  const handleResizePointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setIsResizing(true);
  }, []);

  useEffect(() => {
    if (!isResizing) return;
    const onMove = (e: globalThis.PointerEvent) => {
      setSidebarWidth(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, e.clientX)));
    };
    const onUp = () => setIsResizing(false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [isResizing]);

  const {
    grid,
    tensionMode,
    inputWeightGrid,
    sourceWeightGrids,
    regionOverlays,
    nerveCircles,
    regions,
    tensionRegions,
    regionId,
    inputId,
    labels,
    neuronLabelMap,
    orchestratorState,
    state,
    stats,
    perf,
    generation,
    experimentActive,
    inspectMode,
    connectionMap,
    inspectedCell,
    inspectedRegionId,
    inspectInfo,
    inspectHistory,
    brushSize,
    brushMode,
    normalizedConfig,
    start,
    reconnect,
    updateConfig,
    paint,
    step,
    play,
    pause,
    reset,
    inspect,
    toggleInspectMode,
    toggleTensionMode,
    increaseBrushSize,
    decreaseBrushSize,
    toggleBrushMode,
  } = useExperiment();

  // When backend sends back the normalized config, update the editor
  useEffect(() => {
    if (normalizedConfig) {
      setConfig(normalizedConfig as unknown as ExperimentConfig);
    }
  }, [normalizedConfig]);

  // Fetch experiments + metadata on mount
  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/api/experiments`).then((r) => r.json()),
      fetch(`${API_URL}/api/metadata`).then((r) => r.json()),
    ])
      .then(([expData, metaData]: [Experiment[], Metadata]) => {
        setExperiments(expData);
        setMetadata(metaData);
        if (expData.length > 0) {
          const savedId = Number(localStorage.getItem("neuroflow_last_experiment_id"));
          const initial = expData.find((e) => e.id === savedId) ?? expData[0];
          setSelectedExperimentId(initial.id);

          fetch(`${API_URL}/api/experiments/${initial.id}`)
            .then((r) => r.json())
            .then((detail: ExperimentDetail) => {
              if (selectedExperimentIdRef.current !== initial.id) return;
              loadHistory(initial.id, detail.config);
            })
            .catch(() => {});
        }
      })
      .catch(() => {});
  }, []);

  const hasGrid = grid.length > 0 || Object.keys(regions).length > 0;

  const saveExperimentConfig = useCallback((experimentId: number, cfg: ExperimentConfig) => {
    fetch(`${API_URL}/api/experiments/${experimentId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: cfg }),
    }).catch(() => {});
  }, []);

  // Persist current config for this experiment to the DB (debounced autosave)
  const sessionTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    if (selectedExperimentId == null) return;
    clearTimeout(sessionTimerRef.current);
    sessionTimerRef.current = setTimeout(
      () => saveExperimentConfig(selectedExperimentId, config),
      500,
    );
    return () => clearTimeout(sessionTimerRef.current);
  }, [config, selectedExperimentId, saveExperimentConfig]);

  // Soft config sync: update running experiment when certain nested fields change
  const prevConfigRef = useRef(config);
  const liveTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    const prev = prevConfigRef.current;
    prevConfigRef.current = config;

    if (!experimentActive || !hasGrid) return;

    const changed = extractSoftFingerprint(config) !== extractSoftFingerprint(prev);

    if (!changed) return;

    clearTimeout(liveTimerRef.current);
    liveTimerRef.current = setTimeout(() => updateConfig(config), 80);
    return () => clearTimeout(liveTimerRef.current);
  }, [config, updateConfig, experimentActive, hasGrid]);

  const handleSelectExperiment = useCallback(
    (id: number) => {
      setSelectedExperimentId(id);
      localStorage.setItem("neuroflow_last_experiment_id", String(id));
      fetch(`${API_URL}/api/experiments/${id}`)
        .then((r) => r.json())
        .then((detail: ExperimentDetail) => {
          if (selectedExperimentIdRef.current !== id) return;
          loadHistory(id, detail.config);
        })
        .catch(() => {});
    },
    [loadHistory],
  );

  const handleRevert = useCallback(() => {
    if (selectedExperimentId == null) return;
    fetch(`${API_URL}/api/experiments/${selectedExperimentId}`)
      .then((r) => r.json())
      .then((detail: ExperimentDetail) => setConfig(detail.config))
      .catch(() => {});
  }, [selectedExperimentId]);

  const handleCreateExperiment = useCallback(() => {
    fetch(`${API_URL}/api/experiments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Untitled", config: DEFAULT_CONFIG }),
    })
      .then((r) => r.json())
      .then((detail: ExperimentDetail) => {
        setExperiments((prev) => [...prev, detail]);
        setSelectedExperimentId(detail.id);
        localStorage.setItem("neuroflow_last_experiment_id", String(detail.id));
        setConfig(detail.config);
        setRunHistory([]);
        setRunIndex(-1);
      })
      .catch(() => {});
  }, []);

  const handleRenameExperiment = useCallback((id: number, name: string) => {
    fetch(`${API_URL}/api/experiments/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    })
      .then((r) => r.json())
      .then((detail: ExperimentDetail) => {
        setExperiments((prev) => prev.map((e) => (e.id === id ? detail : e)));
      })
      .catch(() => {});
  }, []);

  const handleDeleteExperiment = useCallback(
    (id: number) => {
      fetch(`${API_URL}/api/experiments/${id}`, { method: "DELETE" })
        .then(() => {
          setExperiments((prev) => {
            const remaining = prev.filter((e) => e.id !== id);
            if (selectedExperimentIdRef.current === id) {
              const next = remaining[0];
              if (next) {
                handleSelectExperiment(next.id);
              } else {
                setSelectedExperimentId(null);
                setRunHistory([]);
                setRunIndex(-1);
              }
            }
            return remaining;
          });
        })
        .catch(() => {});
    },
    [handleSelectExperiment],
  );

  const handleReorderExperiment = useCallback((id: number, direction: "up" | "down") => {
    setExperiments((prev) => {
      const sorted = [...prev].sort((a, b) => a.position - b.position);
      const idx = sorted.findIndex((e) => e.id === id);
      const swapIdx = direction === "up" ? idx - 1 : idx + 1;
      if (idx === -1 || swapIdx < 0 || swapIdx >= sorted.length) return prev;

      const a = sorted[idx];
      const b = sorted[swapIdx];
      fetch(`${API_URL}/api/experiments/${a.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ position: b.position }),
      }).catch(() => {});
      fetch(`${API_URL}/api/experiments/${b.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ position: a.position }),
      }).catch(() => {});

      return prev
        .map((e) => {
          if (e.id === a.id) return { ...e, position: b.position };
          if (e.id === b.id) return { ...e, position: a.position };
          return e;
        })
        .sort((x, y) => x.position - y.position);
    });
  }, []);

  const handleStart = useCallback(() => {
    if (selectedExperimentId == null) return;
    start(config);
    saveExecution(selectedExperimentId, config);
  }, [start, config, saveExecution, selectedExperimentId]);

  const handleRefresh = useCallback(() => {
    if (selectedExperimentId == null) return;
    reconnect(config);
    saveExecution(selectedExperimentId, config);
  }, [reconnect, config, saveExecution, selectedExperimentId]);

  const computeBrushCells = useCallback(
    (x: number, y: number, regionId?: string): { x: number; y: number }[] => {
      const offsets = generateCircleBrush(brushSize);
      let w: number, h: number;
      if (regionId && regions[regionId]) {
        const g = regions[regionId];
        h = g.length;
        w = g[0]?.length ?? 1;
      } else {
        const g = getConfigGrid(config);
        w = g.width;
        h = g.height;
      }
      return offsets
        .map(([dx, dy]) => ({ x: x + dx, y: y + dy }))
        .filter((c) => c.x >= 0 && c.x < w && c.y >= 0 && c.y < h);
    },
    [brushSize, config, regions]
  );

  const handleCellClick = useCallback(
    (x: number, y: number, regionId?: string) => {
      if (inspectMode) {
        inspect(x, y, regionId);
      } else {
        const cells = computeBrushCells(x, y, regionId);
        const value = brushMode === "activate" ? 1.0 : 0.0;
        // origin/radius are the raw (unexpanded) brush center+size — the
        // backend uses these only to record a "loop" pattern into config,
        // when the region has one configured.
        paint(cells, value, regionId, { x, y }, Math.floor(brushSize / 2));
      }
    },
    [inspectMode, inspect, computeBrushCells, brushMode, paint, brushSize]
  );

  // Coalesces paints from rapid drag movement into at most one `paint` per
  // rendered frame — the backend re-serializes full brain state on every
  // paint, so one message per mousemove event was the source of the freeze.
  const dragFrameRef = useRef<{
    cellsByKey: Map<string, { x: number; y: number }>;
    lastPoint: { x: number; y: number } | null;
    regionId?: string;
    scheduled: boolean;
  } | null>(null);

  const flushDragFrame = useCallback(() => {
    const frame = dragFrameRef.current;
    if (!frame) return;
    frame.scheduled = false;
    if (frame.cellsByKey.size === 0) return;
    const cells = Array.from(frame.cellsByKey.values());
    const lastPoint = frame.lastPoint;
    frame.cellsByKey.clear();
    const value = brushMode === "activate" ? 1.0 : 0.0;
    paint(cells, value, frame.regionId, lastPoint ?? undefined, Math.floor(brushSize / 2));
  }, [brushMode, paint, brushSize]);

  const handleCellDrag = useCallback(
    (x: number, y: number, regionId?: string) => {
      if (inspectMode) {
        // Drag the observer across neurons — PixelCanvas already dedupes
        // consecutive calls to the same cell, so this fires once per cell entered.
        inspect(x, y, regionId);
        return;
      }
      const cells = computeBrushCells(x, y, regionId);
      if (!dragFrameRef.current) {
        dragFrameRef.current = { cellsByKey: new Map(), lastPoint: null, regionId, scheduled: false };
      }
      const frame = dragFrameRef.current;
      frame.regionId = regionId;
      frame.lastPoint = { x, y };
      for (const cell of cells) frame.cellsByKey.set(`${cell.x},${cell.y}`, cell);
      if (!frame.scheduled) {
        frame.scheduled = true;
        requestAnimationFrame(flushDragFrame);
      }
    },
    [inspectMode, inspect, computeBrushCells, flushDragFrame]
  );

  // Clears the live draw-cursor stamp on mouse-up/mouse-leave so nothing
  // stays painted once the user stops — draw regions rebuild from zero
  // every tick and only show the cursor while actively painting.
  const handleCellDragEnd = useCallback(
    (_cells: { x: number; y: number }[], regionId?: string) => {
      if (inspectMode) return;
      const value = brushMode === "activate" ? 1.0 : 0.0;
      paint([], value, regionId);
    },
    [inspectMode, brushMode, paint]
  );

  const handlePlay = useCallback(
    () => play(10, stepsPerTick),
    [play, stepsPerTick]
  );

  const handleStep = useCallback(
    () => step(stepsPerTick),
    [step, stepsPerTick]
  );

  const connected = state !== "disconnected";
  const isInitializing = state === "initializing";


  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        background: "#0a0a0f",
        color: "#e0e0ff",
        fontFamily:
          "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        userSelect: isResizing ? "none" : undefined,
      }}
    >
      <Sidebar
        experiments={experiments}
        selectedExperimentId={selectedExperimentId}
        config={config}
        metadata={metadata}
        state={state}
        stats={stats}
        onSelectExperiment={handleSelectExperiment}
        onConfigChange={setConfig}
        onStart={handleStart}
        onRefresh={handleRefresh}
        connected={connected}
        experimentActive={experimentActive && hasGrid}
        width={sidebarWidth}
        onPrevRun={goPrev}
        onNextRun={goNext}
        canGoPrev={canGoPrev}
        canGoNext={canGoNext}
        runPosition={runIndex >= 0 ? runIndex + 1 : 0}
        runTotal={runHistory.length}
        onRevert={handleRevert}
        onCreateExperiment={handleCreateExperiment}
        onRenameExperiment={handleRenameExperiment}
        onDeleteExperiment={handleDeleteExperiment}
        onReorderExperiment={handleReorderExperiment}
      />

      {/* Resize handle */}
      <div
        onPointerDown={handleResizePointerDown}
        style={{
          width: "6px",
          cursor: "col-resize",
          background: isResizing ? "#4cc9f0" : "#2a2a3e",
          transition: isResizing ? "none" : "background 0.15s",
          flexShrink: 0,
          position: "relative",
          zIndex: 20,
        }}
        onMouseEnter={(e) => {
          if (!isResizing) (e.currentTarget as HTMLElement).style.background = "#4cc9f080";
        }}
        onMouseLeave={(e) => {
          if (!isResizing) (e.currentTarget as HTMLElement).style.background = "#2a2a3e";
        }}
      />

      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: "20px 30px",
          gap: "12px",
          overflow: "hidden",
        }}
      >
        <div style={{ flexShrink: 0 }}>
          <Controls
            state={state}
            stats={stats}
            perf={perf}
            generation={generation}
            stepsPerTick={stepsPerTick}
            orchestratorState={orchestratorState}
            onPlay={handlePlay}
            onPause={pause}
            onStep={handleStep}
            onReset={reset}
            onStepsPerTickChange={setStepsPerTick}
          />
        </div>

        <div
          style={{
            flex: 1,
            minHeight: 0,
            background: "#0d0d14",
            borderRadius: "8px",
            border: "1px solid #1a1a2e",
            overflow: "hidden",
            position: "relative",
          }}
        >
          {hasGrid ? (
            <Scene
              regions={regions}
              tensionRegions={tensionRegions}
              regionId={regionId}
              inputId={inputId}
              labels={labels}
              neuronLabelMap={neuronLabelMap}
              tensionMode={tensionMode}
              connectionMap={connectionMap}
              inspectedCell={inspectedCell}
              inspectedRegionId={inspectedRegionId}
              inspectInfo={inspectInfo}
              inspectHistory={inspectHistory}
              inputWeightGrid={inputWeightGrid}
              sourceWeightGrids={sourceWeightGrids}
              regionOverlays={regionOverlays}
              nerveCircles={nerveCircles}
              onCellClick={handleCellClick}
              onCellDrag={handleCellDrag}
              onCellDragEnd={handleCellDragEnd}
              brushSize={brushSize}
              brushMode={brushMode}
              inspectMode={inspectMode}
              canInspect={state === "ready" || state === "paused" || state === "running"}
              onIncrease={increaseBrushSize}
              onDecrease={decreaseBrushSize}
              onToggleMode={toggleBrushMode}
              onToggleInspect={toggleInspectMode}
              onToggleTension={toggleTensionMode}
              isInitializing={isInitializing}
            />
          ) : (
            <div
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                color: "#444",
                fontSize: "0.9rem",
              }}
            >
              {isInitializing ? (
                <>
                  <div className="neuro-spinner" style={{ margin: "0 auto 12px" }} />
                  <p>Building network...</p>
                </>
              ) : (
                <>
                  <p style={{ fontSize: "2rem", marginBottom: "12px" }}>
                    {connected ? "\uD83E\uDDE0" : "\u23F3"}
                  </p>
                  <p>
                    {connected
                      ? "Select an experiment and start"
                      : "Connecting to server..."}
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
