/** NeuroFlow — Main Application Layout. */

import { useState, useEffect, useCallback, useRef, type PointerEvent as ReactPointerEvent } from "react";
import { Sidebar } from "./components/Sidebar";
import { Controls } from "./components/Controls";
import { Scene } from "./components/Scene";
import { useExperiment } from "./hooks/useExperiment";
import { generateSquareBrush } from "./brushes";
import type { ConfigTemplate, ExperimentConfig, Metadata } from "./types";

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
    const tissue = (any.regions as Array<Record<string, unknown>>).find(r => r.wiring);
    const g = tissue?.grid as { width?: number; height?: number } | undefined;
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
      regions: (any.regions as Record<string, unknown>[]).map((r) => {
        const w = (r.wiring as Record<string, unknown>) ?? {};
        return {
          umbral: r.umbral,
          spiking: r.spiking,
          process_mode: w.process_mode,
          tension_function: w.tension_function,
          learning_rate: w.learning_rate,
          source: r.source,
        };
      }),
      connections: ((any.connections as Record<string, unknown>[]) ?? []).map((c) => ({
        learning: c.learning,
      })),
    });
  }
  // flat format
  return JSON.stringify({
    learning: cfg.learning,
    noise: cfg.noise,
    spiking: cfg.spiking,
    process_mode: cfg.wiring?.process_mode,
    tension_function: cfg.wiring?.tension_function,
    input: cfg.input,
  });
}

const SIDEBAR_DEFAULT = 380;
const SIDEBAR_MIN = 280;
const SIDEBAR_MAX = 700;

function App() {
  const [templates, setTemplates] = useState<ConfigTemplate[]>([]);
  const [metadata, setMetadata] = useState<Metadata | undefined>(undefined);
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [config, setConfig] = useState<ExperimentConfig>(DEFAULT_CONFIG);
  const [stepsPerTick, setStepsPerTick] = useState(1);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);
  const [isResizing, setIsResizing] = useState(false);

  // ── Execution history ──
  const [runHistory, setRunHistory] = useState<ExperimentConfig[]>([]);
  const [runIndex, setRunIndex] = useState(-1);

  const canGoPrev = runIndex > 0;
  const canGoNext = runIndex >= 0 && runIndex < runHistory.length - 1;

  const selectedTemplateRef = useRef(selectedTemplate);
  selectedTemplateRef.current = selectedTemplate;

  const loadHistory = useCallback((templateId: string, sessionConfig?: ExperimentConfig, fallbackConfig?: ExperimentConfig) => {
    fetch(`${API_URL}/api/templates/${templateId}/config/history?preset=_default`)
      .then((r) => r.json())
      .then((data: { history: { config: ExperimentConfig }[] }) => {
        if (selectedTemplateRef.current !== templateId) return;
        const configs = data.history.map((h) => h.config);
        setRunHistory(configs);
        if (sessionConfig) {
          setConfig(sessionConfig);
          setRunIndex(configs.length > 0 ? configs.length - 1 : -1);
        } else if (configs.length > 0) {
          setRunIndex(configs.length - 1);
          setConfig(configs[configs.length - 1]);
        } else {
          setRunIndex(-1);
          if (fallbackConfig) setConfig(fallbackConfig);
        }
      })
      .catch(() => {});
  }, []);

  const saveExecution = useCallback(
    (templateId: string, cfg: ExperimentConfig) => {
      fetch(`${API_URL}/api/templates/${templateId}/config?preset=_default`, {
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
    outputWeightGrid,
    nociceptorWeightGrid,
    regions,
    tensionRegions,
    tissueId,
    inputId,
    labels,
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
    brushSize,
    brushMode,
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

  // Fetch templates + metadata on mount
  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/api/templates`).then((r) => r.json()),
      fetch(`${API_URL}/api/metadata`).then((r) => r.json()),
    ])
      .then(([tplData, metaData]: [ConfigTemplate[], Metadata]) => {
        setTemplates(tplData);
        setMetadata(metaData);
        if (tplData.length > 0) {
          const savedId = localStorage.getItem("neuroflow_last_template");
          const initialTpl = tplData.find((t) => t.id === savedId) ?? tplData[0];
          const firstId = initialTpl.id;
          setSelectedTemplate(firstId);
          setConfig(initialTpl.config);

          fetch(`${API_URL}/api/session/last/${firstId}`)
            .then((r) => r.json())
            .then((data: { config: ExperimentConfig | null }) => {
              if (selectedTemplateRef.current !== firstId) return;
              loadHistory(firstId, data.config ?? undefined, initialTpl.config);
            })
            .catch(() => loadHistory(firstId, undefined, initialTpl.config));
        }
      })
      .catch(() => {});
  }, []);

  const hasGrid = grid.length > 0 || Object.keys(regions).length > 0;

  const saveLastSession = useCallback((templateId: string, cfg: ExperimentConfig) => {
    if (!templateId) return;
    fetch(`${API_URL}/api/session/last/${templateId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    }).catch(() => {});
  }, []);

  // Persist last config for this template to local session file (debounced)
  const sessionTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    if (!selectedTemplate) return;
    clearTimeout(sessionTimerRef.current);
    sessionTimerRef.current = setTimeout(
      () => saveLastSession(selectedTemplate, config),
      500,
    );
    return () => clearTimeout(sessionTimerRef.current);
  }, [config, selectedTemplate, saveLastSession]);

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

  const handleSelectTemplate = useCallback(
    (id: string) => {
      setSelectedTemplate(id);
      localStorage.setItem("neuroflow_last_template", id);
      const tpl = templates.find((t) => t.id === id);
      fetch(`${API_URL}/api/session/last/${id}`)
        .then((r) => r.json())
        .then((data: { config: ExperimentConfig | null }) => {
          if (selectedTemplateRef.current !== id) return;
          loadHistory(id, data.config ?? undefined, tpl?.config);
        })
        .catch(() => loadHistory(id, undefined, tpl?.config));
    },
    [templates, loadHistory],
  );

  const handleLoadSession = useCallback(() => {
    if (!selectedTemplate) return;
    fetch(`${API_URL}/api/session/last/${selectedTemplate}`)
      .then((r) => r.json())
      .then((data: { config: ExperimentConfig | null }) => {
        if (data.config) setConfig(data.config);
      })
      .catch(() => {});
  }, [selectedTemplate]);

  const handleLoadDefault = useCallback(() => {
    fetch(`${API_URL}/api/templates/refresh`, { method: "POST" })
      .then((r) => r.json())
      .then((tpls: ConfigTemplate[]) => {
        const tpl = tpls.find((t) => t.id === selectedTemplate);
        if (tpl) setConfig(tpl.config);
      })
      .catch(() => {});
  }, [selectedTemplate]);

  const handleStart = useCallback(() => {
    start(config);
    saveExecution(selectedTemplate, config);
  }, [start, config, saveExecution, selectedTemplate]);

  const handleRefresh = useCallback(() => {
    reconnect(config);
    saveExecution(selectedTemplate, config);
  }, [reconnect, config, saveExecution, selectedTemplate]);

  const applyBrush = useCallback(
    (x: number, y: number, regionId?: string) => {
      if (inspectMode) return;
      const offsets = generateSquareBrush(brushSize);
      const value = brushMode === "activate" ? 1.0 : 0.0;
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
      const cells = offsets
        .map(([dx, dy]) => ({ x: x + dx, y: y + dy }))
        .filter((c) => c.x >= 0 && c.x < w && c.y >= 0 && c.y < h);
      paint(cells, value, regionId);
    },
    [inspectMode, brushSize, brushMode, config, paint, regions]
  );

  const handleCellClick = useCallback(
    (x: number, y: number, regionId?: string) => {
      if (inspectMode) {
        inspect(x, y, regionId);
      } else {
        applyBrush(x, y, regionId);
      }
    },
    [inspectMode, inspect, applyBrush]
  );

  const handleCellDrag = useCallback(
    (x: number, y: number, regionId?: string) => {
      if (inspectMode) return;
      applyBrush(x, y, regionId);
    },
    [inspectMode, applyBrush]
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
  const hasConnectionMap = connectionMap != null;

  const colorSwatch = (bg: string, border?: string): React.CSSProperties => ({
    display: "inline-block",
    width: "10px",
    height: "10px",
    background: bg,
    border: border || "none",
    borderRadius: "2px",
    marginRight: "4px",
    verticalAlign: "middle",
  });


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
        templates={templates}
        selectedTemplate={selectedTemplate}
        config={config}
        metadata={metadata}
        state={state}
        stats={stats}
        onSelectTemplate={handleSelectTemplate}
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
        onLoadDefault={handleLoadDefault}
        onLoadSession={handleLoadSession}
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
              tissueId={tissueId}
              inputId={inputId}
              labels={labels}
              tensionMode={tensionMode}
              connectionMap={connectionMap}
              inspectedCell={inspectedCell}
              inspectedRegionId={inspectedRegionId}
              inspectInfo={inspectInfo}
              inputWeightGrid={inputWeightGrid}
              outputWeightGrid={outputWeightGrid}
              nociceptorWeightGrid={nociceptorWeightGrid}
              onCellClick={handleCellClick}
              onCellDrag={handleCellDrag}
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
                      ? "Select a template and start"
                      : "Connecting to server..."}
                  </p>
                </>
              )}
            </div>
          )}
        </div>

        <div
          style={{
            flexShrink: 0,
            display: "flex",
            gap: "16px",
            fontSize: "0.7rem",
            color: "#444",
            justifyContent: "center",
            paddingBottom: "4px",
          }}
        >
          {hasConnectionMap ? (
            <>
              <span>
                <span style={colorSwatch("#00ff00")} />
                Excitatory (+1)
              </span>
              <span>
                <span style={colorSwatch("#000000", "1px solid #333")} />
                Neutral (0)
              </span>
              <span>
                <span style={colorSwatch("#8b00ff")} />
                Inhibitory (-1)
              </span>
              <span>
                <span style={colorSwatch("#111111", "1px solid #333")} />
                No connection
              </span>
            </>
          ) : tensionMode ? (
            <>
              <span>
                <span style={colorSwatch("#ff8c00")} />
                Excitation (+1)
              </span>
              <span>
                <span style={colorSwatch("#0a0a0a", "1px solid #333")} />
                Neutral (0)
              </span>
              <span>
                <span style={colorSwatch("#5000ff")} />
                Inhibition (-1)
              </span>
            </>
          ) : (
            <>
              <span>
                <span style={colorSwatch("#ffffff")} />
                Active
              </span>
              <span>
                <span style={colorSwatch("#0a0a0a", "1px solid #333")} />
                Inactive
              </span>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
