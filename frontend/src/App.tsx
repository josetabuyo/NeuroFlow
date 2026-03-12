/** NeuroFlow — Main Application Layout. */

import { useState, useEffect, useCallback, useRef, type PointerEvent as ReactPointerEvent } from "react";
import { PixelCanvas } from "./components/PixelCanvas";
import { MiniGrid } from "./components/MiniGrid";
import { Sidebar } from "./components/Sidebar";
import { Controls } from "./components/Controls";
import { BrushPalette } from "./components/BrushPalette";
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

  const loadHistory = useCallback((templateId: string) => {
    fetch(`${API_URL}/api/templates/${templateId}/config/history?preset=_default`)
      .then((r) => r.json())
      .then((data: { history: { config: ExperimentConfig }[] }) => {
        if (selectedTemplateRef.current !== templateId) return;
        const configs = data.history.map((h) => h.config);
        setRunHistory(configs);
        if (configs.length > 0) {
          setRunIndex(configs.length - 1);
          setConfig(configs[configs.length - 1]);
        } else {
          setRunIndex(-1);
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
    tensionGrid,
    tensionMode,
    inputFrame,
    inputWeightGrid,
    inputWeightDims,
    state,
    stats,
    perf,
    generation,
    experimentActive,
    inspectMode,
    connectionMap,
    inspectedCell,
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
          const firstId = tplData[0].id;
          setSelectedTemplate(firstId);
          setConfig(tplData[0].config);

          // Load history for first template
          fetch(`${API_URL}/api/templates/${firstId}/config/history?preset=_default`)
            .then((r) => r.json())
            .then((histData: { history: { config: ExperimentConfig }[] }) => {
              const configs = histData.history.map((h) => h.config);
              setRunHistory(configs);
              if (configs.length > 0) {
                setRunIndex(configs.length - 1);
                setConfig(configs[configs.length - 1]);
              }
            })
            .catch(() => {});
        }
      })
      .catch(() => {});
  }, []);

  const hasGrid = grid.length > 0;

  // Soft config sync: update running experiment when certain nested fields change
  const prevConfigRef = useRef(config);
  const liveTimerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    const prev = prevConfigRef.current;
    prevConfigRef.current = config;

    if (!experimentActive || !hasGrid) return;

    const changed =
      JSON.stringify(config.learning) !== JSON.stringify(prev.learning) ||
      JSON.stringify(config.noise) !== JSON.stringify(prev.noise) ||
      JSON.stringify(config.spiking) !== JSON.stringify(prev.spiking) ||
      JSON.stringify(config.wiring?.process_mode) !== JSON.stringify(prev.wiring?.process_mode) ||
      JSON.stringify(config.wiring?.tension_function) !== JSON.stringify(prev.wiring?.tension_function) ||
      JSON.stringify(config.input?.text) !== JSON.stringify(prev.input?.text) ||
      JSON.stringify(config.input?.font) !== JSON.stringify(prev.input?.font) ||
      JSON.stringify(config.input?.font_size) !== JSON.stringify(prev.input?.font_size) ||
      JSON.stringify(config.input?.frames_per_char) !== JSON.stringify(prev.input?.frames_per_char);

    if (!changed) return;

    clearTimeout(liveTimerRef.current);
    liveTimerRef.current = setTimeout(() => updateConfig(config), 80);
    return () => clearTimeout(liveTimerRef.current);
  }, [config, updateConfig, experimentActive, hasGrid]);

  const handleSelectTemplate = useCallback(
    (id: string) => {
      setSelectedTemplate(id);
      const tpl = templates.find((t) => t.id === id);
      if (!tpl) return;
      setConfig(tpl.config);
      loadHistory(id);
    },
    [templates, loadHistory],
  );

  const handleStart = useCallback(() => {
    start(config);
    saveExecution(selectedTemplate, config);
  }, [start, config, saveExecution, selectedTemplate]);

  const handleRefresh = useCallback(() => {
    reconnect(config);
    saveExecution(selectedTemplate, config);
  }, [reconnect, config, saveExecution, selectedTemplate]);

  const applyBrush = useCallback(
    (x: number, y: number) => {
      if (inspectMode) return;
      const offsets = generateSquareBrush(brushSize);
      const value = brushMode === "activate" ? 1.0 : 0.0;
      const cells = offsets
        .map(([dx, dy]) => ({ x: x + dx, y: y + dy }))
        .filter(
          (c) =>
            c.x >= 0 && c.x < config.grid.width && c.y >= 0 && c.y < config.grid.height
        );
      paint(cells, value);
    },
    [inspectMode, brushSize, brushMode, config, paint]
  );

  const handleCellClick = useCallback(
    (x: number, y: number) => {
      if (inspectMode) {
        inspect(x, y);
      } else {
        applyBrush(x, y);
      }
    },
    [inspectMode, inspect, applyBrush]
  );

  const handleCellDrag = useCallback(
    (x: number, y: number) => {
      if (inspectMode) return;
      applyBrush(x, y);
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
        inputFrame={inputFrame}
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
        <div
          style={{
            flex: 1,
            minHeight: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#0d0d14",
            borderRadius: "8px",
            border: "1px solid #1a1a2e",
            overflow: "hidden",
            padding: "16px",
            position: "relative",
          }}
        >
          {hasGrid ? (
            <>
              <div style={{ display: "flex", gap: "16px", width: "100%", height: "100%", minHeight: 0, alignItems: "center", justifyContent: "center" }}>
                <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", justifyContent: "center", height: "100%", position: "relative" }}>
                  <PixelCanvas
                    grid={grid}
                    tensionGrid={tensionGrid}
                    tensionMode={tensionMode}
                    width={config.grid.width}
                    height={config.grid.height}
                    connectionMap={connectionMap}
                    inspectedCell={inspectedCell}
                    onCellClick={handleCellClick}
                    onCellDrag={handleCellDrag}
                  />
                  <BrushPalette
                    brushSize={brushSize}
                    brushMode={brushMode}
                    inspectMode={inspectMode}
                    tensionMode={tensionMode}
                    canInspect={
                      state === "ready" ||
                      state === "paused" ||
                      state === "running"
                    }
                    onIncrease={increaseBrushSize}
                    onDecrease={decreaseBrushSize}
                    onToggleMode={toggleBrushMode}
                    onToggleInspect={toggleInspectMode}
                    onToggleTension={toggleTensionMode}
                  />
                </div>
                {inputFrame && (
                  <div style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px",
                    alignItems: "center",
                    flexShrink: 0,
                    maxHeight: "100%",
                    overflow: "auto",
                  }}>
                    <MiniGrid
                      label="Input"
                      grid={inputFrame}
                      width={inputFrame[0]?.length ?? 0}
                      height={inputFrame.length}
                      maxSize={140}
                      subtitle={stats?.current_char ? `"${stats.current_char}" ${(stats.frame_in_char ?? 0) + 1}/${stats.frames_per_char ?? "?"}` : undefined}
                    />
                    {inputWeightGrid && inputWeightDims && (
                      <MiniGrid
                        label="Learned"
                        grid={inputWeightGrid}
                        width={inputWeightDims.width}
                        height={inputWeightDims.height}
                        maxSize={140}
                        colorMode="weight"
                        subtitle={inspectedCell ? `Neuron (${inspectedCell.x}, ${inspectedCell.y})` : undefined}
                      />
                    )}
                  </div>
                )}
              </div>
              {isInitializing && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "rgba(10, 10, 15, 0.75)",
                    borderRadius: "8px",
                    zIndex: 10,
                  }}
                >
                  <div style={{ textAlign: "center", color: "#888" }}>
                    <div className="neuro-spinner" />
                    <p style={{ marginTop: "12px", fontSize: "0.85rem" }}>
                      Building network...
                    </p>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div
              style={{
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
