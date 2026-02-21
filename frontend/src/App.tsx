/** NeuroFlow — Main Application Layout. */

import { useState, useEffect, useCallback } from "react";
import { PixelCanvas } from "./components/PixelCanvas";
import { Sidebar } from "./components/Sidebar";
import { Controls } from "./components/Controls";
import { BrushPalette } from "./components/BrushPalette";
import { useExperiment } from "./hooks/useExperiment";
import { generateSquareBrush } from "./brushes";
import type { ExperimentInfo, ExperimentConfig } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "";

const DEFAULT_EXPERIMENTS: ExperimentInfo[] = [
  {
    id: "deamons_lab",
    name: "Deamons Lab",
    description:
      "Configurable mask connectivity lab with balance control",
    masks: [
      { id: "simple", name: "Kohonen Simple", description: "Moore r=1, corona r=2-4, 8 inh. dendrites.", center: "Moore r=1 (8 neighbors)", corona: "r=2-4, 8 blocks 3x3", dendrites_inh: 8 },
      { id: "wide_hat", name: "Wide Hat", description: "Moore r=1, corona r=2-7, 8 inh. dendrites.", center: "Moore r=1 (8 neighbors)", corona: "r=2-7, large corona", dendrites_inh: 8 },
      { id: "narrow_hat", name: "Narrow Hat", description: "Moore r=1, corona r=2-3, 8 inh. dendrites.", center: "Moore r=1 (8 neighbors)", corona: "r=2-3, close corona", dendrites_inh: 8 },
      { id: "big_center", name: "Big Center", description: "Moore r=2 (24 neighbors), corona r=4-7, 8 inh. dendrites.", center: "Moore r=2 (24 neighbors)", corona: "r=4-7, far corona", dendrites_inh: 8 },
      { id: "cross_center", name: "Cross Center", description: "Von Neumann r=1 (4 neighbors), corona r=2-4, 4 inh. dendrites.", center: "Von Neumann r=1 (4 neighbors)", corona: "r=2-4, 4 cardinal blocks", dendrites_inh: 4 },
      { id: "one_dendrite", name: "One Dendrite", description: "Moore r=1, corona r=2-4 in 1 single inh. dendrite.", center: "Moore r=1 (8 neighbors)", corona: "r=2-4, all in 1 dendrite", dendrites_inh: 1 },
      { id: "fine_grain", name: "Fine Grain", description: "Moore r=1, corona r=2-4, 16 inh. sectors.", center: "Moore r=1 (8 neighbors)", corona: "r=2-4, 16 sectors", dendrites_inh: 16 },
      { id: "double_ring", name: "Double Ring", description: "Moore r=1, ring r=2-3 (-1) + ring r=5-7 (-0.5).", center: "Moore r=1 (8 neighbors)", corona: "r=2-3 (-1) + r=5-7 (-0.5)", dendrites_inh: 16 },
      { id: "soft_inhibit", name: "Soft Inhibition", description: "Moore r=1, corona r=2-4, inh. weight -0.5.", center: "Moore r=1 (8 neighbors)", corona: "r=2-4, weight -0.5", dendrites_inh: 8 },
      { id: "strong_center", name: "Strong Center", description: "Moore r=1 x2 exc. dendrites, corona r=2-4.", center: "Moore r=1 (2 exc. dendrites)", corona: "r=2-4, weight -1", dendrites_inh: 8 },
    ],
    default_config: { width: 50, height: 50, balance: 0.0 },
  },
];

const DEFAULT_SELECTED = "deamons_lab";

function resolveConfig(exp: ExperimentInfo): ExperimentConfig {
  const cfg = { ...exp.default_config };
  if (!cfg.mask && exp.masks?.length) cfg.mask = exp.masks[0].id;
  return cfg;
}

function App() {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>(DEFAULT_EXPERIMENTS);
  const [selectedExp, setSelectedExp] = useState(DEFAULT_SELECTED);
  const [config, setConfig] = useState<ExperimentConfig>(
    resolveConfig(DEFAULT_EXPERIMENTS.find((e) => e.id === DEFAULT_SELECTED) ?? DEFAULT_EXPERIMENTS[0])
  );
  const [stepsPerTick, setStepsPerTick] = useState(1);

  const {
    grid,
    tensionGrid,
    tensionMode,
    state,
    stats,
    perf,
    generation,
    activeExperiment,
    inspectMode,
    connectionMap,
    inspectedCell,
    brushSize,
    brushMode,
    start,
    reconnect,
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

  // Fetch experiments list
  useEffect(() => {
    fetch(`${API_URL}/api/experiments`)
      .then((r) => r.json())
      .then((data: ExperimentInfo[]) => {
        if (data.length > 0) {
          setExperiments(data);
          const selected = data.find((e) => e.id === DEFAULT_SELECTED) ?? data[0];
          setConfig(resolveConfig(selected));
        }
      })
      .catch(() => {
        // Use defaults if API is unavailable
      });
  }, []);

  // WebSocket auto-connects via useExperiment hook

  const handleSelectExperiment = useCallback(
    (id: string) => {
      setSelectedExp(id);
      const exp = experiments.find((e) => e.id === id);
      if (exp) setConfig(resolveConfig(exp));
    },
    [experiments]
  );

  const handleStart = useCallback(() => {
    start(selectedExp, config);
  }, [start, selectedExp, config]);

  const handleReconnect = useCallback(() => {
    reconnect(config);
  }, [reconnect, config]);

  const applyBrush = useCallback(
    (x: number, y: number) => {
      if (inspectMode) return;
      const offsets = generateSquareBrush(brushSize);
      const value = brushMode === "activate" ? 1.0 : 0.0;
      const cells = offsets
        .map(([dx, dy]) => ({ x: x + dx, y: y + dy }))
        .filter(
          (c) =>
            c.x >= 0 && c.x < config.width && c.y >= 0 && c.y < config.height
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
  const hasGrid = grid.length > 0;
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
      }}
    >
      <Sidebar
        experiments={experiments}
        selectedExperiment={selectedExp}
        config={config}
        state={state}
        onSelectExperiment={handleSelectExperiment}
        onConfigChange={setConfig}
        onStart={handleStart}
        onReconnect={handleReconnect}
        connected={connected}
        experimentActive={hasGrid && activeExperiment === selectedExp}
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
              <PixelCanvas
                grid={grid}
                tensionGrid={tensionGrid}
                tensionMode={tensionMode}
                width={config.width}
                height={config.height}
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
                      ? "Select an experiment and start"
                      : "Connecting to server..."}
                  </p>
                </>
              )}
            </div>
          )}
        </div>

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

        <div
          style={{
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
