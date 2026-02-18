/** NeuroFlow — Main Application Layout. */

import { useState, useEffect, useCallback } from "react";
import { PixelCanvas } from "./components/PixelCanvas";
import { Sidebar } from "./components/Sidebar";
import { Controls } from "./components/Controls";
import { BrushPalette } from "./components/BrushPalette";
import { useExperiment } from "./hooks/useExperiment";
import { BRUSHES } from "./brushes";
import type { ExperimentInfo, ExperimentConfig } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "";

const DEFAULT_EXPERIMENTS: ExperimentInfo[] = [
  {
    id: "von_neumann",
    name: "Autómata Elemental (Von Neumann)",
    description: "Autómata celular elemental 1D (reglas de Wolfram)",
    rules: [111, 30, 90, 110],
    default_config: { width: 50, height: 50, rule: 111 },
  },
  {
    id: "kohonen",
    name: "Kohonen (Competencia Lateral 2D)",
    description:
      "Mapa autoorganizado con excitación local e inhibición lateral",
    default_config: { width: 30, height: 30 },
  },
  {
    id: "kohonen_balanced",
    name: "Kohonen Balanceado",
    description:
      "Kohonen con balance configurable del Fuzzy OR",
    default_config: { width: 30, height: 30, balance: 0.0 },
  },
  {
    id: "kohonen_lab",
    name: "Kohonen Lab",
    description:
      "Laboratorio de conexionados con máscara y balance configurables",
    masks: [
      { id: "simple", name: "Kohonen Simple", description: "Moore r=1, corona r=2-4, 8 dendritas inh.", center: "Moore r=1 (8 vecinos)", corona: "r=2-4, 8 bloques 3x3", dendrites_inh: 8 },
      { id: "wide_hat", name: "Sombrero Ancho", description: "Moore r=1, corona r=2-7, 8 dendritas inh.", center: "Moore r=1 (8 vecinos)", corona: "r=2-7, corona grande", dendrites_inh: 8 },
      { id: "narrow_hat", name: "Sombrero Estrecho", description: "Moore r=1, corona r=2-3, 8 dendritas inh.", center: "Moore r=1 (8 vecinos)", corona: "r=2-3, corona cercana", dendrites_inh: 8 },
      { id: "big_center", name: "Centro Grande", description: "Moore r=2 (24 vecinos), corona r=4-7, 8 dendritas inh.", center: "Moore r=2 (24 vecinos)", corona: "r=4-7, corona lejana", dendrites_inh: 8 },
      { id: "cross_center", name: "Cruz Central", description: "Von Neumann r=1 (4 vecinos), corona r=2-4, 4 dendritas inh.", center: "Von Neumann r=1 (4 vecinos)", corona: "r=2-4, 4 bloques cardinales", dendrites_inh: 4 },
      { id: "one_dendrite", name: "Una Dendrita", description: "Moore r=1, corona r=2-4 en 1 sola dendrita inh.", center: "Moore r=1 (8 vecinos)", corona: "r=2-4, todo en 1 dendrita", dendrites_inh: 1 },
      { id: "fine_grain", name: "Grano Fino", description: "Moore r=1, corona r=2-4, 16 sectores inh.", center: "Moore r=1 (8 vecinos)", corona: "r=2-4, 16 sectores", dendrites_inh: 16 },
      { id: "double_ring", name: "Doble Anillo", description: "Moore r=1, anillo r=2-3 (-1) + anillo r=5-7 (-0.5).", center: "Moore r=1 (8 vecinos)", corona: "r=2-3 (-1) + r=5-7 (-0.5)", dendrites_inh: 16 },
      { id: "soft_inhibit", name: "Inhibicion Suave", description: "Moore r=1, corona r=2-4, peso inh. -0.5.", center: "Moore r=1 (8 vecinos)", corona: "r=2-4, peso -0.5", dendrites_inh: 8 },
      { id: "strong_center", name: "Centro Fuerte", description: "Moore r=1 x2 dendritas exc., corona r=2-4.", center: "Moore r=1 (2 dendritas exc.)", corona: "r=2-4, peso -1", dendrites_inh: 8 },
    ],
    default_config: { width: 30, height: 30, mask: "simple", balance: 0.0 },
  },
];

function App() {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>(DEFAULT_EXPERIMENTS);
  const [selectedExp, setSelectedExp] = useState("von_neumann");
  const [config, setConfig] = useState<ExperimentConfig>({
    width: 50,
    height: 50,
    rule: 111,
  });
  const [stepsPerTick, setStepsPerTick] = useState(1);

  const {
    grid,
    state,
    stats,
    perf,
    generation,
    activeExperiment,
    inspectMode,
    connectionMap,
    inspectedCell,
    selectedBrush,
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
    setSelectedBrush,
    toggleBrushMode,
  } = useExperiment();

  // Fetch experiments list
  useEffect(() => {
    fetch(`${API_URL}/api/experiments`)
      .then((r) => r.json())
      .then((data: ExperimentInfo[]) => {
        if (data.length > 0) {
          setExperiments(data);
          setConfig(data[0].default_config);
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
      if (exp) setConfig(exp.default_config);
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
      const brush = BRUSHES.find((b) => b.id === selectedBrush);
      if (!brush) return;
      const value = brushMode === "activate" ? 1.0 : 0.0;
      const cells = brush.offsets
        .map(([dx, dy]) => ({ x: x + dx, y: y + dy }))
        .filter(
          (c) =>
            c.x >= 0 && c.x < config.width && c.y >= 0 && c.y < config.height
        );
      paint(cells, value);
    },
    [inspectMode, selectedBrush, brushMode, config, paint]
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
                width={config.width}
                height={config.height}
                {...(selectedExp !== "kohonen" && selectedExp !== "kohonen_balanced" && selectedExp !== "kohonen_lab"
                  ? { inputRow: config.height - 1, outputRow: 0 }
                  : {})}
                connectionMap={connectionMap}
                inspectedCell={inspectedCell}
                onCellClick={handleCellClick}
                onCellDrag={handleCellDrag}
              />
              <BrushPalette
                brushes={BRUSHES}
                selectedBrush={selectedBrush}
                brushMode={brushMode}
                disabled={inspectMode}
                onSelectBrush={setSelectedBrush}
                onToggleMode={toggleBrushMode}
              />
            </>
          ) : (
            <div
              style={{
                textAlign: "center",
                color: "#444",
                fontSize: "0.9rem",
              }}
            >
              <p style={{ fontSize: "2rem", marginBottom: "12px" }}>
                {connected ? "\uD83E\uDDE0" : "\u23F3"}
              </p>
              <p>
                {connected
                  ? "Selecciona un experimento e inicia"
                  : "Conectando al servidor..."}
              </p>
            </div>
          )}
        </div>

        <Controls
          state={state}
          stats={stats}
          perf={perf}
          generation={generation}
          inspectMode={inspectMode}
          stepsPerTick={stepsPerTick}
          onPlay={handlePlay}
          onPause={pause}
          onStep={handleStep}
          onReset={reset}
          onToggleInspect={toggleInspectMode}
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
                Excitatorio (+1)
              </span>
              <span>
                <span style={colorSwatch("#000000", "1px solid #333")} />
                Neutro (0)
              </span>
              <span>
                <span style={colorSwatch("#8b00ff")} />
                Inhibitorio (-1)
              </span>
              <span>
                <span style={colorSwatch("#111111", "1px solid #333")} />
                Sin conexión
              </span>
            </>
          ) : (
            <>
              {selectedExp !== "kohonen" && selectedExp !== "kohonen_balanced" && selectedExp !== "kohonen_lab" && (
                <>
                  <span>
                    <span style={colorSwatch("#4cc9f0")} />
                    ENTRADA
                  </span>
                  <span>
                    <span style={colorSwatch("#f72585")} />
                    SALIDA
                  </span>
                </>
              )}
              <span>
                <span style={colorSwatch("#ffffff")} />
                Activa
              </span>
              <span>
                <span style={colorSwatch("#0a0a0a", "1px solid #333")} />
                Inactiva
              </span>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
