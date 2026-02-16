/** NeuroFlow — Main Application Layout. */

import { useState, useEffect, useCallback } from "react";
import { PixelCanvas } from "./components/PixelCanvas";
import { Sidebar } from "./components/Sidebar";
import { Controls } from "./components/Controls";
import { useExperiment } from "./hooks/useExperiment";
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
];

function App() {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>(DEFAULT_EXPERIMENTS);
  const [selectedExp, setSelectedExp] = useState("von_neumann");
  const [config, setConfig] = useState<ExperimentConfig>({
    width: 50,
    height: 50,
    rule: 111,
  });

  const {
    grid,
    state,
    stats,
    generation,
    inspectMode,
    connectionMap,
    inspectedCell,
    start,
    click,
    step,
    play,
    pause,
    reset,
    inspect,
    toggleInspectMode,
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

  const handleCellClick = useCallback(
    (x: number, y: number) => {
      if (inspectMode) {
        inspect(x, y);
      } else {
        click(x, y);
      }
    },
    [inspectMode, inspect, click]
  );

  const handlePlay = useCallback(() => play(10), [play]);

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
        connected={connected}
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
          }}
        >
          {hasGrid ? (
            <PixelCanvas
              grid={grid}
              width={config.width}
              height={config.height}
              {...(selectedExp !== "kohonen"
                ? { inputRow: config.height - 1, outputRow: 0 }
                : {})}
              connectionMap={connectionMap}
              inspectedCell={inspectedCell}
              onCellClick={handleCellClick}
            />
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
          generation={generation}
          inspectMode={inspectMode}
          onPlay={handlePlay}
          onPause={pause}
          onStep={step}
          onReset={reset}
          onToggleInspect={toggleInspectMode}
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
              {selectedExp !== "kohonen" && (
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
