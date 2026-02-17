/** Controls — Play/Pause/Step/Reset/Inspect + stats display + steps per tick. */

import type { ExperimentState, ExperimentStats, PerfMetrics } from "../types";

interface ControlsProps {
  state: ExperimentState;
  stats: ExperimentStats | null;
  perf: PerfMetrics | null;
  generation: number;
  inspectMode: boolean;
  stepsPerTick: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onToggleInspect: () => void;
  onStepsPerTickChange: (value: number) => void;
}

const btnStyle = (active: boolean, color: string): React.CSSProperties => ({
  padding: "8px 18px",
  background: active ? color : "#1a1a2e",
  color: active ? "#0a0a0a" : "#666",
  border: `1px solid ${active ? color : "#2a2a3e"}`,
  borderRadius: "6px",
  fontSize: "0.85rem",
  fontWeight: 600,
  cursor: active ? "pointer" : "not-allowed",
  transition: "all 0.15s",
  minWidth: "80px",
});

const STEP_OPTIONS = [1, 5, 10, 50, 100, 500, 1000];

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(Math.round(n));
}

export function Controls({
  state,
  stats,
  perf,
  generation,
  inspectMode,
  stepsPerTick,
  onPlay,
  onPause,
  onStep,
  onReset,
  onToggleInspect,
  onStepsPerTickChange,
}: ControlsProps) {
  const hasExperiment = state !== "disconnected";
  const canInteract = state === "ready" || state === "paused";
  const isRunning = state === "running";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 0",
        gap: "12px",
        flexWrap: "wrap",
      }}
    >
      <div style={{ display: "flex", gap: "8px" }}>
        {isRunning ? (
          <button style={btnStyle(true, "#f72585")} onClick={onPause}>
            Pause
          </button>
        ) : (
          <button
            style={btnStyle(canInteract, "#4cc9f0")}
            onClick={onPlay}
            disabled={!canInteract}
          >
            Play
          </button>
        )}
        <button
          style={btnStyle(canInteract, "#7b61ff")}
          onClick={onStep}
          disabled={!canInteract}
        >
          Step
        </button>
        <button
          style={btnStyle(hasExperiment && !isRunning, "#ff9e00")}
          onClick={onReset}
          disabled={!hasExperiment || isRunning}
        >
          Reset
        </button>
        <button
          style={btnStyle(inspectMode, "#ffff00")}
          onClick={onToggleInspect}
          disabled={!canInteract}
        >
          {inspectMode ? "\u2715 Inspeccionar" : "Inspeccionar"}
        </button>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            marginLeft: "8px",
          }}
        >
          <span
            style={{
              fontSize: "0.7rem",
              color: "#888",
              whiteSpace: "nowrap",
            }}
          >
            Steps/tick:
          </span>
          <select
            value={stepsPerTick}
            onChange={(e) => onStepsPerTickChange(Number(e.target.value))}
            style={{
              padding: "4px 8px",
              background: "#1a1a2e",
              border: "1px solid #2a2a3e",
              borderRadius: "4px",
              color: "#e0e0ff",
              fontSize: "0.8rem",
              cursor: "pointer",
            }}
          >
            {STEP_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: "16px",
          fontSize: "0.8rem",
          color: "#888",
          fontFamily: "monospace",
          alignItems: "center",
        }}
      >
        <span>
          Steps:{" "}
          <strong style={{ color: "#e0e0ff" }}>
            {formatNumber(generation)}
            {stats?.total_steps != null && (
              <span style={{ color: "#666" }}>
                {" / "}{stats.total_steps}
              </span>
            )}
          </strong>
        </span>
        <span>
          Activas:{" "}
          <strong style={{ color: "#4cc9f0" }}>
            {stats?.active_cells ?? 0}
          </strong>
        </span>
        {perf && (
          <span
            style={{
              padding: "2px 8px",
              borderRadius: "4px",
              background: "#0d1f0d",
              border: "1px solid #1a3a1a",
            }}
          >
            <span style={{ color: "#666" }}>{perf.elapsed_ms}ms</span>
            {" / "}
            <strong style={{ color: "#4ade80" }}>
              {formatNumber(perf.steps_per_second)} steps/s
            </strong>
          </span>
        )}
        <span
          style={{
            padding: "2px 8px",
            borderRadius: "4px",
            background:
              state === "running"
                ? "#1a3a1a"
                : state === "complete"
                ? "#3a1a1a"
                : "#1a1a2e",
            color:
              state === "running"
                ? "#4ade80"
                : state === "complete"
                ? "#f72585"
                : "#666",
          }}
        >
          {state}
        </span>
      </div>
    </div>
  );
}
