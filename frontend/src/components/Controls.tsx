/** Controls — Play/Pause/Step/Reset + stats display. */

import type { ExperimentState, ExperimentStats } from "../types";

interface ControlsProps {
  state: ExperimentState;
  stats: ExperimentStats | null;
  generation: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
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

export function Controls({
  state,
  stats,
  generation,
  onPlay,
  onPause,
  onStep,
  onReset,
}: ControlsProps) {
  const hasExperiment = state !== "disconnected";
  const canInteract = state === "ready" || state === "paused";
  const isRunning = state === "running";
  const isComplete = state === "complete";

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
      </div>

      <div
        style={{
          display: "flex",
          gap: "20px",
          fontSize: "0.8rem",
          color: "#888",
          fontFamily: "monospace",
        }}
      >
        <span>
          Gen:{" "}
          <strong style={{ color: "#e0e0ff" }}>
            {generation}/{stats?.total_rows ?? "—"}
          </strong>
        </span>
        <span>
          Activas:{" "}
          <strong style={{ color: "#4cc9f0" }}>
            {stats?.active_cells ?? 0}
          </strong>
        </span>
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
