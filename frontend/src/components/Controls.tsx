/** Controls — Play/Pause/Step/Reset + stats display + steps per tick. */

import type { ExperimentState, ExperimentStats, PerfMetrics } from "../types";

interface ControlsProps {
  state: ExperimentState;
  stats: ExperimentStats | null;
  perf: PerfMetrics | null;
  generation: number;
  stepsPerTick: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
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
  stepsPerTick,
  onPlay,
  onPause,
  onStep,
  onReset,
  onStepsPerTickChange,
}: ControlsProps) {
  const hasExperiment = state !== "disconnected";
  const isInitializing = state === "initializing";
  const canInteract = !isInitializing && (state === "ready" || state === "paused");
  const isRunning = state === "running";
  const canStep = canInteract || isRunning;
  const canReset = hasExperiment && !isInitializing;

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
          style={btnStyle(canStep, "#7b61ff")}
          onClick={onStep}
          disabled={!canStep}
        >
          Step
        </button>
        <button
          style={btnStyle(canReset, "#ff9e00")}
          onClick={onReset}
          disabled={!canReset}
        >
          Reset
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
          Active:{" "}
          <strong style={{ color: "#4cc9f0" }}>
            {stats?.active_cells ?? 0}
          </strong>
        </span>
        {stats?.daemon_count != null && (
          <>
            <span
              title="Clusters of >=3 contiguous active neurons"
            >
              Daemons:{" "}
              <strong style={{ color: "#f0a500" }}>
                {stats.daemon_count}
              </strong>
              <span style={{ color: "#555", fontSize: "0.7rem" }}>
                {" "}(~{stats.avg_daemon_size ?? 0})
              </span>
            </span>
            <span
              title="Active neurons not part of any daemon (isolated/small groups)"
              style={{
                color: (stats.noise_cells ?? 0) > (stats.active_cells * 0.3)
                  ? "#f72585"
                  : "#888",
              }}
            >
              Noise:{" "}
              <strong>
                {stats.noise_cells ?? 0}
              </strong>
            </span>
            <span title="Daemon count stability (20-frame sliding window)">
              Stab:{" "}
              <strong
                style={{
                  color:
                    (stats.stability ?? 0) > 0.8
                      ? "#4ade80"
                      : (stats.stability ?? 0) > 0.5
                        ? "#f0a500"
                        : "#f72585",
                }}
              >
                {(stats.stability ?? 0).toFixed(2)}
              </strong>
            </span>
            <span title="Activation contrast inside vs outside daemons">
              Excl:{" "}
              <strong style={{ color: "#7b61ff" }}>
                {(stats.exclusion ?? 0).toFixed(2)}
              </strong>
            </span>
          </>
        )}
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
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            background:
              state === "running"
                ? "#1a3a1a"
                : state === "initializing"
                ? "#1a1a3a"
                : state === "complete"
                ? "#3a1a1a"
                : "#1a1a2e",
            color:
              state === "running"
                ? "#4ade80"
                : state === "initializing"
                ? "#4cc9f0"
                : state === "complete"
                ? "#f72585"
                : "#666",
          }}
        >
          {isInitializing && <span className="neuro-spinner-sm" />}
          {state}
        </span>
      </div>
    </div>
  );
}
