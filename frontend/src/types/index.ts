/** Shared types for NeuroFlow frontend. */

export interface MaskPresetInfo {
  id: string;
  name: string;
  description: string;
  center: string;
  corona: string;
  dendrites_inh: number;
  preview_grid: (number | null)[][];
}

export interface BalanceModeInfo {
  id: string;
  name: string;
}

export interface ExperimentConfig {
  width: number;
  height: number;
  rule?: number;
  balance?: number;
  balance_mode?: string;
  mask?: string;
}

export interface ExperimentInfo {
  id: string;
  name: string;
  description: string;
  rules?: number[];
  masks?: MaskPresetInfo[];
  balance_modes?: BalanceModeInfo[];
  default_config: ExperimentConfig;
}

export interface PerfMetrics {
  steps: number;
  elapsed_ms: number;
  steps_per_second: number;
}

export interface FrameMessage {
  type: "frame";
  generation: number;
  grid: number[][];
  stats: ExperimentStats;
  perf?: PerfMetrics;
}

export interface StatusMessage {
  type: "status";
  state: "running" | "paused" | "ready" | "complete";
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface ConnectionsMessage {
  type: "connections";
  x: number;
  y: number;
  total_dendritas: number;
  total_sinapsis: number;
  weight_grid: (number | null)[][];
}

export type ServerMessage = FrameMessage | StatusMessage | ErrorMessage | ConnectionsMessage;

export interface ExperimentStats {
  active_cells: number;
  steps: number;
  total_steps?: number;
}

export type ExperimentState = "disconnected" | "ready" | "running" | "paused" | "complete";

export interface BrushShape {
  id: string;
  name: string;
  offsets: [number, number][];
}
