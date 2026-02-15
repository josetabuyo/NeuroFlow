/** Shared types for NeuroFlow frontend. */

export interface ExperimentInfo {
  id: string;
  name: string;
  description: string;
  rules?: number[];
  default_config: ExperimentConfig;
}

export interface ExperimentConfig {
  width: number;
  height: number;
  rule?: number;
}

export interface FrameMessage {
  type: "frame";
  generation: number;
  grid: number[][];
  stats: ExperimentStats;
}

export interface StatusMessage {
  type: "status";
  state: "running" | "paused" | "ready" | "complete";
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = FrameMessage | StatusMessage | ErrorMessage;

export interface ExperimentStats {
  active_cells: number;
  generation: number;
  total_rows: number;
  processed_rows?: number;
}

export type ExperimentState = "disconnected" | "ready" | "running" | "paused" | "complete";
