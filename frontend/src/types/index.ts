/** Shared types for NeuroFlow frontend. */

export interface MaskStats {
  excitatory_synapses: number;
  inhibitory_synapses: number;
  ratio_exc_inh: number;
  excitation_radius: number;
  inhibition_radius: number;
}

export interface MaskPresetInfo {
  id: string;
  name: string;
  description: string;
  center: string;
  corona: string;
  dendrites_inh: number;
  preview_grid?: (number | null)[][];
  mask_stats?: MaskStats;
}

export interface BalanceModeInfo {
  id: string;
  name: string;
}

export interface InputSourceInfo {
  id: string;
  name: string;
}

export interface FontInfo {
  id: string;
  name: string;
  sizes: number[];
  default_size: number;
  description: string;
}

export interface ExperimentConfig {
  description?: string;
  width: number;
  height: number;
  rule?: number;
  balance?: number;
  balance_mode?: string;
  mask?: string;
  input_text?: string;
  input_resolution?: number;
  frames_per_char?: number;
  input_dendrite_weight?: number;
  deamon_exc_weight?: number;
  deamon_inh_weight?: number;
  background_white_noise?: number;
  shift_noise?: boolean;
  noise_inter_char?: boolean;
  input_source?: string;
  font?: string;
  font_size?: number;
  learning?: boolean;
  learning_rate?: number;
  spike_adaptation?: boolean;
  max_active_steps?: number;
  refractory_steps?: number;
  process_mode?: string;
  tension_function?: Record<string, number>;
}

export interface ConfigPreset {
  id: string;
  name: string;
  description: string;
  config: ExperimentConfig;
}

export interface ExperimentInfo {
  id: string;
  name: string;
  description: string;
  rules?: number[];
  masks?: MaskPresetInfo[];
  balance_modes?: BalanceModeInfo[];
  input_sources?: InputSourceInfo[];
  fonts?: FontInfo[];
  default_config: ExperimentConfig;
  config_presets?: ConfigPreset[];
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
  tension_grid?: number[][];
  input_frame?: number[][];
  inspect?: ConnectionsMessage;
}

export interface StatusMessage {
  type: "status";
  state: "running" | "paused" | "ready" | "complete" | "initializing";
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
  input_weight_grid?: number[][] | null;
  input_weight_width?: number;
  input_weight_height?: number;
}

export type ServerMessage = FrameMessage | StatusMessage | ErrorMessage | ConnectionsMessage;

export interface ExperimentStats {
  active_cells: number;
  steps: number;
  total_steps?: number;
  daemon_count?: number;
  avg_daemon_size?: number;
  noise_cells?: number;
  stability?: number;
  exclusion?: number;
  current_char?: string;
  char_index?: number;
  frame_in_char?: number;
  frames_per_char?: number;
  input_resolution?: number;
}

export type ExperimentState = "disconnected" | "initializing" | "ready" | "running" | "paused" | "complete";