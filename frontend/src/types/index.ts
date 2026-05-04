/** Shared types for NeuroFlow frontend. */

export interface MaskStats {
  total_dendrites: number;
  exc_dendrites: number;
  inh_dendrites: number;
  excitatory_synapses: number;
  inhibitory_synapses: number;
  ratio_exc_inh: number;
  excitation_radius: number;
  inhibition_radius: number;
}

export interface DendriteInfo {
  centroid: [number, number];   // [col, row] in grid coords (float)
  avg_effective: number;        // average effective weight (signed)
  cells: [number, number][];    // [col, row] grid coords of each synapse
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
  dendrites?: DendriteInfo[];
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

export interface ProcessModeInfo {
  id: string;
  name: string;
  description: string;
}

/* ── Canonical config schema (regions + connections) ── */

export interface RegionSource {
  type: "ascii";
  text?: string;
  frames_per_char?: number;
  font?: string;
  font_size?: number;
  density?: number;
}

export interface RegionWiring {
  mask?: string;
  dendrite_exc_weight?: number;
  dendrite_inh_weight?: number;
  process_mode?: string;
  tension_function?: Record<string, number>;
  learning_rate?: number;
}

export interface RegionNoise {
  background?: number;
  shift?: boolean;
  inter_char?: boolean;
}

export interface RegionSpiking {
  up_ticks?: number;
  down_ticks?: number;
}

export interface RegionDaemon {
  threshold?: number;
  min_size?: number;
}

export interface Region {
  id: string;
  grid: { width: number; height: number };
  source?: RegionSource;
  wiring?: RegionWiring;
  noise?: RegionNoise;
  spiking?: RegionSpiking;
  daemon?: RegionDaemon;
}

export interface Connection {
  from: string;
  to: string;
  type: "full" | "portion";
  weight: number;
  portion?: [number, number];
  learning_rate?: number;
}

export interface CanonicalExperimentConfig {
  name?: string;
  description?: string;
  regions: Region[];
  connections?: Connection[];
}

/* ── Legacy config format (used by frontend UI components) ── */

export interface ExperimentConfig {
  description?: string;
  grid: { width: number; height: number };
  wiring: {
    mask?: string;
    dendrite_exc_weight?: number;
    dendrite_inh_weight?: number;
    process_mode?: string;
    tension_function?: Record<string, number>;
  };
  input?: {
    source?: string;
    text?: string;
    resolution?: number;
    frames_per_char?: number;
    dendrite_input_weight?: number;
    font?: string;
    font_size?: number;
    density?: number;
    portion?: [number, number];
  };
  noise?: {
    background?: number;
    shift?: boolean;
    inter_char?: boolean;
  };
  learning?: {
    rate?: number;
    lr_exc?: number;
    lr_inh?: number;
    lr_input?: number;
  };
  spiking?: {
    up_ticks?: number;
    down_ticks?: number;
  };
}

export interface ConfigTemplate {
  id: string;
  name: string;
  description: string;
  config: ExperimentConfig;
}

export interface Metadata {
  masks: MaskPresetInfo[];
  fonts: FontInfo[];
  process_modes: ProcessModeInfo[];
  input_sources: InputSourceInfo[];
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
  activation: number;
  tension: number;
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
