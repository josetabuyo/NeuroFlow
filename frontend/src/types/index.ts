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

export interface CentroidTwist {
  center: [number, number];
  direction: "cw" | "ccw";
  /** Scales 0 at center -> max_magnitude at the farthest grid corner. */
  max_magnitude?: number;
  /** Constant magnitude everywhere (instead of scaling with radius). Takes precedence over max_magnitude. */
  fix_magnitude?: number;
}

export interface CentroidConfig {
  /** Uniform static shift [dx, dy]. Stacks with `twist` if both are set. */
  shift?: [number, number] | "twist";
  /** Rotational shift — set this (shift is optional) to enable twist mode. Adds on top of `shift`. */
  twist?: CentroidTwist;
  /** Per-neuron random jitter of the shift. Default false. Ignored when `twist` is set. */
  random?: boolean;
}

export interface RegionSource {
  type: "ascii";
  text?: string;
  frames_per_char?: number;
  font?: string;
  font_size?: number;
  density?: number;
}

/** One dendrite group within a deamon wiring block. No fixed "excitatory"/
 * "inhibitory" keys — `id` is just a label, and `weight`'s sign is the
 * group's polarity (defaults to +1 for a single dendrite, -1 once `sectors`
 * make it partitioned, unless `weight` overrides it). A group's starting
 * ring is always computed: it accumulates right after the previous group's
 * outer ring plus a `gap` of empty rings, taken from this group's own `gap`
 * if set, otherwise the wiring's `gap` (see `DeamonWiring.gap`). */
export interface DeamonGroup {
  id: string;
  weight?: number;
  gap?: number;
  weights?: number[];
  noise?: number;
  density?: number;
  sectors?: number;
  learning?: { rate?: number; exclude_range?: [number, number] };
}

export interface DeamonWiring {
  mask?: string;
  shape?: string;
  gap?: number;
  centroid?: CentroidConfig;
  fixed?: boolean;
  groups?: DeamonGroup[];
  learning?: { rate?: number; exclude_range?: [number, number] };
}

export interface RegionWiring {
  deamon?: DeamonWiring;
  process_mode?: string;
  tension?: { function: Record<string, number> };
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
  noise?: RegionNoise;
  spiking?: RegionSpiking;
  daemon?: RegionDaemon;
  process_mode?: string;
  tension?: { function: Record<string, number> };
  /** Per-neuron weight totals by polarity. Both keys optional; each value is
   * always a magnitude (>= 0) — the key itself is the sole source of sign.
   * When set, every neuron's dendrites of that polarity (daemon and/or
   * nerve) are rescaled proportionally so they sum to these targets. Omitted
   * entirely → no rebalancing, dendrites keep their raw configured weights. */
  delta_weight?: { positive?: number; negative?: number };
}

export interface Connection {
  // inter-region full
  from?: string;
  to?: string;
  full?: {
    weight?: number;
    density?: number;
    learning?: { rate?: number; exclude_range?: [number, number] };
    learning_rate?: number;
  };
  // legacy portion (type: "portion" still used internally)
  type?: string;
  portion?: [number, number];
  weight?: number;
  // intra-region (on-connection)
  on?: string;
  deamon?: DeamonWiring;
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
    tension?: { function: Record<string, number> };
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

export interface Experiment {
  id: number;
  name: string;
  position: number;
  created_at: string;
  updated_at: string;
}

export interface ExperimentDetail extends Experiment {
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

export interface OrchestratorEvent {
  index: number;
  kind: "gradient" | "at" | "inject";
  tick_from?: number;
  tick_to?: number;
  tick?: number;
  tick_end?: number;
  expr?: string;
  value?: number;
  progress?: number;
  region?: string;
  template?: string;
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
  regions?: Record<string, number[][]>;
  tension_regions?: Record<string, number[][]>;
  tissue_id?: string;
  input_id?: string;
  labels?: Record<string, number[][]>;
  neuron_label_map?: Record<string, string[][]>;
  orchestrator?: OrchestratorEvent[];
}

export interface StatusMessage {
  type: "status";
  state: "running" | "paused" | "ready" | "complete" | "initializing";
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface RegionOverlay {
  grid: (number | null)[][];
  density: number;
}

export interface ConnectionsMessage {
  type: "connections";
  x: number;
  y: number;
  region_id?: string;
  activation: number;
  tension: number;
  total_dendritas: number;
  total_sinapsis: number;
  weight_grid: (number | null)[][];
  input_weight_grid?: number[][] | null;
  input_weight_width?: number;
  input_weight_height?: number;
  nerve_circles?: Array<{ cx: number; cy: number; radius: number }> | null;
  region_overlays?: Record<string, RegionOverlay>;
  // Dynamic per-source-region grids: {id}_weight_grid, {id}_weight_width, {id}_weight_height
  [key: string]: unknown;
}

export interface ConfigNormalizedMessage {
  type: "config_normalized";
  config: CanonicalExperimentConfig;
}

export type ServerMessage = FrameMessage | StatusMessage | ErrorMessage | ConnectionsMessage | ConfigNormalizedMessage;

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
