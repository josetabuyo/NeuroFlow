/** WebSocket connection + state management for experiments. */

import { useState, useRef, useCallback, useEffect } from "react";
import type {
  CanonicalExperimentConfig,
  ExperimentConfig,
  ExperimentState,
  ExperimentStats,
  OrchestratorEvent,
  PerfMetrics,
  RegionOverlay,
  ServerMessage,
} from "../types";
import { nextBrushSize, prevBrushSize } from "../brushes";

/** Max samples kept for the live inspect chart — one point per tick, oldest drop off.
 * Must be >= InspectChart's display width (280px) so the chart never runs out of data. */
const INSPECT_HISTORY_MAX = 300;

export interface InspectSample {
  tick: number;
  activation: number;
  tension: number;
}

/** Appends a tick sample to the inspect history ring buffer, dropping the oldest
 * once over `max`. A sample landing on the same tick as the last one (e.g. a paint
 * action while paused re-triggers a frame at the same generation) replaces it
 * in place rather than appending, so the chart reflects the latest value instead
 * of going stale. */
export function appendInspectSample(
  prev: InspectSample[],
  sample: InspectSample,
  max: number = INSPECT_HISTORY_MAX,
): InspectSample[] {
  if (prev.length > 0 && prev[prev.length - 1].tick === sample.tick) {
    return [...prev.slice(0, -1), sample];
  }
  const next = [...prev, sample];
  return next.length > max ? next.slice(-max) : next;
}

function getWsUrl(): string {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/experiment`;
}

interface UseExperimentReturn {
  grid: number[][];
  tensionGrid: number[][] | null;
  tensionMode: boolean;
  inputFrame: number[][] | null;
  inputWeightGrid: number[][] | null;
  inputWeightDims: { width: number; height: number } | null;
  sourceWeightGrids: Record<string, { grid: (number | null)[][], width: number, height: number, density: number }>;
  regionOverlays: Record<string, RegionOverlay>;
  nerveCircles: Array<{ cx: number; cy: number; radius: number }> | null;
  regions: Record<string, number[][]>;
  tensionRegions: Record<string, number[][]>;
  regionId: string;
  inputId: string | null;
  labels: Record<string, number[][]>;
  neuronLabelMap: Record<string, string[][]>;
  orchestratorState: OrchestratorEvent[];
  state: ExperimentState;
  stats: ExperimentStats | null;
  perf: PerfMetrics | null;
  generation: number;
  experimentActive: boolean;
  inspectMode: boolean;
  connectionMap: (number | null)[][] | null;
  inspectedCell: { x: number; y: number } | null;
  inspectedRegionId: string | null;
  inspectInfo: {
    activation: number;
    tension: number;
    total_dendritas: number;
    total_sinapsis: number;
  } | null;
  inspectHistory: InspectSample[];
  brushSize: number;
  brushMode: "activate" | "deactivate";
  normalizedConfig: CanonicalExperimentConfig | null;
  start: (config: ExperimentConfig) => void;
  reconnect: (config: ExperimentConfig) => void;
  updateConfig: (config: Partial<ExperimentConfig>) => void;
  click: (x: number, y: number) => void;
  paint: (
    cells: { x: number; y: number }[],
    value: number,
    regionId?: string,
    origin?: { x: number; y: number },
    radius?: number
  ) => void;
  step: (count?: number) => void;
  play: (fps?: number, stepsPerTick?: number) => void;
  pause: () => void;
  reset: () => void;
  inspect: (x: number, y: number, regionId?: string) => void;
  toggleInspectMode: () => void;
  toggleTensionMode: () => void;
  increaseBrushSize: () => void;
  decreaseBrushSize: () => void;
  toggleBrushMode: () => void;
}

function _parseSourceWeightGrids(
  msg: Record<string, unknown>
): Record<string, { grid: (number | null)[][], width: number, height: number, density: number }> {
  const result: Record<string, { grid: (number | null)[][], width: number, height: number, density: number }> = {};
  for (const key of Object.keys(msg)) {
    if (!key.endsWith("_weight_grid") || key === "input_weight_grid" || key === "weight_grid") continue;
    const id = key.slice(0, -"_weight_grid".length);
    const grid = msg[key] as (number | null)[][];
    const width = (msg[`${id}_weight_width`] as number | undefined) ?? grid[0]?.length ?? 1;
    const height = (msg[`${id}_weight_height`] as number | undefined) ?? grid.length ?? 1;
    let density = 0;
    for (const row of grid) {
      for (const v of row) {
        if (v !== null && Math.abs(v) > density) density = Math.abs(v);
      }
    }
    result[id] = { grid, width, height, density };
  }
  return result;
}

function _parseRegionOverlays(msg: Record<string, unknown>): Record<string, RegionOverlay> {
  return (msg["region_overlays"] as Record<string, RegionOverlay> | undefined) ?? {};
}

export function useExperiment(): UseExperimentReturn {
  const [grid, setGrid] = useState<number[][]>([]);
  const [state, setState] = useState<ExperimentState>("disconnected");
  const [stats, setStats] = useState<ExperimentStats | null>(null);
  const [generation, setGeneration] = useState(0);
  const [inspectMode, setInspectMode] = useState(false);
  const [connectionMap, setConnectionMap] = useState<(number | null)[][] | null>(null);
  const [inspectedCell, setInspectedCell] = useState<{ x: number; y: number } | null>(null);
  const [inspectedRegionId, setInspectedRegionId] = useState<string | null>(null);
  const [inspectInfo, setInspectInfo] = useState<{
    activation: number;
    tension: number;
    total_dendritas: number;
    total_sinapsis: number;
  } | null>(null);
  const [inspectHistory, setInspectHistory] = useState<InspectSample[]>([]);
  // Tracks {x, y, regionId} of the currently-inspected neuron so the per-frame
  // handler can look up its activation/tension without a stale closure.
  const inspectTargetRef = useRef<{ x: number; y: number; regionId: string } | null>(null);
  const generationRef = useRef(0);
  const [perf, setPerf] = useState<PerfMetrics | null>(null);
  const [tensionGrid, setTensionGrid] = useState<number[][] | null>(null);
  const [tensionMode, setTensionMode] = useState(false);
  const [inputFrame, setInputFrame] = useState<number[][] | null>(null);
  const [inputWeightGrid, setInputWeightGrid] = useState<number[][] | null>(null);
  const [inputWeightDims, setInputWeightDims] = useState<{ width: number; height: number } | null>(null);
  const [sourceWeightGrids, setSourceWeightGrids] = useState<Record<string, { grid: (number | null)[][], width: number, height: number, density: number }>>({});
  const [regionOverlays, setRegionOverlays] = useState<Record<string, RegionOverlay>>({});
  const [nerveCircles, setNerveCircles] = useState<Array<{ cx: number; cy: number; radius: number }> | null>(null);
  const [regions, setRegions] = useState<Record<string, number[][]>>({});
  const [tensionRegions, setTensionRegions] = useState<Record<string, number[][]>>({});
  const [regionId, setRegionId] = useState<string>("");
  const [inputId, setInputId] = useState<string | null>(null);
  const [labels, setLabels] = useState<Record<string, number[][]>>({});
  const [neuronLabelMap, setNeuronLabelMap] = useState<Record<string, string[][]>>({});
  const [orchestratorState, setOrchestratorState] = useState<OrchestratorEvent[]>([]);
  const [experimentActive, setExperimentActive] = useState(false);
  const [brushSize, setBrushSize] = useState(13);
  const [brushMode, setBrushMode] = useState<"activate" | "deactivate">("activate");
  const [normalizedConfig, setNormalizedConfig] = useState<CanonicalExperimentConfig | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback((data: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }, []);

  // Single effect manages the WebSocket lifecycle
  useEffect(() => {
    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setState("ready");
    };

    ws.onmessage = (event: MessageEvent) => {
      const msg: ServerMessage = JSON.parse(event.data);
      switch (msg.type) {
        case "frame":
          setGrid(msg.grid);
          setGeneration(msg.generation);
          generationRef.current = msg.generation;
          setStats(msg.stats);
          setPerf(msg.perf ?? null);
          setTensionGrid(msg.tension_grid ?? null);
          setInputFrame(msg.input_frame ?? null);
          if (msg.regions) setRegions(msg.regions);
          if (msg.tension_regions) setTensionRegions(msg.tension_regions);
          if (msg.tissue_id) setRegionId(msg.tissue_id);
          if (msg.input_id !== undefined) setInputId(msg.input_id ?? null);
          setLabels(msg.labels ?? {});
          if (msg.neuron_label_map) setNeuronLabelMap(msg.neuron_label_map);
          setOrchestratorState(msg.orchestrator ?? []);

          // Sample activation/tension for the inspected neuron every tick — regions
          // and tension_regions are broadcast unthrottled, unlike the heavier `inspect`
          // weight-grid payload (throttled to every 3rd frame while playing).
          {
            const target = inspectTargetRef.current;
            if (target) {
              const act = msg.regions?.[target.regionId]?.[target.y]?.[target.x];
              const ten = msg.tension_regions?.[target.regionId]?.[target.y]?.[target.x];
              if (act !== undefined) {
                setInspectHistory(prev =>
                  appendInspectSample(prev, { tick: msg.generation, activation: act, tension: ten ?? 0 })
                );
              }
            }
          }

          if (msg.inspect) {
            setConnectionMap(msg.inspect.weight_grid);
            setInspectedCell({ x: msg.inspect.x, y: msg.inspect.y });
            setInspectedRegionId(msg.inspect.region_id ?? null);
            // inspectTargetRef is set solely by "connections" (the initial inspect click) —
            // this throttled payload must never re-arm it, or an in-flight frame built before
            // an "uninspect" lands could resurrect sampling for a target the user already left.
            setInspectInfo({
              activation: msg.inspect.activation,
              tension: msg.inspect.tension,
              total_dendritas: msg.inspect.total_dendritas,
              total_sinapsis: msg.inspect.total_sinapsis,
            });
            setInputWeightGrid(msg.inspect.input_weight_grid ?? null);
            if (msg.inspect.input_weight_width && msg.inspect.input_weight_height) {
              setInputWeightDims({ width: msg.inspect.input_weight_width as number, height: msg.inspect.input_weight_height as number });
            }
            setSourceWeightGrids(_parseSourceWeightGrids(msg.inspect));
            setRegionOverlays(_parseRegionOverlays(msg.inspect));
            setNerveCircles((msg.inspect.nerve_circles as Array<{ cx: number; cy: number; radius: number }> | null) ?? null);
          }
          break;
        case "connections": {
          setConnectionMap(msg.weight_grid);
          setInspectedCell({ x: msg.x, y: msg.y });
          setInspectedRegionId(msg.region_id ?? null);
          // A new "connections" message always means a freshly inspected neuron
          // (click or drag-to-inspect) — start the live chart over from this tick.
          const regionId = msg.region_id;
          if (regionId) {
            inspectTargetRef.current = { x: msg.x, y: msg.y, regionId };
            setInspectHistory([{ tick: generationRef.current, activation: msg.activation, tension: msg.tension }]);
          } else {
            inspectTargetRef.current = null;
            setInspectHistory([]);
          }
          setInspectInfo({
            activation: msg.activation,
            tension: msg.tension,
            total_dendritas: msg.total_dendritas,
            total_sinapsis: msg.total_sinapsis,
          });
          setInputWeightGrid(msg.input_weight_grid ?? null);
          if (msg.input_weight_width && msg.input_weight_height) {
            setInputWeightDims({ width: msg.input_weight_width as number, height: msg.input_weight_height as number });
          } else {
            setInputWeightDims(null);
          }
          setSourceWeightGrids(_parseSourceWeightGrids(msg));
          setRegionOverlays(_parseRegionOverlays(msg));
          setNerveCircles((msg.nerve_circles as Array<{ cx: number; cy: number; radius: number }> | null) ?? null);
          break;
        }
        case "status":
          setState(msg.state);
          break;
        case "config_normalized":
          setNormalizedConfig(msg.config);
          break;
        case "error":
          console.error("Server error:", msg.message);
          break;
      }
    };

    ws.onclose = () => {
      // Only update state if this is still the current WebSocket
      if (wsRef.current === ws) {
        setState("disconnected");
        wsRef.current = null;
      }
    };

    ws.onerror = () => {
      // Error is already logged by browser; onclose will fire next
    };

    return () => {
      // Cleanup: close WebSocket, clear ref so stale onclose doesn't interfere
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
      ws.close();
    };
  }, []);

  const start = useCallback(
    (config: ExperimentConfig) => {
      setExperimentActive(true);
      setState("initializing");
      setConnectionMap(null);
      setInspectedCell(null);
      setInspectedRegionId(null);
      setInspectInfo(null);
      setInspectHistory([]);
      inspectTargetRef.current = null;
      send({ action: "start", config });
    },
    [send]
  );

  const reconnect = useCallback(
    (config: ExperimentConfig) => {
      setState("initializing");
      setConnectionMap(null);
      setInspectedCell(null);
      setInspectedRegionId(null);
      setInspectInfo(null);
      setInspectHistory([]);
      inspectTargetRef.current = null;
      send({ action: "reconnect", config });
    },
    [send]
  );

  const updateConfig = useCallback(
    (config: Partial<ExperimentConfig>) => {
      send({ action: "update_config", config });
    },
    [send]
  );

  const click = useCallback(
    (x: number, y: number) => {
      send({ action: "click", x, y });
    },
    [send]
  );

  const step = useCallback(
    (count = 1) => send({ action: "step", count }),
    [send]
  );
  const play = useCallback(
    (fps = 10, stepsPerTick = 1) =>
      send({ action: "play", fps, steps_per_tick: stepsPerTick }),
    [send]
  );
  const pause = useCallback(() => send({ action: "pause" }), [send]);
  const reset = useCallback(() => {
    setConnectionMap(null);
    setInspectedCell(null);
    setInspectedRegionId(null);
    setInspectInfo(null);
    setInspectHistory([]);
    inspectTargetRef.current = null;
    setRegionOverlays({});
    setNerveCircles(null);
    setState("initializing");
    send({ action: "reset" });
  }, [send]);

  const paint = useCallback(
    (
      cells: { x: number; y: number }[],
      value: number,
      regionId?: string,
      origin?: { x: number; y: number },
      radius?: number
    ) => {
      send({
        action: "paint",
        cells,
        value,
        ...(regionId ? { region_id: regionId } : {}),
        ...(origin ? { origin, radius: radius ?? 0 } : {}),
      });
    },
    [send]
  );

  const inspect = useCallback(
    (x: number, y: number, regionId?: string) => {
      send({ action: "inspect", x, y, ...(regionId ? { region_id: regionId } : {}) });
    },
    [send]
  );

  const increaseBrushSize = useCallback(() => {
    setBrushSize((s) => nextBrushSize(s));
  }, []);

  const decreaseBrushSize = useCallback(() => {
    setBrushSize((s) => prevBrushSize(s));
  }, []);

  const toggleBrushMode = useCallback(() => {
    setBrushMode((prev) => (prev === "activate" ? "deactivate" : "activate"));
  }, []);

  const toggleInspectMode = useCallback(() => {
    setInspectMode((prev) => {
      if (prev) {
        setConnectionMap(null);
        setInspectedCell(null);
        setInspectedRegionId(null);
        setInspectInfo(null);
        setInspectHistory([]);
        inspectTargetRef.current = null;
        setInputWeightGrid(null);
        setInputWeightDims(null);
        setSourceWeightGrids({});
        setRegionOverlays({});
        setNerveCircles(null);
        send({ action: "uninspect" });
      }
      return !prev;
    });
  }, [send]);

  const toggleTensionMode = useCallback(() => {
    setTensionMode((prev) => !prev);
  }, []);

  return {
    grid,
    tensionGrid,
    tensionMode,
    inputFrame,
    inputWeightGrid,
    inputWeightDims,
    sourceWeightGrids,
    regionOverlays,
    nerveCircles,
    regions,
    tensionRegions,
    regionId,
    inputId,
    labels,
    neuronLabelMap,
    orchestratorState,
    state,
    stats,
    perf,
    generation,
    experimentActive,
    inspectMode,
    connectionMap,
    inspectedCell,
    inspectedRegionId,
    inspectInfo,
    inspectHistory,
    brushSize,
    brushMode,
    normalizedConfig,
    start,
    reconnect,
    updateConfig,
    click,
    paint,
    step,
    play,
    pause,
    reset,
    inspect,
    toggleInspectMode,
    toggleTensionMode,
    increaseBrushSize,
    decreaseBrushSize,
    toggleBrushMode,
  };
}
