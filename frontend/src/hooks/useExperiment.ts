/** WebSocket connection + state management for experiments. */

import { useState, useRef, useCallback, useEffect } from "react";
import type {
  ExperimentConfig,
  ExperimentState,
  ExperimentStats,
  PerfMetrics,
  ServerMessage,
} from "../types";
import { nextBrushSize, prevBrushSize } from "../brushes";

function getWsUrl(): string {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/experiment`;
}

interface UseExperimentReturn {
  grid: number[][];
  tensionGrid: number[][] | null;
  tensionMode: boolean;
  state: ExperimentState;
  stats: ExperimentStats | null;
  perf: PerfMetrics | null;
  generation: number;
  activeExperiment: string | null;
  inspectMode: boolean;
  connectionMap: (number | null)[][] | null;
  inspectedCell: { x: number; y: number } | null;
  brushSize: number;
  brushMode: "activate" | "deactivate";
  start: (experiment: string, config: ExperimentConfig) => void;
  reconnect: (config: ExperimentConfig) => void;
  click: (x: number, y: number) => void;
  paint: (cells: { x: number; y: number }[], value: number) => void;
  step: (count?: number) => void;
  play: (fps?: number, stepsPerTick?: number) => void;
  pause: () => void;
  reset: () => void;
  inspect: (x: number, y: number) => void;
  toggleInspectMode: () => void;
  toggleTensionMode: () => void;
  increaseBrushSize: () => void;
  decreaseBrushSize: () => void;
  toggleBrushMode: () => void;
}

export function useExperiment(): UseExperimentReturn {
  const [grid, setGrid] = useState<number[][]>([]);
  const [state, setState] = useState<ExperimentState>("disconnected");
  const [stats, setStats] = useState<ExperimentStats | null>(null);
  const [generation, setGeneration] = useState(0);
  const [inspectMode, setInspectMode] = useState(false);
  const [connectionMap, setConnectionMap] = useState<(number | null)[][] | null>(null);
  const [inspectedCell, setInspectedCell] = useState<{ x: number; y: number } | null>(null);
  const [perf, setPerf] = useState<PerfMetrics | null>(null);
  const [tensionGrid, setTensionGrid] = useState<number[][] | null>(null);
  const [tensionMode, setTensionMode] = useState(false);
  const [activeExperiment, setActiveExperiment] = useState<string | null>(null);
  const [brushSize, setBrushSize] = useState(1);
  const [brushMode, setBrushMode] = useState<"activate" | "deactivate">("activate");
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
          setStats(msg.stats);
          setPerf(msg.perf ?? null);
          setTensionGrid(msg.tension_grid ?? null);
          break;
        case "connections":
          setConnectionMap(msg.weight_grid);
          setInspectedCell({ x: msg.x, y: msg.y });
          break;
        case "status":
          setState(msg.state);
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
    (experiment: string, config: ExperimentConfig) => {
      setActiveExperiment(experiment);
      setState("initializing");
      send({ action: "start", experiment, config });
    },
    [send]
  );

  const reconnect = useCallback(
    (config: ExperimentConfig) => {
      setState("initializing");
      send({ action: "reconnect", config });
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
    setState("initializing");
    send({ action: "reset" });
  }, [send]);

  const paint = useCallback(
    (cells: { x: number; y: number }[], value: number) => {
      send({ action: "paint", cells, value });
    },
    [send]
  );

  const inspect = useCallback(
    (x: number, y: number) => {
      send({ action: "inspect", x, y });
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
      }
      return !prev;
    });
  }, []);

  const toggleTensionMode = useCallback(() => {
    setTensionMode((prev) => !prev);
  }, []);

  return {
    grid,
    tensionGrid,
    tensionMode,
    state,
    stats,
    perf,
    generation,
    activeExperiment,
    inspectMode,
    connectionMap,
    inspectedCell,
    brushSize,
    brushMode,
    start,
    reconnect,
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
