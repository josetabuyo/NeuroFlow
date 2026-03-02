/** JSON config editor — CodeMirror 6 based, with dark theme matching the app.
 *
 * Displays the flat ExperimentConfig as a nested JSON structure for readability,
 * and converts back to flat on edit. Zero backend changes required.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";
import type { ExperimentConfig } from "../types";

/* ── Nested config type (display only) ──────────────────────────── */

interface NestedConfig {
  grid: { width: number; height: number };
  wiring: {
    mask?: string;
    process_mode?: string;
    deamon_exc_weight?: number;
    deamon_inh_weight?: number;
    balance?: number;
    balance_mode?: string;
    rule?: number;
  };
  input?: {
    source?: string;
    dendrite_weight?: number;
    ascii?: {
      font?: string;
      font_size?: number;
      text?: string;
      resolution?: number;
      frames_per_char?: number;
    };
  };
  noise?: {
    white?: boolean;
    prob?: number;
    shift?: boolean;
    inter_char?: boolean;
  };
  learning?: {
    enabled?: boolean;
    rate?: number;
  };
  spiking?: {
    enabled?: boolean;
    max_active_steps?: number;
    refractory_steps?: number;
  };
}

/* ── Flat ↔ Nested conversion ───────────────────────────────────── */

function toNested(c: ExperimentConfig): NestedConfig {
  const n: NestedConfig = {
    grid: { width: c.width, height: c.height },
    wiring: {},
  };

  if (c.mask !== undefined) n.wiring.mask = c.mask;
  if (c.process_mode !== undefined) n.wiring.process_mode = c.process_mode;
  if (c.deamon_exc_weight !== undefined) n.wiring.deamon_exc_weight = c.deamon_exc_weight;
  if (c.deamon_inh_weight !== undefined) n.wiring.deamon_inh_weight = c.deamon_inh_weight;
  if (c.balance !== undefined) n.wiring.balance = c.balance;
  if (c.balance_mode !== undefined) n.wiring.balance_mode = c.balance_mode;
  if (c.rule !== undefined) n.wiring.rule = c.rule;

  const hasInput = c.input_source !== undefined || c.input_dendrite_weight !== undefined
    || c.font !== undefined || c.input_text !== undefined || c.input_resolution !== undefined;
  if (hasInput) {
    n.input = {};
    if (c.input_source !== undefined) n.input.source = c.input_source;
    if (c.input_dendrite_weight !== undefined) n.input.dendrite_weight = c.input_dendrite_weight;

    const hasAscii = c.font !== undefined || c.font_size !== undefined
      || c.input_text !== undefined || c.input_resolution !== undefined || c.frames_per_char !== undefined;
    if (hasAscii) {
      n.input.ascii = {};
      if (c.font !== undefined) n.input.ascii.font = c.font;
      if (c.font_size !== undefined) n.input.ascii.font_size = c.font_size;
      if (c.input_text !== undefined) n.input.ascii.text = c.input_text;
      if (c.input_resolution !== undefined) n.input.ascii.resolution = c.input_resolution;
      if (c.frames_per_char !== undefined) n.input.ascii.frames_per_char = c.frames_per_char;
    }
  }

  const hasNoise = c.white_noise !== undefined || c.noise_prob !== undefined
    || c.shift_noise !== undefined || c.inter_char_noise !== undefined;
  if (hasNoise) {
    n.noise = {};
    if (c.white_noise !== undefined) n.noise.white = c.white_noise;
    if (c.noise_prob !== undefined) n.noise.prob = c.noise_prob;
    if (c.shift_noise !== undefined) n.noise.shift = c.shift_noise;
    if (c.inter_char_noise !== undefined) n.noise.inter_char = c.inter_char_noise;
  }

  const hasLearning = c.learning !== undefined || c.learning_rate !== undefined;
  if (hasLearning) {
    n.learning = {};
    if (c.learning !== undefined) n.learning.enabled = c.learning;
    if (c.learning_rate !== undefined) n.learning.rate = c.learning_rate;
  }

  const hasSpiking = c.spike_adaptation !== undefined || c.max_active_steps !== undefined
    || c.refractory_steps !== undefined;
  if (hasSpiking) {
    n.spiking = {};
    if (c.spike_adaptation !== undefined) n.spiking.enabled = c.spike_adaptation;
    if (c.max_active_steps !== undefined) n.spiking.max_active_steps = c.max_active_steps;
    if (c.refractory_steps !== undefined) n.spiking.refractory_steps = c.refractory_steps;
  }

  return n;
}

function toFlat(n: NestedConfig): ExperimentConfig {
  const c: ExperimentConfig = {
    width: n.grid.width,
    height: n.grid.height,
  };

  if (n.wiring.mask !== undefined) c.mask = n.wiring.mask;
  if (n.wiring.process_mode !== undefined) c.process_mode = n.wiring.process_mode;
  if (n.wiring.deamon_exc_weight !== undefined) c.deamon_exc_weight = n.wiring.deamon_exc_weight;
  if (n.wiring.deamon_inh_weight !== undefined) c.deamon_inh_weight = n.wiring.deamon_inh_weight;
  if (n.wiring.balance !== undefined) c.balance = n.wiring.balance;
  if (n.wiring.balance_mode !== undefined) c.balance_mode = n.wiring.balance_mode;
  if (n.wiring.rule !== undefined) c.rule = n.wiring.rule;

  if (n.input) {
    if (n.input.source !== undefined) c.input_source = n.input.source;
    if (n.input.dendrite_weight !== undefined) c.input_dendrite_weight = n.input.dendrite_weight;
    if (n.input.ascii) {
      if (n.input.ascii.font !== undefined) c.font = n.input.ascii.font;
      if (n.input.ascii.font_size !== undefined) c.font_size = n.input.ascii.font_size;
      if (n.input.ascii.text !== undefined) c.input_text = n.input.ascii.text;
      if (n.input.ascii.resolution !== undefined) c.input_resolution = n.input.ascii.resolution;
      if (n.input.ascii.frames_per_char !== undefined) c.frames_per_char = n.input.ascii.frames_per_char;
    }
  }

  if (n.noise) {
    if (n.noise.white !== undefined) c.white_noise = n.noise.white;
    if (n.noise.prob !== undefined) c.noise_prob = n.noise.prob;
    if (n.noise.shift !== undefined) c.shift_noise = n.noise.shift;
    if (n.noise.inter_char !== undefined) c.inter_char_noise = n.noise.inter_char;
  }

  if (n.learning) {
    if (n.learning.enabled !== undefined) c.learning = n.learning.enabled;
    if (n.learning.rate !== undefined) c.learning_rate = n.learning.rate;
  }

  if (n.spiking) {
    if (n.spiking.enabled !== undefined) c.spike_adaptation = n.spiking.enabled;
    if (n.spiking.max_active_steps !== undefined) c.max_active_steps = n.spiking.max_active_steps;
    if (n.spiking.refractory_steps !== undefined) c.refractory_steps = n.spiking.refractory_steps;
  }

  return c;
}

/* ── Stable JSON serialization ──────────────────────────────────── */

function nestedStringify(config: ExperimentConfig): string {
  return JSON.stringify(toNested(config), null, 2);
}

/* ── CodeMirror theme ───────────────────────────────────────────── */

const neuroHighlight = HighlightStyle.define([
  { tag: tags.propertyName, color: "#8a8aad" },
  { tag: tags.string, color: "#4cc9f0" },
  { tag: tags.number, color: "#06d6a0" },
  { tag: tags.bool, color: "#ffd166" },
  { tag: tags.null, color: "#666" },
  { tag: tags.punctuation, color: "#555" },
  { tag: tags.brace, color: "#6a6a8e" },
  { tag: tags.squareBracket, color: "#6a6a8e" },
]);

const neuroTheme = EditorView.theme({
  "&": {
    backgroundColor: "#0d0d14",
    color: "#e0e0ff",
    fontSize: "0.8rem",
    fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
  },
  ".cm-content": {
    caretColor: "#4cc9f0",
    padding: "8px 0",
  },
  ".cm-cursor": {
    borderLeftColor: "#4cc9f0",
  },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
    backgroundColor: "#1e3a5f !important",
  },
  ".cm-gutters": {
    backgroundColor: "#0a0a10",
    color: "#444",
    border: "none",
    borderRight: "1px solid #1a1a2e",
  },
  ".cm-activeLineGutter": {
    backgroundColor: "#12122a",
  },
  ".cm-activeLine": {
    backgroundColor: "#12122a",
  },
  ".cm-foldPlaceholder": {
    backgroundColor: "#1a1a2e",
    color: "#666",
    border: "none",
  },
  ".cm-matchingBracket": {
    backgroundColor: "#1e3a5f",
    outline: "1px solid #4cc9f0",
  },
});

const extensions = [json(), neuroTheme, syntaxHighlighting(neuroHighlight)];

/* ── Component ──────────────────────────────────────────────────── */

interface JsonConfigEditorProps {
  config: ExperimentConfig;
  onChange: (config: ExperimentConfig) => void;
}

export function JsonConfigEditor({ config, onChange }: JsonConfigEditorProps) {
  const [text, setText] = useState(() => nestedStringify(config));
  const [parseError, setParseError] = useState<string | null>(null);
  const lastExternalConfig = useRef(config);
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>();

  const configJson = useMemo(() => nestedStringify(config), [config]);

  useEffect(() => {
    if (configJson !== nestedStringify(lastExternalConfig.current)) {
      lastExternalConfig.current = config;
      setText(configJson);
      setParseError(null);
    }
  }, [config, configJson]);

  const handleChange = useCallback(
    (value: string) => {
      setText(value);
      clearTimeout(debounceTimer.current);
      debounceTimer.current = setTimeout(() => {
        try {
          const parsed = JSON.parse(value);
          if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
            setParseError("Config must be a JSON object");
            return;
          }
          if (!parsed.grid || typeof parsed.grid.width !== "number" || typeof parsed.grid.height !== "number") {
            setParseError("grid.width and grid.height are required");
            return;
          }
          setParseError(null);
          const flat = toFlat(parsed as NestedConfig);
          lastExternalConfig.current = flat;
          onChange(flat);
        } catch (e) {
          setParseError((e as Error).message);
        }
      }, 300);
    },
    [onChange],
  );

  useEffect(() => () => clearTimeout(debounceTimer.current), []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h3
          style={{
            fontSize: "0.75rem",
            textTransform: "uppercase",
            color: "#888",
            margin: 0,
            letterSpacing: "0.1em",
          }}
        >
          Config
        </h3>
        {parseError && (
          <span style={{ fontSize: "0.6rem", color: "#ef476f", maxWidth: "160px", textAlign: "right", lineHeight: "1.2" }}>
            JSON Error
          </span>
        )}
      </div>
      <div
        style={{
          borderRadius: "6px",
          border: `1px solid ${parseError ? "#ef476f" : "#2a2a3e"}`,
          overflow: "hidden",
          transition: "border-color 0.2s",
        }}
      >
        <CodeMirror
          value={text}
          onChange={handleChange}
          extensions={extensions}
          theme="none"
          basicSetup={{
            lineNumbers: false,
            foldGutter: true,
            bracketMatching: true,
            closeBrackets: true,
            autocompletion: false,
            highlightActiveLine: true,
            highlightSelectionMatches: false,
            searchKeymap: false,
          }}
          style={{ maxHeight: "60vh", overflow: "auto" }}
        />
      </div>
      {parseError && (
        <span
          style={{
            fontSize: "0.6rem",
            color: "#ef476f",
            lineHeight: "1.3",
            padding: "0 4px",
            wordBreak: "break-word",
          }}
        >
          {parseError}
        </span>
      )}
    </div>
  );
}
