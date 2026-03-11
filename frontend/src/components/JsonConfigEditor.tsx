/** JSON config editor — CodeMirror 6 based, with dark theme matching the app.
 *
 * Displays the flat ExperimentConfig as a nested JSON structure for readability,
 * and converts back to flat on edit. Zero backend changes required.
 *
 * Provides context-aware autocomplete for fields with known options
 * (masks, fonts, process_mode, etc.) derived from the ExperimentInfo.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting, syntaxTree } from "@codemirror/language";
import { autocompletion } from "@codemirror/autocomplete";
import type { CompletionContext, CompletionResult } from "@codemirror/autocomplete";
import { tags } from "@lezer/highlight";
import type { ExperimentConfig, ExperimentInfo } from "../types";

/* ── Nested config type (display only) ──────────────────────────── */

interface NestedConfig {
  description?: string;
  grid: { width: number; height: number };
  wiring: {
    mask?: string;
    process_mode?: string;
    dendrite_weight?: number;
    deamon_exc_weight?: number;
    deamon_inh_weight?: number;
    balance?: number;
    balance_mode?: string;
    rule?: number;
    tension_function?: Record<string, number>;
  };
  input?: {
    source?: string;
    ascii?: {
      font?: string;
      font_size?: number;
      text?: string;
      resolution?: number;
      frames_per_char?: number;
    };
  };
  noise?: {
    background_white_noise?: number;
    shift?: boolean;
    noise_inter_char?: boolean;
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
    ...(c.description !== undefined ? { description: c.description } : {}),
    grid: { width: c.width, height: c.height },
    wiring: {},
  };

  if (c.mask !== undefined) n.wiring.mask = c.mask;
  if (c.process_mode !== undefined) n.wiring.process_mode = c.process_mode;
  if (c.input_dendrite_weight !== undefined) n.wiring.dendrite_weight = c.input_dendrite_weight;
  if (c.deamon_exc_weight !== undefined) n.wiring.deamon_exc_weight = c.deamon_exc_weight;
  if (c.deamon_inh_weight !== undefined) n.wiring.deamon_inh_weight = c.deamon_inh_weight;
  if (c.balance !== undefined) n.wiring.balance = c.balance;
  if (c.balance_mode !== undefined) n.wiring.balance_mode = c.balance_mode;
  if (c.rule !== undefined) n.wiring.rule = c.rule;
  if (c.tension_function !== undefined) n.wiring.tension_function = c.tension_function;

  const hasInput = c.input_source !== undefined
    || c.font !== undefined
    || (c.input_text !== undefined && c.input_text !== "")
    || c.input_resolution !== undefined;
  if (hasInput) {
    n.input = {};
    if (c.input_source !== undefined) n.input.source = c.input_source;

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

  const hasNoise = c.background_white_noise !== undefined
    || c.shift_noise !== undefined || c.noise_inter_char !== undefined;
  if (hasNoise) {
    n.noise = {};
    if (c.background_white_noise !== undefined) n.noise.background_white_noise = c.background_white_noise;
    if (c.shift_noise !== undefined) n.noise.shift = c.shift_noise;
    if (c.noise_inter_char !== undefined) n.noise.noise_inter_char = c.noise_inter_char;
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
    ...(n.description !== undefined ? { description: n.description } : {}),
    width: n.grid.width,
    height: n.grid.height,
  };

  if (n.wiring.mask !== undefined) c.mask = n.wiring.mask;
  if (n.wiring.process_mode !== undefined) c.process_mode = n.wiring.process_mode;
  if (n.wiring.dendrite_weight !== undefined) c.input_dendrite_weight = n.wiring.dendrite_weight;
  if (n.wiring.deamon_exc_weight !== undefined) c.deamon_exc_weight = n.wiring.deamon_exc_weight;
  if (n.wiring.deamon_inh_weight !== undefined) c.deamon_inh_weight = n.wiring.deamon_inh_weight;
  if (n.wiring.balance !== undefined) c.balance = n.wiring.balance;
  if (n.wiring.balance_mode !== undefined) c.balance_mode = n.wiring.balance_mode;
  if (n.wiring.rule !== undefined) c.rule = n.wiring.rule;
  if (n.wiring.tension_function !== undefined) c.tension_function = n.wiring.tension_function;

  if (n.input) {
    if (n.input.source !== undefined) c.input_source = n.input.source;
    if (n.input.ascii) {
      if (n.input.ascii.font !== undefined) c.font = n.input.ascii.font;
      if (n.input.ascii.font_size !== undefined) c.font_size = n.input.ascii.font_size;
      if (n.input.ascii.text !== undefined) c.input_text = n.input.ascii.text;
      if (n.input.ascii.resolution !== undefined) c.input_resolution = n.input.ascii.resolution;
      if (n.input.ascii.frames_per_char !== undefined) c.frames_per_char = n.input.ascii.frames_per_char;
    }
  } else {
    c.input_text = "";
  }

  if (n.noise) {
    if (n.noise.background_white_noise !== undefined) c.background_white_noise = n.noise.background_white_noise;
    if (n.noise.shift !== undefined) c.shift_noise = n.noise.shift;
    if (n.noise.noise_inter_char !== undefined) c.noise_inter_char = n.noise.noise_inter_char;
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

/* ── Autocomplete: JSON path detection + options ────────────────── */

interface OptionItem {
  value: string;
  label: string;
  detail?: string;
}

type OptionsMap = Record<string, OptionItem[]>;

function buildOptionsMap(exp: ExperimentInfo | undefined): OptionsMap {
  const map: OptionsMap = {};

  map["wiring.process_mode"] = [
    { value: "avg_vs_avg", label: "avg_vs_avg", detail: "Avg excitatory vs avg inhibitory" },
    { value: "min_vs_max", label: "min_vs_max", detail: "Best exc vs best inh" },
    { value: "sum", label: "sum", detail: "All dendrites summed" },
  ];

  if (exp?.masks) {
    map["wiring.mask"] = exp.masks.map((m) => ({
      value: m.id,
      label: m.id,
      detail: m.name,
    }));
  }

  if (exp?.balance_modes) {
    map["wiring.balance_mode"] = exp.balance_modes.map((m) => ({
      value: m.id,
      label: m.id,
      detail: m.name,
    }));
  }

  if (exp?.input_sources) {
    map["input.source"] = exp.input_sources.map((s) => ({
      value: s.id,
      label: s.id,
      detail: s.name,
    }));
  }

  if (exp?.fonts) {
    map["input.ascii.font"] = exp.fonts.map((f) => ({
      value: f.id,
      label: f.id,
      detail: `${f.name} — ${f.description}`,
    }));
  }

  return map;
}

function getJsonPath(context: CompletionContext): string[] {
  const tree = syntaxTree(context.state);
  const path: string[] = [];
  let node = tree.resolveInner(context.pos, -1);

  if (node.type.name === "PropertyName") return [];

  while (node.parent) {
    node = node.parent;
    if (node.type.name === "Property") {
      const nameNode = node.getChild("PropertyName");
      if (nameNode) {
        const raw = context.state.doc.sliceString(nameNode.from, nameNode.to);
        path.unshift(raw.replace(/^"|"$/g, ""));
      }
    }
  }

  return path;
}

function makeCompletionSource(optionsMap: OptionsMap) {
  return (context: CompletionContext): CompletionResult | null => {
    const tree = syntaxTree(context.state);
    const node = tree.resolveInner(context.pos, -1);

    if (node.type.name !== "String") return null;

    const path = getJsonPath(context);
    const pathKey = path.join(".");
    const available = optionsMap[pathKey];
    if (!available || available.length === 0) return null;

    const from = node.from + 1;
    const to = node.to - 1;

    return {
      from,
      to,
      filter: true,
      options: available.map((opt) => ({
        label: opt.value,
        detail: opt.detail,
        type: "enum",
      })),
    };
  };
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
  ".cm-tooltip": {
    backgroundColor: "#1a1a2e",
    border: "1px solid #2a2a3e",
    borderRadius: "4px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
  },
  ".cm-tooltip-autocomplete": {
    "& > ul": {
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      fontSize: "0.78rem",
    },
    "& > ul > li": {
      padding: "4px 8px",
      color: "#e0e0ff",
    },
    "& > ul > li[aria-selected]": {
      backgroundColor: "#1e3a5f",
      color: "#4cc9f0",
    },
  },
  ".cm-completionLabel": {
    color: "#4cc9f0",
  },
  ".cm-completionDetail": {
    color: "#666",
    fontStyle: "normal",
    marginLeft: "8px",
  },
  ".cm-completionIcon-enum": {
    "&::after": { content: "'◇'" },
    color: "#06d6a0",
  },
});

const baseExtensions = [json(), neuroTheme, syntaxHighlighting(neuroHighlight), EditorView.lineWrapping];

/* ── Component ──────────────────────────────────────────────────── */

interface JsonConfigEditorProps {
  config: ExperimentConfig;
  onChange: (config: ExperimentConfig) => void;
  experimentInfo?: ExperimentInfo;
}

export function JsonConfigEditor({ config, onChange, experimentInfo }: JsonConfigEditorProps) {
  const [text, setText] = useState(() => nestedStringify(config));
  const [parseError, setParseError] = useState<string | null>(null);
  const lastExternalConfig = useRef(config);
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>();

  const configJson = useMemo(() => nestedStringify(config), [config]);

  const extensions = useMemo(() => {
    const optionsMap = buildOptionsMap(experimentInfo);
    const completionSource = makeCompletionSource(optionsMap);
    return [
      ...baseExtensions,
      autocompletion({
        override: [completionSource],
        activateOnTyping: true,
        icons: true,
      }),
    ];
  }, [experimentInfo]);

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
