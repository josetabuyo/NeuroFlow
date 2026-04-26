/** JSON config editor — CodeMirror 6 based, with dark theme matching the app.
 *
 * Works directly with the nested ExperimentConfig format.
 * No flat/nested conversion needed — what you see is what the backend gets.
 *
 * Provides context-aware autocomplete for fields with known options
 * (masks, fonts, process_mode, etc.) derived from Metadata.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting, syntaxTree } from "@codemirror/language";
import { autocompletion } from "@codemirror/autocomplete";
import type { CompletionContext, CompletionResult } from "@codemirror/autocomplete";
import { tags } from "@lezer/highlight";
import type { ExperimentConfig, Metadata } from "../types";

/* ── Autocomplete: JSON path detection + options ────────────────── */

interface OptionItem {
  value: string;
  label: string;
  detail?: string;
}

type OptionsMap = Record<string, OptionItem[]>;

function buildOptionsMap(metadata: Metadata | undefined): OptionsMap {
  const map: OptionsMap = {};

  if (metadata?.process_modes) {
    map["wiring.process_mode"] = metadata.process_modes.map((m) => ({
      value: m.id,
      label: m.id,
      detail: m.description,
    }));
  } else {
    map["wiring.process_mode"] = [
      { value: "avg_vs_avg", label: "avg_vs_avg", detail: "Avg excitatory vs avg inhibitory" },
      { value: "min_vs_max", label: "min_vs_max", detail: "Best exc vs best inh" },
      { value: "sum", label: "sum", detail: "All dendrites summed" },
    ];
  }

  if (metadata?.masks) {
    map["wiring.mask"] = metadata.masks.map((m) => ({
      value: m.id,
      label: m.id,
      detail: m.name,
    }));
  }

  if (metadata?.input_sources) {
    map["input.source"] = metadata.input_sources.map((s) => ({
      value: s.id,
      label: s.id,
      detail: s.name,
    }));
  }

  if (metadata?.fonts) {
    map["input.font"] = metadata.fonts.map((f) => ({
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
  metadata?: Metadata;
}

export function JsonConfigEditor({ config, onChange, metadata }: JsonConfigEditorProps) {
  const stringify = (c: ExperimentConfig) => JSON.stringify(c, null, 2);

  const [text, setText] = useState(() => stringify(config));
  const [parseError, setParseError] = useState<string | null>(null);
  const lastExternalConfig = useRef(config);
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>();

  const configJson = useMemo(() => stringify(config), [config]);

  const extensions = useMemo(() => {
    const optionsMap = buildOptionsMap(metadata);
    const completionSource = makeCompletionSource(optionsMap);
    return [
      ...baseExtensions,
      autocompletion({
        override: [completionSource],
        activateOnTyping: true,
        icons: true,
      }),
    ];
  }, [metadata]);

  useEffect(() => {
    if (configJson !== stringify(lastExternalConfig.current)) {
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
          if (!parsed.wiring || typeof parsed.wiring !== "object") {
            setParseError("wiring section is required");
            return;
          }
          setParseError(null);
          lastExternalConfig.current = parsed;
          onChange(parsed as ExperimentConfig);
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
          resize: "vertical",
          height: "440px",
          display: "flex",
          flexDirection: "column",
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
          style={{ flex: 1, height: "100%", overflow: "auto" }}
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
