/** Sidebar — experiment selector and configuration. */

import { useState, useEffect } from "react";
import type { ExperimentInfo, ExperimentConfig } from "../types";

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px",
  background: "#1a1a2e",
  border: "1px solid #2a2a3e",
  borderRadius: "4px",
  color: "#e0e0ff",
  fontSize: "0.9rem",
};

/** Numeric input that allows empty field while editing and normalizes , to . */
function NumericInput({
  value,
  onChange,
  min,
  max,
  step,
  integer,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
}) {
  const [text, setText] = useState(String(value));

  useEffect(() => {
    setText(String(value));
  }, [value]);

  const commit = (raw: string) => {
    const normalized = raw.replace(",", ".");
    const parsed = integer ? parseInt(normalized, 10) : parseFloat(normalized);
    if (Number.isNaN(parsed)) return;
    let clamped = parsed;
    if (min !== undefined && clamped < min) clamped = min;
    if (max !== undefined && clamped > max) clamped = max;
    onChange(clamped);
    setText(String(clamped));
  };

  return (
    <input
      type="text"
      inputMode={integer ? "numeric" : "decimal"}
      value={text}
      step={step}
      onChange={(e) => setText(e.target.value.replace(",", "."))}
      onBlur={() => commit(text)}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit(text);
      }}
      style={inputStyle}
    />
  );
}

interface SidebarProps {
  experiments: ExperimentInfo[];
  selectedExperiment: string;
  config: ExperimentConfig;
  onSelectExperiment: (id: string) => void;
  onConfigChange: (config: ExperimentConfig) => void;
  onStart: () => void;
  onReconnect?: () => void;
  connected: boolean;
  experimentActive?: boolean;
}

export function Sidebar({
  experiments,
  selectedExperiment,
  config,
  onSelectExperiment,
  onConfigChange,
  onStart,
  onReconnect,
  connected,
  experimentActive,
}: SidebarProps) {
  const selectedExp = experiments.find((e) => e.id === selectedExperiment);
  const hasMasks = selectedExp?.masks && selectedExp.masks.length > 0;
  const canReconnect = experimentActive && hasMasks;

  return (
    <aside
      style={{
        width: "260px",
        minWidth: "260px",
        background: "#12121a",
        borderRight: "1px solid #2a2a3e",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        overflowY: "auto",
      }}
    >
      <div>
        <h1
          style={{
            fontSize: "1.3rem",
            fontWeight: 700,
            color: "#e0e0ff",
            margin: 0,
            letterSpacing: "0.05em",
          }}
        >
          NeuroFlow
        </h1>
        <span
          style={{
            fontSize: "0.7rem",
            color: "#666",
            fontFamily: "monospace",
          }}
        >
          v0.1.0
        </span>
      </div>

      <div>
        <h3
          style={{
            fontSize: "0.75rem",
            textTransform: "uppercase",
            color: "#888",
            marginBottom: "10px",
            letterSpacing: "0.1em",
          }}
        >
          Experimentos
        </h3>
        {experiments.map((exp) => (
          <button
            key={exp.id}
            onClick={() => onSelectExperiment(exp.id)}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              padding: "10px 12px",
              marginBottom: "4px",
              background:
                selectedExperiment === exp.id ? "#1e1e3a" : "transparent",
              border:
                selectedExperiment === exp.id
                  ? "1px solid #4cc9f0"
                  : "1px solid transparent",
              borderRadius: "6px",
              color: selectedExperiment === exp.id ? "#4cc9f0" : "#aaa",
              cursor: "pointer",
              fontSize: "0.85rem",
              transition: "all 0.15s",
            }}
          >
            {exp.name}
          </button>
        ))}
      </div>

      {selectedExp && (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {selectedExp.rules && selectedExp.rules.length > 0 && (
            <div>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Regla
              </label>
              <select
                value={config.rule ?? ""}
                onChange={(e) =>
                  onConfigChange({ ...config, rule: Number(e.target.value) })
                }
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "#1a1a2e",
                  border: "1px solid #2a2a3e",
                  borderRadius: "4px",
                  color: "#e0e0ff",
                  fontSize: "0.9rem",
                }}
              >
                {selectedExp.rules.map((r) => (
                  <option key={r} value={r}>
                    Rule {r}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div style={{ display: "flex", gap: "8px" }}>
            <div style={{ flex: 1 }}>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Ancho
              </label>
              <NumericInput
                value={config.width}
                min={5}
                max={200}
                integer
                onChange={(v) => onConfigChange({ ...config, width: v })}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Alto
              </label>
              <NumericInput
                value={config.height}
                min={5}
                max={200}
                integer
                onChange={(v) => onConfigChange({ ...config, height: v })}
              />
            </div>
          </div>

          {hasMasks && (
            <div>
              <label
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  color: "#888",
                  display: "block",
                  marginBottom: "6px",
                  letterSpacing: "0.1em",
                }}
              >
                Conexionado
              </label>
              <div
                style={{
                  maxHeight: "220px",
                  overflowY: "auto",
                  display: "flex",
                  flexDirection: "column",
                  gap: "3px",
                }}
              >
                {selectedExp!.masks!.map((m) => (
                  <button
                    key={m.id}
                    onClick={() =>
                      onConfigChange({ ...config, mask: m.id })
                    }
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      padding: "6px 8px",
                      background:
                        config.mask === m.id ? "#1a1a3a" : "transparent",
                      border:
                        config.mask === m.id
                          ? "1px solid #f72585"
                          : "1px solid #222",
                      borderRadius: "4px",
                      color: config.mask === m.id ? "#f72585" : "#999",
                      cursor: "pointer",
                      fontSize: "0.78rem",
                      transition: "all 0.15s",
                    }}
                  >
                    <span
                      style={{
                        fontWeight: 600,
                        display: "block",
                        lineHeight: 1.3,
                      }}
                    >
                      {m.name}
                    </span>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        color: config.mask === m.id ? "#c7608b" : "#555",
                        display: "block",
                        lineHeight: 1.3,
                      }}
                    >
                      {m.description}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {selectedExp.balance_modes && selectedExp.balance_modes.length > 0 && (
            <div>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Modo de balance
              </label>
              <select
                value={config.balance_mode ?? "none"}
                onChange={(e) =>
                  onConfigChange({ ...config, balance_mode: e.target.value })
                }
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "#1a1a2e",
                  border: "1px solid #2a2a3e",
                  borderRadius: "4px",
                  color: "#e0e0ff",
                  fontSize: "0.9rem",
                }}
              >
                {selectedExp.balance_modes.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {selectedExp.default_config.balance !== undefined &&
            config.balance_mode !== undefined &&
            config.balance_mode !== "none" && (
            <div>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Balance
              </label>
              <NumericInput
                value={config.balance ?? 0}
                min={-1}
                max={1}
                step={0.1}
                onChange={(v) =>
                  onConfigChange({ ...config, balance: v })
                }
              />
              <span
                style={{
                  fontSize: "0.65rem",
                  color: "#555",
                  marginTop: "2px",
                  display: "block",
                }}
              >
                0 = sin cambio, + excitatorio, − inhibitorio
              </span>
            </div>
          )}

          {selectedExp.default_config.balance !== undefined &&
            !selectedExp.balance_modes && (
            <div>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Balance
              </label>
              <NumericInput
                value={config.balance ?? 0}
                min={-1}
                max={1}
                step={0.1}
                onChange={(v) =>
                  onConfigChange({ ...config, balance: v })
                }
              />
              <span
                style={{
                  fontSize: "0.65rem",
                  color: "#555",
                  marginTop: "2px",
                  display: "block",
                }}
              >
                0 = sin cambio, + excitatorio, − inhibitorio
              </span>
            </div>
          )}

          {selectedExp.init_modes && selectedExp.init_modes.length > 0 && (
            <div>
              <label
                style={{
                  fontSize: "0.75rem",
                  color: "#888",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Inicialización
              </label>
              <select
                value={config.init ?? "random"}
                onChange={(e) =>
                  onConfigChange({ ...config, init: e.target.value })
                }
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "#1a1a2e",
                  border: "1px solid #2a2a3e",
                  borderRadius: "4px",
                  color: "#e0e0ff",
                  fontSize: "0.9rem",
                }}
              >
                {selectedExp.init_modes.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={onStart}
            disabled={!connected}
            style={{
              padding: "10px",
              background: connected ? "#4cc9f0" : "#333",
              color: connected ? "#0a0a0a" : "#666",
              border: "none",
              borderRadius: "6px",
              fontSize: "0.9rem",
              fontWeight: 600,
              cursor: connected ? "pointer" : "not-allowed",
              transition: "all 0.15s",
            }}
          >
            {connected ? "Iniciar Experimento" : "Conectando..."}
          </button>

          {canReconnect && onReconnect && (
            <button
              onClick={onReconnect}
              style={{
                padding: "8px",
                background: "transparent",
                color: "#f72585",
                border: "1px solid #f72585",
                borderRadius: "6px",
                fontSize: "0.8rem",
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              Reconectar
            </button>
          )}
        </div>
      )}

      <div style={{ marginTop: "auto", fontSize: "0.7rem", color: "#444" }}>
        <p>
          {selectedExp?.rules
            ? "Click en la fila inferior (azul) para activar neuronas de entrada."
            : "Click en cualquier celda para activar/desactivar neuronas."}
        </p>
        <p style={{ marginTop: "4px" }}>
          Usa Play para ver la propagacion automatica.
        </p>
      </div>
    </aside>
  );
}
