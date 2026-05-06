/** Draggable, resizable panel that wraps a neural region's canvas. */

import { useRef, useCallback, type ReactNode } from "react";

export interface BoxLayout {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface LayerBoxProps {
  id: string;
  label: string;
  layout: BoxLayout;
  onUpdate: (id: string, layout: BoxLayout) => void;
  children: ReactNode;
  minSize?: number;
  /** Glow border when this box is relevant to the current inspection. */
  highlighted?: boolean;
}

export const HEADER_H = 26;
const HANDLE = 10;

export function LayerBox({
  id,
  label,
  layout,
  onUpdate,
  children,
  minSize = 60,
  highlighted = false,
}: LayerBoxProps) {
  const { x, y, w, h } = layout;

  // ── Move (drag header) ────────────────────────────────────────────
  const moveRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);

  const onMoveDown = useCallback(
    (e: React.PointerEvent) => {
      e.stopPropagation();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      moveRef.current = { sx: e.clientX, sy: e.clientY, ox: x, oy: y };
    },
    [x, y],
  );

  const onMoveMove = useCallback(
    (e: React.PointerEvent) => {
      if (!moveRef.current) return;
      const { sx, sy, ox, oy } = moveRef.current;
      onUpdate(id, { x: ox + e.clientX - sx, y: oy + e.clientY - sy, w, h });
    },
    [id, onUpdate, w, h],
  );

  const onMoveUp = useCallback(() => { moveRef.current = null; }, []);

  // ── Resize (edge / corner handles) ───────────────────────────────
  const resizeRef = useRef<{
    sx: number; sy: number;
    ox: number; oy: number; ow: number; oh: number;
    edge: string;
  } | null>(null);

  const onResizeDown = useCallback(
    (edge: string) => (e: React.PointerEvent) => {
      e.stopPropagation();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      resizeRef.current = { sx: e.clientX, sy: e.clientY, ox: x, oy: y, ow: w, oh: h, edge };
    },
    [x, y, w, h],
  );

  const onResizeMove = useCallback(
    (e: React.PointerEvent) => {
      if (!resizeRef.current) return;
      const { sx, sy, ox, oy, ow, oh, edge } = resizeRef.current;
      const dx = e.clientX - sx;
      const dy = e.clientY - sy;

      let nx = ox, ny = oy, nw = ow, nh = oh;

      if (edge.includes("e")) nw = Math.max(minSize, ow + dx);
      if (edge.includes("s")) nh = Math.max(minSize, oh + dy);
      if (edge.includes("w")) { nw = Math.max(minSize, ow - dx); nx = ox + ow - nw; }
      if (edge.includes("n")) { nh = Math.max(minSize, oh - dy); ny = oy + oh - nh; }

      onUpdate(id, { x: nx, y: ny, w: nw, h: nh });
    },
    [id, onUpdate, minSize],
  );

  const onResizeUp = useCallback(() => { resizeRef.current = null; }, []);

  const handle = (edge: string, style: React.CSSProperties) => (
    <div
      key={edge}
      onPointerDown={onResizeDown(edge)}
      onPointerMove={onResizeMove}
      onPointerUp={onResizeUp}
      style={{ position: "absolute", ...style, zIndex: 6, cursor: `${edge}-resize` }}
    />
  );

  const accent = highlighted ? "#7c3aed" : "#2a2a3e";
  const glow   = highlighted ? "0 0 14px rgba(124,58,237,0.35)" : "none";

  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, userSelect: "none" }}>

      {/* ── Header ── */}
      <div
        onPointerDown={onMoveDown}
        onPointerMove={onMoveMove}
        onPointerUp={onMoveUp}
        style={{
          height: HEADER_H,
          background: "#14142a",
          borderRadius: "6px 6px 0 0",
          border: `1px solid ${accent}`,
          borderBottom: "none",
          display: "flex",
          alignItems: "center",
          padding: "0 10px",
          gap: "7px",
          cursor: "grab",
          transition: "border-color 0.2s",
        }}
      >
        <span style={{ color: highlighted ? "#a78bfa" : "#2e2e52", fontSize: "0.8rem", lineHeight: 1 }}>⠿</span>
        <span style={{ fontSize: "0.62rem", color: highlighted ? "#a78bfa" : "#555", letterSpacing: "0.1em", textTransform: "uppercase" }}>
          {label}
        </span>
      </div>

      {/* ── Content ── */}
      <div
        style={{
          width: "100%",
          height: h,
          border: `1px solid ${accent}`,
          borderRadius: "0 0 6px 6px",
          overflow: "hidden",
          background: "#0d0d14",
          position: "relative",
          transition: "border-color 0.2s, box-shadow 0.2s",
          boxShadow: glow,
        }}
      >
        {children}
      </div>

      {/* ── Resize handles ── */}
      {/* corners */}
      {handle("se", { right: 0,  bottom: 0,        width: HANDLE*2, height: HANDLE*2 })}
      {handle("sw", { left: 0,   bottom: 0,        width: HANDLE*2, height: HANDLE*2 })}
      {handle("ne", { right: 0,  top: HEADER_H,    width: HANDLE*2, height: HANDLE*2 })}
      {handle("nw", { left: 0,   top: HEADER_H,    width: HANDLE*2, height: HANDLE*2 })}
      {/* edges */}
      {handle("e",  { right: 0,  top: HEADER_H + HANDLE*2, bottom: HANDLE*2, width: HANDLE })}
      {handle("w",  { left: 0,   top: HEADER_H + HANDLE*2, bottom: HANDLE*2, width: HANDLE })}
      {handle("s",  { left: HANDLE*2, right: HANDLE*2,    bottom: 0,         height: HANDLE })}
      {handle("n",  { left: HANDLE*2, right: HANDLE*2,    top: HEADER_H,     height: HANDLE })}
    </div>
  );
}
