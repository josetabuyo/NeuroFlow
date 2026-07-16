/** Live scrolling time-series of activation + surface tension for the inspected neuron.
 * One pixel per frame — each new sample enters at the right edge; the trace grows
 * leftward until the window is full, then scrolls right-to-left like an oscilloscope. */

import { useEffect, useRef } from "react";
import type { InspectSample } from "../hooks/useExperiment";

export const INSPECT_CHART_WIDTH = 280;
export const INSPECT_CHART_HEIGHT = 90;

const BG = "#050508";
const ZERO_LINE = "#1a1a2e";
const ACTIVATION_COLOR = "#e8e8f0";
const TENSION_POS = "#ff8c00";
const TENSION_NEG = "#8b5cf6";

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function InspectChart({ history }: { history: InspectSample[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const targetWidth = INSPECT_CHART_WIDTH * dpr;
    const targetHeight = INSPECT_CHART_HEIGHT * dpr;
    // See PixelCanvas.tsx — avoid reallocating the GPU backing store every
    // frame when the size hasn't actually changed.
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, INSPECT_CHART_WIDTH, INSPECT_CHART_HEIGHT);

    // Zero/mid reference line (tension baseline)
    const midY = INSPECT_CHART_HEIGHT / 2;
    ctx.strokeStyle = ZERO_LINE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY + 0.5);
    ctx.lineTo(INSPECT_CHART_WIDTH, midY + 0.5);
    ctx.stroke();

    if (history.length === 0) return;

    // Show only the most recent WIDTH samples — a scrolling window, 1px/frame.
    const visible = history.slice(-INSPECT_CHART_WIDTH);
    const xOffset = INSPECT_CHART_WIDTH - visible.length; // newest sample always at the right edge

    const yActivation = (v: number) => INSPECT_CHART_HEIGHT - clamp(v, 0, 1) * INSPECT_CHART_HEIGHT;
    const yTension = (v: number) => midY - clamp(v, -1, 1) * midY;

    // Activation curve
    ctx.strokeStyle = ACTIVATION_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    visible.forEach((s, i) => {
      const x = xOffset + i;
      const y = yActivation(s.activation);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Tension curve — color switches per-segment by sign
    ctx.lineWidth = 1;
    for (let i = 1; i < visible.length; i++) {
      const prev = visible[i - 1];
      const curr = visible[i];
      ctx.strokeStyle = curr.tension >= 0 ? TENSION_POS : TENSION_NEG;
      ctx.beginPath();
      ctx.moveTo(xOffset + i - 1, yTension(prev.tension));
      ctx.lineTo(xOffset + i, yTension(curr.tension));
      ctx.stroke();
    }
  }, [history]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: INSPECT_CHART_WIDTH,
        height: INSPECT_CHART_HEIGHT,
        borderRadius: 3,
        border: "1px solid #1a1a2e",
        display: "block",
      }}
    />
  );
}
