import { describe, it, expect } from "vitest";
import { appendInspectSample, type InspectSample } from "./useExperiment";

describe("appendInspectSample", () => {
  it("appends a new tick sample", () => {
    const prev: InspectSample[] = [{ tick: 1, activation: 0, tension: 0 }];
    const next = appendInspectSample(prev, { tick: 2, activation: 1, tension: 0.5 });
    expect(next).toHaveLength(2);
    expect(next[1]).toEqual({ tick: 2, activation: 1, tension: 0.5 });
  });

  it("replaces the last sample when a new one lands on the same tick (e.g. painting while paused)", () => {
    const prev: InspectSample[] = [{ tick: 5, activation: 0.2, tension: -0.1 }];
    const next = appendInspectSample(prev, { tick: 5, activation: 0.9, tension: 0.9 });
    expect(next).toHaveLength(1);
    expect(next[0]).toEqual({ tick: 5, activation: 0.9, tension: 0.9 });
  });

  it("drops the oldest sample once over the max", () => {
    const prev: InspectSample[] = [
      { tick: 1, activation: 0, tension: 0 },
      { tick: 2, activation: 0, tension: 0 },
      { tick: 3, activation: 0, tension: 0 },
    ];
    const next = appendInspectSample(prev, { tick: 4, activation: 1, tension: 1 }, 3);
    expect(next).toHaveLength(3);
    expect(next.map(s => s.tick)).toEqual([2, 3, 4]);
  });

  it("builds up from empty", () => {
    const next = appendInspectSample([], { tick: 0, activation: 0.5, tension: 0.1 });
    expect(next).toEqual([{ tick: 0, activation: 0.5, tension: 0.1 }]);
  });
});
