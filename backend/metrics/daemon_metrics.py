"""Daemon detection and stability scoring — HUD-only, decoupled from the
simulation's generation counter and from the play loop's frame rate.

Callers own the cadence: DaemonMetrics just needs `compute()` invoked
whenever a fresh reading is wanted, and scores stability over a real
wall-clock window, not a count of ticks.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

DEFAULT_DAEMON_THRESHOLD = 0.5
DEFAULT_MIN_DAEMON_SIZE = 3
DEFAULT_STABILITY_WINDOW_SECONDS = 10.0


def detect_daemons(
    values: torch.Tensor,
    width: int,
    height: int,
    threshold: float,
    min_size: int = DEFAULT_MIN_DAEMON_SIZE,
) -> tuple[int, set[int], set[int], list[int]]:
    """Detect daemons as connected components of active neurons (8-connectivity)."""
    n = width * height
    active = (values[:n] > threshold).tolist()
    visited = [False] * n
    daemon_indices: set[int] = set()
    noise_indices: set[int] = set()
    sizes: list[int] = []

    for idx in range(n):
        if active[idx] and not visited[idx]:
            queue = [idx]
            visited[idx] = True
            cluster: list[int] = []
            head = 0
            while head < len(queue):
                cidx = queue[head]
                head += 1
                cluster.append(cidx)
                cx, cy = cidx % width, cidx // width
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            nidx = ny * width + nx
                            if active[nidx] and not visited[nidx]:
                                visited[nidx] = True
                                queue.append(nidx)
            if len(cluster) >= min_size:
                daemon_indices.update(cluster)
                sizes.append(len(cluster))
            else:
                noise_indices.update(cluster)

    return len(sizes), daemon_indices, noise_indices, sizes


@dataclass
class DaemonMetrics:
    """Stateful daemon-count/stability tracker for one experiment.

    `compute()` is the only entry point; stability is scored from the
    coefficient of variation of daemon counts recorded within the last
    `window_seconds` of real time — call cadence and simulation speed don't
    affect what "stable" means.
    """

    threshold: float = DEFAULT_DAEMON_THRESHOLD
    min_size: int = DEFAULT_MIN_DAEMON_SIZE
    window_seconds: float = DEFAULT_STABILITY_WINDOW_SECONDS
    _history: deque[tuple[float, int]] = field(default_factory=deque, repr=False)

    def reset(self) -> None:
        """Clear stability history — call on experiment reset/rebuild."""
        self._history.clear()

    def compute(self, values: torch.Tensor, width: int, height: int) -> dict[str, Any]:
        """Snapshot -> {daemon_count, avg_daemon_size, noise_cells, exclusion, stability}."""
        n = width * height
        vals = values[:n]

        count, daemon_indices, noise_indices, sizes = detect_daemons(
            values, width, height, self.threshold, self.min_size,
        )
        avg_size = round(sum(sizes) / len(sizes), 1) if sizes else 0.0

        if daemon_indices:
            daemon_mask = torch.zeros(n, dtype=torch.bool)
            daemon_mask[list(daemon_indices)] = True
            inside_mean = vals[daemon_mask].mean().item()
            outside = vals[~daemon_mask]
            outside_mean = outside.mean().item() if outside.numel() > 0 else 0.0
            exclusion = inside_mean - outside_mean
        else:
            exclusion = 0.0

        return {
            "daemon_count": count,
            "avg_daemon_size": avg_size,
            "noise_cells": len(noise_indices),
            "exclusion": round(exclusion, 3),
            "stability": self._score_stability(count),
        }

    def _score_stability(self, count: int) -> float:
        now = time.monotonic()
        self._history.append((now, count))
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 2:
            return 0.0

        counts = [c for _, c in self._history]
        mean_c = sum(counts) / len(counts)
        if mean_c > 0:
            variance = sum((c - mean_c) ** 2 for c in counts) / len(counts)
            cv = (variance ** 0.5) / mean_c
            return round(max(0.0, min(1.0, 1.0 - cv)), 3)
        return 1.0 if all(c == 0 for c in counts) else 0.0
