"""Battery protocol: build probes, run them through a subject, reduce to statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BatteryResult:
    battery: str
    family: str
    n_items: int
    raw: pd.DataFrame
    summary: dict
    probes_preview: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timing_s: float = 0.0
    config: dict = field(default_factory=dict)
    experimental: bool = False


class Battery:
    """Subclasses set `id`, `family`, `description`, `sources` and implement `_run`."""

    id = "base"
    family = "lm"
    description = ""
    sources: list[str] = []
    experimental = False

    def __init__(self, n: int | None = None, seed: int = 0, null: bool = False, **cfg):
        self.n = n
        self.seed = seed
        self.null = null
        self.cfg = cfg

    def config(self) -> dict:
        return {"n": self.n, "seed": self.seed, "null": self.null, **self.cfg}

    def run(self, subject) -> BatteryResult:
        if getattr(subject, "family", None) != self.family:
            raise TypeError(f"battery {self.id} expects a {self.family} subject, got {subject.family}")
        t0 = time.perf_counter()
        result = self._run(subject)
        result.timing_s = round(time.perf_counter() - t0, 2)
        result.config = self.config()
        result.experimental = self.experimental
        return result

    def _run(self, subject) -> BatteryResult:  # pragma: no cover - abstract
        raise NotImplementedError


def preview(rows: list[dict], k: int = 3) -> list[dict]:
    return [dict(r) for r in rows[:k]]
