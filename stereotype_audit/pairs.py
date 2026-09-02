"""The stereo pair: two texts identical except for one demographic cue."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

CUES_PACKAGE = "stereotype_audit.cues"
TEMPLATES_PACKAGE = "stereotype_audit.templates"


@dataclass(frozen=True)
class StereoPair:
    pair_id: str
    axis: str
    template_id: str
    left: str
    right: str
    cue_left: str
    cue_right: str
    group_left: str
    group_right: str
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def swapped(self) -> StereoPair:
        """The same pair with channels exchanged (used for order-balanced scoring)."""
        return StereoPair(
            pair_id=self.pair_id + ":swap",
            axis=self.axis,
            template_id=self.template_id,
            left=self.right,
            right=self.left,
            cue_left=self.cue_right,
            cue_right=self.cue_left,
            group_left=self.group_right,
            group_right=self.group_left,
            meta=dict(self.meta),
        )


def load_cues(name: str) -> dict:
    """Load a cue list shipped with the package (see docs/cues.md for provenance)."""
    text = resources.files(CUES_PACKAGE).joinpath(f"{name}.json").read_text(encoding="utf-8")
    return json.loads(text)


def load_templates(name: str) -> dict:
    text = resources.files(TEMPLATES_PACKAGE).joinpath(f"{name}.json").read_text(encoding="utf-8")
    return json.loads(text)


def fill(template: str, **slots: str) -> str:
    """Fill `{slot}` and `{Slot}` placeholders; the capitalised form upper-cases the first letter."""
    out = template
    for key, value in slots.items():
        out = out.replace("{" + key + "}", value)
        cap = value[:1].upper() + value[1:]
        out = out.replace("{" + key[:1].upper() + key[1:] + "}", cap)
    return out


def _pair_id(*parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def make_pairs(
    axis: str,
    templates: list[dict],
    cue_groups: dict[str, list[str]],
    group_pairs: list[tuple[str, str]],
    seed: int = 0,
    max_pairs: int | None = None,
    extra_slots: dict[str, str] | None = None,
) -> list[StereoPair]:
    """Expand templates × cue groups into stereo pairs.

    For each (group_left, group_right) and each template, every cue in the left
    group is paired with every cue in the right group. `max_pairs` subsamples the
    result deterministically.
    """
    extra = extra_slots or {}
    pairs: list[StereoPair] = []
    for tpl in templates:
        for g_left, g_right in group_pairs:
            for c_left, c_right in itertools.product(cue_groups[g_left], cue_groups[g_right]):
                left = fill(tpl["text"], cue=c_left, **extra)
                right = fill(tpl["text"], cue=c_right, **extra)
                if left == right:
                    continue
                pairs.append(
                    StereoPair(
                        pair_id=_pair_id(axis, tpl["id"], c_left, c_right, json.dumps(extra, sort_keys=True)),
                        axis=axis,
                        template_id=tpl["id"],
                        left=left,
                        right=right,
                        cue_left=c_left,
                        cue_right=c_right,
                        group_left=g_left,
                        group_right=g_right,
                        meta={"slots": extra} if extra else {},
                    )
                )
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(pairs), size=max_pairs, replace=False))
        pairs = [pairs[i] for i in keep]
    return pairs


def shuffle_cues(pairs: list[StereoPair], seed: int = 0) -> list[StereoPair]:
    """Null construction: randomly swap the two channels of each pair.

    Group labels travel with the cues, so a battery run on the shuffled set has
    no systematic left/right alignment and should show no effect.
    """
    rng = np.random.default_rng(seed)
    flips = rng.random(len(pairs)) < 0.5
    return [p.swapped() if f else p for p, f in zip(pairs, flips, strict=True)]


def write_pairs(pairs: list[StereoPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.as_dict(), ensure_ascii=False) + "\n")
