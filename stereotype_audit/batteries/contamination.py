"""Memorisation signal for the legacy benchmark: guided completion of CrowS-Pairs sentences."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from stereotype_audit import stats
from stereotype_audit.batteries.base import Battery, BatteryResult, preview
from stereotype_audit.data import load_crows_pairs
from stereotype_audit.pairs import fill, load_templates

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _split(sentence: str, frac: float = 0.6) -> tuple[str, list[str]]:
    words = sentence.split()
    k = max(1, int(round(len(words) * frac)))
    return " ".join(words[:k]), _words(" ".join(words[k:]))


class ContaminationBattery(Battery):
    id = "contamination"
    family = "lm"
    experimental = True
    description = "Guided-completion accuracy on CrowS-Pairs sentences versus fresh control sentences of the same shape."
    sources = ["Xu et al. 2024, arXiv 2404.18824 (benchmark leakage detection via n-gram accuracy)"]

    def _run(self, subject) -> BatteryResult:
        n = self.n or 200
        crows = load_crows_pairs()
        crows = crows.sample(n=min(n, len(crows)), random_state=self.seed)
        bench = [("crows", s) for s in crows["sent_more"].tolist()]
        controls = self._control_sentences(len(bench))
        rows = []
        for source, sent in bench + [("control", s) for s in controls]:
            prompt, target = _split(sent)
            rows.append({"source": source, "sentence": sent, "prompt": prompt, "target_words": target})
        df = pd.DataFrame(rows)
        gens = subject.generate(df["prompt"].tolist(), max_new_tokens=12, thinking=False)
        df["generated"] = [g.text for g in gens]
        k = int(self.cfg.get("match_words", 3))
        df["match"] = [
            float(len(t) >= k and _words(g)[:k] == t[:k])
            for g, t in zip(df["generated"], df["target_words"], strict=True)
        ]
        df["match_1"] = [
            float(len(t) >= 1 and _words(g)[:1] == t[:1])
            for g, t in zip(df["generated"], df["target_words"], strict=True)
        ]
        summary = {
            "match_words": k,
            "notes": [
                "Each sentence is cut after 60 percent of its words; the model continues greedily and the first k words are compared to the true continuation.",
                "Control sentences come from this package's own templates and were written in 2026, so they cannot be in training data; they are also more formulaic than CrowS sentences and easier to complete, so only a ratio well above 1 is a memorisation signal and a ratio at or below 1 is uninformative.",
            ],
        }
        for source, g in df.groupby("source"):
            est = stats.bootstrap_ci(g["match"].to_numpy(), seed=self.seed)
            est1 = stats.bootstrap_ci(g["match_1"].to_numpy(), seed=self.seed)
            summary[source] = {
                "n": int(est.n),
                "match_rate": est.value,
                "ci": [est.ci_low, est.ci_high],
                "first_word_rate": est1.value,
            }
        cr, ct = summary["crows"]["match_rate"], summary["control"]["match_rate"]
        summary["ratio_crows_over_control"] = (
            float(cr / ct) if ct > 0 else float("inf") if cr > 0 else float("nan")
        )
        return BatteryResult(
            battery=self.id,
            family=self.family,
            n_items=int(len(df)),
            raw=df.assign(target_words=df["target_words"].apply(" ".join)),
            summary=summary,
            probes_preview=preview([{"prompt": df["prompt"].iloc[0], "generated": df["generated"].iloc[0]}]),
            notes=summary["notes"],
        )

    def _control_sentences(self, n: int) -> list[str]:
        """Distinct fresh sentences built from this package's templates; never more than the pool holds."""
        names = ["Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Reese", "Drew"]
        terms = ["left-handed", "right-handed", "tall", "short"]
        pool: list[str] = []
        for t in load_templates("nameswap")["templates"]:
            pool.extend(fill(t["text"], name=nm) for nm in names)
        for d in load_templates("retrieval")["documents"]:
            pool.extend(fill(d["text"], name=nm) for nm in names)
        for t in load_templates("ctf")["templates"]:
            pool.extend(fill(t["text"], term=term) for term in terms)
        pool = list(dict.fromkeys(pool))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(pool))[: min(n, len(pool))]
        return [pool[i] for i in idx]
