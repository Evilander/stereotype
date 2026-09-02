"""SEAT: WEAT on sentence embeddings of bleached templates around a single word."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stereotype_audit import stats
from stereotype_audit.batteries.base import Battery, BatteryResult, preview
from stereotype_audit.pairs import fill, load_cues, load_templates


def _words(source: str, key: str) -> list[str]:
    data = load_cues(source)
    table = data.get("groups") or data.get("sets")
    return table[key]


class SEATBattery(Battery):
    id = "seat"
    family = "embed"
    description = (
        "Sentence Encoder Association Test: WEAT effect sizes on embeddings of bleached template sentences."
    )
    sources = [
        "May et al. 2019, arXiv 1903.10561 (SEAT)",
        "Caliskan, Bryson & Narayanan 2017, Science (WEAT)",
    ]

    def _run(self, subject) -> BatteryResult:
        spec = load_templates("seat")
        templates = spec["templates"]
        if self.n is not None:
            templates = templates[: max(1, self.n)]
        rng = np.random.default_rng(self.seed)
        rows = []
        summary = {
            "tests": {},
            "notes": [
                "s(w) = mean cosine to attribute-set A sentences minus mean cosine to attribute-set B sentences; d is the WEAT effect size between the two target sets with a permutation p over target labels.",
                "Each word is embedded inside every template; the template embeddings are averaged before computing cosines, as in SEAT's word-level variant.",
            ],
        }
        for test in spec["tests"]:
            tx, ty = test["targets"]
            ta, tb = test["attributes"]
            words = {
                "X": _words(test["target_source"], tx),
                "Y": _words(test["target_source"], ty),
                "A": _words("iat_attributes", ta),
                "B": _words("iat_attributes", tb),
            }
            if self.null:
                pooled = words["X"] + words["Y"]
                rng.shuffle(pooled)
                words["X"], words["Y"] = pooled[: len(words["X"])], pooled[len(words["X"]) :]
            texts, index = [], []
            for role, lst in words.items():
                for w in lst:
                    for tpl in templates:
                        texts.append(fill(tpl["text"], word=w))
                        index.append((role, w, tpl["id"]))
            emb = subject.encode(texts, kind="passage")
            df = pd.DataFrame(index, columns=["role", "word", "template_id"])
            df["vec"] = list(emb)
            mean_vec = df.groupby(["role", "word"])["vec"].apply(lambda v: np.mean(np.stack(v), axis=0))
            mean_vec = mean_vec.apply(lambda v: v / np.linalg.norm(v))
            A = np.stack(mean_vec.loc["A"].to_list())
            B = np.stack(mean_vec.loc["B"].to_list())
            s = {}
            for role in ("X", "Y"):
                M = np.stack(mean_vec.loc[role].to_list())
                s[role] = M @ A.T
                s[role] = s[role].mean(axis=1) - (M @ B.T).mean(axis=1)
            d = stats.weat_effect_size(s["X"], s["Y"])
            p1, p2 = stats.weat_permutation_p(s["X"], s["Y"], n_perm=5000, seed=self.seed)
            boots = []
            for _ in range(1000):
                bx = rng.choice(s["X"], size=s["X"].size, replace=True)
                by = rng.choice(s["Y"], size=s["Y"].size, replace=True)
                boots.append(stats.weat_effect_size(bx, by))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            summary["tests"][test["id"]] = {
                "targets": [tx, ty],
                "attributes": [ta, tb],
                "n_targets": [int(s["X"].size), int(s["Y"].size)],
                "effect_size_d": d,
                "d_ci": [float(lo), float(hi)],
                "perm_p_one_sided": p1,
                "perm_p_two_sided": p2,
                "stereotypical_direction": test["stereotypical_direction"],
            }
            for role in ("X", "Y"):
                for w, val in zip(mean_vec.loc[role].index, s[role], strict=True):
                    rows.append({"test": test["id"], "role": role, "word": w, "s": float(val)})
        return BatteryResult(
            battery=self.id,
            family=self.family,
            n_items=int(len(rows)),
            raw=pd.DataFrame(rows),
            summary=summary,
            probes_preview=preview([{"sentence": fill(templates[0]["text"], word="Emily")}]),
            notes=summary["notes"],
        )
