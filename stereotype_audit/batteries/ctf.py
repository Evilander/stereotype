"""Counterfactual token fairness for classifiers: identity terms in neutral sentences."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from stereotype_audit import stats
from stereotype_audit.batteries.base import Battery, BatteryResult, preview
from stereotype_audit.pairs import fill, load_cues, load_templates


class CTFBattery(Battery):
    id = "ctf"
    family = "clf"
    description = "Score gaps between identity terms and neutral reference terms in the same sentences; false-positive flip rates."
    sources = [
        "Dixon et al. 2018, AIES (unintended bias, identity terms)",
        "Garg et al. 2019, AIES (counterfactual token fairness)",
    ]

    def _run(self, subject) -> BatteryResult:
        spec = load_templates("ctf")
        terms = load_cues("identity_terms")
        templates = spec["templates"]
        if self.n is not None:
            templates = templates[: max(1, self.n)]
        rows = []
        for tpl in templates:
            for ref in terms["reference_terms"]:
                rows.append(
                    {
                        "template_id": tpl["id"],
                        "axis": "reference",
                        "term": ref,
                        "text": fill(tpl["text"], term=ref),
                    }
                )
            for axis, lst in terms["axes"].items():
                for term in lst:
                    rows.append(
                        {
                            "template_id": tpl["id"],
                            "axis": axis,
                            "term": term,
                            "text": fill(tpl["text"], term=term),
                        }
                    )
        df = pd.DataFrame(rows)
        if self.null:
            # shuffle axis labels among identity terms inside each template; reference rows stay reference
            rng = np.random.default_rng(self.seed)
            for _, idx in df[df["axis"] != "reference"].groupby("template_id").groups.items():
                df.loc[idx, "axis"] = rng.permutation(df.loc[idx, "axis"].to_numpy())
        df["score"] = subject.scores(df["text"].tolist())
        threshold = float(self.cfg.get("threshold", 0.5))
        df["flagged"] = (df["score"] >= threshold).astype(float)
        ref = df[df["axis"] == "reference"].groupby("template_id")["score"].mean()
        df["gap_vs_reference"] = df["score"] - df["template_id"].map(ref)
        summary = {
            "target_label": subject.label_name,
            "threshold": threshold,
            "reference_mean_score": float(ref.mean()),
            "reference_flag_rate": float(df.loc[df["axis"] == "reference", "flagged"].mean()),
            "reference_terms": {
                t: float(v) for t, v in df[df["axis"] == "reference"].groupby("term")["score"].mean().items()
            },
            "terms": {},
            "axes": {},
            "notes": [
                "gap = mean score for the term minus the mean score of the reference terms in the same templates; positive means the classifier treats the neutral sentence as more toxic because of the identity term.",
                "flag_rate = share of templates scored at or above the threshold; every sentence is neutral, so any flag is a false positive.",
                "ctf_gap = mean absolute score difference between all pairs of identity terms on the same axis within a template.",
                "reference_terms lists each neutral reference term's mean score so the spread of the baseline itself is visible.",
            ],
        }
        for term, g in df[df["axis"] != "reference"].groupby("term"):
            est = stats.bootstrap_ci(g["gap_vs_reference"].to_numpy(), seed=self.seed)
            summary["terms"][term] = {
                "axis": g["axis"].iloc[0],
                "gap": est.value,
                "ci": [est.ci_low, est.ci_high],
                "flag_rate": float(g["flagged"].mean()),
                "mean_score": float(g["score"].mean()),
                "n_templates": int(est.n),
            }
        for axis, g in df[df["axis"] != "reference"].groupby("axis"):
            gaps = g.groupby("template_id")["gap_vs_reference"].mean().to_numpy()
            est = stats.bootstrap_ci(gaps, seed=self.seed)
            ctf = []
            for _, gt in g.groupby("template_id"):
                per_term = gt.groupby("term")["score"].mean().to_numpy()
                ctf.extend(abs(x - y) for x, y in itertools.combinations(per_term, 2))
            summary["axes"][axis] = {
                "gap": est.value,
                "ci": [est.ci_low, est.ci_high],
                "flag_rate": float(g["flagged"].mean()),
                "ctf_gap": float(np.mean(ctf)) if ctf else float("nan"),
                "worst_term": str(g.groupby("term")["score"].mean().idxmax()),
                "n_terms": int(g["term"].nunique()),
            }
        return BatteryResult(
            battery=self.id,
            family=self.family,
            n_items=int(len(df)),
            raw=df,
            summary=summary,
            probes_preview=preview(df[["text", "axis"]].head(3).to_dict("records")),
            notes=summary["notes"],
        )
