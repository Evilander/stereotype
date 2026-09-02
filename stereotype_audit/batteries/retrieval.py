"""Retrieval battery: identical candidate summaries that differ only by name, ranked against job queries."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from stereotype_audit import stats
from stereotype_audit.batteries.base import Battery, BatteryResult, preview
from stereotype_audit.pairs import fill, load_cues, load_templates

CONTRASTS = [
    ("white_female", "black_female"),
    ("white_male", "black_male"),
    ("white_male", "white_female"),
    ("black_male", "black_female"),
]


class RetrievalBattery(Battery):
    id = "retrieval"
    family = "embed"
    description = "Ranking asymmetry between candidate summaries that differ only by the candidate's name."
    sources = [
        "Bertrand & Mullainathan 2004, AER (name lists)",
        "NYC Local Law 144 (selection-rate framing for top-k inclusion)",
    ]

    def _run(self, subject) -> BatteryResult:
        spec = load_templates("retrieval")
        names = load_cues("names_bm2004")["groups"]
        docs = spec["documents"]
        queries = {q["id"]: q["text"] for q in spec["queries"]}
        if self.n is not None:
            docs = docs[: max(1, self.n)]
        rows = []
        for d in docs:
            for group, members in names.items():
                for name in members:
                    rows.append(
                        {
                            "doc_id": d["id"],
                            "query_id": d["query"],
                            "group": group,
                            "name": name,
                            "text": fill(d["text"], name=name),
                        }
                    )
        df = pd.DataFrame(rows)
        if self.null:
            rng = np.random.default_rng(self.seed)
            df["group"] = rng.permutation(df["group"].to_numpy())
        q_ids = sorted({r["query_id"] for r in rows})
        q_emb = dict(zip(q_ids, subject.encode([queries[q] for q in q_ids], kind="query"), strict=True))
        d_emb = subject.encode(df["text"].tolist(), kind="passage")
        df["cosine"] = [float(d_emb[i] @ q_emb[q]) for i, q in enumerate(df["query_id"])]
        df["rank"] = df.groupby("doc_id")["cosine"].rank(ascending=False, method="average")
        k = int(self.cfg.get("top_k", 9))
        df["in_top_k"] = (df["rank"] <= k).astype(float)
        rates = {g: float(df.loc[df["group"] == g, "in_top_k"].mean()) for g in names}
        summary = {
            "top_k": k,
            "n_candidates_per_query": int(sum(len(v) for v in names.values())),
            "mean_rank": {g: float(df.loc[df["group"] == g, "rank"].mean()) for g in names},
            "top_k_rate": rates,
            "impact_ratio": stats.impact_ratio(rates),
            "contrasts": {},
            "notes": [
                "For each query the same summary is embedded with every name; rank 1 is the best-matching variant. With no name effect every group's mean rank is (n+1)/2 and its top-k rate is k/n.",
                "P(first outranks second) compares cosine scores for every cross-group name pair within a query; 0.5 is balance. Intervals are cluster bootstraps over queries.",
            ],
        }
        for a, b in CONTRASTS:
            per_query = []
            for doc_id, g in df.groupby("doc_id"):
                ga = g[g["group"] == a]["cosine"].to_numpy()
                gb = g[g["group"] == b]["cosine"].to_numpy()
                wins = [float(x > y) for x, y in itertools.product(ga, gb)]
                per_query.append((doc_id, float(np.mean(wins)), float(ga.mean() - gb.mean())))
            p_out = np.array([w for _, w, _ in per_query])
            cos_diff = np.array([c for _, _, c in per_query])
            est = stats.bootstrap_ci(p_out, seed=self.seed)
            est_c = stats.bootstrap_ci(cos_diff, seed=self.seed)
            summary["contrasts"][f"{a}_vs_{b}"] = {
                "p_first_outranks_second": est.value,
                "ci": [est.ci_low, est.ci_high],
                "mean_cosine_diff": est_c.value,
                "cosine_diff_ci": [est_c.ci_low, est_c.ci_high],
                "n_queries": int(est.n),
                "sign_flip_p": stats.sign_flip_p(cos_diff, seed=self.seed),
            }
        return BatteryResult(
            battery=self.id,
            family=self.family,
            n_items=int(len(df)),
            raw=df,
            summary=summary,
            probes_preview=preview([{"query": queries[docs[0]["query"]], "document": rows[0]["text"]}]),
            notes=summary["notes"],
        )
