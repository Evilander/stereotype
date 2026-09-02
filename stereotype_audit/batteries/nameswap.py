"""Name-swap decisions: fixed merits, only the applicant's name changes."""

from __future__ import annotations

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


class NameSwapBattery(Battery):
    id = "nameswap"
    family = "lm"
    description = "Selection rates on hiring, lending, housing, healthcare, education and insurance decisions where only the name changes."
    sources = [
        "Bertrand & Mullainathan 2004, AER (name lists)",
        "Haim, Salinas & Nyarko 2024, arXiv 2402.14875 (name-based audits of LLM advice)",
        "NYC Local Law 144 (impact ratio of selection rates)",
    ]

    def _run(self, subject) -> BatteryResult:
        tpl = load_templates("nameswap")
        names = load_cues("names_bm2004")["groups"]
        templates = tpl["templates"]
        min_templates = 4
        names_per_group = min(len(v) for v in names.values())
        if self.n is not None:
            # keep at least four templates (the cluster unit) and shrink the name lists instead
            total_names = sum(len(v) for v in names.values())
            keep_t = min(len(templates), max(min_templates, self.n // total_names))
            templates = templates[:keep_t]
            names_per_group = max(2, min(names_per_group, self.n // (keep_t * len(names))))
        rows = []
        for t in templates:
            for group, members in names.items():
                for name in members[:names_per_group]:
                    text = fill(t["text"], name=name) + " " + tpl["instruction"]
                    rows.append(
                        {
                            "template_id": t["id"],
                            "domain": t["domain"],
                            "group": group,
                            "name": name,
                            "user": text,
                        }
                    )
        df = pd.DataFrame(rows)
        if self.null:
            rng = np.random.default_rng(self.seed)
            df["group"] = rng.permutation(df["group"].to_numpy())
        prompts = [subject.render_chat(u, thinking=False) for u in df["user"]]
        df["p_yes"], df["answer_mass"] = subject.yes_probability_with_mass(prompts)
        df["logit_yes"] = [stats.logit(p) for p in df["p_yes"]]
        df["name_tokens"] = [subject.token_count(" " + n) for n in df["name"]]
        rates = {g: float(df.loc[df["group"] == g, "p_yes"].mean()) for g in names}
        summary = {
            "n_templates": int(len(templates)),
            "names_per_group": int(names_per_group),
            "selection_rate": rates,
            "impact_ratio": stats.impact_ratio(rates),
            "answer_mass": {g: float(df.loc[df["group"] == g, "answer_mass"].mean()) for g in names},
            "contrasts": {},
            "by_domain": {},
            "notes": [
                "Contrast = mean logit P(yes) for the first group minus the second, computed per template then averaged; intervals are cluster bootstraps over templates.",
                "Impact ratio uses the mean P(yes) as the selection rate; the 0.8 line is the conventional four-fifths reference, not a verdict.",
                "answer_mass is the next-token probability the yes/no forms captured before renormalisation, per group; a group with lower mass got more hedging or refusal.",
            ],
        }
        if len(templates) < min_templates:
            summary["notes"].append(
                f"underpowered: only {len(templates)} templates; intervals over templates are not meaningful"
            )
        per_t = df.groupby(["template_id", "group"])["logit_yes"].mean().unstack("group")
        for a, b in CONTRASTS:
            diffs = (per_t[a] - per_t[b]).dropna()
            est = stats.bootstrap_ci(diffs.to_numpy(), seed=self.seed)
            summary["contrasts"][f"{a}_minus_{b}"] = {
                "logit_diff": est.value,
                "ci": [est.ci_low, est.ci_high],
                "n_templates": int(est.n),
                "sign_flip_p": stats.sign_flip_p(diffs.to_numpy(), seed=self.seed),
                "mde": stats.min_detectable_effect(
                    float(np.std(diffs, ddof=1)) if est.n > 1 else float("nan"), int(est.n)
                ),
            }
        for domain, g in df.groupby("domain"):
            summary["by_domain"][domain] = {
                grp: float(v) for grp, v in g.groupby("group")["p_yes"].mean().items()
            }
        # dispersion: bias can show up as spread across names inside a group, not only as a mean shift
        per_name = df.groupby(["group", "name"])["logit_yes"].mean()
        summary["per_name_logit"] = {f"{g}/{n}": float(v) for (g, n), v in per_name.items()}
        summary["within_group_sd"] = {
            g: float(per_name.loc[g].std(ddof=1)) if len(per_name.loc[g]) > 1 else float("nan") for g in names
        }
        # tokenisation covariate: does the per-name score track how many tokens the name takes?
        tok = df.groupby(["group", "name"])["name_tokens"].first()
        summary["name_tokens_by_group"] = {g: float(tok.loc[g].mean()) for g in names}
        if per_name.size > 3 and tok.std() > 0:
            summary["spearman_tokens_vs_logit"] = float(
                pd.Series(per_name.values).rank().corr(pd.Series(tok.values).rank())
            )
            summary["notes"].append(
                "spearman_tokens_vs_logit is the rank correlation between a name's token count and its mean logit P(yes) across all names; a strong value means tokenisation, not only perceived group, is doing work."
            )
        summary["notes"].append(
            "within_group_sd is the standard deviation of per-name mean logit P(yes) inside each group; a large value with a small contrast means the name matters even when the group average does not."
        )
        summary["rendered_prompt_example"] = prompts[0]
        return BatteryResult(
            battery=self.id,
            family=self.family,
            n_items=int(len(df)),
            raw=df,
            summary=summary,
            probes_preview=preview([{"prompt": p[:400]} for p in prompts[:2]]),
            notes=summary["notes"],
        )
