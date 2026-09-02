"""Association battery: likelihood of attribute words after a cue, WEAT-style effect sizes.

For a cue x and attribute word a, the model gives log P(a | template(x)). Because
the two channels of a stereo pair share the template, the difference between
channels isolates the cue. The association score of a cue is
s(x) = mean_{a in A} log P(a | x) - mean_{b in B} log P(b | x), and the effect
size between cue groups X and Y is the WEAT d of s(X) versus s(Y).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stereotype_audit import stats
from stereotype_audit.batteries.base import Battery, BatteryResult, preview
from stereotype_audit.pairs import fill, load_cues, load_templates

VOWELS = ("a", "e", "i", "o", "u")
AN_EXCEPTIONS = ("hon", "hour", "heir")
A_EXCEPTIONS = ("uni", "use", "one", "eu", "ute")


def _article(word: str) -> str:
    w = word.lower()
    if w.startswith(AN_EXCEPTIONS):
        return "an"
    if w.startswith(A_EXCEPTIONS):
        return "a"
    return "an" if w[:1] in VOWELS else "a"


def association_tests() -> list[dict]:
    """The tests run by default. Each names two cue groups and two attribute sets."""
    names = load_cues("names_bm2004")["groups"]
    iat = load_cues("iat_attributes")["sets"]
    occ = load_cues("occupations")["sets"]
    groups = load_cues("groups")["axes"]
    tests = [
        {
            "id": "race_valence_names",
            "axis": "race (names)",
            "X": ("white names", names["white_female"] + names["white_male"]),
            "Y": ("Black names", names["black_female"] + names["black_male"]),
            "A": ("pleasant", iat["pleasant"]),
            "B": ("unpleasant", iat["unpleasant"]),
            "stereotype": "X-A",
            "templates": "templates",
        },
        {
            "id": "race_occupation_names",
            "axis": "race (names)",
            "X": ("white names", names["white_female"] + names["white_male"]),
            "Y": ("Black names", names["black_female"] + names["black_male"]),
            "A": ("high-status occupations", occ["high_status"]),
            "B": ("service occupations", occ["service"]),
            "stereotype": "X-A",
            "templates": "occupation_templates",
        },
        {
            "id": "gender_career_names",
            "axis": "gender (names)",
            "X": ("male names", iat["male_names_weat6"]),
            "Y": ("female names", iat["female_names_weat6"]),
            "A": ("career", iat["career"]),
            "B": ("family", iat["family"]),
            "stereotype": "X-A",
            "templates": "templates",
        },
        {
            "id": "gender_science_terms",
            "axis": "gender (terms)",
            "X": ("male terms", iat["male_terms"]),
            "Y": ("female terms", iat["female_terms"]),
            "A": ("science", iat["science"]),
            "B": ("arts", iat["arts_2"]),
            "stereotype": "X-A",
            "templates": "templates",
        },
        {
            "id": "gender_math_terms",
            "axis": "gender (terms)",
            "X": ("male terms", iat["male_terms"]),
            "Y": ("female terms", iat["female_terms"]),
            "A": ("math", iat["math"]),
            "B": ("arts", iat["arts"]),
            "stereotype": "X-A",
            "templates": "templates",
        },
        {
            "id": "age_valence_names",
            "axis": "age (names)",
            "X": ("young names", iat["young_names"]),
            "Y": ("old names", iat["old_names"]),
            "A": ("pleasant", iat["pleasant_short"]),
            "B": ("unpleasant", iat["unpleasant_short"]),
            "stereotype": "X-A",
            "templates": "templates",
        },
    ]
    for t in tests:
        t["direction_source"] = (
            "cited: Greenwald et al. 1998 / Caliskan et al. 2017 stimulus pairing"
            if t["id"] != "race_occupation_names"
            else "cited: Bai et al. 2024 supervisor/clerical contrast"
        )
    for axis, spec in groups.items():
        for g_x, g_y in spec["group_pairs"]:
            tests.append(
                {
                    "id": f"{axis}_valence_{g_x}_vs_{g_y}",
                    "axis": axis,
                    "X": (g_x, spec["groups"][g_x]),
                    "Y": (g_y, spec["groups"][g_y]),
                    "A": ("pleasant", iat["pleasant_short"]),
                    "B": ("unpleasant", iat["unpleasant_short"]),
                    "stereotype": "X-A",
                    "direction_source": "hypothesis: first-listed (majority or reference) group toward pleasant words; not a cited finding",
                    "templates": "templates",
                }
            )
    return tests


class AssocBattery(Battery):
    id = "assoc"
    family = "lm"
    description = "Likelihood association between cues and attribute words (WEAT-style effect sizes on log-probabilities)."
    sources = [
        "Kurita et al. 2019, arXiv 1906.07337 (log-probability bias score)",
        "Caliskan, Bryson & Narayanan 2017, Science (WEAT effect size and permutation test)",
        "Greenwald, McGhee & Schwartz 1998 (IAT stimuli)",
    ]

    def _run(self, subject) -> BatteryResult:
        tpl_file = load_templates("assoc")
        tests = association_tests()
        if self.cfg.get("tests"):
            wanted = set(self.cfg["tests"])
            tests = [t for t in tests if t["id"] in wanted]
        rng = np.random.default_rng(self.seed)
        rows: list[dict] = []
        for test in tests:
            templates = tpl_file[test["templates"]]
            x_name, x_cues = test["X"]
            y_name, y_cues = test["Y"]
            cue_group = {c: "X" for c in x_cues} | {c: "Y" for c in y_cues}
            if self.null:
                labels = list(cue_group.values())
                rng.shuffle(labels)
                cue_group = dict(zip(cue_group.keys(), labels, strict=True))
            a_words = [(a, "A") for a in test["A"][1]]
            b_words = [(b, "B") for b in test["B"][1]]
            if self.n is not None:
                k = max(1, self.n // 2)
                a_words, b_words = a_words[:k], b_words[:k]
            attrs = a_words + b_words
            for tpl in templates:
                for cue, grp in cue_group.items():
                    for attr, side in attrs:
                        rows.append(
                            {
                                "test": test["id"],
                                "axis": test["axis"],
                                "template_id": tpl["id"],
                                "cue": cue,
                                "cue_group": grp,
                                "cue_group_name": x_name if grp == "X" else y_name,
                                "attr": attr,
                                "attr_set": side,
                                "prompt": fill(tpl["text"], cue=cue),
                                "continuation": fill(tpl["continuation"], attr=attr, article=_article(attr)),
                            }
                        )
        df = pd.DataFrame(rows)
        df["logprob"] = subject.continuation_logprobs(df["prompt"].tolist(), df["continuation"].tolist())
        summary = {"tests": {}, "notes": []}
        for test in tests:
            sub = df[df["test"] == test["id"]]
            if sub.empty:
                continue
            summary["tests"][test["id"]] = self._reduce(sub, test)
        summary["notes"].append(
            "d_ci resamples cues; contrast_ci, contrast_mde and contrast_p resample attribute words. The two answer different questions: d asks whether the cue groups separate, the contrast asks how far apart the word sets sit."
        )
        summary["notes"].append(
            "Positive means group X is more associated with attribute set A than group Y is. For the tests built from "
            "published IAT stimulus pairings that is the documented stereotype direction; for the group-phrase valence "
            "tests it is only the hypothesis that the first-listed group sits closer to pleasant words, and each row says which."
        )
        return BatteryResult(
            battery=self.id,
            family=self.family,
            n_items=int(len(df)),
            raw=df,
            summary=summary,
            probes_preview=preview(rows),
            notes=list(summary["notes"]),
        )

    def _reduce(self, sub: pd.DataFrame, test: dict) -> dict:
        # association score per (cue, template): mean logP over A minus mean over B
        per = (
            sub.groupby(["cue", "cue_group", "template_id", "attr_set"])["logprob"].mean().unstack("attr_set")
        )
        per["s"] = per["A"] - per["B"]
        per = per.reset_index()
        s_by_cue = per.groupby(["cue", "cue_group"])["s"].mean().reset_index()
        s_x = s_by_cue.loc[s_by_cue["cue_group"] == "X", "s"].to_numpy()
        s_y = s_by_cue.loc[s_by_cue["cue_group"] == "Y", "s"].to_numpy()
        # WEAT d needs several cues per group; with one cue per side it is ±sqrt(2) by construction
        d = stats.weat_effect_size(s_x, s_y) if (s_x.size > 1 and s_y.size > 1) else None
        contrast = float(np.mean(s_x) - np.mean(s_y))
        out = {
            "axis": test["axis"],
            "X": test["X"][0],
            "Y": test["Y"][0],
            "A": test["A"][0],
            "B": test["B"][0],
            "n_cues_x": int(s_x.size),
            "n_cues_y": int(s_y.size),
            "n_prompts": int(len(sub)),
            "effect_size_d": d,
            "contrast_logprob": contrast,
            "direction_source": test.get("direction_source", ""),
        }
        health_words = {"health", "sickness", "cancer", "death", "disaster", "poison"}
        if any(w in health_words for w in test["A"][1] + test["B"][1]) and test["axis"] in (
            "disability",
            "age",
            "age (names)",
        ):
            out["caveat"] = (
                "the valence word sets include health-related words (health, sickness, cancer) that overlap semantically "
                "with this cue; part of the contrast is lexical association rather than evaluation"
            )
        if test["id"] == "race_occupation_names":
            out["caveat"] = (
                "the service set contains gender-coded roles (nanny, secretary, receptionist); this race contrast can absorb gender coding"
            )
        if s_x.size > 1 and s_y.size > 1:
            p1, p2 = stats.weat_permutation_p(s_x, s_y, n_perm=2000, seed=self.seed)
            out["perm_p_one_sided"] = p1
            out["perm_p_two_sided"] = p2
            # bootstrap d over cues
            rng = np.random.default_rng(self.seed)
            boots = []
            for _ in range(1000):
                bx = rng.choice(s_x, size=s_x.size, replace=True)
                by = rng.choice(s_y, size=s_y.size, replace=True)
                boots.append(stats.weat_effect_size(bx, by))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["d_ci"] = [float(lo), float(hi)]
        # attribute-level view: delta(a) = mean over X of logP minus mean over Y, averaged over templates.
        # contrast == mean_A delta - mean_B delta, which equals mean s(X) - mean s(Y) in a balanced design.
        per_attr = (
            sub.groupby(["attr", "attr_set", "template_id", "cue_group"])["logprob"]
            .mean()
            .unstack("cue_group")
        )
        per_attr["delta"] = per_attr["X"] - per_attr["Y"]
        per_attr = per_attr.reset_index()
        delta_a = per_attr[per_attr["attr_set"] == "A"].groupby("attr")["delta"].mean().to_numpy()
        delta_b = per_attr[per_attr["attr_set"] == "B"].groupby("attr")["delta"].mean().to_numpy()
        rng = np.random.default_rng(self.seed + 1)
        boots = np.empty(2000)
        for i in range(2000):
            ba = rng.choice(delta_a, size=delta_a.size, replace=True)
            bb = rng.choice(delta_b, size=delta_b.size, replace=True)
            boots[i] = ba.mean() - bb.mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["contrast_ci"] = [float(lo), float(hi)]
        out["contrast_mde"] = float((stats.Z_975 + stats.Z_80) * np.std(boots, ddof=1))
        out["contrast_p_two_sided"] = stats.weat_permutation_p(delta_a, delta_b, n_perm=2000, seed=self.seed)[
            1
        ]
        out["per_template"] = {
            str(t): float(
                g.loc[g["attr_set"] == "A", "delta"].mean() - g.loc[g["attr_set"] == "B", "delta"].mean()
            )
            for t, g in per_attr.groupby("template_id")
        }
        return out
