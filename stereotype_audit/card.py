"""The Stereotype Card: one JSON contract, rendered to Markdown and self-contained HTML."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

from stereotype_audit import CARD_SCHEMA_VERSION, __version__


@dataclass
class Row:
    battery: str
    label: str
    value: float
    ci: tuple[float, float] | None
    n: int | None
    p: float | None
    center: float
    span: float
    unit: str
    note: str = ""
    p_adj: float | None = None


MIN_CLUSTERS = 4


def _finish_rows(rows: list[Row]) -> list[Row]:
    """Holm-adjust the p-values within a battery and flag rows whose n is too small to interpret."""
    from stereotype_audit.stats import holm

    pvals = {i: r.p for i, r in enumerate(rows) if r.p is not None and math.isfinite(r.p)}
    adjusted = holm({str(i): p for i, p in pvals.items()}) if pvals else {}
    for i, r in enumerate(rows):
        if str(i) in adjusted:
            r.p_adj = adjusted[str(i)]
        if r.n is not None and r.ci is not None and r.n < MIN_CLUSTERS:
            r.note = f"underpowered (n={r.n}); " + r.note
    return rows


def _f(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _ci(ci) -> str:
    if not ci:
        return ""
    return f"[{_f(ci[0])}, {_f(ci[1])}]"


def _with_mde(note: str, mde, unit: str) -> str:
    """Append the minimum detectable effect so an interval covering zero is read against the run's power."""
    if mde is None or (isinstance(mde, float) and not math.isfinite(mde)):
        return note
    return f"{note}; MDE {_f(mde)} {unit}"


def highlights(battery: str, s: dict) -> list[Row]:
    """The rows worth putting on the front of the card for each battery."""
    rows: list[Row] = []
    if battery == "assoc":
        for tid, t in s.get("tests", {}).items():
            label = f"{tid}: {t['X']} vs {t['Y']} on {t['A']}/{t['B']}"
            src = t.get("direction_source", "")
            direction = (
                "positive = documented stereotype direction"
                if src.startswith("cited")
                else "positive = first group toward A (hypothesis, uncited)"
            )
            note = direction + (f"; {t['caveat']}" if t.get("caveat") else "")
            if t.get("effect_size_d") is not None:
                rows.append(
                    Row(
                        battery,
                        label,
                        t["effect_size_d"],
                        tuple(t.get("d_ci", ())) or None,
                        t["n_prompts"],
                        t.get("perm_p_two_sided"),
                        0.0,
                        2.0,
                        "d",
                        _with_mde(note, t.get("contrast_mde"), "logprob"),
                    )
                )
            else:
                rows.append(
                    Row(
                        battery,
                        label,
                        t["contrast_logprob"],
                        tuple(t.get("contrast_ci", ())) or None,
                        t["n_prompts"],
                        t.get("contrast_p_two_sided"),
                        0.0,
                        4.0,
                        "logprob",
                        _with_mde(
                            note + " (one cue per group: log-probability contrast, no d)",
                            t.get("contrast_mde"),
                            "logprob",
                        ),
                    )
                )
    elif battery == "decision":
        for attr, a in s.get("attributes", {}).items():
            for lv, v in a.get("levels", {}).items():
                rows.append(
                    Row(
                        battery,
                        f"{attr}={lv} vs baseline",
                        v["discrimination_score"],
                        tuple(v["ci"]),
                        v["n_questions"],
                        v["sign_flip_p"],
                        0.0,
                        1.5,
                        "logit",
                        _with_mde("positive favours the level", v.get("mde"), "logit"),
                    )
                )
            mass = a.get("answer_mass")
            if mass:
                lo, hi = min(mass.values()), max(mass.values())
                rows.append(
                    Row(
                        battery,
                        f"{attr}: yes/no answer mass by level",
                        (lo + hi) / 2,
                        (lo, hi),
                        None,
                        None,
                        0.5,
                        0.5,
                        "prob",
                        f"range across levels [{_f(lo, 2)}, {_f(hi, 2)}]; low or uneven mass means hedging or refusal differs by level",
                    )
                )
    elif battery == "nameswap":
        for k, v in s.get("contrasts", {}).items():
            rows.append(
                Row(
                    battery,
                    k.replace("_minus_", " − "),
                    v["logit_diff"],
                    tuple(v["ci"]),
                    v["n_templates"],
                    v["sign_flip_p"],
                    0.0,
                    1.5,
                    "logit",
                    _with_mde(
                        "positive favours the first group"
                        + (
                            f"; token-matched {_f(v['token_matched_logit_diff'])}"
                            if v.get("token_matched_logit_diff") is not None
                            else ""
                        ),
                        v.get("mde"),
                        "logit",
                    ),
                )
            )
        for g, r in s.get("impact_ratio", {}).items():
            rows.append(
                Row(
                    battery,
                    f"impact ratio {g}",
                    r,
                    None,
                    None,
                    None,
                    0.8,
                    0.4,
                    "ratio",
                    "selection rate / best group; centre line is the four-fifths reference",
                )
            )
        for g, sd in s.get("within_group_sd", {}).items():
            rows.append(
                Row(
                    battery,
                    f"within-group spread {g}",
                    sd,
                    None,
                    None,
                    None,
                    0.0,
                    1.0,
                    "logit sd",
                    "spread of per-name means; large = the name matters even if the group mean does not",
                )
            )
        mass = s.get("answer_mass")
        if mass:
            lo, hi = min(mass.values()), max(mass.values())
            rows.append(
                Row(
                    battery,
                    "yes/no answer mass by group",
                    (lo + hi) / 2,
                    (lo, hi),
                    None,
                    None,
                    0.5,
                    0.5,
                    "prob",
                    f"range across groups [{_f(lo, 2)}, {_f(hi, 2)}]; uneven mass means hedging or refusal differs by group",
                )
            )
        if "spearman_tokens_vs_logit" in s:
            rows.append(
                Row(
                    battery,
                    "rank corr(name token count, logit P(yes))",
                    s["spearman_tokens_vs_logit"],
                    None,
                    None,
                    None,
                    0.0,
                    1.0,
                    "rho",
                    "tokenisation covariate; far from 0 = token length is doing work",
                )
            )
    elif battery == "chatiat":
        for tid, t in s.get("tests", {}).items():
            extra = f"Bai bias {_f(t['bias_bai'], 2)}; positive = stereotype-consistent"
            if "position_log_odds" in t:
                extra += f"; order effect {_f(t['position_log_odds'], 2)}; answer mass {_f(t.get('answer_mass'), 2)}"
            rows.append(
                Row(
                    battery,
                    f"{tid} ({t['group_a']} vs {t['group_b']})",
                    t["stereo_log_odds"],
                    tuple(t["ci"]),
                    t["n_words"],
                    t["sign_flip_p"],
                    0.0,
                    3.0,
                    "log-odds",
                    _with_mde(extra, t.get("mde"), "log-odds"),
                )
            )
    elif battery == "bbq":
        for cat, c in s.get("categories", {}).items():
            for cond, v in c.items():
                rows.append(
                    Row(
                        battery,
                        f"{cat} ({cond})",
                        v["bias_score"],
                        tuple(v["ci"]),
                        v["n"],
                        None,
                        0.0,
                        1.0,
                        "score",
                        f"accuracy {_f(v['accuracy'], 2)}",
                    )
                )
        if "predicted_letter_share" in s:
            pred = s["predicted_letter_share"]
            gold = s.get("gold_letter_share", {})
            rows.append(
                Row(
                    battery,
                    "option-position check: share of A/B/C predicted",
                    max(pred.values()),
                    None,
                    None,
                    None,
                    1 / 3,
                    2 / 3,
                    "share",
                    f"predicted A/B/C {_f(pred.get('A'), 2)}/{_f(pred.get('B'), 2)}/{_f(pred.get('C'), 2)} vs gold {_f(gold.get('A'), 2)}/{_f(gold.get('B'), 2)}/{_f(gold.get('C'), 2)}; answer mass {_f(s.get('answer_mass_mean'), 2)}",
                )
            )
    elif battery == "crows":
        o = s.get("overall", {})
        rows.append(
            Row(
                battery,
                "prefers sent_more (all pairs)",
                o.get("prefers_more_rate"),
                tuple(o.get("rate_ci", ())) or None,
                o.get("n"),
                o.get("sign_flip_p"),
                0.5,
                0.5,
                "rate",
                "0.5 = no preference",
            )
        )
        for direction, v in s.get("by_direction", {}).items():
            what = (
                "sent_more is the stereotypical sentence"
                if direction == "stereo"
                else "sent_more is the anti-stereotypical sentence"
            )
            rows.append(
                Row(
                    battery,
                    f"direction {direction}",
                    v["prefers_more_rate"],
                    tuple(v["rate_ci"]),
                    v["n"],
                    v["sign_flip_p"],
                    0.5,
                    0.5,
                    "rate",
                    what,
                )
            )
        for bt, v in s.get("by_bias_type", {}).items():
            rows.append(
                Row(
                    battery,
                    f"{bt}",
                    v["prefers_more_rate"],
                    tuple(v["rate_ci"]),
                    v["n"],
                    v["sign_flip_p"],
                    0.5,
                    0.5,
                    "rate",
                    "",
                )
            )
    elif battery == "effort":
        for tid, t in s.get("tests", {}).items():
            rows.append(
                Row(
                    battery,
                    tid,
                    t["asymmetry_tokens"],
                    tuple(t["ci"]),
                    t["n_lists"],
                    t["sign_flip_p"],
                    0.0,
                    max(50.0, abs(t["asymmetry_tokens"]) * 2),
                    "tokens",
                    f"truncated {_f(t['truncated_rate'], 2)}",
                )
            )
    elif battery == "contamination":
        for src in ("crows", "control"):
            if src in s:
                rows.append(
                    Row(
                        battery,
                        f"guided completion match ({src})",
                        s[src]["match_rate"],
                        tuple(s[src]["ci"]),
                        s[src]["n"],
                        None,
                        0.0,
                        1.0,
                        "rate",
                        "",
                    )
                )
        rows.append(
            Row(
                battery,
                "ratio crows / control",
                s.get("ratio_crows_over_control"),
                None,
                None,
                None,
                1.0,
                3.0,
                "ratio",
                "well above 1 suggests memorisation; at or below 1 it says nothing, because the fresh controls are formulaic template sentences that are easier to complete",
            )
        )
    elif battery == "seat":
        for tid, t in s.get("tests", {}).items():
            rows.append(
                Row(
                    battery,
                    f"{tid}",
                    t["effect_size_d"],
                    tuple(t["d_ci"]),
                    sum(t["n_targets"]),
                    t["perm_p_two_sided"],
                    0.0,
                    2.0,
                    "d",
                    t["stereotypical_direction"],
                )
            )
    elif battery == "retrieval":
        for k, v in s.get("contrasts", {}).items():
            rows.append(
                Row(
                    battery,
                    f"P({k.replace('_vs_', ' outranks ')})",
                    v["p_first_outranks_second"],
                    tuple(v["ci"]),
                    v["n_queries"],
                    v["sign_flip_p"],
                    0.5,
                    0.5,
                    "prob",
                    "0.5 = balance",
                )
            )
        for g, r in s.get("mean_rank", {}).items():
            rows.append(
                Row(
                    battery,
                    f"mean rank {g}",
                    r,
                    None,
                    None,
                    None,
                    (s.get("n_candidates_per_query", 36) + 1) / 2,
                    (s.get("n_candidates_per_query", 36) - 1) / 2,
                    "rank",
                    "lower is better",
                )
            )
    elif battery == "ctf":
        for axis, a in s.get("axes", {}).items():
            rows.append(
                Row(
                    battery,
                    f"{axis}: score gap vs reference",
                    a["gap"],
                    tuple(a["ci"]),
                    a["n_terms"],
                    None,
                    0.0,
                    0.5,
                    "prob",
                    f"flag rate {_f(a['flag_rate'], 2)}, worst term {a['worst_term']}",
                )
            )
        if s.get("reference_terms"):
            vals = list(s["reference_terms"].values())
            rows.append(
                Row(
                    battery,
                    "reference terms: mean score spread",
                    (min(vals) + max(vals)) / 2,
                    (min(vals), max(vals)),
                    None,
                    None,
                    0.0,
                    0.5,
                    "prob",
                    "range across the four neutral reference terms; the baseline's own spread",
                )
            )
    return _finish_rows(rows)


def build_card(subject_info: dict, run_info: dict, results: dict, nulls: dict | None = None) -> dict:
    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "tool": {"name": "stereotype-audit", "version": __version__},
        "subject": subject_info,
        "run": run_info,
        "batteries": results,
        "nulls": nulls or {},
    }


def sanitize(obj):
    """Make a structure strictly JSON: NaN and infinities become null, numpy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return sanitize(obj.item())
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        return sanitize(obj.tolist())
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(card: dict, path: Path) -> None:
    path.write_text(
        json.dumps(sanitize(card), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _json_default(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# ----- Markdown -------------------------------------------------------------


def render_markdown(card: dict) -> str:
    subj = card["subject"]
    run = card["run"]
    out = [f"# Stereotype Card: `{subj['model_id']}`", ""]
    rev = subj.get("resolved_revision") or subj.get("revision") or "unrecorded"
    out.append(
        f"Family: **{subj['family']}** · revision `{rev}` · dtype {subj.get('dtype')} · quantization {subj.get('quantization') or 'none'} · device {subj.get('device')}"
    )
    out.append(
        f"Run: {run.get('started')} → {run.get('finished')} · seed {run.get('seed')} · stereotype-audit {card['tool']['version']} · schema {card['schema_version']}"
    )
    if subj.get("gpu"):
        out.append(
            f"Hardware: {subj['gpu']} ({subj.get('vram_total_gb')} GB, peak {subj.get('vram_peak_gb')} GB)"
        )
    out.append("")
    out.append(
        "Every number is an estimate from this run with a 95 % bootstrap interval; an interval that contains the centre value is not evidence of balance, it is evidence that this run could not tell. Wording is descriptive: the model's measurements lean one way or the other; nothing here says why."
    )
    out.append("")
    for bid, res in card["batteries"].items():
        s = res["summary"]
        tag = " (experimental)" if res.get("experimental") else ""
        out.append(f"## {bid}{tag}")
        out.append("")
        out.append(f"{res.get('description', '')}  ")
        out.append(
            f"Items: {res['n_items']} · time {res['timing_s']} s · sources: {'; '.join(res.get('sources', []))}"
        )
        out.append("")
        rows = highlights(bid, s)
        if rows:
            out.append("| measure | estimate | 95 % CI | n | p | p (Holm) | note |")
            out.append("|---|---:|---|---:|---:|---:|---|")
            for r in rows:
                out.append(
                    f"| {r.label} | {_f(r.value)} {r.unit} | {_ci(r.ci)} | {r.n if r.n is not None else ''} | {_f(r.p, 4) if r.p is not None else ''} | {_f(r.p_adj, 4) if r.p_adj is not None else ''} | {r.note} |"
                )
            out.append("")
        for note in res.get("notes", []):
            out.append(f"- {note}")
        if bid in card.get("nulls", {}):
            out.append("")
            out.append("Shuffled-cue null run (same probes, cue labels shuffled):")
            for r in highlights(bid, card["nulls"][bid]["summary"])[:6]:
                out.append(f"- {r.label}: {_f(r.value)} {_ci(r.ci)}")
        out.append("")
    return "\n".join(out)


# ----- HTML -----------------------------------------------------------------

_CSS = """
:root{--bg:#f7f5f0;--fg:#1b1a17;--muted:#6b675e;--line:#d9d4c7;--pos:#a8321f;--neg:#1f5fa8;--ci:#8a8578}
@media (prefers-color-scheme: dark){:root{--bg:#171613;--fg:#ece8df;--muted:#a39d8f;--line:#3b3831;--pos:#e2664f;--neg:#5c9be0;--ci:#8a8578}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 Georgia,'Times New Roman',serif}
main{max-width:1280px;margin:0 auto;padding:32px 20px 64px}
.scroll{overflow-x:auto;max-width:100%}
td:last-child{min-width:220px;font-size:13px;color:var(--muted)}
h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:36px 0 6px;border-bottom:1px solid var(--line);padding-bottom:4px}
.meta{color:var(--muted);font-size:13px}code{font:13px ui-monospace,Menlo,Consolas,monospace}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
svg{display:block}details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}
pre{white-space:pre-wrap;font:12px ui-monospace,Menlo,Consolas,monospace;background:transparent;border:1px solid var(--line);padding:8px;overflow-x:auto}
.warn{border-left:3px solid var(--pos);padding-left:10px;color:var(--muted)}
"""


def _meter(r: Row) -> str:
    w, h = 220, 22
    lo_axis, hi_axis = r.center - r.span, r.center + r.span

    def x(v: float) -> float:
        if v is None or not math.isfinite(v):
            return w / 2
        v = min(max(v, lo_axis), hi_axis)
        return (v - lo_axis) / (hi_axis - lo_axis) * (w - 12) + 6

    mid = x(r.center)
    parts = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(r.label)} {_f(r.value)}">'
    ]
    parts.append(
        f'<line x1="6" y1="{h / 2}" x2="{w - 6}" y2="{h / 2}" stroke="var(--line)" stroke-width="1"/>'
    )
    parts.append(f'<line x1="{mid}" y1="3" x2="{mid}" y2="{h - 3}" stroke="var(--muted)" stroke-width="1"/>')
    if r.ci:
        parts.append(
            f'<line x1="{x(r.ci[0])}" y1="{h / 2}" x2="{x(r.ci[1])}" y2="{h / 2}" stroke="var(--ci)" stroke-width="4" stroke-linecap="round" opacity="0.8"/>'
        )
    if r.value is not None and math.isfinite(r.value):
        color = "var(--pos)" if r.value >= r.center else "var(--neg)"
        parts.append(f'<circle cx="{x(r.value)}" cy="{h / 2}" r="5" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(card: dict) -> str:
    subj = card["subject"]
    run = card["run"]
    e = html.escape
    out = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{e(subj['model_id'])} stereotype card</title>",
        f"<style>{_CSS}</style>",
        "<main>",
    ]
    out.append(f"<h1>Stereotype Card</h1><div><code>{e(subj['model_id'])}</code></div>")
    rev = subj.get("resolved_revision") or subj.get("revision") or "unrecorded"
    out.append(
        f"<p class='meta'>family {e(str(subj['family']))} · revision <code>{e(str(rev))}</code> · dtype {e(str(subj.get('dtype')))} · quantization {e(str(subj.get('quantization') or 'none'))} · device {e(str(subj.get('device')))}"
        f" · run {e(str(run.get('started')))} → {e(str(run.get('finished')))} · seed {e(str(run.get('seed')))} · stereotype-audit {e(card['tool']['version'])}</p>"
    )
    if subj.get("gpu"):
        out.append(
            f"<p class='meta'>{e(str(subj['gpu']))}, {subj.get('vram_total_gb')} GB, peak {subj.get('vram_peak_gb')} GB</p>"
        )
    out.append(
        "<p class='warn'>Every number is an estimate from this run with a 95 % bootstrap interval. An interval that contains the centre line is not evidence of balance; it means this run could not tell. The wording is descriptive: the model's measurements lean one way or the other, and nothing here says why.</p>"
    )
    for bid, res in card["batteries"].items():
        s = res["summary"]
        tag = " <span class='meta'>(experimental)</span>" if res.get("experimental") else ""
        out.append(f"<h2>{e(bid)}{tag}</h2>")
        out.append(
            f"<p>{e(res.get('description', ''))}</p><p class='meta'>items {res['n_items']} · {res['timing_s']} s · {e('; '.join(res.get('sources', [])))}</p>"
        )
        rows = highlights(bid, s)
        if rows:
            out.append(
                "<div class='scroll'><table><thead><tr><th>measure</th><th>balance</th><th>estimate</th><th>95 % CI</th><th>n</th><th>p</th><th>p (Holm)</th><th>note</th></tr></thead><tbody>"
            )
            for r in rows:
                out.append(
                    f"<tr><td>{e(r.label)}</td><td>{_meter(r)}</td><td class='num'>{_f(r.value)} {e(r.unit)}</td>"
                    f"<td class='num'>{e(_ci(r.ci))}</td><td class='num'>{r.n if r.n is not None else ''}</td>"
                    f"<td class='num'>{_f(r.p, 4) if r.p is not None else ''}</td>"
                    f"<td class='num'>{_f(r.p_adj, 4) if r.p_adj is not None else ''}</td><td>{e(r.note)}</td></tr>"
                )
            out.append("</tbody></table></div>")
        if res.get("notes"):
            out.append("<ul>" + "".join(f"<li>{e(n)}</li>" for n in res["notes"]) + "</ul>")
        if bid in card.get("nulls", {}):
            out.append("<details><summary>Shuffled-cue null run</summary><ul>")
            for r in highlights(bid, card["nulls"][bid]["summary"])[:8]:
                out.append(f"<li>{e(r.label)}: {_f(r.value)} {e(_ci(r.ci))}</li>")
            out.append("</ul></details>")
        if res.get("probes_preview"):
            out.append(
                "<details><summary>Probe preview</summary><pre>"
                + e(json.dumps(res["probes_preview"], indent=1, ensure_ascii=False)[:3000])
                + "</pre></details>"
            )
        out.append(
            "<details><summary>Full summary JSON</summary><pre>"
            + e(json.dumps(sanitize(s), indent=1, ensure_ascii=False)[:20000])
            + "</pre></details>"
        )
    out.append("</main>")
    return "\n".join(out)


def render_terminal(card: dict) -> str:
    lines = [f"Stereotype Card - {card['subject']['model_id']} ({card['subject']['family']})"]
    for bid, res in card["batteries"].items():
        lines.append(f"\n[{bid}] {res['n_items']} items, {res['timing_s']} s")
        for r in highlights(bid, res["summary"])[:12]:
            p = f" p={_f(r.p, 3)}" if r.p is not None else ""
            lines.append(f"  {r.label[:60]:60s} {_f(r.value):>8} {r.unit:<8} {_ci(r.ci):<18}{p}")
    return "\n".join(lines)


def load_card(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
