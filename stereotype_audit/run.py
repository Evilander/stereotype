"""Orchestration: load a subject, run batteries, write raw measurements and the card."""

from __future__ import annotations

import datetime as dt
import json
import shlex
import sys
from pathlib import Path

from stereotype_audit.batteries.assoc import AssocBattery
from stereotype_audit.batteries.base import Battery, BatteryResult
from stereotype_audit.batteries.bbq import BBQBattery
from stereotype_audit.batteries.chatiat import ChatIATBattery
from stereotype_audit.batteries.contamination import ContaminationBattery
from stereotype_audit.batteries.crows import CrowsBattery
from stereotype_audit.batteries.ctf import CTFBattery
from stereotype_audit.batteries.decision import DecisionBattery
from stereotype_audit.batteries.effort import EffortBattery
from stereotype_audit.batteries.nameswap import NameSwapBattery
from stereotype_audit.batteries.retrieval import RetrievalBattery
from stereotype_audit.batteries.seat import SEATBattery
from stereotype_audit.card import build_card, render_html, render_markdown, sanitize, write_json

BATTERIES: dict[str, type[Battery]] = {
    b.id: b
    for b in (
        AssocBattery,
        DecisionBattery,
        NameSwapBattery,
        ChatIATBattery,
        BBQBattery,
        CrowsBattery,
        EffortBattery,
        ContaminationBattery,
        SEATBattery,
        RetrievalBattery,
        CTFBattery,
    )
}

DEFAULT_BATTERIES = {
    "lm": ["assoc", "decision", "nameswap", "chatiat", "bbq", "crows"],
    "embed": ["seat", "retrieval"],
    "clf": ["ctf"],
}

# bbq has no cue to shuffle (its stereotype labels come from the dataset), so it has no null run
NULL_CAPABLE = {"assoc", "decision", "nameswap", "chatiat", "crows", "seat", "retrieval", "ctf"}


def detect_family(model_id: str, revision: str | None = None) -> str:
    """embed if the repo ships sentence-transformers modules, clf if the architecture is a sequence classifier, else lm."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from transformers import AutoConfig

    try:
        hf_hub_download(model_id, "modules.json", revision=revision)
        return "embed"
    except EntryNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - offline or gated: fall through to config
        pass
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    archs = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]
    if any("forsequenceclassification" in a for a in archs):
        return "clf"
    return "lm"


def load_subject(model_id: str, family: str | None = None, **kwargs):
    family = family or detect_family(model_id, kwargs.get("revision"))
    if family == "lm":
        from stereotype_audit.subjects.causal_lm import CausalLMSubject

        return CausalLMSubject(
            model_id,
            revision=kwargs.get("revision"),
            dtype=kwargs.get("dtype", "auto"),
            quant=kwargs.get("quant"),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 16),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
    if family == "embed":
        from stereotype_audit.subjects.embedder import EmbedderSubject

        return EmbedderSubject(
            model_id,
            revision=kwargs.get("revision"),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 64),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
    if family == "clf":
        from stereotype_audit.subjects.classifier import ClassifierSubject

        return ClassifierSubject(
            model_id,
            revision=kwargs.get("revision"),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 64),
            label=kwargs.get("label"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
    raise ValueError(f"unknown family {family!r}")


def _result_record(res: BatteryResult, battery_cls: type[Battery]) -> dict:
    return {
        "description": battery_cls.description,
        "sources": list(battery_cls.sources),
        "experimental": res.experimental,
        "n_items": res.n_items,
        "timing_s": res.timing_s,
        "config": res.config,
        "summary": res.summary,
        "probes_preview": res.probes_preview,
        "notes": res.notes,
    }


def audit(
    model_id: str,
    out_dir: str | Path,
    batteries: list[str] | None = None,
    family: str | None = None,
    n: int | None = None,
    seed: int = 0,
    nulls: bool = False,
    battery_cfg: dict | None = None,
    subject_kwargs: dict | None = None,
    log=print,
    resume: bool = False,
) -> dict:
    """Run batteries on a model and write the card. With `resume`, batteries whose summary already
    exists in `out_dir` are loaded instead of recomputed, so an interrupted run continues."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    subject = load_subject(model_id, family, **(subject_kwargs or {}))
    fam = subject.family
    chosen = batteries or DEFAULT_BATTERIES[fam]
    unknown = [b for b in chosen if b not in BATTERIES]
    if unknown:
        raise ValueError(f"unknown batteries: {unknown}; known: {sorted(BATTERIES)}")
    wrong = [b for b in chosen if BATTERIES[b].family != fam]
    if wrong:
        raise ValueError(f"batteries {wrong} do not apply to a {fam} subject")
    results: dict[str, dict] = {}
    null_results: dict[str, dict] = {}
    reused: list[str] = []
    cfg_all = battery_cfg or {}
    for bid in chosen:
        cls = BATTERIES[bid]
        cfg = dict(cfg_all.get(bid, {}))
        summary_path = out / f"{bid}.summary.json"
        if resume and summary_path.exists():
            results[bid] = json.loads(summary_path.read_text(encoding="utf-8"))
            reused.append(bid)
            log(f"[{bid}] reused {summary_path.name}")
        else:
            log(f"[{bid}] running on {model_id} …")
            res = cls(n=n, seed=seed, **cfg).run(subject)
            log(f"[{bid}] {res.n_items} items in {res.timing_s} s")
            res.raw.to_parquet(out / f"{bid}.raw.parquet", index=False)
            results[bid] = _result_record(res, cls)
            summary_path.write_text(
                json.dumps(sanitize(results[bid]), indent=2, allow_nan=False, ensure_ascii=False),
                encoding="utf-8",
            )
        if nulls and bid in NULL_CAPABLE:
            null_path = out / f"{bid}.null.summary.json"
            if resume and null_path.exists():
                null_results[bid] = json.loads(null_path.read_text(encoding="utf-8"))
                log(f"[{bid}] reused {null_path.name}")
            else:
                log(f"[{bid}] shuffled-cue null …")
                nres = cls(n=n, seed=seed + 1000, null=True, **cfg).run(subject)
                null_results[bid] = _result_record(nres, cls)
                nres.raw.to_parquet(out / f"{bid}.null.raw.parquet", index=False)
                null_path.write_text(
                    json.dumps(sanitize(null_results[bid]), indent=2, allow_nan=False, ensure_ascii=False),
                    encoding="utf-8",
                )
    finished = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    info = subject.info().as_dict()
    run_info = {
        "started": started,
        "finished": finished,
        "seed": seed,
        "n": n,
        "batteries": chosen,
        "reused_from_earlier_run": reused,
        "nulls": nulls,
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "python": sys.version.split()[0],
    }
    card = build_card(info, run_info, results, null_results)
    write_json(card, out / "card.json")
    (out / "card.md").write_text(render_markdown(card), encoding="utf-8")
    (out / "card.html").write_text(render_html(card), encoding="utf-8")
    log(f"card written to {out / 'card.json'}")
    return card
