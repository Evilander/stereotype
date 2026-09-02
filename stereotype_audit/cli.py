"""Command line: stereotype audit | card | compare | probes | batteries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stereotype_audit import __version__


def _parse_cfg(items: list[str] | None) -> dict:
    """--cfg battery.key=value (value parsed as JSON when possible)."""
    out: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        battery, _, field = key.partition(".")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out.setdefault(battery, {})[field] = value
    return out


def cmd_audit(args) -> int:
    from stereotype_audit.run import audit

    subject_kwargs = {
        "revision": args.revision,
        "dtype": args.dtype,
        "quant": args.quant,
        "device": args.device,
        "batch_size": args.batch_size,
        "trust_remote_code": args.trust_remote_code,
        "label": args.label,
    }
    out_dir = args.out or Path("runs") / args.model.replace("/", "__")
    card = audit(
        args.model,
        out_dir,
        batteries=args.batteries.split(",") if args.batteries else None,
        family=args.family,
        n=args.n,
        seed=args.seed,
        nulls=args.nulls,
        resume=args.resume,
        battery_cfg=_parse_cfg(args.cfg),
        subject_kwargs=subject_kwargs,
        log=lambda m: print(m, file=sys.stderr),
    )
    from stereotype_audit.card import render_terminal

    print(render_terminal(card))
    return 0


def cmd_card(args) -> int:
    from stereotype_audit.card import load_card, render_html, render_markdown, render_terminal

    card = load_card(Path(args.run_dir) / "card.json")
    if args.format == "md":
        print(render_markdown(card))
    elif args.format == "html":
        print(render_html(card))
    else:
        print(render_terminal(card))
    return 0


def cmd_rerender(args) -> int:
    from stereotype_audit.card import load_card, render_html, render_markdown

    run_dir = Path(args.run_dir)
    card = load_card(run_dir / "card.json")
    (run_dir / "card.md").write_text(render_markdown(card), encoding="utf-8")
    (run_dir / "card.html").write_text(render_html(card), encoding="utf-8")
    print(f"re-rendered {run_dir / 'card.md'} and card.html")
    return 0


def cmd_compare(args) -> int:
    from stereotype_audit.card import _ci, _f, highlights, load_card

    cards = [load_card(Path(d) / "card.json") for d in args.run_dirs]
    ids = [c["subject"]["model_id"] for c in cards]
    print("| measure | " + " | ".join(f"`{i}`" for i in ids) + " |")
    print("|---|" + "---:|" * len(ids))
    batteries = []
    for c in cards:
        for b in c["batteries"]:
            if b not in batteries:
                batteries.append(b)
    for b in batteries:
        rows_per = [
            {r.label: r for r in highlights(b, c["batteries"][b]["summary"])} if b in c["batteries"] else {}
            for c in cards
        ]
        labels = []
        for rp in rows_per:
            for label in rp:
                if label not in labels:
                    labels.append(label)
        for label in labels:
            cells = []
            for rp in rows_per:
                r = rp.get(label)
                cells.append(f"{_f(r.value)} {_ci(r.ci)}" if r else "")
            print(f"| {b}: {label} | " + " | ".join(cells) + " |")
    return 0


def cmd_probes(args) -> int:
    from stereotype_audit.batteries.assoc import association_tests
    from stereotype_audit.pairs import fill, load_cues, load_templates

    if args.battery == "assoc":
        tpl = load_templates("assoc")["templates"]
        count = 0
        for test in association_tests():
            for cue in test["X"][1] + test["Y"][1]:
                for t in tpl:
                    for attr in test["A"][1][:1] + test["B"][1][:1]:
                        print(
                            f"{test['id']}\t{fill(t['text'], cue=cue)}{fill(t['continuation'], attr=attr, article='a')}"
                        )
                        count += 1
        print(f"# {count} example probes (one attribute per set shown)", file=sys.stderr)
    elif args.battery == "nameswap":
        spec = load_templates("nameswap")
        names = load_cues("names_bm2004")["groups"]
        for t in spec["templates"]:
            for group, members in names.items():
                print(f"{t['id']}\t{group}\t{fill(t['text'], name=members[0])} {spec['instruction']}")
    elif args.battery == "ctf":
        spec = load_templates("ctf")
        terms = load_cues("identity_terms")
        for t in spec["templates"]:
            for axis, lst in terms["axes"].items():
                print(f"{t['id']}\t{axis}\t{fill(t['text'], term=lst[0])}")
    else:
        print(json.dumps(load_templates(args.battery), indent=2))
    return 0


def cmd_batteries(args) -> int:
    from stereotype_audit.run import BATTERIES, DEFAULT_BATTERIES

    for bid, cls in BATTERIES.items():
        tag = " (experimental)" if cls.experimental else ""
        print(f"{bid:14s} {cls.family:6s}{tag}\n    {cls.description}")
    print("\ndefaults:", json.dumps(DEFAULT_BATTERIES))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stereotype", description="Stereo-pair bias auditor for Hugging Face text models."
    )
    p.add_argument("--version", action="version", version=f"stereotype-audit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="run batteries on a model and write a Stereotype Card")
    a.add_argument("model", help="Hugging Face model id or local path")
    a.add_argument("--family", choices=["lm", "embed", "clf"], help="override auto-detection")
    a.add_argument("--batteries", help="comma-separated battery ids (default: family defaults)")
    a.add_argument("--out", help="output directory (default runs/<model>)")
    a.add_argument("--n", type=int, help="cap items per battery (quick runs)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--nulls", action="store_true", help="also run shuffled-cue null versions")
    a.add_argument(
        "--resume", action="store_true", help="reuse battery summaries already in the output directory"
    )
    a.add_argument("--cfg", action="append", help="battery option, e.g. --cfg bbq.per_category=200")
    a.add_argument("--revision")
    a.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    a.add_argument("--quant", choices=["4bit"])
    a.add_argument("--device")
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--label", help="classifier target label")
    a.add_argument("--trust-remote-code", action="store_true")
    a.set_defaults(func=cmd_audit)

    c = sub.add_parser("card", help="render an existing run")
    c.add_argument("run_dir")
    c.add_argument("--format", choices=["terminal", "md", "html"], default="terminal")
    c.set_defaults(func=cmd_card)

    rr = sub.add_parser("rerender", help="rewrite card.md and card.html from an existing card.json")
    rr.add_argument("run_dir")
    rr.set_defaults(func=cmd_rerender)

    cmp_ = sub.add_parser("compare", help="side-by-side highlights of several runs as Markdown")
    cmp_.add_argument("run_dirs", nargs="+")
    cmp_.set_defaults(func=cmd_compare)

    pr = sub.add_parser("probes", help="print the probes a battery uses")
    pr.add_argument("battery")
    pr.set_defaults(func=cmd_probes)

    b = sub.add_parser("batteries", help="list batteries")
    b.set_defaults(func=cmd_batteries)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; the card uses ordinary Unicode punctuation.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
