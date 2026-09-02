import json

from stereotype_audit.card import (
    build_card,
    highlights,
    render_html,
    render_markdown,
    render_terminal,
    write_json,
)


def _fake_card():
    subject = {
        "model_id": "org/model",
        "family": "lm",
        "dtype": "bfloat16",
        "quantization": None,
        "device": "cuda",
    }
    run = {"started": "2026-09-01T00:00:00+00:00", "finished": "2026-09-01T00:10:00+00:00", "seed": 0}
    results = {
        "nameswap": {
            "description": "d",
            "sources": ["s"],
            "experimental": False,
            "n_items": 432,
            "timing_s": 1.0,
            "config": {},
            "summary": {
                "selection_rate": {"white_female": 0.6, "black_female": 0.5},
                "impact_ratio": {"white_female": 1.0, "black_female": 0.8333},
                "contrasts": {
                    "white_female_minus_black_female": {
                        "logit_diff": 0.4,
                        "ci": [0.1, 0.7],
                        "n_templates": 12,
                        "sign_flip_p": 0.01,
                        "mde": 0.2,
                    }
                },
            },
            "probes_preview": [{"prompt": "x"}],
            "notes": ["note"],
        },
        "assoc": {
            "description": "d",
            "sources": [],
            "experimental": False,
            "n_items": 10,
            "timing_s": 1.0,
            "config": {},
            "summary": {
                "tests": {
                    "t": {
                        "X": "x",
                        "Y": "y",
                        "A": "a",
                        "B": "b",
                        "effect_size_d": float("nan"),
                        "n_prompts": 10,
                        "contrast_ci": [0, 1],
                    }
                }
            },
            "probes_preview": [],
            "notes": [],
        },
    }
    return build_card(subject, run, results, {"nameswap": results["nameswap"]})


def test_highlights_and_renderers(tmp_path):
    card = _fake_card()
    rows = highlights("nameswap", card["batteries"]["nameswap"]["summary"])
    assert any(r.label.startswith("white_female") for r in rows)
    md = render_markdown(card)
    assert "Stereotype Card" in md and "white_female − black_female" in md and "Shuffled-cue null" in md
    html = render_html(card)
    assert (
        "<svg" in html and "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html
    )
    assert "n/a" in html  # NaN effect size renders as n/a, never as NaN
    term = render_terminal(card)
    assert "[nameswap]" in term
    write_json(card, tmp_path / "card.json")
    loaded = json.loads((tmp_path / "card.json").read_text(encoding="utf-8"))
    assert loaded["batteries"]["assoc"]["summary"]["tests"]["t"]["effect_size_d"] is None
