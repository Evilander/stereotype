"""Every battery runs end to end on the tiny fixture model with a small n and produces schema-shaped output."""

import numpy as np
import pytest

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def smol():
    from stereotype_audit.subjects.causal_lm import CausalLMSubject

    try:
        return CausalLMSubject(SMOL, device="cpu", batch_size=32)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"fixture model unavailable: {err}")


def _check(res, battery_id):
    assert res.battery == battery_id
    assert res.n_items > 0 and len(res.raw) == res.n_items
    assert isinstance(res.summary, dict) and res.timing_s >= 0
    assert res.probes_preview


def test_assoc_and_null(smol):
    from stereotype_audit.batteries.assoc import AssocBattery

    res = AssocBattery(n=6, tests=["gender_career_names"]).run(smol)
    _check(res, "assoc")
    t = res.summary["tests"]["gender_career_names"]
    assert np.isfinite(t["effect_size_d"]) and len(t["contrast_ci"]) == 2
    assert set(res.raw["attr_set"]) == {"A", "B"}
    again = AssocBattery(n=6, tests=["gender_career_names"]).run(smol)
    assert again.summary["tests"]["gender_career_names"]["effect_size_d"] == pytest.approx(
        t["effect_size_d"], abs=1e-6
    )
    null = AssocBattery(n=6, tests=["gender_career_names"], null=True).run(smol)
    assert "effect_size_d" in null.summary["tests"]["gender_career_names"]


def test_decision(smol):
    from stereotype_audit.batteries.decision import DecisionBattery

    res = DecisionBattery(n=140).run(smol)
    _check(res, "decision")
    s = res.summary
    assert s["baseline_rows_present"] == s["n_questions"] >= 1
    assert "Black" in s["attributes"]["race"]["levels"]
    assert 0.0 <= s["overall_p_yes"]["value"] <= 1.0
    assert 'either "yes" or "no"' in s["rendered_prompt_example"]


def test_nameswap(smol):
    from stereotype_audit.batteries.nameswap import NameSwapBattery

    res = NameSwapBattery(n=36).run(smol)
    _check(res, "nameswap")
    assert set(res.summary["selection_rate"]) == {"white_female", "white_male", "black_female", "black_male"}
    assert max(res.summary["impact_ratio"].values()) == 1.0


def test_chatiat(smol):
    from stereotype_audit.batteries.chatiat import ChatIATBattery

    res = ChatIATBattery(n=4, pairs_per_test=2).run(smol)
    _check(res, "chatiat")
    for t in res.summary["tests"].values():
        assert -1.0 <= t["bias_bai"] <= 1.0
        assert 0.0 <= t["stereotypical_rate"] <= 1.0


def test_bbq(smol):
    from stereotype_audit.batteries.bbq import BBQBattery

    res = BBQBattery(per_category=8, categories=["Age"]).run(smol)
    _check(res, "bbq")
    age = res.summary["categories"]["Age"]
    assert set(age) == {"ambig", "disambig"}
    assert -1.0 <= age["disambig"]["bias_score"] <= 1.0


def test_crows(smol):
    from stereotype_audit.batteries.crows import CrowsBattery

    res = CrowsBattery(n=30).run(smol)
    _check(res, "crows")
    assert 0.0 <= res.summary["overall"]["prefers_more_rate"] <= 1.0


def test_contamination(smol):
    from stereotype_audit.batteries.contamination import ContaminationBattery

    res = ContaminationBattery(n=10).run(smol)
    _check(res, "contamination")
    assert res.summary["crows"]["n"] == 10 and res.summary["control"]["n"] == 10


def test_effort_refuses_without_thinking(smol):
    from stereotype_audit.batteries.effort import EffortBattery

    with pytest.raises(RuntimeError):
        EffortBattery(n=1).run(smol)
