import numpy as np
import pytest


def test_crows_loads_from_cache_or_network():
    from stereotype_audit.data import load_crows_pairs

    try:
        df = load_crows_pairs()
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"CrowS-Pairs unavailable: {err}")
    assert len(df) == 1508
    assert set(df["stereo_antistereo"]) == {"stereo", "antistereo"}
    assert df["bias_type"].nunique() == 9


def test_discrim_eval_baseline_present():
    from stereotype_audit.data import discrim_eval_baseline, load_discrim_eval

    try:
        df = load_discrim_eval("explicit")
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"Discrim-Eval unavailable: {err}")
    assert len(df) == 9450
    base = discrim_eval_baseline(df)
    assert len(base) == 70


def test_bbq_label_components():
    from stereotype_audit.data import _nationality_from_text, bbq_label_components

    assert {"f", "black"} <= bbq_label_components("F-Black")
    assert {"lowses", "hispanic"} <= bbq_label_components("lowSES-Hispanic")
    assert "m" in bbq_label_components("man") and "f" in bbq_label_components("woman")
    assert "trans" not in bbq_label_components("nontrans")
    assert "old" not in bbq_label_components("nonOld")
    assert _nationality_from_text("The South American man") == "southamerican"
    assert _nationality_from_text("The British friend") == "british"


def test_bbq_matcher_singles_out_one_answer_on_clean_categories():
    from stereotype_audit.data import bbq_is_biased_answer, bbq_n_biased_options, bbq_unknown_index, load_bbq

    try:
        df = load_bbq(
            ["Religion", "Age", "Gender_identity", "Nationality", "Race_ethnicity"], per_category=120, seed=1
        )
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"BBQ unavailable: {err}")
    for cat, g in df.groupby("category"):
        rate = np.mean([bbq_n_biased_options(r) == 1 for _, r in g.iterrows()])
        floor = 0.85 if cat == "Gender_identity" else 0.98
        assert rate >= floor, f"{cat}: only {rate:.2f} of items single out one biased answer"
    for _, row in df.iterrows():
        u = bbq_unknown_index(row)
        assert bbq_is_biased_answer(row, u) is False
