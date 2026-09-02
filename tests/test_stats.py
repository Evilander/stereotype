import numpy as np
import pytest

from stereotype_audit import stats


def test_bootstrap_ci_covers_true_mean_most_of_the_time():
    rng = np.random.default_rng(1)
    covered = 0
    trials = 200
    for t in range(trials):
        sample = rng.normal(loc=0.3, scale=1.0, size=40)
        est = stats.bootstrap_ci(sample, n_boot=500, seed=t)
        if est.ci_low <= 0.3 <= est.ci_high:
            covered += 1
    assert covered / trials >= 0.90


def test_bootstrap_ci_is_deterministic_and_handles_edge_cases():
    a = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=3)
    b = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=3)
    assert a == b
    assert a.n == 4 and a.value == 2.5
    single = stats.bootstrap_ci([7.0])
    assert single.value == single.ci_low == single.ci_high == 7.0
    empty = stats.bootstrap_ci([])
    assert np.isnan(empty.value) and empty.n == 0
    with_nan = stats.bootstrap_ci([1.0, float("nan"), 3.0])
    assert with_nan.n == 2 and with_nan.value == 2.0


def test_cluster_bootstrap_widens_for_dependent_items():
    rng = np.random.default_rng(5)
    # six clusters with distinct means; items inside a cluster are near-identical
    groups = np.repeat(np.arange(6), 50)
    values = rng.normal(size=6)[groups] + rng.normal(scale=0.01, size=300)
    naive = stats.bootstrap_ci(values, seed=1)
    clustered = stats.cluster_bootstrap_ci(values, groups, seed=1)
    assert (clustered.ci_high - clustered.ci_low) > 3 * (naive.ci_high - naive.ci_low)


def test_weat_effect_size_matches_hand_computation():
    s_x = np.array([0.5, 0.7, 0.6])
    s_y = np.array([-0.1, 0.0, 0.1])
    pooled = np.concatenate([s_x, s_y])
    expected = (s_x.mean() - s_y.mean()) / pooled.std(ddof=1)
    assert stats.weat_effect_size(s_x, s_y) == pytest.approx(expected, abs=1e-9)
    assert stats.weat_effect_size(s_x, s_x) == 0.0


def test_permutation_p_small_for_clear_separation_and_large_for_none():
    rng = np.random.default_rng(2)
    x = rng.normal(1.0, 0.2, 30)
    y = rng.normal(0.0, 0.2, 30)
    one, two = stats.weat_permutation_p(x, y, n_perm=500, seed=0)
    assert one < 0.01 and two < 0.01
    one_null, two_null = stats.weat_permutation_p(y, rng.normal(0.0, 0.2, 30), n_perm=500, seed=0)
    assert two_null > 0.05


def test_sign_flip_p():
    diffs = np.full(40, 0.5) + np.random.default_rng(0).normal(scale=0.1, size=40)
    assert stats.sign_flip_p(diffs, n_perm=500) < 0.01
    null = np.random.default_rng(1).normal(scale=1.0, size=40)
    assert stats.sign_flip_p(null, n_perm=500) > 0.05


def test_holm_adjustment():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert np.isnan(adj["d"])


def test_impact_ratio_and_mde():
    ratios = stats.impact_ratio({"a": 0.5, "b": 0.4, "c": 0.25})
    assert ratios["a"] == 1.0 and ratios["b"] == pytest.approx(0.8) and ratios["c"] == pytest.approx(0.5)
    assert stats.min_detectable_effect(1.0, 100) == pytest.approx(0.2802, abs=1e-3)
    assert np.isnan(stats.min_detectable_effect(1.0, 0))


def test_cohens_d_and_helpers():
    assert stats.cohens_d_paired([1.0, 1.0, 1.0]) == 0.0 or np.isinf(stats.cohens_d_paired([1.0, 1.0, 1.0]))
    assert stats.cohens_d_paired([0.0, 0.0]) == 0.0
    assert stats.logit(0.5) == 0.0
    assert stats.log_sum_exp([np.log(0.25), np.log(0.25)]) == pytest.approx(np.log(0.5))
