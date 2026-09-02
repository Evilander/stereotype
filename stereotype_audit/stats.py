"""Statistics for paired contrasts: bootstrap intervals, effect sizes, permutation tests.

Everything here is deterministic given a seed and works on plain numpy arrays so
batteries can stay free of model code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a percentile-bootstrap interval."""

    value: float
    ci_low: float
    ci_high: float
    n: int
    method: str

    def as_dict(self) -> dict:
        return asdict(self)

    def contains_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def bootstrap_ci(
    values,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "percentile-bootstrap",
) -> Estimate:
    """Percentile bootstrap of `stat` over independent items."""
    arr = _finite(values)
    n = arr.size
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, method)
    point = float(stat(arr))
    if n == 1:
        return Estimate(point, point, point, 1, method)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(arr[row]) for row in idx], dtype=np.float64)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), n, method)


def cluster_bootstrap_ci(
    values,
    groups,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Bootstrap that resamples whole clusters (for example templates) rather than items.

    Items inside a cluster share a template and are not independent; resampling
    clusters keeps that dependence intact.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    grp = np.asarray(groups).ravel()
    keep = np.isfinite(arr)
    arr, grp = arr[keep], grp[keep]
    labels = np.unique(grp)
    if arr.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, "cluster-bootstrap")
    point = float(stat(arr))
    if labels.size == 1:
        return bootstrap_ci(arr, stat, n_boot, alpha, seed, method="percentile-bootstrap")
    members = [arr[grp == g] for g in labels]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, labels.size, size=labels.size)
        boots[b] = stat(np.concatenate([members[i] for i in pick]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # n is the number of clusters: that is the effective sample size the interval rests on
    return Estimate(point, float(lo), float(hi), int(labels.size), "cluster-bootstrap")


def cohens_d_paired(diffs) -> float:
    """Standardised mean of paired differences (mean / sd, ddof=1)."""
    arr = _finite(diffs)
    if arr.size < 2:
        return float("nan")
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return 0.0 if float(np.mean(arr)) == 0.0 else float("inf") * np.sign(float(np.mean(arr)))
    return float(np.mean(arr) / sd)


def weat_effect_size(s_x, s_y) -> float:
    """WEAT effect size (Caliskan et al. 2017): (mean s_x - mean s_y) / std(s_x ∪ s_y, ddof=1)."""
    sx, sy = _finite(s_x), _finite(s_y)
    pooled = np.concatenate([sx, sy])
    if sx.size == 0 or sy.size == 0 or pooled.size < 2:
        return float("nan")
    sd = float(np.std(pooled, ddof=1))
    if sd == 0.0:
        return 0.0
    return float((np.mean(sx) - np.mean(sy)) / sd)


def weat_permutation_p(s_x, s_y, n_perm: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Permutation test over target labels.

    Returns (one_sided, two_sided). One-sided is the WEAT convention:
    P[(mean s_x' - mean s_y') > observed] under random relabelling.
    """
    sx, sy = _finite(s_x), _finite(s_y)
    pooled = np.concatenate([sx, sy])
    nx = sx.size
    if nx == 0 or sy.size == 0:
        return float("nan"), float("nan")
    observed = float(np.mean(sx) - np.mean(sy))
    rng = np.random.default_rng(seed)
    count_one = 0
    count_two = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = float(np.mean(perm[:nx]) - np.mean(perm[nx:]))
        if stat > observed:
            count_one += 1
        if abs(stat) >= abs(observed):
            count_two += 1
    return (count_one + 1) / (n_perm + 1), (count_two + 1) / (n_perm + 1)


def sign_flip_p(diffs, n_perm: int = 2000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for a mean of paired differences."""
    arr = _finite(diffs)
    if arr.size == 0:
        return float("nan")
    observed = abs(float(np.mean(arr)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, arr.size))
    stats = np.abs((signs * arr).mean(axis=1))
    return float((np.sum(stats >= observed) + 1) / (n_perm + 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment; NaNs are passed through untouched."""
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    out = {k: v for k, v in pvalues.items() if not np.isfinite(v)}
    m = len(items)
    if m == 0:
        return out
    items.sort(key=lambda kv: kv[1])
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def impact_ratio(rates: dict[str, float]) -> dict[str, float]:
    """Selection rate of each group divided by the highest group rate (NYC Local Law 144 usage)."""
    finite = {k: v for k, v in rates.items() if np.isfinite(v)}
    if not finite:
        return {k: float("nan") for k in rates}
    top = max(finite.values())
    if top <= 0:
        return {k: float("nan") for k in rates}
    return {k: (v / top if np.isfinite(v) else float("nan")) for k, v in rates.items()}


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest mean paired difference a two-sided test at `alpha` would detect with `power`.

    Uses the normal approximation (z_{1-alpha/2} + z_{power}) * sd / sqrt(n).
    Only alpha=0.05 and power=0.8 are tabulated here.
    """
    if n <= 0 or not np.isfinite(sd):
        return float("nan")
    if alpha != 0.05 or power != 0.8:
        raise ValueError("only alpha=0.05, power=0.8 are supported")
    return float((Z_975 + Z_80) * sd / np.sqrt(n))


def logit(p: float, eps: float = 1e-9) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def log_sum_exp(values) -> float:
    arr = _finite(values)
    if arr.size == 0:
        return float("-inf")
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))
