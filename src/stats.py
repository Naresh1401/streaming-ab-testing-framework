"""
Core statistics engine for streaming A/B experiments.

Implements frequentist tests, CUPED variance reduction, sequential testing,
and sample-size planning — all calibrated for typical streaming-platform metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats as sp_stats
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from typing import Literal


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Container for a single statistical test outcome."""
    test_name: str
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    lift_abs: float
    lift_rel_pct: float
    is_significant: bool
    power_achieved: float | None = None
    alpha: float = 0.05


@dataclass
class SequentialResult:
    """Result of an O'Brien-Fleming sequential check at a single look."""
    look_number: int
    total_looks: int
    nominal_alpha: float
    spending_boundary: float
    observed_z: float
    reject_null: bool
    cumulative_alpha_spent: float


@dataclass
class CUPEDResult:
    """Before / after comparison from CUPED adjustment."""
    raw_lift: float
    cuped_lift: float
    raw_se: float
    cuped_se: float
    variance_reduction_pct: float
    equivalent_extra_users_pct: float
    p_value: float
    ci_lower: float
    ci_upper: float


# ---------------------------------------------------------------------------
# Two-proportion z-test
# ---------------------------------------------------------------------------

def two_proportion_z_test(
    successes_ctrl: int,
    n_ctrl: int,
    successes_var: int,
    n_var: int,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided z-test for two independent proportions.

    Used for binary streaming metrics such as subscription_start,
    feature_adoption, autoplay_accepted, clicked_recommendation.
    """
    p1 = successes_ctrl / n_ctrl
    p2 = successes_var / n_var
    p_pool = (successes_ctrl + successes_var) / (n_ctrl + n_var)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_ctrl + 1 / n_var))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = 2 * sp_stats.norm.sf(abs(z))

    # CI on the difference using un-pooled SE
    se_diff = np.sqrt(p1 * (1 - p1) / n_ctrl + p2 * (1 - p2) / n_var)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    lift_rel = (diff / p1 * 100) if p1 > 0 else 0.0

    # Achieved power via normal approximation
    effect_size = abs(diff) / (se_diff if se_diff > 0 else 1e-9)
    power = 1 - sp_stats.norm.cdf(z_crit - effect_size)

    return TestResult(
        test_name="Two-Proportion Z-Test",
        statistic=z,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        lift_abs=diff,
        lift_rel_pct=lift_rel,
        is_significant=p_value < alpha,
        power_achieved=power,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------

def welch_t_test(
    control: np.ndarray | pd.Series,
    variant: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test for continuous metrics.

    Used for watch_time_minutes, content_completion_rate, etc.
    """
    control = np.asarray(control, dtype=float)
    variant = np.asarray(variant, dtype=float)

    t_stat, p_value = sp_stats.ttest_ind(control, variant, equal_var=False)

    mean_c, mean_v = control.mean(), variant.mean()
    diff = mean_v - mean_c

    se = np.sqrt(control.var(ddof=1) / len(control) + variant.var(ddof=1) / len(variant))
    # Welch-Satterthwaite df
    s1, s2 = control.var(ddof=1), variant.var(ddof=1)
    n1, n2 = len(control), len(variant)
    df_num = (s1 / n1 + s2 / n2) ** 2
    df_den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)

    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
    lift_rel = (diff / mean_c * 100) if mean_c != 0 else 0.0

    # Achieved power
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohen_d = abs(diff) / pooled_std if pooled_std > 0 else 0.0
    try:
        power = tt_ind_solve_power(
            effect_size=cohen_d,
            nobs1=n1,
            ratio=n2 / n1,
            alpha=alpha,
        )
    except Exception:
        power = None

    return TestResult(
        test_name="Welch's t-Test",
        statistic=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        lift_abs=diff,
        lift_rel_pct=lift_rel,
        is_significant=p_value < alpha,
        power_achieved=power,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------

def mann_whitney_u_test(
    control: np.ndarray | pd.Series,
    variant: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> TestResult:
    """Mann-Whitney U test for heavily skewed distributions.

    Streaming usage distributions are notoriously right-skewed:
    most users stream 0-2 times/week, power users stream 50+.
    """
    control = np.asarray(control, dtype=float)
    variant = np.asarray(variant, dtype=float)

    u_stat, p_value = sp_stats.mannwhitneyu(control, variant, alternative="two-sided")

    # Hodges-Lehmann estimator for the location shift
    diff_median = np.median(variant) - np.median(control)
    mean_c, mean_v = control.mean(), variant.mean()
    diff_mean = mean_v - mean_c
    lift_rel = (diff_mean / mean_c * 100) if mean_c != 0 else 0.0

    # Bootstrap 95 % CI on the mean difference
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(2000):
        bc = rng.choice(control, size=len(control), replace=True)
        bv = rng.choice(variant, size=len(variant), replace=True)
        boot_diffs.append(bv.mean() - bc.mean())
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return TestResult(
        test_name="Mann-Whitney U Test",
        statistic=u_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        lift_abs=diff_mean,
        lift_rel_pct=lift_rel,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# CUPED variance reduction
# ---------------------------------------------------------------------------

def cuped_adjust(
    metric: np.ndarray | pd.Series,
    covariate: np.ndarray | pd.Series,
    variant_flag: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> CUPEDResult:
    """CUPED (Controlled-experiment Using Pre-Experiment Data).

    Uses pre-experiment watch_time (or any covariate correlated with the
    outcome) to reduce variance in the treatment-effect estimate.

    Parameters
    ----------
    metric : post-experiment metric (e.g. watch_time day 1-14)
    covariate : pre-experiment metric (e.g. watch_time day -14 to 0)
    variant_flag : 0 = control, 1 = variant
    """
    metric = np.asarray(metric, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    variant_flag = np.asarray(variant_flag, dtype=int)

    # Compute theta = Cov(Y, X) / Var(X)
    cov_yx = np.cov(metric, covariate)[0, 1]
    var_x = np.var(covariate, ddof=1)
    theta = cov_yx / var_x if var_x > 0 else 0.0

    # Adjusted metric
    metric_adj = metric - theta * (covariate - covariate.mean())

    ctrl_mask = variant_flag == 0
    var_mask = variant_flag == 1

    # --- Raw ---
    raw_ctrl = metric[ctrl_mask]
    raw_var = metric[var_mask]
    raw_lift = raw_var.mean() - raw_ctrl.mean()
    raw_se = np.sqrt(raw_ctrl.var(ddof=1) / ctrl_mask.sum() + raw_var.var(ddof=1) / var_mask.sum())

    # --- CUPED ---
    cuped_ctrl = metric_adj[ctrl_mask]
    cuped_var = metric_adj[var_mask]
    cuped_lift = cuped_var.mean() - cuped_ctrl.mean()
    cuped_se = np.sqrt(cuped_ctrl.var(ddof=1) / ctrl_mask.sum() + cuped_var.var(ddof=1) / var_mask.sum())

    var_reduction = 1 - (cuped_se ** 2) / (raw_se ** 2) if raw_se > 0 else 0.0
    equiv_extra_users = (1 / (1 - var_reduction) - 1) * 100 if var_reduction < 1 else float("inf")

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = cuped_lift - z_crit * cuped_se
    ci_upper = cuped_lift + z_crit * cuped_se
    z = cuped_lift / cuped_se if cuped_se > 0 else 0.0
    p_value = 2 * sp_stats.norm.sf(abs(z))

    return CUPEDResult(
        raw_lift=raw_lift,
        cuped_lift=cuped_lift,
        raw_se=raw_se,
        cuped_se=cuped_se,
        variance_reduction_pct=var_reduction * 100,
        equivalent_extra_users_pct=equiv_extra_users,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


# ---------------------------------------------------------------------------
# Sequential testing — O'Brien-Fleming alpha spending
# ---------------------------------------------------------------------------

def obrien_fleming_boundary(look: int, total_looks: int, alpha: float = 0.05) -> float:
    """Return the z-boundary at a given *look* under O'Brien-Fleming spending.

    Uses the Lan-DeMets approximation to the OBF spending function:
        alpha_spent(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))
    where t = look / total_looks  (information fraction).
    """
    t = look / total_looks
    z_full = sp_stats.norm.ppf(1 - alpha / 2)
    return z_full / np.sqrt(t)


def sequential_test(
    control: np.ndarray | pd.Series,
    variant: np.ndarray | pd.Series,
    look: int,
    total_looks: int,
    alpha: float = 0.05,
) -> SequentialResult:
    """Run a single sequential look using O'Brien-Fleming spending.

    Parameters
    ----------
    control, variant : arrays of metric values observed *so far*.
    look : current look number (1-indexed).
    total_looks : planned number of interim analyses.
    """
    control = np.asarray(control, dtype=float)
    variant = np.asarray(variant, dtype=float)

    diff = variant.mean() - control.mean()
    se = np.sqrt(control.var(ddof=1) / len(control) + variant.var(ddof=1) / len(variant))
    z_obs = diff / se if se > 0 else 0.0

    boundary = obrien_fleming_boundary(look, total_looks, alpha)

    # Cumulative alpha spent (Lan-DeMets OBF approximation)
    t = look / total_looks
    z_full = sp_stats.norm.ppf(1 - alpha / 2)
    cum_alpha = 2 - 2 * sp_stats.norm.cdf(z_full / np.sqrt(t))

    return SequentialResult(
        look_number=look,
        total_looks=total_looks,
        nominal_alpha=alpha,
        spending_boundary=boundary,
        observed_z=z_obs,
        reject_null=abs(z_obs) >= boundary,
        cumulative_alpha_spent=cum_alpha,
    )


# ---------------------------------------------------------------------------
# Minimum Detectable Effect (MDE)
# ---------------------------------------------------------------------------

def minimum_detectable_effect(
    n_per_arm: int,
    baseline_rate: float | None = None,
    baseline_std: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: Literal["proportion", "continuous"] = "proportion",
) -> float:
    """Return the smallest absolute effect detectable with *n_per_arm* users.

    For proportions, returns the minimum detectable difference in rates.
    For continuous metrics, returns the minimum detectable difference in means.
    """
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    if metric_type == "proportion":
        if baseline_rate is None:
            raise ValueError("baseline_rate required for proportion metric")
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n_per_arm)
        return (z_alpha + z_beta) * se
    else:
        if baseline_std is None:
            raise ValueError("baseline_std required for continuous metric")
        se = baseline_std * np.sqrt(2 / n_per_arm)
        return (z_alpha + z_beta) * se


# ---------------------------------------------------------------------------
# Required sample size
# ---------------------------------------------------------------------------

def required_sample_size(
    baseline_rate: float | None = None,
    baseline_std: float | None = None,
    mde: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: Literal["proportion", "continuous"] = "proportion",
) -> int:
    """Return the number of users required *per arm*.

    Parameters
    ----------
    mde : minimum detectable effect (absolute difference).
    """
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    if metric_type == "proportion":
        if baseline_rate is None:
            raise ValueError("baseline_rate required for proportion metric")
        p = baseline_rate
        n = 2 * (z_alpha + z_beta) ** 2 * p * (1 - p) / (mde ** 2)
    else:
        if baseline_std is None:
            raise ValueError("baseline_std required for continuous metric")
        n = 2 * (z_alpha + z_beta) ** 2 * baseline_std ** 2 / (mde ** 2)

    return int(np.ceil(n))
