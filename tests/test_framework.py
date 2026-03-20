"""
Test suite for the streaming A/B testing framework.

Covers:
- p-value accuracy against known analytical solutions
- CUPED variance reduction on controlled synthetic data
- SRM detection on a known imbalance
- Novelty effect flagging on artificially inflated day-1 data
- Corrections module (BH and Holm-Šídák)
- Guardrail verdicts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from src.stats import (
    two_proportion_z_test,
    welch_t_test,
    mann_whitney_u_test,
    cuped_adjust,
    sequential_test,
    obrien_fleming_boundary,
    minimum_detectable_effect,
    required_sample_size,
)
from src.corrections import benjamini_hochberg, holm_sidak
from src.experiment import StreamingExperimentAnalyzer
from src.guardrails import run_guardrails


# =========================================================================
#  Stats engine
# =========================================================================


class TestTwoProportionZ:
    """Verify two-proportion z-test p-values against scipy."""

    def test_known_proportions(self):
        # p1=0.10, p2=0.12, n=10 000 each
        n = 10_000
        p1, p2 = 0.10, 0.12
        s1, s2 = int(p1 * n), int(p2 * n)

        result = two_proportion_z_test(s1, n, s2, n)

        # Replicate with scipy's proportions_ztest logic manually
        p_pool = (s1 + s2) / (2 * n)
        se = np.sqrt(p_pool * (1 - p_pool) * 2 / n)
        z = (s2 / n - s1 / n) / se
        expected_p = 2 * sp_stats.norm.sf(abs(z))

        assert abs(result.p_value - expected_p) < 1e-6
        assert result.is_significant  # 2 pp lift with 10 k should be sig

    def test_no_difference(self):
        result = two_proportion_z_test(500, 5000, 500, 5000)
        assert not result.is_significant
        assert abs(result.lift_abs) < 1e-9

    def test_ci_contains_zero_when_not_significant(self):
        result = two_proportion_z_test(100, 1000, 102, 1000)
        # With very small lift, CI should contain 0
        assert result.ci_lower <= 0 <= result.ci_upper


class TestWelchT:
    """Verify Welch's t-test against scipy."""

    def test_against_scipy(self):
        rng = np.random.default_rng(0)
        ctrl = rng.normal(50, 10, 5000)
        var = rng.normal(52, 10, 5000)

        result = welch_t_test(ctrl, var)
        _, scipy_p = sp_stats.ttest_ind(ctrl, var, equal_var=False)

        assert abs(result.p_value - scipy_p) < 1e-10
        assert result.is_significant

    def test_identical_distributions(self):
        rng = np.random.default_rng(1)
        data = rng.normal(100, 15, 10_000)
        ctrl, var = data[:5000], data[5000:]

        result = welch_t_test(ctrl, var)
        assert not result.is_significant


class TestMannWhitney:
    """Verify Mann-Whitney U against scipy."""

    def test_against_scipy(self):
        rng = np.random.default_rng(2)
        ctrl = rng.exponential(2, 5000)
        var = rng.exponential(2.5, 5000)

        result = mann_whitney_u_test(ctrl, var)
        u, scipy_p = sp_stats.mannwhitneyu(ctrl, var, alternative="two-sided")

        assert abs(result.p_value - scipy_p) < 1e-10
        assert result.statistic == u


# =========================================================================
#  CUPED
# =========================================================================


class TestCUPED:
    """Verify CUPED reduces variance on a controlled synthetic dataset."""

    def test_variance_reduction(self):
        rng = np.random.default_rng(42)
        n = 20_000
        # Pre-experiment covariate correlated with outcome (r ≈ 0.7)
        pre = rng.normal(50, 10, n)
        noise = rng.normal(0, 7, n)
        treatment = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        post = pre * 0.7 + noise + treatment * 2  # true lift = 2

        result = cuped_adjust(post, pre, treatment)

        # CUPED SE must be smaller than raw SE
        assert result.cuped_se < result.raw_se
        # Variance reduction should be substantial (>30 %)
        assert result.variance_reduction_pct > 30
        # CUPED lift should be close to the true effect of 2
        assert abs(result.cuped_lift - 2.0) < 0.5

    def test_no_covariate_correlation(self):
        """When covariate is pure noise, CUPED should not help much."""
        rng = np.random.default_rng(99)
        n = 10_000
        pre = rng.normal(0, 1, n)  # uncorrelated noise
        post = rng.normal(5, 2, n)
        treatment = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

        result = cuped_adjust(post, pre, treatment)
        # Variance reduction near zero
        assert result.variance_reduction_pct < 10


# =========================================================================
#  SRM
# =========================================================================


class TestSRM:
    """Verify SRM detection catches a known imbalance."""

    def _make_df(self, n_ctrl: int, n_var: int) -> pd.DataFrame:
        return pd.DataFrame({
            "variant": ["control"] * n_ctrl + ["variant"] * n_var,
            "metric": np.random.default_rng(0).normal(0, 1, n_ctrl + n_var),
        })

    def test_detects_imbalance(self):
        # 50 000 vs 48 000 — a meaningful imbalance at scale
        df = self._make_df(50_000, 48_000)
        analyzer = StreamingExperimentAnalyzer()
        analyzer.load(df, "variant", "metric", "continuous")
        srm = analyzer.check_srm()

        assert srm.is_imbalanced
        assert srm.p_value < 0.001

    def test_no_false_alarm(self):
        df = self._make_df(50_000, 50_000)
        analyzer = StreamingExperimentAnalyzer()
        analyzer.load(df, "variant", "metric", "continuous")
        srm = analyzer.check_srm()

        assert not srm.is_imbalanced


# =========================================================================
#  Novelty effect
# =========================================================================

class TestNovelty:
    """Verify novelty-effect check flags artificially inflated day-1 results."""

    def test_flags_novelty(self):
        rng = np.random.default_rng(7)
        n = 5000
        dates_early = pd.date_range("2025-01-01", periods=3, freq="D")
        dates_late = pd.date_range("2025-01-08", periods=7, freq="D")

        rows = []
        for d in dates_early:
            # Early: big lift for variant
            rows.append(pd.DataFrame({
                "user_id": range(n),
                "variant": ["control"] * (n // 2) + ["variant"] * (n // 2),
                "event_date": d,
                "metric": np.concatenate([
                    rng.normal(10, 2, n // 2),
                    rng.normal(14, 2, n // 2),  # +4 lift
                ]),
            }))
        for d in dates_late:
            # Late: lift gone
            rows.append(pd.DataFrame({
                "user_id": range(n),
                "variant": ["control"] * (n // 2) + ["variant"] * (n // 2),
                "event_date": d,
                "metric": np.concatenate([
                    rng.normal(10, 2, n // 2),
                    rng.normal(10.5, 2, n // 2),  # +0.5 lift
                ]),
            }))
        df = pd.concat(rows, ignore_index=True)

        analyzer = StreamingExperimentAnalyzer()
        analyzer.load(df, "variant", "metric", "continuous")
        result = analyzer.novelty_effect_check()

        assert result.has_novelty_effect
        assert result.early_lift > result.late_lift
        assert result.lift_decay_pct > 20

    def test_no_novelty_when_stable(self):
        rng = np.random.default_rng(8)
        n = 4000
        dates = pd.date_range("2025-01-01", periods=14, freq="D")
        rows = []
        for d in dates:
            rows.append(pd.DataFrame({
                "user_id": range(n),
                "variant": ["control"] * (n // 2) + ["variant"] * (n // 2),
                "event_date": d,
                "metric": np.concatenate([
                    rng.normal(10, 2, n // 2),
                    rng.normal(11, 2, n // 2),
                ]),
            }))
        df = pd.concat(rows, ignore_index=True)

        analyzer = StreamingExperimentAnalyzer()
        analyzer.load(df, "variant", "metric", "continuous")
        result = analyzer.novelty_effect_check()

        assert not result.has_novelty_effect


# =========================================================================
#  Sequential testing
# =========================================================================

class TestSequential:
    def test_boundary_decreases(self):
        """OBF boundary should decrease over time (more lenient later)."""
        boundaries = [obrien_fleming_boundary(i, 10) for i in range(1, 11)]
        for i in range(1, len(boundaries)):
            assert boundaries[i] <= boundaries[i - 1]

    def test_final_boundary_near_nominal(self):
        """At the last look, boundary should be close to z_alpha/2."""
        b = obrien_fleming_boundary(10, 10, alpha=0.05)
        z_crit = sp_stats.norm.ppf(0.975)
        assert abs(b - z_crit) < 0.01


# =========================================================================
#  Corrections
# =========================================================================

class TestBenjaminiHochberg:
    def test_all_significant(self):
        result = benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05)
        assert all(result.is_significant)

    def test_none_significant(self):
        result = benjamini_hochberg([0.5, 0.6, 0.7], alpha=0.05)
        assert not any(result.is_significant)

    def test_adjusted_at_least_raw(self):
        raw = [0.01, 0.04, 0.06]
        result = benjamini_hochberg(raw)
        for r, a in zip(raw, result.adjusted_p_values):
            assert a >= r - 1e-12  # adjusted should be ≥ raw


class TestHolmSidak:
    def test_all_significant(self):
        result = holm_sidak([0.001, 0.002, 0.003], alpha=0.05)
        assert all(result.is_significant)

    def test_none_significant(self):
        result = holm_sidak([0.5, 0.6, 0.7], alpha=0.05)
        assert not any(result.is_significant)


# =========================================================================
#  Guardrails
# =========================================================================

class TestGuardrails:
    def test_ship_when_safe(self):
        rng = np.random.default_rng(10)
        n = 50_000
        # Use the same draws split evenly so control & variant are near-identical
        cancelled = rng.binomial(1, 0.002, 2 * n)
        completion = rng.beta(6, 3, 2 * n)
        df = pd.DataFrame({
            "variant": ["control"] * n + ["variant"] * n,
            "cancelled_flag": cancelled,
            "completion_pct": completion,
        })
        result = run_guardrails(df)
        assert result.overall_decision == "SHIP"

    def test_stop_on_cancellation_spike(self):
        rng = np.random.default_rng(11)
        n = 50_000
        df = pd.DataFrame({
            "variant": ["control"] * n + ["variant"] * n,
            "cancelled_flag": np.concatenate([
                rng.binomial(1, 0.002, n),
                rng.binomial(1, 0.010, n),  # 5× spike
            ]),
            "completion_pct": rng.beta(6, 3, 2 * n),
        })
        result = run_guardrails(df)
        assert result.overall_decision == "STOP"


# =========================================================================
#  Sample size & MDE
# =========================================================================

class TestSampleSize:
    def test_proportion_round_trip(self):
        """Sample size → MDE → sample size should round-trip."""
        n = required_sample_size(baseline_rate=0.10, mde=0.01, alpha=0.05, power=0.80)
        mde_back = minimum_detectable_effect(n, baseline_rate=0.10, alpha=0.05, power=0.80)
        assert abs(mde_back - 0.01) < 0.001

    def test_continuous_round_trip(self):
        n = required_sample_size(baseline_std=15.0, mde=2.0, alpha=0.05, power=0.80,
                                 metric_type="continuous")
        mde_back = minimum_detectable_effect(n, baseline_std=15.0, alpha=0.05, power=0.80,
                                             metric_type="continuous")
        assert abs(mde_back - 2.0) < 0.5
