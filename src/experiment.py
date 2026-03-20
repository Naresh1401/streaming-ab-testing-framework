"""
StreamingExperimentAnalyzer — the main experiment runner.

Wraps the stats engine, corrections, and guardrails into a single
ergonomic interface for analysing streaming A/B tests end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats as sp_stats
from typing import Literal

from src.stats import (
    two_proportion_z_test,
    welch_t_test,
    mann_whitney_u_test,
    cuped_adjust,
    sequential_test,
    TestResult,
    CUPEDResult,
    SequentialResult,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FrequentistOutput:
    test_result: TestResult
    sequential: SequentialResult | None = None


@dataclass
class BayesianOutput:
    prob_variant_wins: float
    expected_loss: float
    credible_interval_95: tuple[float, float]
    posterior_control_mean: float
    posterior_variant_mean: float


@dataclass
class SRMResult:
    chi2: float
    p_value: float
    n_control: int
    n_variant: int
    expected_ratio: float
    observed_ratio: float
    is_imbalanced: bool
    message: str


@dataclass
class NoveltyResult:
    early_lift: float
    late_lift: float
    lift_decay_pct: float
    has_novelty_effect: bool
    message: str


@dataclass
class LongRunProjection:
    daily_incremental_conversions: float
    annual_incremental_conversions: float
    annual_revenue_impact: float
    message: str


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class StreamingExperimentAnalyzer:
    """End-to-end analyser for a single streaming experiment arm comparison."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.variant_col: str = "variant"
        self.metric_col: str = ""
        self.metric_type: Literal["proportion", "continuous"] = "proportion"
        self._control_label = "control"
        self._variant_label = "variant"

    # ----- loading --------------------------------------------------------

    def load(
        self,
        df: pd.DataFrame,
        variant_col: str = "variant",
        metric_col: str = "autoplay_accepted",
        metric_type: Literal["proportion", "continuous"] = "proportion",
        control_label: str = "control",
        variant_label: str = "variant",
    ) -> "StreamingExperimentAnalyzer":
        """Load experiment data."""
        self.df = df.copy()
        self.variant_col = variant_col
        self.metric_col = metric_col
        self.metric_type = metric_type
        self._control_label = control_label
        self._variant_label = variant_label
        return self

    # ----- helpers --------------------------------------------------------

    def _split(self) -> tuple[pd.Series, pd.Series]:
        ctrl = self.df.loc[self.df[self.variant_col] == self._control_label, self.metric_col]
        var = self.df.loc[self.df[self.variant_col] == self._variant_label, self.metric_col]
        return ctrl, var

    # ----- frequentist ----------------------------------------------------

    def run_frequentist(self, alpha: float = 0.05) -> FrequentistOutput:
        """Run the appropriate frequentist test based on metric_type."""
        ctrl, var = self._split()

        if self.metric_type == "proportion":
            result = two_proportion_z_test(
                successes_ctrl=int(ctrl.sum()),
                n_ctrl=len(ctrl),
                successes_var=int(var.sum()),
                n_var=len(var),
                alpha=alpha,
            )
        else:
            result = welch_t_test(ctrl.values, var.values, alpha=alpha)

        return FrequentistOutput(test_result=result)

    def run_mann_whitney(self, alpha: float = 0.05) -> TestResult:
        """Run Mann-Whitney U for skewed continuous metrics."""
        ctrl, var = self._split()
        return mann_whitney_u_test(ctrl.values, var.values, alpha=alpha)

    # ----- bayesian -------------------------------------------------------

    def run_bayesian(self, n_samples: int = 100_000) -> BayesianOutput:
        """Bayesian analysis using conjugate priors.

        - Proportions: Beta-Binomial (Beta(1,1) prior).
        - Continuous: Normal-Normal with empirical prior.
        """
        ctrl, var = self._split()
        rng = np.random.default_rng(42)

        if self.metric_type == "proportion":
            # Beta posterior
            a_c, b_c = 1 + ctrl.sum(), 1 + len(ctrl) - ctrl.sum()
            a_v, b_v = 1 + var.sum(), 1 + len(var) - var.sum()
            post_ctrl = rng.beta(a_c, b_c, size=n_samples)
            post_var = rng.beta(a_v, b_v, size=n_samples)
        else:
            # Normal posterior (known variance approximation with large n)
            mu_c, se_c = ctrl.mean(), ctrl.std(ddof=1) / np.sqrt(len(ctrl))
            mu_v, se_v = var.mean(), var.std(ddof=1) / np.sqrt(len(var))
            post_ctrl = rng.normal(mu_c, se_c, size=n_samples)
            post_var = rng.normal(mu_v, se_v, size=n_samples)

        diff = post_var - post_ctrl
        prob_wins = float((diff > 0).mean())
        expected_loss = float(np.maximum(-diff, 0).mean())
        ci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))

        return BayesianOutput(
            prob_variant_wins=prob_wins,
            expected_loss=expected_loss,
            credible_interval_95=ci,
            posterior_control_mean=float(post_ctrl.mean()),
            posterior_variant_mean=float(post_var.mean()),
        )

    # ----- CUPED ----------------------------------------------------------

    def apply_cuped(
        self,
        covariate_col: str = "pre_experiment_watch_time",
        alpha: float = 0.05,
    ) -> CUPEDResult:
        """Apply CUPED variance reduction using a pre-experiment covariate."""
        metric = self.df[self.metric_col].values
        covariate = self.df[covariate_col].values
        variant_flag = (self.df[self.variant_col] == self._variant_label).astype(int).values
        return cuped_adjust(metric, covariate, variant_flag, alpha=alpha)

    # ----- SRM ------------------------------------------------------------

    def check_srm(self, expected_ratio: float = 1.0) -> SRMResult:
        """Sample Ratio Mismatch detection via chi-squared goodness-of-fit.

        An SRM means something went *wrong* with randomisation — the
        results cannot be trusted until the root cause is found.
        """
        counts = self.df[self.variant_col].value_counts()
        n_ctrl = int(counts.get(self._control_label, 0))
        n_var = int(counts.get(self._variant_label, 0))
        total = n_ctrl + n_var

        expected_ctrl = total * (1 / (1 + expected_ratio))
        expected_var = total - expected_ctrl

        chi2 = ((n_ctrl - expected_ctrl) ** 2 / expected_ctrl
                + (n_var - expected_var) ** 2 / expected_var)
        p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)
        is_imbalanced = p_value < 0.001  # strict threshold

        obs_ratio = n_var / n_ctrl if n_ctrl > 0 else float("inf")

        if is_imbalanced:
            msg = (
                f"⚠️  SRM DETECTED (p={p_value:.2e}). Observed ratio "
                f"{obs_ratio:.4f} vs expected {expected_ratio:.4f}. "
                "Randomisation may be broken — investigate before trusting results."
            )
        else:
            msg = (
                f"✓ No SRM detected (p={p_value:.4f}). "
                f"Observed ratio {obs_ratio:.4f} ≈ expected {expected_ratio:.4f}."
            )

        return SRMResult(
            chi2=chi2,
            p_value=p_value,
            n_control=n_ctrl,
            n_variant=n_var,
            expected_ratio=expected_ratio,
            observed_ratio=obs_ratio,
            is_imbalanced=is_imbalanced,
            message=msg,
        )

    # ----- Novelty effect -------------------------------------------------

    def novelty_effect_check(
        self,
        date_col: str = "event_date",
        early_days: int = 3,
        late_days: int = 7,
    ) -> NoveltyResult:
        """Compare treatment effect in the first *early_days* vs the last *late_days*.

        Streaming users exhibit strong novelty effects — new thumbnails,
        new UI, new algorithms all look better on day 1 than day 14.
        """
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()

        early_mask = df[date_col] <= min_date + pd.Timedelta(days=early_days - 1)
        late_mask = df[date_col] >= max_date - pd.Timedelta(days=late_days - 1)

        def _lift(subset: pd.DataFrame) -> float:
            c = subset.loc[subset[self.variant_col] == self._control_label, self.metric_col]
            v = subset.loc[subset[self.variant_col] == self._variant_label, self.metric_col]
            return v.mean() - c.mean() if len(c) > 0 and len(v) > 0 else 0.0

        early_lift = _lift(df[early_mask])
        late_lift = _lift(df[late_mask])
        decay = ((early_lift - late_lift) / abs(early_lift) * 100) if early_lift != 0 else 0.0
        has_novelty = decay > 20  # >20 % decay signals novelty

        if has_novelty:
            msg = (
                f"⚠️  Novelty effect detected: early lift {early_lift:.4f} → "
                f"late lift {late_lift:.4f} ({decay:+.1f}% decay). "
                "Consider extending the experiment or discounting early data."
            )
        else:
            msg = (
                f"✓ No significant novelty effect: early lift {early_lift:.4f}, "
                f"late lift {late_lift:.4f} ({decay:+.1f}% change)."
            )

        return NoveltyResult(
            early_lift=early_lift,
            late_lift=late_lift,
            lift_decay_pct=decay,
            has_novelty_effect=has_novelty,
            message=msg,
        )

    # ----- Long-run projection --------------------------------------------

    def compute_long_run_projection(
        self,
        daily_eligible_users: int = 500_000,
        revenue_per_conversion: float = 119.88,
        projection_days: int = 365,
    ) -> LongRunProjection:
        """Project metric change to subscriber / revenue impact.

        Parameters
        ----------
        daily_eligible_users : average DAU in the experiment funnel.
        revenue_per_conversion : annual revenue per converted user.
        projection_days : forecast horizon (default 365).
        """
        ctrl, var = self._split()
        lift_abs = var.mean() - ctrl.mean()

        daily_inc = lift_abs * daily_eligible_users
        annual_inc = daily_inc * projection_days
        annual_rev = annual_inc * revenue_per_conversion

        msg = (
            f"Projected impact: {lift_abs:+.6f} absolute lift × "
            f"{daily_eligible_users:,} daily users = {daily_inc:,.0f} "
            f"incremental conversions/day → {annual_inc:,.0f}/year "
            f"→ ${annual_rev:,.0f} additional ARR."
        )

        return LongRunProjection(
            daily_incremental_conversions=daily_inc,
            annual_incremental_conversions=annual_inc,
            annual_revenue_impact=annual_rev,
            message=msg,
        )
