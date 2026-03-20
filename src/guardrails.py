"""
Guardrail metrics for streaming experiments.

Every experiment must pass guardrail checks before shipping.
A single guardrail failure blocks the launch regardless of how
impressive the primary metric looks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from src.stats import two_proportion_z_test, welch_t_test


Decision = Literal["SHIP", "HOLD", "STOP"]


@dataclass
class GuardrailVerdict:
    metric_name: str
    decision: Decision
    p_value: float
    lift_abs: float
    lift_rel_pct: float
    threshold_description: str
    justification: str


@dataclass
class GuardrailSummary:
    verdicts: list[GuardrailVerdict]
    overall_decision: Decision
    overall_justification: str


# ---------------------------------------------------------------------------
# Individual guardrail checks
# ---------------------------------------------------------------------------

def _check_cancellation_rate(
    df: pd.DataFrame,
    variant_col: str = "variant",
    metric_col: str = "cancelled_flag",
    control_label: str = "control",
    variant_label: str = "variant",
    alpha: float = 0.05,
) -> GuardrailVerdict:
    """subscriber_cancellation_rate: MUST NOT increase (catastrophic)."""
    ctrl = df.loc[df[variant_col] == control_label, metric_col]
    var = df.loc[df[variant_col] == variant_label, metric_col]

    result = two_proportion_z_test(
        successes_ctrl=int(ctrl.sum()), n_ctrl=len(ctrl),
        successes_var=int(var.sum()), n_var=len(var),
        alpha=alpha,
    )

    if result.lift_abs > 0 and result.is_significant:
        decision: Decision = "STOP"
        justification = (
            f"Cancellation rate increased by {result.lift_abs:.4f} "
            f"({result.lift_rel_pct:+.2f}%, p={result.p_value:.4f}). "
            "This is a catastrophic guardrail — experiment must be stopped."
        )
    elif result.lift_abs > 0 and not result.is_significant:
        decision = "HOLD"
        justification = (
            f"Cancellation rate shows a non-significant increase of "
            f"{result.lift_abs:.4f} (p={result.p_value:.4f}). "
            "Monitor closely; do not ship until more data confirms safety."
        )
    else:
        decision = "SHIP"
        justification = (
            f"Cancellation rate is flat or decreased ({result.lift_abs:+.4f}, "
            f"p={result.p_value:.4f}). Guardrail passed."
        )

    return GuardrailVerdict(
        metric_name="subscriber_cancellation_rate",
        decision=decision,
        p_value=result.p_value,
        lift_abs=result.lift_abs,
        lift_rel_pct=result.lift_rel_pct,
        threshold_description="Must not significantly increase",
        justification=justification,
    )


def _check_completion_rate(
    df: pd.DataFrame,
    variant_col: str = "variant",
    metric_col: str = "completion_pct",
    control_label: str = "control",
    variant_label: str = "variant",
    min_threshold: float = 0.60,
    alpha: float = 0.05,
) -> GuardrailVerdict:
    """content_completion_rate: must stay above threshold."""
    ctrl = df.loc[df[variant_col] == control_label, metric_col]
    var = df.loc[df[variant_col] == variant_label, metric_col]

    result = welch_t_test(ctrl.values, var.values, alpha=alpha)
    variant_mean = var.mean()

    if variant_mean < min_threshold and result.lift_abs < 0 and result.is_significant:
        decision: Decision = "STOP"
        justification = (
            f"Completion rate dropped to {variant_mean:.4f} (below {min_threshold}), "
            f"Δ = {result.lift_abs:+.4f}, p = {result.p_value:.4f}. "
            "Below minimum acceptable threshold — STOP."
        )
    elif result.lift_abs < 0 and result.is_significant:
        decision = "HOLD"
        justification = (
            f"Completion rate decreased significantly ({result.lift_abs:+.4f}, "
            f"p = {result.p_value:.4f}) but remains above threshold "
            f"({variant_mean:.4f} ≥ {min_threshold}). Investigate further."
        )
    else:
        decision = "SHIP"
        justification = (
            f"Completion rate is stable ({result.lift_abs:+.4f}, "
            f"p = {result.p_value:.4f}, variant mean = {variant_mean:.4f}). "
            "Guardrail passed."
        )

    return GuardrailVerdict(
        metric_name="content_completion_rate",
        decision=decision,
        p_value=result.p_value,
        lift_abs=result.lift_abs,
        lift_rel_pct=result.lift_rel_pct,
        threshold_description=f"Must stay above {min_threshold:.0%}",
        justification=justification,
    )


def _check_support_contacts(
    df: pd.DataFrame,
    variant_col: str = "variant",
    metric_col: str = "support_contacts",
    control_label: str = "control",
    variant_label: str = "variant",
    alpha: float = 0.05,
) -> GuardrailVerdict | None:
    """customer_support_contacts: spike = bad UX."""
    if metric_col not in df.columns:
        return None

    ctrl = df.loc[df[variant_col] == control_label, metric_col]
    var = df.loc[df[variant_col] == variant_label, metric_col]
    result = welch_t_test(ctrl.values, var.values, alpha=alpha)

    if result.lift_abs > 0 and result.is_significant:
        decision: Decision = "HOLD"
        justification = (
            f"Support contacts increased by {result.lift_abs:+.4f} "
            f"(p = {result.p_value:.4f}). Possible UX issue — investigate."
        )
    else:
        decision = "SHIP"
        justification = (
            f"Support contacts stable ({result.lift_abs:+.4f}, "
            f"p = {result.p_value:.4f}). Guardrail passed."
        )

    return GuardrailVerdict(
        metric_name="customer_support_contacts",
        decision=decision,
        p_value=result.p_value,
        lift_abs=result.lift_abs,
        lift_rel_pct=result.lift_rel_pct,
        threshold_description="Must not spike",
        justification=justification,
    )


def _check_error_rate(
    df: pd.DataFrame,
    variant_col: str = "variant",
    metric_col: str = "error_flag",
    control_label: str = "control",
    variant_label: str = "variant",
    alpha: float = 0.05,
) -> GuardrailVerdict | None:
    """Technical guardrail: error rate must not increase."""
    if metric_col not in df.columns:
        return None

    ctrl = df.loc[df[variant_col] == control_label, metric_col]
    var = df.loc[df[variant_col] == variant_label, metric_col]
    result = two_proportion_z_test(
        successes_ctrl=int(ctrl.sum()), n_ctrl=len(ctrl),
        successes_var=int(var.sum()), n_var=len(var),
        alpha=alpha,
    )

    if result.lift_abs > 0 and result.is_significant:
        decision: Decision = "STOP"
        justification = (
            f"Error rate increased by {result.lift_abs:+.4f} "
            f"(p = {result.p_value:.4f}). Technical failure — STOP."
        )
    else:
        decision = "SHIP"
        justification = (
            f"Error rate stable ({result.lift_abs:+.4f}, "
            f"p = {result.p_value:.4f}). Guardrail passed."
        )

    return GuardrailVerdict(
        metric_name="error_rate",
        decision=decision,
        p_value=result.p_value,
        lift_abs=result.lift_abs,
        lift_rel_pct=result.lift_rel_pct,
        threshold_description="Must not increase",
        justification=justification,
    )


# ---------------------------------------------------------------------------
# Aggregated guardrail check
# ---------------------------------------------------------------------------

_DECISION_RANK: dict[Decision, int] = {"SHIP": 0, "HOLD": 1, "STOP": 2}


def run_guardrails(
    df: pd.DataFrame,
    variant_col: str = "variant",
    control_label: str = "control",
    variant_label: str = "variant",
    alpha: float = 0.05,
) -> GuardrailSummary:
    """Run all applicable guardrails and return an overall recommendation."""
    checks = [
        _check_cancellation_rate(df, variant_col, control_label=control_label, variant_label=variant_label, alpha=alpha),
        _check_completion_rate(df, variant_col, control_label=control_label, variant_label=variant_label, alpha=alpha),
        _check_support_contacts(df, variant_col, control_label=control_label, variant_label=variant_label, alpha=alpha),
        _check_error_rate(df, variant_col, control_label=control_label, variant_label=variant_label, alpha=alpha),
    ]
    verdicts = [c for c in checks if c is not None]

    if not verdicts:
        return GuardrailSummary(
            verdicts=[],
            overall_decision="SHIP",
            overall_justification="No guardrail metrics available to check.",
        )

    worst = max(verdicts, key=lambda v: _DECISION_RANK[v.decision])
    overall = worst.decision

    if overall == "SHIP":
        justification = "All guardrails passed. Safe to ship."
    elif overall == "HOLD":
        failing = [v.metric_name for v in verdicts if v.decision in ("HOLD", "STOP")]
        justification = (
            f"Guardrail concern(s) on: {', '.join(failing)}. "
            "Collect more data or investigate before shipping."
        )
    else:
        failing = [v.metric_name for v in verdicts if v.decision == "STOP"]
        justification = (
            f"CRITICAL guardrail failure(s): {', '.join(failing)}. "
            "Do NOT ship. Investigate root cause immediately."
        )

    return GuardrailSummary(
        verdicts=verdicts,
        overall_decision=overall,
        overall_justification=justification,
    )
