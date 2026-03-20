"""
Auto-report generator — produces a Markdown experiment report
in the style used by streaming company experimentation teams.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal

from src.experiment import (
    FrequentistOutput,
    BayesianOutput,
    SRMResult,
    NoveltyResult,
    LongRunProjection,
)
from src.stats import CUPEDResult
from src.guardrails import GuardrailSummary


@dataclass
class ReportConfig:
    experiment_name: str
    hypothesis: str
    owner: str
    start_date: str
    end_date: str
    primary_metric: str
    total_users: int
    daily_eligible_users: int = 500_000
    revenue_per_conversion: float = 119.88


def generate_report(
    config: ReportConfig,
    freq: FrequentistOutput,
    bayes: BayesianOutput,
    cuped: CUPEDResult | None,
    srm: SRMResult,
    novelty: NoveltyResult,
    projection: LongRunProjection,
    guardrails: GuardrailSummary,
) -> str:
    """Return a complete Markdown experiment report."""

    tr = freq.test_result
    sig_label = "**Statistically Significant** ✅" if tr.is_significant else "Not Significant ❌"
    confidence = _confidence_level(tr, bayes, guardrails)

    sections = [
        _header(config),
        _executive_summary(config, tr, bayes, guardrails, confidence),
        _primary_metric(config, tr, sig_label),
        _bayesian_section(bayes),
        _cuped_section(cuped),
        _srm_section(srm),
        _novelty_section(novelty),
        _guardrail_section(guardrails),
        _business_impact(projection),
        _recommendation(tr, bayes, guardrails, confidence),
        _next_steps(tr, guardrails, confidence),
        _footer(),
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _header(c: ReportConfig) -> str:
    return f"""# Experiment Report: {c.experiment_name}

| Field | Value |
|---|---|
| **Hypothesis** | {c.hypothesis} |
| **Owner** | {c.owner} |
| **Dates** | {c.start_date} → {c.end_date} |
| **Primary Metric** | {c.primary_metric} |
| **Total Users** | {c.total_users:,} |
| **Generated** | {datetime.date.today().isoformat()} |"""


def _executive_summary(
    c: ReportConfig,
    tr,
    bayes: BayesianOutput,
    guardrails: GuardrailSummary,
    confidence: str,
) -> str:
    return f"""## Executive Summary

The **{c.experiment_name}** experiment measured the impact on **{c.primary_metric}**.

- **Absolute lift**: {tr.lift_abs:+.4f}
- **Relative lift**: {tr.lift_rel_pct:+.2f}%
- **p-value**: {tr.p_value:.4f}
- **P(variant wins)**: {bayes.prob_variant_wins:.1%}
- **Guardrail status**: {guardrails.overall_decision}
- **Recommendation confidence**: {confidence}"""


def _primary_metric(c: ReportConfig, tr, sig_label: str) -> str:
    power_str = f"{tr.power_achieved:.1%}" if tr.power_achieved else "N/A"
    return f"""## Primary Metric: {c.primary_metric}

| Measure | Value |
|---|---|
| Test | {tr.test_name} |
| Statistic | {tr.statistic:.4f} |
| p-value | {tr.p_value:.6f} |
| Result | {sig_label} |
| Absolute lift | {tr.lift_abs:+.6f} |
| Relative lift | {tr.lift_rel_pct:+.2f}% |
| 95% CI | [{tr.ci_lower:+.6f}, {tr.ci_upper:+.6f}] |
| Achieved power | {power_str} |"""


def _bayesian_section(b: BayesianOutput) -> str:
    return f"""## Bayesian Analysis

| Measure | Value |
|---|---|
| P(variant > control) | {b.prob_variant_wins:.1%} |
| Expected loss (if ship variant) | {b.expected_loss:.6f} |
| 95% Credible Interval | [{b.credible_interval_95[0]:+.6f}, {b.credible_interval_95[1]:+.6f}] |

> A P(variant wins) above 95% is strong evidence; 90-95% is moderate."""


def _cuped_section(cuped: CUPEDResult | None) -> str:
    if cuped is None:
        return "## CUPED Variance Reduction\n\n_Not applied (no pre-experiment covariate available)._"
    return f"""## CUPED Variance Reduction

| Measure | Raw | CUPED |
|---|---|---|
| Lift estimate | {cuped.raw_lift:+.6f} | {cuped.cuped_lift:+.6f} |
| Standard error | {cuped.raw_se:.6f} | {cuped.cuped_se:.6f} |

- **Variance reduced by {cuped.variance_reduction_pct:.1f}%**
- Equivalent to running **{cuped.equivalent_extra_users_pct:.0f}% more users**
- CUPED p-value: {cuped.p_value:.6f}
- CUPED 95% CI: [{cuped.ci_lower:+.6f}, {cuped.ci_upper:+.6f}]"""


def _srm_section(srm: SRMResult) -> str:
    icon = "⚠️" if srm.is_imbalanced else "✅"
    return f"""## Sample Ratio Mismatch (SRM) {icon}

| Measure | Value |
|---|---|
| Control n | {srm.n_control:,} |
| Variant n | {srm.n_variant:,} |
| Observed ratio | {srm.observed_ratio:.4f} |
| Expected ratio | {srm.expected_ratio:.4f} |
| Chi² | {srm.chi2:.4f} |
| p-value | {srm.p_value:.4f} |

{srm.message}"""


def _novelty_section(n: NoveltyResult) -> str:
    icon = "⚠️" if n.has_novelty_effect else "✅"
    return f"""## Novelty Effect Check {icon}

| Period | Lift |
|---|---|
| First 3 days | {n.early_lift:+.6f} |
| Last 7 days | {n.late_lift:+.6f} |
| Decay | {n.lift_decay_pct:+.1f}% |

{n.message}"""


def _guardrail_section(g: GuardrailSummary) -> str:
    rows = []
    for v in g.verdicts:
        icon = {"SHIP": "🟢", "HOLD": "🟡", "STOP": "🔴"}[v.decision]
        rows.append(
            f"| {v.metric_name} | {icon} {v.decision} | {v.lift_abs:+.4f} | "
            f"{v.p_value:.4f} | {v.threshold_description} |"
        )
    table = "\n".join(rows)
    return f"""## Guardrail Metrics

| Metric | Decision | Lift | p-value | Threshold |
|---|---|---|---|---|
{table}

**Overall**: {g.overall_decision} — {g.overall_justification}"""


def _business_impact(p: LongRunProjection) -> str:
    return f"""## Business Impact Projection

{p.message}

| Horizon | Incremental Conversions | Revenue Impact |
|---|---|---|
| Daily | {p.daily_incremental_conversions:,.0f} | — |
| Annual | {p.annual_incremental_conversions:,.0f} | **${p.annual_revenue_impact:,.0f}** |"""


def _recommendation(tr, bayes, guardrails, confidence) -> str:
    if guardrails.overall_decision == "STOP":
        rec = "**STOP** — Guardrail failure. Do not ship."
    elif not tr.is_significant:
        rec = "**HOLD** — Primary metric not significant. Collect more data or iterate."
    elif guardrails.overall_decision == "HOLD":
        rec = "**HOLD** — Primary metric wins but guardrail concerns remain."
    else:
        rec = "**SHIP** — Primary metric significant, guardrails clean."

    return f"""## Recommendation

{rec}

**Confidence level**: {confidence}"""


def _next_steps(tr, guardrails, confidence) -> str:
    steps = []
    if guardrails.overall_decision == "STOP":
        steps.append("1. Kill the experiment immediately")
        steps.append("2. Investigate guardrail failure root cause")
        steps.append("3. Fix and redesign before re-testing")
    elif not tr.is_significant:
        steps.append("1. Extend experiment duration or increase traffic allocation")
        steps.append("2. Review if MDE was realistic")
        steps.append("3. Consider iterating on the treatment before re-running")
    elif guardrails.overall_decision == "HOLD":
        steps.append("1. Investigate guardrail metric concerns")
        steps.append("2. Consider a follow-up targeted experiment")
        steps.append("3. Ship only after guardrail concerns are resolved")
    else:
        steps.append("1. Ship to 100% traffic")
        steps.append("2. Monitor guardrails for 7 days post-launch")
        steps.append("3. Archive experiment and update documentation")

    return "## Next Steps\n\n" + "\n".join(steps)


def _footer() -> str:
    return (
        "---\n"
        "*Generated by the Streaming A/B Testing Framework — "
        "Experimentation Science Team*"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_level(tr, bayes: BayesianOutput, guardrails: GuardrailSummary) -> str:
    if guardrails.overall_decision == "STOP":
        return "High (guardrail failure is clear)"
    score = 0
    if tr.is_significant:
        score += 1
    if bayes.prob_variant_wins > 0.95:
        score += 1
    if guardrails.overall_decision == "SHIP":
        score += 1
    power = tr.power_achieved
    if power is not None and power > 0.80:
        score += 1
    if score >= 3:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"
