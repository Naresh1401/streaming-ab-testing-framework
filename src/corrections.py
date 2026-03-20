"""
Multiple testing corrections for streaming experiments.

When a single experiment measures watch_time, completion_rate, skip_rate,
and share_rate simultaneously, raw p-values must be corrected.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import stats as sp_stats


@dataclass
class CorrectionResult:
    """Holds original and adjusted p-values with per-metric decisions."""
    method: str
    metric_names: list[str]
    raw_p_values: list[float]
    adjusted_p_values: list[float]
    is_significant: list[bool]
    alpha: float


# ---------------------------------------------------------------------------
# Benjamini-Hochberg (FDR control)
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: list[float] | np.ndarray,
    metric_names: list[str] | None = None,
    alpha: float = 0.05,
) -> CorrectionResult:
    """Benjamini-Hochberg procedure for controlling the False Discovery Rate.

    Ideal for testing a *family* of non-guardrail metrics together
    (e.g. watch_time, completion_rate, skip_rate, share_rate).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if metric_names is None:
        metric_names = [f"metric_{i}" for i in range(m)]

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # Adjusted p-values (step-up)
    adj = np.empty(m)
    adj[-1] = sorted_p[-1]  # placeholder; will fill via reverse pass
    adjusted_sorted = np.empty(m)
    adjusted_sorted[m - 1] = sorted_p[m - 1]
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i + 1], sorted_p[i] * m / (i + 1))
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original order
    adjusted = np.empty(m)
    adjusted[sorted_idx] = adjusted_sorted

    return CorrectionResult(
        method="Benjamini-Hochberg (FDR)",
        metric_names=list(metric_names),
        raw_p_values=p.tolist(),
        adjusted_p_values=adjusted.tolist(),
        is_significant=(adjusted < alpha).tolist(),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Holm-Šídák (family-wise error rate control)
# ---------------------------------------------------------------------------

def holm_sidak(
    p_values: list[float] | np.ndarray,
    metric_names: list[str] | None = None,
    alpha: float = 0.05,
) -> CorrectionResult:
    """Holm-Šídák step-down procedure for strong FWER control.

    Use this for *guardrail* metrics where a single false positive
    could cause a catastrophic ship decision (e.g. cancellation rate
    falsely appearing non-significant).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if metric_names is None:
        metric_names = [f"guardrail_{i}" for i in range(m)]

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        adjusted_sorted[i] = 1.0 - (1.0 - sorted_p[i]) ** (m - i)

    # Enforce monotonicity (step-down)
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[sorted_idx] = adjusted_sorted

    return CorrectionResult(
        method="Holm-Šídák (FWER)",
        metric_names=list(metric_names),
        raw_p_values=p.tolist(),
        adjusted_p_values=adjusted.tolist(),
        is_significant=(adjusted < alpha).tolist(),
        alpha=alpha,
    )
