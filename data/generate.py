"""
Synthetic streaming experiment data generator.

Produces three realistic experiments with 50 000 users per arm,
daily granularity, and the exact column schema required by the
experiment runner and dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(2024)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_ids(n: int, prefix: str) -> np.ndarray:
    return np.array([f"{prefix}_{i:06d}" for i in range(n)])


def _date_range(start: str, days: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=days, freq="D")


def _random_content_ids(n: int) -> np.ndarray:
    return RNG.choice([f"content_{i:04d}" for i in range(500)], size=n)


def _subscription_plans(n: int, annual_frac: float = 0.25, free_frac: float = 0.10) -> np.ndarray:
    probs = [free_frac, 1 - annual_frac - free_frac, annual_frac]
    return RNG.choice(["free", "monthly", "annual"], size=n, p=probs)


# ---------------------------------------------------------------------------
# Experiment 1 — Autoplay countdown timer (12 s → 5 s)
# ---------------------------------------------------------------------------

def generate_autoplay_experiment(n_per_arm: int = 50_000, days: int = 14) -> pd.DataFrame:
    """Autoplay acceptance rate: control 72 %, variant 76 %.

    Guardrail: content_completion_rate must not drop.
    Novelty: first 3 days show inflated lift (~6 pp) that settles to 4 pp.
    """
    dates = _date_range("2025-06-01", days)
    rows = []

    for variant_label, base_accept, n in [("control", 0.72, n_per_arm), ("variant", 0.76, n_per_arm)]:
        uids = _user_ids(n, f"ap_{variant_label}")
        pre_watch = RNG.lognormal(mean=3.5, sigma=0.6, size=n).clip(1, 300)
        plans = _subscription_plans(n)

        for day_idx, date in enumerate(dates):
            # Novelty: variant gets +2 pp lift on days 0-2
            novelty_boost = 0.02 if (variant_label == "variant" and day_idx < 3) else 0.0
            accept_rate = base_accept + novelty_boost

            accepted = RNG.binomial(1, accept_rate, size=n)
            watch = RNG.lognormal(mean=3.5, sigma=0.6, size=n).clip(0, 300)
            completion = RNG.beta(6, 3, size=n)  # mean ~0.67
            clicked_rec = RNG.binomial(1, 0.031, size=n)
            cancelled = RNG.binomial(1, 0.002, size=n)

            day_df = pd.DataFrame({
                "user_id": uids,
                "variant": variant_label,
                "experiment": "autoplay_countdown",
                "event_date": date,
                "content_id": _random_content_ids(n),
                "watch_time_minutes": np.round(watch, 2),
                "completion_pct": np.round(completion, 4),
                "autoplay_accepted": accepted,
                "clicked_recommendation": clicked_rec,
                "subscription_plan": plans,
                "cancelled_flag": cancelled,
                "pre_experiment_watch_time": np.round(pre_watch, 2),
            })
            rows.append(day_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Experiment 2 — Personalised thumbnail algorithm
# ---------------------------------------------------------------------------

def generate_thumbnail_experiment(n_per_arm: int = 50_000, days: int = 21) -> pd.DataFrame:
    """CTR: control 3.1 %, variant 3.6 %.

    Secondary: watch_time_minutes control 47 min, variant 49 min.
    Guardrail: subscriber_cancellation_rate must not increase.
    """
    dates = _date_range("2025-07-01", days)
    rows = []

    configs = [
        ("control", 0.031, 47.0),
        ("variant", 0.036, 49.0),
    ]

    for variant_label, ctr, avg_watch in configs:
        n = n_per_arm
        uids = _user_ids(n, f"th_{variant_label}")
        pre_watch = RNG.lognormal(mean=3.5, sigma=0.6, size=n).clip(1, 300)
        plans = _subscription_plans(n)

        for day_idx, date in enumerate(dates):
            clicked = RNG.binomial(1, ctr, size=n)
            watch = RNG.lognormal(mean=np.log(avg_watch) - 0.18, sigma=0.6, size=n).clip(0, 300)
            completion = RNG.beta(6, 3, size=n)
            autoplay = RNG.binomial(1, 0.72, size=n)
            cancelled = RNG.binomial(1, 0.002, size=n)

            day_df = pd.DataFrame({
                "user_id": uids,
                "variant": variant_label,
                "experiment": "personalised_thumbnails",
                "event_date": date,
                "content_id": _random_content_ids(n),
                "watch_time_minutes": np.round(watch, 2),
                "completion_pct": np.round(completion, 4),
                "autoplay_accepted": autoplay,
                "clicked_recommendation": clicked,
                "subscription_plan": plans,
                "cancelled_flag": cancelled,
                "pre_experiment_watch_time": np.round(pre_watch, 2),
            })
            rows.append(day_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Experiment 3 — Annual plan upsell placement
# ---------------------------------------------------------------------------

def generate_upsell_experiment(n_per_arm: int = 50_000, days: int = 30) -> pd.DataFrame:
    """Annual plan conversion: control 1.8 %, variant 2.3 %.

    Secondary: monthly_plan_cancellation must not spike.
    """
    dates = _date_range("2025-08-01", days)
    rows = []

    for variant_label, conv_rate, n in [("control", 0.018, n_per_arm), ("variant", 0.023, n_per_arm)]:
        uids = _user_ids(n, f"up_{variant_label}")
        pre_watch = RNG.lognormal(mean=3.5, sigma=0.6, size=n).clip(1, 300)

        # Start everyone as monthly for this test
        plans = np.where(RNG.random(n) < conv_rate, "annual", "monthly")
        cancel_base = 0.003 if variant_label == "control" else 0.0032

        for day_idx, date in enumerate(dates):
            watch = RNG.lognormal(mean=3.5, sigma=0.6, size=n).clip(0, 300)
            completion = RNG.beta(6, 3, size=n)
            autoplay = RNG.binomial(1, 0.72, size=n)
            clicked = RNG.binomial(1, 0.032, size=n)
            cancelled = RNG.binomial(1, cancel_base, size=n)

            day_df = pd.DataFrame({
                "user_id": uids,
                "variant": variant_label,
                "experiment": "annual_upsell_placement",
                "event_date": date,
                "content_id": _random_content_ids(n),
                "watch_time_minutes": np.round(watch, 2),
                "completion_pct": np.round(completion, 4),
                "autoplay_accepted": autoplay,
                "clicked_recommendation": clicked,
                "subscription_plan": plans,
                "cancelled_flag": cancelled,
                "pre_experiment_watch_time": np.round(pre_watch, 2),
            })
            rows.append(day_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def generate_all(out_dir: str | Path = "data") -> dict[str, Path]:
    """Generate all three experiments and write to CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generators = {
        "autoplay_countdown.csv": generate_autoplay_experiment,
        "personalised_thumbnails.csv": generate_thumbnail_experiment,
        "annual_upsell_placement.csv": generate_upsell_experiment,
    }

    paths = {}
    for fname, gen_fn in generators.items():
        path = out / fname
        df = gen_fn()
        df.to_csv(path, index=False)
        rows = len(df)
        users = df["user_id"].nunique()
        print(f"  ✓ {fname}: {rows:,} rows, {users:,} unique users → {path}")
        paths[fname] = path

    return paths


if __name__ == "__main__":
    print("Generating synthetic streaming experiments …")
    generate_all()
    print("Done.")
