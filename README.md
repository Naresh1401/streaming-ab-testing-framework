# 🎬 Streaming A/B Testing Framework

> A production-ready statistical experimentation framework for streaming platforms — built with the rigour expected at Netflix, Spotify, Disney+, and similar data-driven companies.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

## Why Netflix, Spotify and YouTube Run 1,000+ Experiments Simultaneously

Every pixel you see on a streaming platform has been A/B tested. The thumbnail that made you click? Tested across 50 variants. The autoplay countdown? Optimised through dozens of experiments. The subscription upsell at the end of your episode? Positioned by an experiment that ran for 30 days across millions of users.

At scale, even a **0.1 percentage-point improvement** in click-through rate translates to millions of dollars in annual revenue. But running experiments at this scale introduces statistical pitfalls that most off-the-shelf tools don't handle:

- **Peeking at results daily** inflates false positive rates from 5% to 30%+
- **Noisy streaming metrics** (watch time varies wildly by user) demand variance reduction techniques
- **Novelty effects** make every change look good for the first 3 days
- **Multiple metric testing** without correction leads to spurious "wins"

This framework addresses all of these problems.

---

## The Peeking Problem

Imagine you're running an experiment on a new recommendation algorithm. You check the dashboard on Day 3 — p-value is 0.04. Ship it?

**No.** If you planned to run for 14 days but checked every day, your effective significance level isn't 5% — it's closer to **30%**. You've been "peeking", and each peek is another lottery ticket for a false positive.

This framework implements **O'Brien-Fleming alpha spending**, which gives you a mathematically valid way to check results before the experiment ends. Early looks require very strong evidence (z > 4.0 on Day 1 of 14), while the final look uses nearly the full alpha budget. You can peek responsibly.

```
Day  1/14: boundary z = 5.14  (almost impossible to cross by chance)
Day  7/14: boundary z = 2.64  (strong evidence required)
Day 14/14: boundary z = 1.96  (standard threshold)
```

---

## CUPED Explained (No Equations)

Streaming metrics are *noisy*. One user watches 0 minutes; another watches 300. This massive variance makes it hard to detect real effects.

**CUPED** (Controlled-experiment Using Pre-Experiment Data) is like grading on a curve. If a student scored 90% on the midterm and 92% on the final, you're pretty confident they slightly improved. If another student scored 60% on both, you know they stayed flat. By *accounting for each user's baseline behaviour*, you strip away the noise and see the signal more clearly.

In practice: CUPED uses each user's **pre-experiment watch time** to predict their post-experiment behaviour, then measures the treatment effect on the *residual* (the part that baseline behaviour can't explain). This typically **reduces variance by 30-50%**, which is equivalent to running the experiment with 40-100% *more* users — for free.

---

## Novelty Effect: The Biggest Trap in Streaming Experimentation

You deploy a new personalised thumbnail algorithm. Day 1: click-through rate jumps 20%. Day 3: still up 15%. You ship it.

Two weeks later, the lift has decayed to 3%. What happened?

**Novelty effect.** Users notice *anything* new. A different thumbnail catches the eye simply because it's different, not because it's better. In streaming, this is the #1 reason experiments over-estimate long-term impact.

This framework includes an automatic **novelty effect check** that compares the treatment effect in the first 3 days vs. the last 7 days. If the lift decays by more than 20%, it flags the result and warns you to extend the experiment.

---

## Architecture

```
streaming-ab-testing-framework/
│
├── src/
│   ├── stats.py           # Core statistics engine (z-test, t-test, Mann-Whitney,
│   │                       #   CUPED, sequential testing, MDE, sample size)
│   ├── corrections.py     # Multiple testing corrections (BH-FDR, Holm-Šídák)
│   ├── experiment.py      # StreamingExperimentAnalyzer — the main runner
│   ├── guardrails.py      # Guardrail metrics (cancellation, completion, errors)
│   └── report.py          # Auto-generated Markdown experiment reports
│
├── data/
│   └── generate.py        # Synthetic data: 3 realistic streaming experiments
│
├── dashboard/
│   └── app.py             # Streamlit dashboard — upload CSV → full analysis
│
├── tests/
│   └── test_framework.py  # Comprehensive test suite
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic experiments

```bash
python -m data.generate
```

### 3. Run a full analysis in 5 lines

```python
import pandas as pd
from src.experiment import StreamingExperimentAnalyzer

df = pd.read_csv("data/autoplay_countdown.csv")
analyzer = StreamingExperimentAnalyzer()
analyzer.load(df, variant_col="variant", metric_col="autoplay_accepted", metric_type="proportion")

print(analyzer.run_frequentist())
print(analyzer.run_bayesian())
print(analyzer.check_srm())
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Synthetic Experiments & Key Results

### Experiment 1 — Autoplay Countdown Timer (12 s → 5 s)

| Metric | Control | Variant | Lift | p-value | Decision |
|---|---|---|---|---|---|
| autoplay_acceptance_rate | 72.0% | 76.0% | +4.0 pp | < 0.001 | **SHIP** ✅ |
| content_completion_rate (guardrail) | ~66.7% | ~66.7% | ~0 pp | > 0.05 | Passed |

> **Novelty alert**: First 3 days show ~6 pp lift (vs. 4 pp steady-state). The framework detects and flags this automatically.

### Experiment 2 — Personalised Thumbnail Algorithm

| Metric | Control | Variant | Lift | p-value | Decision |
|---|---|---|---|---|---|
| click_through_rate | 3.1% | 3.6% | +0.5 pp (+16.1% relative) | < 0.001 | **SHIP** ✅ |
| watch_time_minutes | 47 min | 49 min | +2.0 min | < 0.001 | Supporting |
| subscriber_cancellation_rate (guardrail) | ~0.2% | ~0.2% | ~0 | > 0.05 | Passed |

### Experiment 3 — Annual Plan Upsell Placement

| Metric | Control | Variant | Lift | p-value | Decision |
|---|---|---|---|---|---|
| annual_plan_conversion_rate | 1.8% | 2.3% | +0.5 pp (+27.8% relative) | < 0.001 | **SHIP** ✅ |
| monthly_plan_cancellation (guardrail) | ~0.3% | ~0.32% | +0.02 pp | > 0.05 | Passed |

> **Business impact**: +0.5 pp lift × 500K daily users × 365 days × $119.88/annual plan = **~$1.4M additional ARR**.

---

## Framework Capabilities

| Capability | Implementation |
|---|---|
| Two-proportion z-test | Binary metrics (subscribe, click, adopt) |
| Welch's t-test | Continuous metrics (watch time, completion rate) |
| Mann-Whitney U | Skewed distributions (streams per week) |
| CUPED variance reduction | Pre-experiment covariate adjustment |
| Sequential testing | O'Brien-Fleming alpha spending |
| Bayesian analysis | Beta-Binomial / Normal posteriors |
| SRM detection | Chi-squared goodness-of-fit |
| Novelty effect check | Early vs. late lift comparison |
| Multiple testing | Benjamini-Hochberg (FDR), Holm-Šídák (FWER) |
| Guardrail system | Cancellation, completion, support contacts, errors |
| Auto-report | Markdown with business impact projections |
| Interactive dashboard | Streamlit with Plotly charts |

---

## License

MIT
