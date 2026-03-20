"""
Streamlit dashboard for streaming A/B experiment analysis.

Launch:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.experiment import StreamingExperimentAnalyzer
from src.guardrails import run_guardrails
from src.stats import sequential_test, obrien_fleming_boundary
from src.report import generate_report, ReportConfig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Streaming A/B Testing Dashboard",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Streaming A/B Testing Dashboard")
st.markdown("Upload an experiment CSV to get a full frequentist + Bayesian analysis, "
            "CUPED variance reduction, guardrail checks, and a downloadable report.")

# ---------------------------------------------------------------------------
# Sidebar — Upload & settings
# ---------------------------------------------------------------------------

st.sidebar.header("Experiment Setup")
uploaded = st.sidebar.file_uploader("Upload experiment CSV", type=["csv"])

if uploaded is None:
    st.info("👈 Upload an experiment CSV to begin (or generate one with `python -m data.generate`).")
    st.stop()

df = pd.read_csv(uploaded)

st.sidebar.success(f"Loaded {len(df):,} rows")

variant_col = st.sidebar.selectbox("Variant column", df.columns, index=list(df.columns).index("variant") if "variant" in df.columns else 0)
variants = sorted(df[variant_col].unique())
control_label = st.sidebar.selectbox("Control label", variants, index=0)
variant_label = st.sidebar.selectbox("Variant label", [v for v in variants if v != control_label], index=0)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
metric_col = st.sidebar.selectbox("Primary metric", numeric_cols)
metric_type = st.sidebar.radio("Metric type", ["proportion", "continuous"])
alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

has_covariate = "pre_experiment_watch_time" in df.columns
has_date = "event_date" in df.columns

# Experiment metadata for report
exp_name = st.sidebar.text_input("Experiment name", "Unnamed Experiment")
exp_owner = st.sidebar.text_input("Owner", "Experimentation Team")
exp_hypothesis = st.sidebar.text_input("Hypothesis", "Variant improves the primary metric")

# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

analyzer = StreamingExperimentAnalyzer()
analyzer.load(df, variant_col, metric_col, metric_type, control_label, variant_label)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

tab_freq, tab_bayes, tab_cuped, tab_seq, tab_ts, tab_guard, tab_report = st.tabs([
    "Frequentist", "Bayesian", "CUPED", "Sequential", "Time Series", "Guardrails", "Report",
])

# ---- Frequentist ---------------------------------------------------------

with tab_freq:
    st.header("Frequentist Analysis")
    freq = analyzer.run_frequentist(alpha=alpha)
    tr = freq.test_result

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("p-value", f"{tr.p_value:.6f}")
    col2.metric("Absolute lift", f"{tr.lift_abs:+.6f}")
    col3.metric("Relative lift", f"{tr.lift_rel_pct:+.2f}%")
    col4.metric("Power", f"{tr.power_achieved:.1%}" if tr.power_achieved else "N/A")

    if tr.is_significant:
        st.success(f"Result is **statistically significant** at α = {alpha} ({tr.test_name})")
    else:
        st.warning(f"Result is **not significant** at α = {alpha} ({tr.test_name})")

    st.markdown(f"**95% CI**: [{tr.ci_lower:+.6f}, {tr.ci_upper:+.6f}]")

    # SRM
    st.subheader("Sample Ratio Mismatch (SRM)")
    srm = analyzer.check_srm()
    if srm.is_imbalanced:
        st.error(srm.message)
    else:
        st.success(srm.message)

    # Novelty
    if has_date:
        st.subheader("Novelty Effect Check")
        novelty = analyzer.novelty_effect_check()
        if novelty.has_novelty_effect:
            st.warning(novelty.message)
        else:
            st.success(novelty.message)
    else:
        novelty = None

# ---- Bayesian ------------------------------------------------------------

with tab_bayes:
    st.header("Bayesian Analysis")
    bayes = analyzer.run_bayesian()

    col1, col2 = st.columns(2)

    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bayes.prob_variant_wins * 100,
        title={"text": "P(Variant Wins)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#636EFA"},
            "steps": [
                {"range": [0, 50], "color": "#FECACA"},
                {"range": [50, 90], "color": "#FEF3C7"},
                {"range": [90, 95], "color": "#D1FAE5"},
                {"range": [95, 100], "color": "#6EE7B7"},
            ],
            "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.8, "value": 95},
        },
    ))
    fig_gauge.update_layout(height=300)
    col1.plotly_chart(fig_gauge, use_container_width=True)

    col2.metric("P(variant wins)", f"{bayes.prob_variant_wins:.1%}")
    col2.metric("Expected loss", f"{bayes.expected_loss:.6f}")
    col2.metric("95% Credible Interval",
                f"[{bayes.credible_interval_95[0]:+.6f}, {bayes.credible_interval_95[1]:+.6f}]")

# ---- CUPED ---------------------------------------------------------------

with tab_cuped:
    st.header("CUPED Variance Reduction")
    if has_covariate:
        cuped = analyzer.apply_cuped(alpha=alpha)

        col1, col2, col3 = st.columns(3)
        col1.metric("Variance reduction", f"{cuped.variance_reduction_pct:.1f}%")
        col2.metric("Equivalent extra users", f"{cuped.equivalent_extra_users_pct:.0f}%")
        col3.metric("CUPED p-value", f"{cuped.p_value:.6f}")

        st.markdown(
            f"**Raw SE**: {cuped.raw_se:.6f} → **CUPED SE**: {cuped.cuped_se:.6f}  \n"
            f"**Raw lift**: {cuped.raw_lift:+.6f} → **CUPED lift**: {cuped.cuped_lift:+.6f}  \n"
            f"**CUPED 95% CI**: [{cuped.ci_lower:+.6f}, {cuped.ci_upper:+.6f}]"
        )

        # Bar chart comparing SE
        fig_se = go.Figure(data=[
            go.Bar(name="Raw", x=["Standard Error"], y=[cuped.raw_se], marker_color="#EF4444"),
            go.Bar(name="CUPED", x=["Standard Error"], y=[cuped.cuped_se], marker_color="#10B981"),
        ])
        fig_se.update_layout(barmode="group", height=300, title="Standard Error: Raw vs CUPED")
        st.plotly_chart(fig_se, use_container_width=True)
    else:
        cuped = None
        st.info("No `pre_experiment_watch_time` column found — CUPED requires a pre-experiment covariate.")

# ---- Sequential testing --------------------------------------------------

with tab_seq:
    st.header("Sequential Testing (O'Brien-Fleming)")
    if has_date:
        df_sorted = df.copy()
        df_sorted["event_date"] = pd.to_datetime(df_sorted["event_date"])
        dates = sorted(df_sorted["event_date"].unique())
        total_looks = len(dates)

        boundaries = []
        z_values = []
        look_dates = []
        reject_flags = []

        for i, d in enumerate(dates, 1):
            subset = df_sorted[df_sorted["event_date"] <= d]
            ctrl = subset.loc[subset[variant_col] == control_label, metric_col].values
            var = subset.loc[subset[variant_col] == variant_label, metric_col].values
            if len(ctrl) < 2 or len(var) < 2:
                continue
            seq = sequential_test(ctrl, var, look=i, total_looks=total_looks, alpha=alpha)
            boundaries.append(seq.spending_boundary)
            z_values.append(seq.observed_z)
            look_dates.append(d)
            reject_flags.append(seq.reject_null)

        fig_seq = go.Figure()
        fig_seq.add_trace(go.Scatter(x=look_dates, y=boundaries, mode="lines", name="OBF Boundary (+)", line=dict(color="red", dash="dash")))
        fig_seq.add_trace(go.Scatter(x=look_dates, y=[-b for b in boundaries], mode="lines", name="OBF Boundary (−)", line=dict(color="red", dash="dash")))
        fig_seq.add_trace(go.Scatter(x=look_dates, y=z_values, mode="lines+markers", name="Observed Z"))
        fig_seq.update_layout(title="Sequential Test: Z-statistic vs O'Brien-Fleming Boundary", xaxis_title="Date", yaxis_title="Z", height=400)
        st.plotly_chart(fig_seq, use_container_width=True)

        if any(reject_flags):
            first_reject = look_dates[reject_flags.index(True)]
            st.success(f"Boundary crossed on **{pd.Timestamp(first_reject).date()}** — early stopping justified.")
        else:
            st.info("Boundary not crossed yet — continue collecting data.")
    else:
        st.info("No `event_date` column found — sequential analysis requires daily data.")

# ---- Time series ---------------------------------------------------------

with tab_ts:
    st.header("Daily Metric Time Series")
    if has_date:
        df_ts = df.copy()
        df_ts["event_date"] = pd.to_datetime(df_ts["event_date"])
        daily = df_ts.groupby(["event_date", variant_col])[metric_col].mean().reset_index()

        fig_ts = px.line(daily, x="event_date", y=metric_col, color=variant_col,
                         title=f"Daily Mean {metric_col} by Variant",
                         labels={"event_date": "Date", metric_col: metric_col})
        fig_ts.update_layout(height=400)
        st.plotly_chart(fig_ts, use_container_width=True)

        # Cumulative
        daily_cum = df_ts.sort_values("event_date").groupby(variant_col).apply(
            lambda g: g.set_index("event_date")[metric_col].expanding().mean().reset_index(),
            include_groups=False,
        ).reset_index(level=0)
        fig_cum = px.line(daily_cum, x="event_date", y=metric_col, color=variant_col,
                          title=f"Cumulative Mean {metric_col} by Variant")
        fig_cum.update_layout(height=400)
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("No `event_date` column — cannot plot time series.")

# ---- Guardrails ----------------------------------------------------------

with tab_guard:
    st.header("Guardrail Metrics")
    guard = run_guardrails(df, variant_col, control_label, variant_label, alpha=alpha)

    for v in guard.verdicts:
        icon = {"SHIP": "🟢", "HOLD": "🟡", "STOP": "🔴"}[v.decision]
        with st.expander(f"{icon} {v.metric_name} — {v.decision}"):
            st.markdown(f"**Threshold**: {v.threshold_description}")
            st.markdown(f"**Lift**: {v.lift_abs:+.6f} ({v.lift_rel_pct:+.2f}%)")
            st.markdown(f"**p-value**: {v.p_value:.4f}")
            st.markdown(f"**Justification**: {v.justification}")

    st.subheader("Overall Guardrail Decision")
    overall_icon = {"SHIP": "🟢", "HOLD": "🟡", "STOP": "🔴"}[guard.overall_decision]
    st.markdown(f"### {overall_icon} {guard.overall_decision}")
    st.markdown(guard.overall_justification)

# ---- Report --------------------------------------------------------------

with tab_report:
    st.header("Experiment Report")

    if has_date:
        dates_parsed = pd.to_datetime(df["event_date"])
        start_date = str(dates_parsed.min().date())
        end_date = str(dates_parsed.max().date())
    else:
        start_date = "N/A"
        end_date = "N/A"

    config = ReportConfig(
        experiment_name=exp_name,
        hypothesis=exp_hypothesis,
        owner=exp_owner,
        start_date=start_date,
        end_date=end_date,
        primary_metric=metric_col,
        total_users=df["user_id"].nunique() if "user_id" in df.columns else len(df),
    )

    projection = analyzer.compute_long_run_projection()

    if novelty is None:
        # Create a placeholder
        from src.experiment import NoveltyResult as NR
        novelty = NR(early_lift=0, late_lift=0, lift_decay_pct=0, has_novelty_effect=False, message="No date column available.")

    report_md = generate_report(config, freq, bayes, cuped, srm, novelty, projection, guard)

    st.markdown(report_md)

    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report_md,
        file_name=f"experiment_report_{exp_name.replace(' ', '_').lower()}.md",
        mime="text/markdown",
    )
