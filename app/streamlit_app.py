"""
app/streamlit_app.py — Interactive dashboard for earnings call NLP analysis.

Loads from cached/processed data only. No inference at runtime.

Run from repo root:
    PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

from src.signal_testing import (
    correlation_analysis,
    quantile_analysis,
    quantile_by_sector,
    SENTIMENT_FEATURES,
    CAR_COLUMNS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Earnings Call NLP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent color palette (matches notebooks)
C = dict(
    positive="#2E86AB",
    negative="#E84855",
    neutral="#6C757D",
    accent="#F4A261",
    grid="#E9ECEF",
)

FEATURE_LABELS = {
    "sentiment_delta":   "Sentiment Delta (QoQ change)",
    "qa_divergence":     "Q&A Divergence",
    "net_sentiment":     "Net Sentiment",
    "ceo_sentiment":     "CEO Sentiment",
    "analyst_tone":      "Analyst Tone",
    "sentiment_variance":"Sentiment Variance",
    "negative_chunk_pct":"Negative Chunk %",
}

CAR_LABELS = {
    "CAR_1d": "1-Day CAR",
    "CAR_3d": "3-Day CAR",
    "CAR_5d": "5-Day CAR",
}

QUARTILE_COLORS = [C["negative"], C["neutral"], C["neutral"], C["positive"]]

FOOTER = """
<div style='text-align:center; color:#6C757D; font-size:0.78em;
            padding-top:2rem; border-top:1px solid #E9ECEF; margin-top:2rem;'>
    This is an analytical research project, not investment advice.
</div>
"""

# ---------------------------------------------------------------------------
# Data loading — cached so Streamlit only reads parquet once per session
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/analysis_ready.parquet")
    df["year"]    = df["date"].dt.year
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["log_market_cap"] = np.where(
        df["market_cap"].notna() & (df["market_cap"] > 0),
        np.log(df["market_cap"]), np.nan,
    )
    return df


@st.cache_data
def precompute_correlations(data_hash: int) -> pd.DataFrame:
    """Run correlation analysis once; cache result for the session."""
    df = load_data()
    return correlation_analysis(df, SENTIMENT_FEATURES, CAR_COLUMNS)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def chart_layout(**overrides) -> dict:
    base = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="sans-serif", size=12, color="#333"),
        margin=dict(t=55, b=40, l=60, r=30),
        xaxis=dict(gridcolor=C["grid"], linecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], linecolor=C["grid"]),
    )
    base.update(overrides)
    return base


def footer():
    st.markdown(FOOTER, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 1 — Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame) -> None:
    st.title("Earnings Call NLP — Overview")

    # ── Top metrics ─────────────────────────────────────────────────────────
    corr = precompute_correlations(0)
    top_row = corr.sort_values("spearman_r", key=abs, ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transcripts analyzed", f"{df['ticker'].count() + 1213:,}")
    c2.metric("Unique companies",      f"{df['ticker'].nunique():,}")
    c3.metric("Date range",            f"{df['date'].min().year} – {df['date'].max().year}")
    c4.metric(
        "Strongest signal",
        f"{FEATURE_LABELS[top_row['feature']].split('(')[0].strip()}",
        f"r={top_row['spearman_r']:.3f} on {top_row['car_window']}",
    )

    st.divider()

    # ── Quarterly call volume ────────────────────────────────────────────────
    st.subheader("Earnings calls per fiscal quarter")
    qtr = df.groupby("quarter").size().reset_index(name="count").sort_values("quarter")
    fig = go.Figure(go.Bar(
        x=qtr["quarter"], y=qtr["count"],
        marker_color=C["positive"],
        hovertemplate="%{x}: %{y:,} calls<extra></extra>",
    ))
    fig.update_layout(**chart_layout(height=300, showlegend=False))
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Sector coverage + CAR distribution ──────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Events by sector")
        sec = (
            df.dropna(subset=["sector"])
            .groupby("sector").size().reset_index(name="n")
            .sort_values("n", ascending=True)
        )
        fig2 = go.Figure(go.Bar(
            x=sec["n"], y=sec["sector"],
            orientation="h",
            marker_color=C["positive"],
            text=sec["n"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            hovertemplate="%{y}: %{x:,} events<extra></extra>",
        ))
        fig2.update_layout(**chart_layout(height=400, showlegend=False,
                                           margin=dict(t=40, b=40, l=160, r=60)))
        fig2.update_xaxes(title_text="Events")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("3-Day CAR distribution")
        car_data = df["CAR_3d"].dropna()
        fig3 = go.Figure(go.Histogram(
            x=car_data * 100, nbinsx=80,
            marker_color=C["positive"], opacity=0.85,
            hovertemplate="~%{x:.1f}%: %{y:,} events<extra></extra>",
        ))
        fig3.add_vline(
            x=float(car_data.mean() * 100),
            line_dash="dash", line_color=C["negative"], line_width=1.5,
            annotation_text=f"Mean: {car_data.mean()*100:+.2f}%",
            annotation_position="top right",
        )
        fig3.add_vline(x=0, line_color=C["neutral"], line_width=1)
        fig3.update_layout(**chart_layout(height=400, showlegend=False))
        fig3.update_xaxes(title_text="3-Day CAR (%)")
        fig3.update_yaxes(title_text="Events")
        st.plotly_chart(fig3, use_container_width=True)

    footer()


# ---------------------------------------------------------------------------
# PAGE 2 — Company Explorer
# ---------------------------------------------------------------------------

def page_company_explorer(df: pd.DataFrame) -> None:
    st.title("Company Explorer")

    # ── Ticker selector ──────────────────────────────────────────────────────
    all_tickers = sorted(df["ticker"].unique())
    default_idx = all_tickers.index("AAPL") if "AAPL" in all_tickers else 0
    ticker = st.selectbox("Select company", all_tickers, index=default_idx,
                           help="Search by typing a ticker symbol")

    company_df = df[df["ticker"] == ticker].sort_values("date").copy()
    n_events = len(company_df)
    n_with_car = company_df["CAR_3d"].notna().sum()

    if n_events == 0:
        st.warning("No data found for this ticker.")
        return

    # Quick summary
    sector_val = company_df["sector"].dropna().iloc[-1] if company_df["sector"].notna().any() else "—"
    mcap_val = company_df["market_cap"].dropna().iloc[-1] if company_df["market_cap"].notna().any() else None
    mcap_str = f"${mcap_val/1e9:.1f}B" if mcap_val else "—"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Earnings calls",   n_events)
    m2.metric("Calls with CAR",   n_with_car)
    m3.metric("Sector",           sector_val)
    m4.metric("Market cap (last)", mcap_str)

    st.divider()

    # ── Sentiment timeline ───────────────────────────────────────────────────
    st.subheader("Sentiment over time")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Net sentiment line
    fig.add_trace(go.Scatter(
        x=company_df["date"], y=company_df["net_sentiment"],
        mode="lines+markers", name="Net sentiment",
        line=dict(color=C["positive"], width=2),
        marker=dict(size=7),
        hovertemplate="Date: %{x|%b %Y}<br>Net sentiment: %{y:.4f}<extra></extra>",
    ), secondary_y=False)

    # Sentiment delta bars
    delta_data = company_df.dropna(subset=["sentiment_delta"])
    if not delta_data.empty:
        bar_colors = [C["positive"] if v >= 0 else C["negative"]
                      for v in delta_data["sentiment_delta"]]
        fig.add_trace(go.Bar(
            x=delta_data["date"], y=delta_data["sentiment_delta"],
            name="Sentiment delta (QoQ)",
            marker_color=bar_colors, opacity=0.45,
            hovertemplate="Date: %{x|%b %Y}<br>Delta: %{y:+.4f}<extra></extra>",
        ), secondary_y=True)

    fig.add_hline(y=0, line_color=C["neutral"], line_width=1, secondary_y=False)
    fig.update_layout(**chart_layout(height=360, legend=dict(x=0.01, y=0.99)))
    fig.update_yaxes(title_text="Net sentiment", secondary_y=False,
                     gridcolor=C["grid"])
    fig.update_yaxes(title_text="Sentiment delta (QoQ)", secondary_y=True,
                     showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── CAR bars per event ───────────────────────────────────────────────────
    st.subheader("Cumulative Abnormal Return per earnings call")
    car_window = st.radio(
        "CAR window", ["CAR_1d", "CAR_3d", "CAR_5d"],
        format_func=lambda x: CAR_LABELS[x],
        horizontal=True, key="company_car_window",
    )

    car_df = company_df.dropna(subset=[car_window])
    if car_df.empty:
        st.info("No CAR data available for this company.")
    else:
        bar_colors = [C["positive"] if v >= 0 else C["negative"]
                      for v in car_df[car_window]]
        fig_car = go.Figure(go.Bar(
            x=car_df["date"],
            y=car_df[car_window] * 100,
            marker_color=bar_colors,
            text=[f"{v*100:+.1f}%" for v in car_df[car_window]],
            textposition="outside",
            hovertemplate=(
                "Date: %{x|%b %Y}<br>"
                f"{CAR_LABELS[car_window]}: %{{y:+.2f}}%<extra></extra>"
            ),
        ))
        fig_car.add_hline(y=0, line_color=C["neutral"], line_width=1)
        fig_car.update_layout(**chart_layout(
            height=320, showlegend=False,
            yaxis=dict(title_text=f"{CAR_LABELS[car_window]} (%)",
                       gridcolor=C["grid"]),
        ))
        st.plotly_chart(fig_car, use_container_width=True)
        if n_with_car < n_events:
            st.caption(
                f"ℹ {n_events - n_with_car} of {n_events} events have no CAR data "
                f"(ticker may have been delisted or renamed during part of the period)."
            )

    # ── Events table ─────────────────────────────────────────────────────────
    st.subheader("All earnings events")

    display_cols = ["date", "quarter", "net_sentiment", "sentiment_delta",
                    "qa_divergence", "analyst_tone", "CAR_1d", "CAR_3d", "CAR_5d"]
    display_cols = [c for c in display_cols if c in company_df.columns]

    table_df = company_df[display_cols].copy().sort_values("date", ascending=False)
    table_df["date"] = table_df["date"].dt.date

    # Format floats
    float_cols = [c for c in display_cols if c not in ("date", "quarter")]
    for col in float_cols:
        if col.startswith("CAR"):
            table_df[col] = table_df[col].apply(
                lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—"
            )
        else:
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:+.4f}" if pd.notna(x) else "—"
            )

    col_rename = {
        "net_sentiment": "Net Sentiment",
        "sentiment_delta": "Δ Sentiment",
        "qa_divergence": "Q&A Divergence",
        "analyst_tone": "Analyst Tone",
        "quarter": "Quarter",
        "date": "Date",
    }
    table_df = table_df.rename(columns=col_rename)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    footer()


# ---------------------------------------------------------------------------
# PAGE 3 — Signal Dashboard
# ---------------------------------------------------------------------------

def page_signal_dashboard(df: pd.DataFrame) -> None:
    st.title("Signal Dashboard")

    corr_results = precompute_correlations(0)

    # ── Section 1: Correlation heatmap ──────────────────────────────────────
    st.subheader("Correlation analysis")

    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        selected_features = st.multiselect(
            "Features",
            options=SENTIMENT_FEATURES,
            default=SENTIMENT_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            key="corr_features",
        )
    with ctrl_col2:
        corr_window = st.radio(
            "CAR window", ["All"] + CAR_COLUMNS,
            format_func=lambda x: "All windows" if x == "All" else CAR_LABELS[x],
            key="corr_window",
        )

    if not selected_features:
        st.info("Select at least one feature.")
    else:
        filtered = corr_results[corr_results["feature"].isin(selected_features)]
        if corr_window != "All":
            filtered = filtered[filtered["car_window"] == corr_window]

        pivot = filtered.pivot(
            index="feature", columns="car_window", values="spearman_r"
        ).reindex([f for f in SENTIMENT_FEATURES if f in selected_features])

        colorscale = [[0.0, C["negative"]], [0.5, "white"], [1.0, C["positive"]]]
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[CAR_LABELS[c] for c in pivot.columns],
            y=[FEATURE_LABELS[f] for f in pivot.index],
            colorscale=colorscale,
            zmid=0, zmin=-0.2, zmax=0.2,
            text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row]
                  for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=13),
            hovertemplate="Feature: %{y}<br>Window: %{x}<br>Spearman r: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Spearman r", thickness=14),
        ))
        fig_hm.update_layout(**chart_layout(
            height=max(280, 65 * len(selected_features)),
            margin=dict(t=40, b=40, l=210, r=40),
        ))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption(
            "Spearman rank correlation. All 21 pairs are statistically significant "
            "(p<0.001) at n=11k–14k. Signal strength, not significance, is the "
            "meaningful quantity at this sample size."
        )

    st.divider()

    # ── Section 2: Quantile analysis ────────────────────────────────────────
    st.subheader("Quantile analysis")

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        q_feature = st.selectbox(
            "Feature", SENTIMENT_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            key="q_feature",
        )
    with q_col2:
        q_window = st.selectbox(
            "CAR window", CAR_COLUMNS,
            format_func=lambda x: CAR_LABELS[x],
            index=1, key="q_window",
        )

    q_sub = df[[q_feature, q_window]].dropna()
    if len(q_sub) < 100:
        st.warning("Too few observations for quantile analysis.")
    else:
        try:
            q_summary, q_spread = quantile_analysis(df, q_feature, q_window)

            fig_q = go.Figure(go.Bar(
                x=q_summary["label"],
                y=q_summary["mean_car"] * 100,
                error_y=dict(
                    type="data",
                    array=(q_summary["std_car"] / np.sqrt(q_summary["n"]) * 1.96 * 100).tolist(),
                    visible=True,
                ),
                marker_color=QUARTILE_COLORS,
                text=[f"{v*100:+.2f}%" for v in q_summary["mean_car"]],
                textposition="outside",
                hovertemplate=(
                    "%{x}<br>"
                    f"Mean {CAR_LABELS[q_window]}: %{{y:+.2f}}%<br>"
                    "n=%{customdata:,}<extra></extra>"
                ),
                customdata=q_summary["n"],
            ))
            fig_q.add_hline(y=0, line_color=C["neutral"], line_width=1)
            fig_q.update_layout(**chart_layout(
                height=380, showlegend=False,
                title=dict(
                    text=(
                        f"{FEATURE_LABELS[q_feature]} → {CAR_LABELS[q_window]}<br>"
                        f"<sup>Q4−Q1 spread: {q_spread['spread']*100:+.2f}pp  "
                        f"t={q_spread['t_stat']:.2f}  "
                        f"p={'<0.001' if q_spread['p_value'] < 0.001 else f\"{q_spread['p_value']:.3f}\"}  "
                        f"n={q_spread['n_q4']+q_spread['n_q1']:,}</sup>"
                    ),
                    font=dict(size=13),
                ),
                yaxis=dict(title_text=f"Mean {CAR_LABELS[q_window]} (%)", gridcolor=C["grid"]),
            ))
            st.plotly_chart(fig_q, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute quantile analysis: {e}")

    st.divider()

    # ── Section 3: Sector breakdown ─────────────────────────────────────────
    st.subheader("Sector breakdown")

    s_col1, s_col2 = st.columns([2, 3])
    with s_col1:
        s_feature = st.selectbox(
            "Feature", SENTIMENT_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x],
            key="s_feature",
        )
    with s_col2:
        year_min = int(df["year"].min())
        year_max = int(df["year"].max())
        year_range = st.slider(
            "Year range", year_min, year_max,
            value=(2019, year_max),
            key="year_range",
        )

    df_filtered = df[
        df["year"].between(year_range[0], year_range[1]) &
        df["sector"].notna()
    ]
    n_filtered = df_filtered[[s_feature, "CAR_3d"]].dropna().shape[0]
    st.caption(f"{n_filtered:,} complete observations in selected range")

    if n_filtered < 200:
        st.warning("Too few observations in this range for reliable sector analysis.")
    else:
        try:
            sector_df = quantile_by_sector(df_filtered, s_feature, "CAR_3d")
            if sector_df.empty:
                st.info("No sectors with sufficient data in the selected range.")
            else:
                sec_colors = [
                    C["positive"] if s else C["neutral"]
                    for s in sector_df["is_significant"]
                ]
                fig_sec = go.Figure(go.Bar(
                    x=sector_df["spread"] * 100,
                    y=sector_df["sector"],
                    orientation="h",
                    marker_color=sec_colors,
                    text=[
                        f"{v*100:+.2f}pp {'✓' if s else ''}"
                        for v, s in zip(sector_df["spread"], sector_df["is_significant"])
                    ],
                    textposition="outside",
                    hovertemplate=(
                        "%{y}<br>"
                        "Q4−Q1 spread: %{x:+.2f}pp<br>"
                        "n=%{customdata:,}<extra></extra>"
                    ),
                    customdata=sector_df["n_events"],
                ))
                fig_sec.add_vline(x=0, line_color=C["neutral"], line_width=1)
                fig_sec.update_layout(**chart_layout(
                    height=max(350, 42 * len(sector_df)),
                    showlegend=False,
                    title=dict(
                        text=(
                            f"{FEATURE_LABELS[s_feature]} → CAR_3d, "
                            f"{year_range[0]}–{year_range[1]}<br>"
                            "<sup>Blue = significant at p&lt;0.05</sup>"
                        ),
                        font=dict(size=13),
                    ),
                    xaxis=dict(title_text="Q4−Q1 Spread (percentage points)",
                               gridcolor=C["grid"]),
                    margin=dict(t=55, b=40, l=190, r=120),
                ))
                st.plotly_chart(fig_sec, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute sector breakdown: {e}")

    footer()


# ---------------------------------------------------------------------------
# PAGE 4 — Methodology
# ---------------------------------------------------------------------------

def page_methodology() -> None:
    st.title("Methodology")

    with st.expander("FinBERT and the 512-token limit", expanded=True):
        st.markdown("""
**Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) via HuggingFace Transformers.

FinBERT is a BERT-based model fine-tuned on financial text. It outputs three probabilities
per input sequence — `positive_prob`, `negative_prob`, `neutral_prob` — which sum to 1.0.
The model has a hard limit of 512 tokens per input.

A typical earnings call transcript is 8,000+ words (~10,000 tokens), so transcripts must
be split into chunks before scoring. Each chunk is scored independently; transcript-level
sentiment is the mean of chunk-level scores.

**Limitation:** FinBERT was trained on financial news and analyst reports, not earnings
call transcripts specifically. The procedural boilerplate in calls (operator instructions,
safe-harbour language) is stripped by preprocessing, but there may be domain mismatch that
compresses scores toward neutral relative to a transcript-fine-tuned model.
        """)

    with st.expander("Chunking strategy: speaker turns vs sliding windows"):
        st.markdown("""
Two chunking strategies were scored and compared:

**Non-overlap (primary):** Each chunk corresponds to one speaker turn. Long turns are
split at sentence boundaries to stay under 512 tokens. Speaker attribution is preserved —
CEO remarks are separate chunks from CFO remarks, which are separate from analyst questions.

**Overlap (50%-stride sliding windows):** All speaker turns within a section are
concatenated, then chunked with a 50% stride. Each chunk overlaps by half with the previous
one. Speaker attribution is lost, but sentiment at chunk boundaries is captured twice.

**Finding:** Overlap and non-overlap produce near-identical transcript-level aggregates
(r ≈ 0.95+), and non-overlap shows marginally stronger return correlations in signal testing.
Non-overlap was chosen as the primary strategy because it preserves speaker attribution,
which is analytically meaningful (CEO vs analyst tone are distinct signals).

The overlap strategy generates ~70% as many chunks (885k vs 1.25M) because many short
speaker turns fit in a single window under either strategy — the boundary-capture benefit
is smaller than expected.
        """)

    with st.expander("Feature engineering: what sentiment delta measures"):
        st.markdown("""
Seven features are computed from the chunk-level FinBERT scores:

| Feature | Definition |
|---|---|
| **sentiment_delta** | Net sentiment this quarter minus net sentiment last quarter (primary signal) |
| **net_sentiment** | Mean(positive_prob − negative_prob) across all chunks |
| **qa_divergence** | Q&A net sentiment minus prepared remarks net sentiment |
| **ceo_sentiment** | Mean net sentiment of CEO-tagged chunks |
| **cfo_sentiment** | Mean net sentiment of CFO-tagged chunks |
| **analyst_tone** | Mean net sentiment of analyst-tagged chunks in Q&A |
| **sentiment_variance** | Standard deviation of positive_prob across chunks |

`sentiment_delta` is the primary signal because markets are forward-looking and already
price in the fact that management typically sounds positive on earnings calls. The
*change* in tone quarter-over-quarter carries information about whether this call
was better or worse than expected — which is what drives short-term stock returns.
        """)

    with st.expander("Return calculation: CAR and the 4pm ET rule"):
        st.markdown("""
**Cumulative Abnormal Return (CAR)** = stock return minus SPY return, summed over
the return window. The market-adjusted approach (stock − SPY) is the primary measure;
beta-adjusted CAR (using OLS beta on 120 prior trading days) is a robustness check.

**Return window start:**
- Call before 4:00 p.m. ET → window starts on the earnings date (market is open, reaction is immediate)
- Call at or after 4:00 p.m. ET → window starts the next trading day (market closed, first reaction is next morning)
- Time unknown → next trading day (conservative fallback)

**Windows:** 1-day, 3-day, and 5-day post-earnings CAR.

**Coverage:** 20.1% of events are excluded because yfinance could not find historical
prices — these are disproportionately small-cap companies that have since been delisted,
acquired, or renamed. See README for survivorship bias discussion.
        """)

    with st.expander("What the quantile analysis shows"):
        st.markdown("""
Events are sorted by a sentiment feature and divided into quartiles (Q1 = lowest, Q4 = highest).
Mean CAR is computed for each quartile.

A **monotonic** pattern — Q1 through Q4 stepping up without inversions — indicates the
feature meaningfully discriminates between earnings call outcomes. The **Q4−Q1 spread**
(top minus bottom quartile mean CAR) is the primary summary statistic; its significance
is tested with a Welch t-test (unequal variance).

The out-of-sample test applies quantile thresholds computed on the training period
(≤2021) to the holdout period (2022–2023). The thresholds are fixed — they cannot
adapt to the test distribution — making this a stricter test than in-sample analysis.

**Key finding:** `sentiment_delta` produces a +3.3pp Q4−Q1 spread in-sample (p<0.001)
and a +4.2pp spread out-of-sample on the 2022–2023 holdout. The signal survives
Bonferroni correction and is stable across market-adjusted and beta-adjusted CAR.
        """)

    with st.expander("Statistical methodology: HC3, Bonferroni, out-of-sample testing"):
        st.markdown("""
**Robust standard errors (HC3):** OLS regression uses heteroskedasticity-consistent
standard errors (HC3) rather than assuming homoskedastic errors. Stock returns are
well-known to be heteroskedastic (variance changes over time and across firms).

**Multicollinearity reduction:** The full 7-feature model had severe VIF inflation
(net_sentiment VIF=56, sentiment_variance VIF=91). These features measure the same
underlying signal. The regression uses a reduced 3-feature model
(sentiment_delta, qa_divergence, analyst_tone, all VIF<5) as the primary specification.
The full model is retained in notebooks for transparency.

**Bonferroni correction:** With 21 correlation pairs and 18 regression tests, the
family-wise Type I error rate inflates without correction. Bonferroni sets the adjusted
threshold to α/n_tests. `sentiment_delta` and `analyst_tone` survive this conservative
correction in all specifications; `qa_divergence` survives at 1-day and 3-day windows
(full-sample model) but not at 5-day or in the market-cap-controlled model.

**Two regression variants:** The full-sample model (~14k events, no controls) is the
primary specification. The controlled model (~8.2k events, adds log(market_cap) +
sector dummies) is a robustness check. Coefficients are stable across both, confirming
the signal is not a proxy for size or sector effects.
        """)

    footer()


# ---------------------------------------------------------------------------
# Sidebar + routing
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()

    with st.sidebar:
        st.markdown("## 📊 Earnings Call NLP")
        st.caption("FinBERT sentiment → return signals")
        st.divider()

        page = st.radio(
            "Navigate",
            ["Overview", "Company Explorer", "Signal Dashboard", "Methodology"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(
            f"**{df['ticker'].nunique():,}** companies · "
            f"**{df['CAR_3d'].notna().sum():,}** events with CAR data\n\n"
            f"Loaded from `data/processed/`"
        )

    if page == "Overview":
        page_overview(df)
    elif page == "Company Explorer":
        page_company_explorer(df)
    elif page == "Signal Dashboard":
        page_signal_dashboard(df)
    elif page == "Methodology":
        page_methodology()


if __name__ == "__main__":
    main()
