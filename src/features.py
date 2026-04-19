"""
src/features.py — Sentiment feature engineering from cached FinBERT scores.

All features are computed from cached parquet files. FinBERT inference is
never re-run here.

Pipeline:
  1. compute_transcript_sentiment()  — chunk → transcript aggregation
  2. compute_sentiment_delta()       — QoQ change in net sentiment (primary signal)
  3. compute_divergence_features()   — Q&A vs prepared remarks, CEO vs CFO, analyst tone
  4. compute_linguistic_features()   — word count, chunk count, negative chunk pct
  5. build_feature_matrix()          — assembles all features into one DataFrame
                                       and saves to data/processed/features.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Transcript-level sentiment aggregation
# ---------------------------------------------------------------------------

def compute_transcript_sentiment(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level FinBERT scores to one row per transcript.

    Computes overall and per-section sentiment statistics. Each transcript
    is identified by its (ticker, date) pair.

    Args:
        scores_df: DataFrame from load_cached_scores() with columns:
            ticker, date, source_section, positive_prob, negative_prob,
            neutral_prob, chunk_text, role.

    Returns:
        DataFrame with one row per (ticker, date) and columns:
            ticker, date,
            mean_positive        — mean positive_prob across all chunks
            mean_negative        — mean negative_prob across all chunks
            mean_neutral         — mean neutral_prob across all chunks
            net_sentiment        — mean(positive_prob - negative_prob)
            sentiment_variance   — std of positive_prob across all chunks
            max_positive         — highest positive_prob chunk score
            min_positive         — lowest positive_prob chunk score (most negative passage)
            pr_net_sentiment     — net sentiment, prepared_remarks chunks only
            qa_net_sentiment     — net sentiment, qa_session chunks only
            chunk_count          — total chunks for this transcript
    """
    scores_df = scores_df.copy()
    scores_df["net"] = scores_df["positive_prob"] - scores_df["negative_prob"]

    # Overall aggregations
    overall = (
        scores_df
        .groupby(["ticker", "date"])
        .agg(
            mean_positive=("positive_prob", "mean"),
            mean_negative=("negative_prob", "mean"),
            mean_neutral=("neutral_prob", "mean"),
            net_sentiment=("net", "mean"),
            sentiment_variance=("positive_prob", "std"),
            max_positive=("positive_prob", "max"),
            min_positive=("positive_prob", "min"),
            chunk_count=("positive_prob", "count"),
        )
        .reset_index()
    )

    # Per-section net sentiment
    section_net = (
        scores_df
        .groupby(["ticker", "date", "source_section"])["net"]
        .mean()
        .unstack("source_section")
        .rename(columns={
            "prepared_remarks": "pr_net_sentiment",
            "qa_session": "qa_net_sentiment",
        })
        .reset_index()
    )

    # Some transcripts have no Q&A — qa_net_sentiment will be NaN for those
    result = overall.merge(section_net, on=["ticker", "date"], how="left")

    logger.info(
        "compute_transcript_sentiment: %d transcripts, %d with Q&A data",
        len(result),
        result["qa_net_sentiment"].notna().sum(),
    )
    return result


# ---------------------------------------------------------------------------
# 2. Quarter-over-quarter sentiment delta (primary signal)
# ---------------------------------------------------------------------------

def compute_sentiment_delta(transcript_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Compute quarter-over-quarter change in net sentiment per company.

    For each (ticker, date), the delta is net_sentiment minus the net_sentiment
    from the immediately preceding earnings event for that ticker. The first
    appearance of each ticker gets NaN (no prior quarter to compare against).

    Gaps in quarterly coverage are handled naturally: the delta is always
    relative to the immediately prior observed call, regardless of how many
    quarters were skipped.

    Args:
        transcript_sentiment_df: Output of compute_transcript_sentiment().

    Returns:
        Input DataFrame with one additional column:
            sentiment_delta — net_sentiment minus prior quarter net_sentiment.
                              NaN for a ticker's first appearance.
    """
    df = transcript_sentiment_df.sort_values(["ticker", "date"]).copy()
    df["sentiment_delta"] = (
        df.groupby("ticker")["net_sentiment"].diff()
    )
    n_with_delta = df["sentiment_delta"].notna().sum()
    logger.info(
        "compute_sentiment_delta: %d / %d transcripts have a prior-quarter delta",
        n_with_delta, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 3. Divergence features
# ---------------------------------------------------------------------------

def compute_divergence_features(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Q&A vs prepared remarks divergence and speaker-role sentiment.

    Three features per transcript:
      - qa_divergence: Q&A net sentiment minus prepared remarks net sentiment.
        Positive = Q&A is more positive than prepared remarks (unusual).
        Negative = Q&A is more negative (analysts pushing back, management hedging).
      - ceo_sentiment: mean net sentiment of CEO-tagged chunks across both sections.
      - cfo_sentiment: mean net sentiment of CFO-tagged chunks.
      - analyst_tone: mean net sentiment of analyst-tagged chunks in Q&A.

    All four are NaN when the relevant speaker/section is absent.

    Args:
        scores_df: Chunk-level scores DataFrame with columns including
            ticker, date, source_section, role, positive_prob, negative_prob.

    Returns:
        DataFrame with one row per (ticker, date) and columns:
            ticker, date, qa_divergence, ceo_sentiment, cfo_sentiment, analyst_tone.
    """
    df = scores_df.copy()
    df["net"] = df["positive_prob"] - df["negative_prob"]

    # Q&A divergence: qa_net - pr_net
    section_net = (
        df.groupby(["ticker", "date", "source_section"])["net"]
        .mean()
        .unstack("source_section")
    )
    section_net.columns.name = None

    divergence = pd.DataFrame(index=section_net.index)
    pr_col = "prepared_remarks" if "prepared_remarks" in section_net.columns else None
    qa_col = "qa_session" if "qa_session" in section_net.columns else None

    if pr_col and qa_col:
        divergence["qa_divergence"] = section_net[qa_col] - section_net[pr_col]
    else:
        divergence["qa_divergence"] = np.nan

    divergence = divergence.reset_index()

    # Speaker-role sentiment
    role_net = (
        df.groupby(["ticker", "date", "role"])["net"]
        .mean()
        .unstack("role")
    )
    role_net.columns.name = None
    role_net = role_net.reset_index()

    role_cols = {
        "ceo": "ceo_sentiment",
        "cfo": "cfo_sentiment",
        "analyst": "analyst_tone",
    }
    for src, dst in role_cols.items():
        if src not in role_net.columns:
            role_net[dst] = np.nan
        else:
            role_net = role_net.rename(columns={src: dst})

    keep_cols = ["ticker", "date"] + [
        dst for dst in role_cols.values() if dst in role_net.columns
    ]
    role_net = role_net[keep_cols]

    result = divergence.merge(role_net, on=["ticker", "date"], how="left")

    logger.info(
        "compute_divergence_features: %d transcripts, "
        "%d with qa_divergence, %d with ceo_sentiment, %d with analyst_tone",
        len(result),
        result["qa_divergence"].notna().sum(),
        result.get("ceo_sentiment", pd.Series()).notna().sum(),
        result.get("analyst_tone", pd.Series()).notna().sum(),
    )
    return result


# ---------------------------------------------------------------------------
# 4. Linguistic / structural features
# ---------------------------------------------------------------------------

def compute_linguistic_features(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Compute word count, chunk count, and negative chunk proportion.

    These are structural features that serve as controls in regression —
    longer transcripts may have more diffuse sentiment; companies that use
    more negative language overall may have a different base rate.

    Args:
        scores_df: Chunk-level scores DataFrame with columns:
            ticker, date, chunk_text, positive_prob, negative_prob.

    Returns:
        DataFrame with one row per (ticker, date) and columns:
            ticker, date, word_count, chunk_count, negative_chunk_pct,
            sentiment_variance (included here to confirm overlap with
            compute_transcript_sentiment).
    """
    df = scores_df.copy()
    df["word_count_chunk"] = df["chunk_text"].str.split().str.len()
    df["is_negative_chunk"] = df["negative_prob"] > 0.5

    result = (
        df.groupby(["ticker", "date"])
        .agg(
            word_count=("word_count_chunk", "sum"),
            chunk_count=("positive_prob", "count"),
            negative_chunk_pct=("is_negative_chunk", "mean"),
            sentiment_variance=("positive_prob", "std"),
        )
        .reset_index()
    )

    logger.info(
        "compute_linguistic_features: %d transcripts, "
        "avg word_count=%.0f, avg negative_chunk_pct=%.3f",
        len(result),
        result["word_count"].mean(),
        result["negative_chunk_pct"].mean(),
    )
    return result


# ---------------------------------------------------------------------------
# 5. Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Chain all feature functions and return one row per earnings event.

    Saves the result to data/processed/features.parquet.

    Feature decisions made at Decision Breakpoint 5:
    - word_count DROPPED in favour of chunk_count. The two are highly
      correlated (r=0.773) and chunk_count is more directly meaningful
      for this pipeline (it reflects the unit of inference, not raw text
      length). Keeping both would introduce redundant multicollinearity.
    - net_sentiment and sentiment_variance are retained despite r=0.704.
      They capture meaningfully different things (level vs. dispersion of
      sentiment within a call). Watch VIF in regression — if VIF > 5 for
      either, revisit dropping sentiment_variance. ⚠️ MULTICOLLINEARITY FLAG.

    Args:
        scores_df: Chunk-level scores from finbert_scores_nooverlap.parquet.

    Returns:
        DataFrame with columns:
            ticker, date, net_sentiment, sentiment_delta, qa_divergence,
            ceo_sentiment, cfo_sentiment, analyst_tone, sentiment_variance,
            negative_chunk_pct, chunk_count.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transcript_sent = compute_transcript_sentiment(scores_df)
    transcript_sent = compute_sentiment_delta(transcript_sent)
    divergence = compute_divergence_features(scores_df)
    linguistic = compute_linguistic_features(scores_df)

    feature_cols_from_transcript = [
        "ticker", "date", "net_sentiment", "sentiment_delta",
        "sentiment_variance",
    ]
    matrix = transcript_sent[feature_cols_from_transcript].copy()

    matrix = matrix.merge(
        divergence[["ticker", "date", "qa_divergence", "ceo_sentiment", "cfo_sentiment", "analyst_tone"]],
        on=["ticker", "date"], how="left",
    )
    matrix = matrix.merge(
        # word_count excluded — collinear with chunk_count (r=0.773)
        linguistic[["ticker", "date", "chunk_count", "negative_chunk_pct"]],
        on=["ticker", "date"], how="left",
    )

    # Canonical column order
    final_cols = [
        "ticker", "date",
        "net_sentiment", "sentiment_delta",
        "qa_divergence", "ceo_sentiment", "cfo_sentiment", "analyst_tone",
        "sentiment_variance", "negative_chunk_pct", "chunk_count",
    ]
    matrix = matrix[final_cols]

    out_path = config.PROCESSED_DIR / "features.parquet"
    matrix.to_parquet(out_path, index=False)
    logger.info("Feature matrix saved: %d rows → %s", len(matrix), out_path)

    return matrix
