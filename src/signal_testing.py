"""
src/signal_testing.py — Correlation analysis and OLS regression for sentiment signals.

Tests whether FinBERT-derived sentiment features predict post-earnings
cumulative abnormal returns (CAR).

Pipeline:
  1. correlation_analysis()   — Pearson + Spearman for each feature × CAR window
  2. regression_analysis()    — OLS with HC3 robust SEs, VIF checks
  3. apply_bonferroni()       — Multiple-testing correction across all tests

Usage:
    PYTHONPATH=. .venv/bin/python3.12 src/signal_testing.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature and target definitions
# ---------------------------------------------------------------------------

SENTIMENT_FEATURES = [
    "sentiment_delta",
    "qa_divergence",
    "net_sentiment",
    "ceo_sentiment",
    "analyst_tone",
    "sentiment_variance",
    "negative_chunk_pct",
]

# Reduced feature set for regression — Decision Breakpoint 7.
#
# The full 7-feature model produced severe multicollinearity:
#   net_sentiment      VIF = 56
#   sentiment_variance VIF = 91 (rises to 159 with log_market_cap)
#   ceo_sentiment      VIF = 13
#   analyst_tone       VIF =  8
#   negative_chunk_pct VIF =  8
#
# These features are not independently estimable in a joint regression — they
# are largely measuring the same underlying signal (overall call positivity),
# inflating standard errors and making individual coefficients unreliable.
# The full model is retained in output for transparency (it documents why
# the reduction was made), but all further analysis uses REDUCED_FEATURES.
#
# The three retained features each capture a distinct construct:
#   sentiment_delta  — quarter-over-quarter tone change (VIF 1.3)
#   qa_divergence    — management defensiveness in Q&A vs. prepared remarks (VIF ~1.5 when isolated)
#   analyst_tone     — sell-side analyst questioning pressure (VIF ~2.0 when isolated)
REDUCED_FEATURES = [
    "sentiment_delta",
    "qa_divergence",
    "analyst_tone",
]

CAR_COLUMNS = ["CAR_1d", "CAR_3d", "CAR_5d"]

CONTROL_VARS = ["log_market_cap"]  # sector dummies added dynamically
# NOTE: earnings_surprise is NOT in the dataset — omitted from controls.
# market_cap is ~53% missing; regression sample will be ~8,200 rows vs ~14,000
# for correlation analysis. Both sample sizes are reported in output.

VIF_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# 1. Correlation analysis
# ---------------------------------------------------------------------------

def correlation_analysis(
    df: pd.DataFrame,
    features: list[str],
    car_columns: list[str],
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations between sentiment features and CAR.

    For each (feature, CAR window) pair, computes both Pearson and Spearman
    correlations on the pairwise-complete observations (rows where both the
    feature and the CAR value are non-null).

    Args:
        df: analysis_ready DataFrame.
        features: list of sentiment feature column names.
        car_columns: list of CAR column names to test against.

    Returns:
        DataFrame with columns:
            feature, car_window, n, pearson_r, pearson_p,
            spearman_r, spearman_p, spearman_vs_pearson_gap
        Sorted by abs(spearman_r) descending.
        Rows where |spearman_r| - |pearson_r| > 0.02 are flagged as
        potentially nonlinear (nonlinear_flag=True).
    """
    records = []

    for car_col in car_columns:
        for feat in features:
            # Pairwise-complete observations
            mask = df[[feat, car_col]].notna().all(axis=1)
            sub = df.loc[mask, [feat, car_col]]
            n = len(sub)

            if n < 30:
                logger.warning(
                    "correlation_analysis: only %d observations for (%s, %s) — skipping",
                    n, feat, car_col,
                )
                continue

            x = sub[feat].values
            y = sub[car_col].values

            p_r, p_p = stats.pearsonr(x, y)
            s_r, s_p = stats.spearmanr(x, y)

            gap = abs(s_r) - abs(p_r)

            records.append({
                "feature": feat,
                "car_window": car_col,
                "n": n,
                "pearson_r": round(p_r, 6),
                "pearson_p": round(p_p, 6),
                "spearman_r": round(s_r, 6),
                "spearman_p": round(s_p, 6),
                "spearman_vs_pearson_gap": round(gap, 6),
            })

    results = pd.DataFrame(records)
    results = results.sort_values("spearman_r", key=abs, ascending=False).reset_index(drop=True)

    # Flag pairs where Spearman materially exceeds Pearson (suggests nonlinearity)
    results["nonlinear_flag"] = results["spearman_vs_pearson_gap"] > 0.02

    logger.info(
        "correlation_analysis: %d feature × window pairs tested; "
        "%d with nonlinear_flag=True",
        len(results),
        results["nonlinear_flag"].sum(),
    )
    return results


# ---------------------------------------------------------------------------
# 2. OLS regression with HC3 robust standard errors
# ---------------------------------------------------------------------------

def regression_analysis(
    df: pd.DataFrame,
    target_car: str,
    sentiment_features: list[str],
    control_vars: list[str] | None = None,
) -> dict:
    """Run OLS regression of CAR on sentiment features plus controls.

    Uses HC3 heteroskedasticity-robust standard errors. Sector fixed effects
    are included as dummies (base category dropped). VIF is computed on the
    design matrix (excluding the constant) to flag multicollinearity.

    NOTE: earnings_surprise is NOT available in the dataset. If it becomes
    available, add it to control_vars before calling this function.

    NOTE: market_cap is ~53% missing. Including log_market_cap as a control
    restricts the sample to ~8,200 rows. Regression is run on this reduced
    sample and the sample size is reported.

    Args:
        df: analysis_ready DataFrame with log_market_cap already computed.
        target_car: name of the CAR column to use as dependent variable
            (e.g. "CAR_3d").
        sentiment_features: list of sentiment feature column names.
        control_vars: list of additional numeric control columns to include.
            Pass ["log_market_cap"] to include market cap. Pass [] or None
            to run without controls (full ~14k sample).

    Returns:
        dict with keys:
            summary_df  — per-coefficient results (coef, se, t, p, ci_lo, ci_hi)
            r_squared   — float
            adj_r_squared — float
            n_obs       — int
            vif_df      — DataFrame with feature, VIF; flags VIF > threshold
            model       — fitted statsmodels RegressionResultsWrapper
    """
    if control_vars is None:
        control_vars = []

    all_features = sentiment_features + control_vars

    # Build sector dummies — always include if sector is available
    # Drop NaN sector rows only when sector is actually used
    sector_dummies = None
    if "sector" in df.columns and df["sector"].notna().any():
        sector_dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=True)
        sector_dummies = sector_dummies.astype(float)

    # Assemble design matrix
    cols_needed = [target_car] + all_features
    mask = df[cols_needed].notna().all(axis=1)
    if sector_dummies is not None:
        # Only require sector non-null if sector column present
        mask &= df["sector"].notna()

    sub = df.loc[mask].copy()

    if sector_dummies is not None:
        sub_dummies = sector_dummies.loc[sub.index]
        X = pd.concat([sub[all_features], sub_dummies], axis=1)
    else:
        X = sub[all_features].copy()

    y = sub[target_car]
    X_const = add_constant(X, has_constant="add")

    model = OLS(y, X_const).fit(cov_type="HC3")

    # Coefficient summary table
    coef_df = pd.DataFrame({
        "coef": model.params,
        "se": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
        "ci_lo": model.conf_int()[0],
        "ci_hi": model.conf_int()[1],
    }).drop(index="const", errors="ignore")
    coef_df = coef_df.round(6)

    # VIF — computed on non-constant columns only
    # Use only numeric/sentinel columns (not the const)
    vif_matrix = X.copy()
    vif_vals = []
    for i, col in enumerate(vif_matrix.columns):
        try:
            v = variance_inflation_factor(vif_matrix.values, i)
        except Exception:
            v = np.nan
        vif_vals.append({"feature": col, "vif": round(v, 2)})

    vif_df = pd.DataFrame(vif_vals)
    vif_df["high_vif"] = vif_df["vif"] > VIF_THRESHOLD

    n_high_vif = vif_df["high_vif"].sum()
    if n_high_vif > 0:
        logger.warning(
            "regression_analysis (%s): %d features with VIF > %.1f: %s",
            target_car,
            n_high_vif,
            VIF_THRESHOLD,
            vif_df.loc[vif_df["high_vif"], "feature"].tolist(),
        )

    logger.info(
        "regression_analysis: target=%s, n=%d, R²=%.4f, adj_R²=%.4f",
        target_car, len(sub), model.rsquared, model.rsquared_adj,
    )

    return {
        "summary_df": coef_df,
        "r_squared": round(model.rsquared, 6),
        "adj_r_squared": round(model.rsquared_adj, 6),
        "n_obs": len(sub),
        "vif_df": vif_df,
        "model": model,
    }


# ---------------------------------------------------------------------------
# 3. Bonferroni correction
# ---------------------------------------------------------------------------

def apply_bonferroni(
    results_df: pd.DataFrame,
    p_col: str = "spearman_p",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Bonferroni correction to a results DataFrame.

    The adjusted significance threshold is alpha / n_tests, where n_tests
    is the total number of rows in results_df (i.e., all tests are counted,
    not just those in a single window). This is intentionally conservative.

    Args:
        results_df: DataFrame with at least one p-value column.
        p_col: name of the column containing the raw p-values to correct.
        alpha: family-wise error rate (default 0.05).

    Returns:
        Input DataFrame with two additional columns:
            bonferroni_threshold  — alpha / n_tests (same value in every row)
            survives_bonferroni   — True if p_col < bonferroni_threshold
    """
    df = results_df.copy()
    n_tests = len(df)
    threshold = alpha / n_tests

    df["bonferroni_threshold"] = round(threshold, 8)
    df["survives_bonferroni"] = df[p_col] < threshold

    n_survive = df["survives_bonferroni"].sum()
    logger.info(
        "apply_bonferroni: %d total tests, threshold=%.6f, "
        "%d survive (%.1f%%)",
        n_tests, threshold, n_survive, 100 * n_survive / n_tests,
    )
    return df


# ---------------------------------------------------------------------------
# Main — runs full pipeline and prints results at each stop point
# ---------------------------------------------------------------------------

def _fmt_corr_table(df: pd.DataFrame) -> str:
    """Format correlation results for display."""
    display_cols = [
        "feature", "car_window", "n",
        "pearson_r", "pearson_p",
        "spearman_r", "spearman_p",
        "spearman_vs_pearson_gap", "nonlinear_flag",
    ]
    return df[display_cols].to_string(index=True)


def _fmt_regression(result: dict, target_car: str) -> str:
    """Format regression results for display."""
    lines = [
        f"\n{'='*70}",
        f"OLS Regression: {target_car} ~ sentiment_features + controls",
        f"{'='*70}",
        f"N = {result['n_obs']:,}   R² = {result['r_squared']:.4f}   "
        f"Adj R² = {result['adj_r_squared']:.4f}",
        f"Standard errors: HC3 (heteroskedasticity-robust)",
        "",
        "Coefficients (sentiment features and controls only):",
    ]

    coef_df = result["summary_df"]
    # Show sentiment features first, then controls, then sector dummies summary
    sentiment_rows = coef_df[coef_df.index.isin(SENTIMENT_FEATURES)]
    control_rows = coef_df[~coef_df.index.isin(SENTIMENT_FEATURES) &
                           ~coef_df.index.str.startswith("sector_")]
    sector_rows = coef_df[coef_df.index.str.startswith("sector_")]

    if not sentiment_rows.empty:
        lines.append(sentiment_rows.to_string())
    if not control_rows.empty:
        lines.append("\nControls:")
        lines.append(control_rows.to_string())
    if not sector_rows.empty:
        lines.append(f"\nSector dummies: {len(sector_rows)} coefficients (not shown)")

    lines += [
        "",
        "VIF check (sentiment features + controls):",
        result["vif_df"][~result["vif_df"]["feature"].str.startswith("sector_")].to_string(index=False),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Load data
    path = config.PROCESSED_DIR / "analysis_ready.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    # Add log_market_cap (log of market cap in millions, NaN where missing)
    df["log_market_cap"] = np.where(
        df["market_cap"].notna() & (df["market_cap"] > 0),
        np.log(df["market_cap"]),
        np.nan,
    )

    # ------------------------------------------------------------------
    # STEP 1: Correlation analysis
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: CORRELATION ANALYSIS")
    print("="*70)
    print(f"Sample: {df[CAR_COLUMNS[0]].notna().sum():,} events with CAR data")
    print(f"Features tested: {SENTIMENT_FEATURES}")
    print(f"CAR windows: {CAR_COLUMNS}")

    corr_results = correlation_analysis(df, SENTIMENT_FEATURES, CAR_COLUMNS)
    print("\nFull correlation table (sorted by |spearman_r| descending):\n")
    print(_fmt_corr_table(corr_results))

    nonlinear = corr_results[corr_results["nonlinear_flag"]]
    if not nonlinear.empty:
        print(f"\n{'='*70}")
        print(f"Nonlinear flags (|spearman_r| - |pearson_r| > 0.02): {len(nonlinear)} pairs")
        print(_fmt_corr_table(nonlinear))
    else:
        print("\nNo pairs flagged as potentially nonlinear (gap <= 0.02).")

    print("\n" + "="*70)
    print("STOP 1: Review correlations above before proceeding.")
    print("Run with --step2 to continue to regression.")
    print("="*70)

    import sys
    if "--step2" not in sys.argv:
        sys.exit(0)

    # ------------------------------------------------------------------
    # STEP 2: Regression — CAR_3d
    #   Full 7-feature model shown first (documents multicollinearity).
    #   Reduced 3-feature model is the primary specification going forward.
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2: REGRESSION — CAR_3d")
    print("="*70)
    print("NOTE: earnings_surprise not in dataset — omitted from controls.")
    print("Full model shown to document multicollinearity; reduced model is primary.")
    print("Each model run in two variants: (A) full sample; (B) + log_market_cap + sector\n")

    # Full model — for reference / VIF documentation only
    reg_3d_full_all = regression_analysis(
        df,
        target_car="CAR_3d",
        sentiment_features=SENTIMENT_FEATURES,
        control_vars=[],
    )
    print(_fmt_regression(reg_3d_full_all, "CAR_3d — FULL model, Model A (no controls) [VIF reference]"))

    reg_3d_ctrl_all = regression_analysis(
        df,
        target_car="CAR_3d",
        sentiment_features=SENTIMENT_FEATURES,
        control_vars=["log_market_cap"],
    )
    print(_fmt_regression(reg_3d_ctrl_all, "CAR_3d — FULL model, Model B (+ log_market_cap + sector) [VIF reference]"))

    # Reduced model — primary specification
    reg_3d_full = regression_analysis(
        df,
        target_car="CAR_3d",
        sentiment_features=REDUCED_FEATURES,
        control_vars=[],
    )
    print(_fmt_regression(reg_3d_full, "CAR_3d — REDUCED model, Model A (full sample, no controls)"))

    reg_3d = regression_analysis(
        df,
        target_car="CAR_3d",
        sentiment_features=REDUCED_FEATURES,
        control_vars=["log_market_cap"],
    )
    print(_fmt_regression(reg_3d, "CAR_3d — REDUCED model, Model B (+ log_market_cap + sector)"))

    print("\n" + "="*70)
    print("STOP 2: Review CAR_3d regressions above before proceeding.")
    print("Run with --step3 to continue to CAR_1d and CAR_5d.")
    print("="*70)

    if "--step3" not in sys.argv:
        sys.exit(0)

    # ------------------------------------------------------------------
    # STEP 3: Regressions — CAR_1d and CAR_5d (reduced model only)
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 3: REGRESSIONS — CAR_1d and CAR_5d (reduced model)")
    print("="*70)

    reg_1d_full = regression_analysis(
        df,
        target_car="CAR_1d",
        sentiment_features=REDUCED_FEATURES,
        control_vars=[],
    )
    print(_fmt_regression(reg_1d_full, "CAR_1d — REDUCED model, Model A (full sample, no controls)"))

    reg_1d = regression_analysis(
        df,
        target_car="CAR_1d",
        sentiment_features=REDUCED_FEATURES,
        control_vars=["log_market_cap"],
    )
    print(_fmt_regression(reg_1d, "CAR_1d — REDUCED model, Model B (+ log_market_cap + sector)"))

    reg_5d_full = regression_analysis(
        df,
        target_car="CAR_5d",
        sentiment_features=REDUCED_FEATURES,
        control_vars=[],
    )
    print(_fmt_regression(reg_5d_full, "CAR_5d — REDUCED model, Model A (full sample, no controls)"))

    reg_5d = regression_analysis(
        df,
        target_car="CAR_5d",
        sentiment_features=REDUCED_FEATURES,
        control_vars=["log_market_cap"],
    )
    print(_fmt_regression(reg_5d, "CAR_5d — REDUCED model, Model B (+ log_market_cap + sector)"))

    print("\n" + "="*70)
    print("STOP 3: Review CAR_1d and CAR_5d regressions above.")
    print("Run with --step4 to apply Bonferroni correction.")
    print("="*70)

    if "--step4" not in sys.argv:
        sys.exit(0)

    # ------------------------------------------------------------------
    # STEP 4: Bonferroni correction across all tests
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4: BONFERRONI CORRECTION")
    print("="*70)

    # Combine all regression p-values into one pool with the correlation results
    # Correlations: 21 pairs × 2 tests (Pearson + Spearman) = 42 tests
    # Regressions: 3 windows × 7 sentiment features = 21 tests
    # Total: 63 tests
    corr_bonf = apply_bonferroni(corr_results, p_col="spearman_p", alpha=0.05)

    # Build regression p-value table — reduced model only (primary specification)
    reg_results = {
        "CAR_1d (full)": reg_1d_full,
        "CAR_1d (ctrl)": reg_1d,
        "CAR_3d (full)": reg_3d_full,
        "CAR_3d (ctrl)": reg_3d,
        "CAR_5d (full)": reg_5d_full,
        "CAR_5d (ctrl)": reg_5d,
    }
    reg_rows = []
    for car, res in reg_results.items():
        for feat in REDUCED_FEATURES:
            if feat in res["summary_df"].index:
                reg_rows.append({
                    "feature": feat,
                    "car_window": car,
                    "coef": res["summary_df"].loc[feat, "coef"],
                    "p_value": res["summary_df"].loc[feat, "p_value"],
                })
    reg_pvals = pd.DataFrame(reg_rows)
    reg_bonf = apply_bonferroni(reg_pvals, p_col="p_value", alpha=0.05)

    print("\nCorrelation results with Bonferroni correction (spearman_p):")
    print(corr_bonf[["feature", "car_window", "spearman_r", "spearman_p",
                      "bonferroni_threshold", "survives_bonferroni"]].to_string(index=False))

    print("\nRegression p-values with Bonferroni correction:")
    print(reg_bonf.to_string(index=False))

    n_corr_survive = corr_bonf["survives_bonferroni"].sum()
    n_reg_survive = reg_bonf["survives_bonferroni"].sum()
    print(f"\nSummary:")
    print(f"  Correlation tests: {n_corr_survive}/{len(corr_bonf)} survive Bonferroni")
    print(f"  Regression tests:  {n_reg_survive}/{len(reg_bonf)} survive Bonferroni")
