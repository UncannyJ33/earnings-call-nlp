"""
src/data_ingestion.py — Load and validate the Motley Fool earnings call dataset.

The raw dataset is a pickle file containing a DataFrame with these original columns:
  date        — call datetime string (e.g. "Aug 27, 2020, 9:00 p.m. ET")
                 ~379 rows have a list value: ['Company name Q# YYYY', 'date string ET']
  exchange    — exchange + ticker string (e.g. "NASDAQ: BILI")
  q           — fiscal quarter label (e.g. "2021-Q3")
  ticker      — ticker symbol (e.g. "BILI")
  transcript  — full transcript text; section headers are literal lines:
                  "Prepared Remarks:"  and  "Questions and Answers:"

Standardized output columns:
  ticker          — original ticker column, uppercased and stripped
  date            — date-only (no time) extracted from the raw date field, as datetime.date
  fiscal_quarter  — renamed from 'q'
  exchange        — exchange prefix only (NYSE / NASDAQ / etc.)
  transcript_text — renamed from 'transcript'
  has_qa_section  — bool: transcript contains a Q&A section
"""

import re
import pickle
import warnings
from pathlib import Path

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_date_string(raw_date: object) -> str | None:
    """Normalize the 'date' field to a plain date string.

    The raw field has two formats:
      1. str  — "Aug 27, 2020, 9:00 p.m. ET"
      2. list — ['Brunswick (BC 0.66%) Q4 2018 ', 'Jan. 31, 2019 11:00 a.m. ET']

    Returns a string like "Aug 27, 2020" (time and timezone stripped), or None
    on failure.

    Args:
        raw_date: The raw value from the 'date' column.

    Returns:
        A cleaned date string, or None if extraction fails.
    """
    if isinstance(raw_date, list):
        # Join list parts and search for a recognizable date pattern.
        combined = " ".join(str(part) for part in raw_date)
    elif isinstance(raw_date, str):
        combined = raw_date
    else:
        return None

    # Strip trailing timezone abbreviation ("ET", "EST", "EDT", etc.)
    combined = re.sub(r"\s+E[SD]?T$", "", combined.strip())

    # Extract the date portion: "Month DD, YYYY" or "Month. DD, YYYY"
    match = re.search(
        r"(Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|"
        r"May\.?|Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|Sep(?:tember)?\.?|"
        r"Oct(?:ober)?\.?|Nov(?:ember)?\.?|Dec(?:ember)?\.?)"
        r"\s+\d{1,2},?\s+\d{4}",
        combined,
        re.IGNORECASE,
    )
    return match.group(0) if match else None


def _parse_dates(raw_series: pd.Series) -> pd.Series:
    """Convert the raw 'date' Series to pandas datetime (date component only).

    Args:
        raw_series: The original 'date' column from the raw DataFrame.

    Returns:
        A Series of pandas Timestamps (NaT for unparseable rows).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        date_strings = raw_series.apply(_extract_date_string)
        return pd.to_datetime(date_strings, format="mixed", errors="coerce").dt.normalize()


def _clean_exchange(raw_exchange: pd.Series) -> pd.Series:
    """Extract just the exchange name (NYSE / NASDAQ / etc.) from the raw field.

    Raw format: "NYSE: GFF" or "NASDAQ: BILI"

    Args:
        raw_exchange: The original 'exchange' column.

    Returns:
        A Series containing just the exchange prefix string.
    """
    return raw_exchange.astype(str).str.extract(r"^(\w+)")[0].str.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_transcripts(filepath: str | Path) -> pd.DataFrame:
    """Load the Motley Fool earnings call dataset and return a clean DataFrame.

    Handles the pickle format used by the Kaggle dataset. Column names are
    standardized and the date field is parsed to a datetime with time stripped.

    Args:
        filepath: Path to the dataset file (expects .pkl for this dataset;
                  .csv and .json are also handled for future flexibility).

    Returns:
        DataFrame with columns:
          ticker, date, fiscal_quarter, exchange, transcript_text, has_qa_section

    Prints:
        A summary of transcript count, date range, unique companies, and
        average transcript length.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".pkl":
        with open(filepath, "rb") as fh:
            raw_df = pickle.load(fh)
    elif suffix == ".csv":
        raw_df = pd.read_csv(filepath)
    elif suffix == ".json":
        raw_df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}")

    # --- Column mapping ---
    # Original → Standardized
    #   ticker     → ticker          (already clean; just strip/uppercase)
    #   date       → date            (parsed to date-only datetime)
    #   q          → fiscal_quarter  (already clean YYYY-QN label)
    #   exchange   → exchange        (extract prefix only)
    #   transcript → transcript_text (rename for clarity)
    clean_df = pd.DataFrame()
    clean_df["ticker"] = raw_df["ticker"].astype(str).str.strip().str.upper()
    clean_df["date"] = _parse_dates(raw_df["date"])
    clean_df["fiscal_quarter"] = raw_df["q"].astype(str).str.strip()
    clean_df["exchange"] = _clean_exchange(raw_df["exchange"])
    clean_df["transcript_text"] = raw_df["transcript"].astype(str)
    clean_df["has_qa_section"] = clean_df["transcript_text"].str.contains(
        "Questions and Answers:", regex=False
    )

    # --- Summary ---
    n_total = len(clean_df)
    date_range_min = clean_df["date"].min()
    date_range_max = clean_df["date"].max()
    n_companies = clean_df["ticker"].nunique()
    avg_length = clean_df["transcript_text"].str.len().mean()

    print("=" * 55)
    print("  Dataset summary")
    print("=" * 55)
    print(f"  Total transcripts   : {n_total:,}")
    print(f"  Date range          : {date_range_min.date()} → {date_range_max.date()}")
    print(f"  Unique tickers      : {n_companies:,}")
    print(f"  Avg transcript len  : {avg_length:,.0f} chars")
    print(f"  Has Q&A section     : {clean_df['has_qa_section'].sum():,} ({clean_df['has_qa_section'].mean()*100:.1f}%)")
    print("=" * 55)

    return clean_df


def validate_data(df: pd.DataFrame) -> dict:
    """Check the cleaned transcript DataFrame for common data quality issues.

    Checks performed:
      1. Missing / empty transcript text
      2. Unparseable dates (NaT)
      3. Duplicate (ticker, date) pairs
      4. Very short transcripts (< 500 chars — likely malformed)

    Args:
        df: DataFrame returned by load_transcripts().

    Returns:
        A dict mapping issue name → count, e.g.
        {"empty_transcripts": 0, "unparseable_dates": 1, ...}

    Prints:
        A formatted report of each issue and its count.
    """
    issues: dict[str, int] = {}

    # 1. Missing / empty transcript text
    empty_mask = df["transcript_text"].isna() | (df["transcript_text"].str.strip() == "")
    issues["empty_transcripts"] = int(empty_mask.sum())

    # 2. Unparseable dates (NaT after parsing)
    unparseable_mask = df["date"].isna()
    issues["unparseable_dates"] = int(unparseable_mask.sum())

    # 3. Duplicate (ticker, date) pairs — same company, same call date
    dup_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    issues["duplicate_ticker_date_rows"] = int(dup_mask.sum())
    issues["duplicate_ticker_date_groups"] = int(
        df[dup_mask].groupby(["ticker", "date"]).ngroups
    ) if dup_mask.sum() > 0 else 0

    # 4. Very short transcripts — likely parse errors or stubs
    short_mask = df["transcript_text"].str.len() < 500
    issues["short_transcripts_lt500"] = int(short_mask.sum())

    # --- Report ---
    print("=" * 55)
    print("  Data quality report")
    print("=" * 55)
    status = lambda n: "OK" if n == 0 else "WARN"
    print(f"  [{status(issues['empty_transcripts'])}]  Empty / null transcripts     : {issues['empty_transcripts']}")
    print(f"  [{status(issues['unparseable_dates'])}]  Unparseable dates (NaT)      : {issues['unparseable_dates']}")
    print(f"  [{status(issues['duplicate_ticker_date_rows'])}]  Duplicate ticker+date rows   : {issues['duplicate_ticker_date_rows']} ({issues['duplicate_ticker_date_groups']} groups)")
    print(f"  [{status(issues['short_transcripts_lt500'])}]  Short transcripts (<500 ch)  : {issues['short_transcripts_lt500']}")
    print("=" * 55)

    return issues


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_transcripts(config.RAW_DIR / "motley-fool-data.pkl")
    issues = validate_data(df)

    print("\n--- 3 transcript samples (first 200 chars each) ---")
    for idx in [0, 500, 5000]:
        row = df.iloc[idx]
        print(f"\n[row {idx}]  ticker={row['ticker']}  date={row['date'].date()}  q={row['fiscal_quarter']}")
        print(f"  {repr(row['transcript_text'][:200])}")
