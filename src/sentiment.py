"""
src/sentiment.py — FinBERT inference and caching pipeline.

Pipeline stages:
  1. load_finbert_model()          — download / load ProsusAI/finbert
  2. score_chunks()                — batched inference with tqdm progress bar
  3. cache_scores()                — save parquet + JSON metadata sidecar
  4. load_cached_scores()          — load and validate from parquet
  5. run_transcript_batch()        — preprocess + score N transcripts, save partial
  6. merge_partial_caches()        — combine all partials into final cache files
  7. run_inference_pipeline()      — single-shot orchestration (kept for small datasets)

Segmented inference (full dataset):
  Full inference on 18,755 transcripts takes ~27 hours on Apple Silicon MPS
  (benchmarked at ~0.055s/chunk, ~110 chunks/transcript combined overlap +
  non-overlap, ~2.07M total chunks). To make this manageable the pipeline
  runs in batches of BATCH_SIZE_TRANSCRIPTS (~1,750) transcripts each, with
  each batch taking roughly 3 hours. Partial results are written immediately
  after each batch so a session can be interrupted and resumed at any point
  without losing completed work. Both non-overlap and overlap chunking
  strategies are scored in every batch because the two approaches are
  compared during signal testing — overlap uses a 50% stride window across
  the full section text and may capture sentiment that falls at non-overlap
  chunk boundaries.

  Run batches with:
      PYTHONPATH=. python src/sentiment.py             # auto-resume next batch
      PYTHONPATH=. python src/sentiment.py --status    # show progress
      PYTHONPATH=. python src/sentiment.py --merge     # assemble final cache files
      PYTHONPATH=. python src/sentiment.py --start 3500  # force a specific start index

Cache layout (data/cache/):
  partials/nooverlap_{start:06d}_{end:06d}.parquet  — batch outputs (non-overlap)
  partials/overlap_{start:06d}_{end:06d}.parquet    — batch outputs (overlap)
  finbert_scores_nooverlap.parquet                  — final merged cache
  finbert_scores_overlap.parquet                    — final merged cache
  finbert_scores_nooverlap.json                     — provenance sidecar
  finbert_scores_overlap.json                       — provenance sidecar
"""

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOOVERLAP_CACHE_FILENAME = "finbert_scores_nooverlap.parquet"
OVERLAP_CACHE_FILENAME   = "finbert_scores_overlap.parquet"
PARTIALS_DIR_NAME        = "partials"

# Transcripts per batch. At ~0.055s/chunk and ~110 chunks/transcript combined,
# each batch takes roughly 3 hours on Apple Silicon MPS.
BATCH_SIZE_TRANSCRIPTS = 1_750

# Columns that must be present in a valid scores cache
_REQUIRED_SCORE_COLS = {"positive_prob", "negative_prob", "neutral_prob"}


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_finbert_model() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, torch.device]:
    """Load ProsusAI/finbert tokenizer and model, configured for inference.

    Selects MPS (Apple Silicon) if available; falls back to CPU.
    The model is placed in eval mode so dropout/BatchNorm behave correctly.

    Returns:
        Tuple of (model, tokenizer, device).
    """
    logger.info("Loading FinBERT: %s", config.FINBERT_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL)
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) acceleration")
    else:
        device = torch.device("cpu")
        logger.info("MPS unavailable — using CPU")

    model = model.to(device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# 2. Batch inference
# ---------------------------------------------------------------------------

def score_chunks(
    chunks_df: pd.DataFrame,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run FinBERT on every chunk and append three probability columns.

    Processes texts in batches for GPU/MPS efficiency. Chunks longer than
    512 tokens are truncated (FinBERT's hard limit); a single aggregate
    warning is emitted at the start if any are detected.

    Args:
        chunks_df: DataFrame containing at minimum a 'chunk_text' column.
            Must also have 'chunk_token_count' if truncation pre-checks
            are desired (present in preprocessing output by default).
        model: Loaded FinBERT model in eval mode.
        tokenizer: Corresponding FinBERT tokenizer.
        device: torch.device to run inference on.
        batch_size: Number of chunks per forward pass.

    Returns:
        Copy of chunks_df with three new float columns:
        positive_prob, negative_prob, neutral_prob. Each row sums to 1.0.
    """
    if chunks_df.empty:
        logger.warning("score_chunks: received empty DataFrame, returning as-is.")
        result = chunks_df.copy()
        for col in ("positive_prob", "negative_prob", "neutral_prob"):
            result[col] = pd.Series(dtype=float)
        return result

    # Warn upfront if any chunks will be truncated (uses preprocessed estimates)
    if "chunk_token_count" in chunks_df.columns:
        n_over_limit = (chunks_df["chunk_token_count"] > config.MAX_TOKENS).sum()
        if n_over_limit > 0:
            warnings.warn(
                f"{n_over_limit} chunk(s) have estimated token count > {config.MAX_TOKENS} "
                f"and will be truncated during tokenization. "
                f"Consider tightening the chunking limit in preprocessing.",
                stacklevel=2,
            )

    # Build label → column name mapping from the model's own config
    # (guards against any id2label ordering differences across model versions)
    id2label: dict[int, str] = model.config.id2label  # e.g. {0:'positive', 1:'negative', 2:'neutral'}

    texts = chunks_df["chunk_text"].tolist()
    n_chunks = len(texts)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    all_probs: list[dict[str, float]] = []

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="FinBERT inference", unit="batch"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_chunks)
            batch_texts = texts[start:end]

            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.MAX_TOKENS,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            logits = model(**encoding).logits          # (batch_size, 3)
            probs = torch.softmax(logits, dim=-1)      # probabilities sum to 1
            probs_list = probs.cpu().tolist()

            for row in probs_list:
                all_probs.append({id2label[i]: row[i] for i in range(len(row))})

    result_df = chunks_df.copy()
    result_df["positive_prob"] = [p["positive"] for p in all_probs]
    result_df["negative_prob"] = [p["negative"] for p in all_probs]
    result_df["neutral_prob"]  = [p["neutral"]  for p in all_probs]

    return result_df


# ---------------------------------------------------------------------------
# 3. Caching (write)
# ---------------------------------------------------------------------------

def cache_scores(
    scored_df: pd.DataFrame,
    cache_path: Path,
    *,
    overlap: bool,
    batch_size: int,
    avg_batch_time_s: float,
) -> None:
    """Save a scored DataFrame to parquet with a JSON metadata sidecar.

    The sidecar file shares the same stem as the parquet but has a .json
    extension. It records model provenance for reproducibility.

    Args:
        scored_df: DataFrame output from score_chunks().
        cache_path: Full destination path for the .parquet file.
        overlap: Whether this cache contains overlap (True) or
            non-overlap (False) chunks.
        batch_size: Batch size used during inference.
        avg_batch_time_s: Average wall-clock seconds per batch.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    scored_df.to_parquet(cache_path, index=False)
    logger.info("Saved %d rows → %s", len(scored_df), cache_path)

    metadata = {
        "model": config.FINBERT_MODEL,
        "inference_date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "chunk_count": len(scored_df),
        "overlap": overlap,
        "batch_size": batch_size,
        "avg_batch_time_s": round(avg_batch_time_s, 4),
    }
    sidecar_path = cache_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata sidecar → %s", sidecar_path)


# ---------------------------------------------------------------------------
# 4. Caching (read)
# ---------------------------------------------------------------------------

def load_cached_scores(cache_path: Path) -> pd.DataFrame:
    """Load and validate a cached FinBERT scores parquet file.

    Args:
        cache_path: Path to the .parquet file produced by cache_scores().

    Returns:
        Validated DataFrame with sentiment probability columns present.

    Raises:
        FileNotFoundError: If cache_path does not exist.
        ValueError: If required probability columns are missing.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    df = pd.read_parquet(cache_path)

    missing_cols = _REQUIRED_SCORE_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Cache file '{cache_path.name}' is missing expected columns: {missing_cols}"
        )

    logger.info("Loaded %d rows from %s", len(df), cache_path.name)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _benchmark_throughput(
    sample_texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    n_warmup: int = 2,
) -> float:
    """Measure average inference time per chunk on a small text sample.

    Runs n_warmup forward passes first to prime MPS/CPU caches before timing.

    Args:
        sample_texts: Short list of chunk texts (aim for ~10).
        model: Loaded FinBERT model in eval mode.
        tokenizer: Corresponding tokenizer.
        device: Target device.
        n_warmup: Number of un-timed warm-up passes.

    Returns:
        Seconds per chunk (float).
    """
    probe = sample_texts[:2]
    for _ in range(n_warmup):
        enc = tokenizer(
            probe, padding=True, truncation=True,
            max_length=config.MAX_TOKENS, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)

    enc = tokenizer(
        sample_texts, padding=True, truncation=True,
        max_length=config.MAX_TOKENS, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**enc)
    elapsed = time.perf_counter() - t0

    return elapsed / len(sample_texts)


def _partials_dir(cache_dir: Path) -> Path:
    """Return the partials subdirectory path (does not create it)."""
    return cache_dir / PARTIALS_DIR_NAME


def _partial_path(cache_dir: Path, overlap: bool, start: int, end: int) -> Path:
    """Return the canonical path for a single batch partial file."""
    prefix = "overlap" if overlap else "nooverlap"
    return _partials_dir(cache_dir) / f"{prefix}_{start:06d}_{end:06d}.parquet"


# ---------------------------------------------------------------------------
# 5. Segmented inference — batch helpers
# ---------------------------------------------------------------------------

def get_batch_progress(cache_dir: Path) -> int:
    """Scan completed partial files and return the next transcript start index.

    Reads partial filenames of the form ``nooverlap_{start:06d}_{end:06d}.parquet``
    in the partials subdirectory. Returns the highest ``end`` index seen, which
    is the correct start index for the next batch. Returns 0 if no partials exist.

    Args:
        cache_dir: Root cache directory (data/cache/).

    Returns:
        Next transcript start index (0 if no work done yet).
    """
    partials = list(_partials_dir(cache_dir).glob("nooverlap_*.parquet"))
    if not partials:
        return 0

    max_end = 0
    for p in partials:
        # filename: nooverlap_000000_001750.parquet
        parts = p.stem.split("_")
        try:
            end_idx = int(parts[2])
            max_end = max(max_end, end_idx)
        except (IndexError, ValueError):
            logger.warning("Could not parse partial filename: %s", p.name)

    return max_end


def run_transcript_batch(
    raw_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    cache_dir: Path,
    batch_size: int = 32,
    model: Optional[AutoModelForSequenceClassification] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Path]:
    """Preprocess and score one batch of transcripts, saving partial parquets.

    Loads the FinBERT model if not provided. Preprocesses only the transcripts
    in raw_df.iloc[start_idx:end_idx], scores both non-overlap and overlap chunks,
    and writes two partial parquet files to data/cache/partials/.

    Skips the batch silently if partial files for this exact index range already
    exist (safe to re-run after an interrupted session).

    Args:
        raw_df: Full raw transcripts DataFrame (all 18,755 rows).
        start_idx: Inclusive start index into raw_df.
        end_idx: Exclusive end index into raw_df.
        cache_dir: Root cache directory (data/cache/).
        batch_size: Chunks per FinBERT forward pass.
        model: Pre-loaded FinBERT model. Loaded fresh if None.
        tokenizer: Pre-loaded tokenizer. Loaded fresh if None.
        device: torch.device. Detected if None.

    Returns:
        Dict with 'nooverlap' and 'overlap' keys mapping to partial file paths.
    """
    from src.preprocessing import run_preprocessing_pipeline  # local import to avoid circulars

    no_path = _partial_path(cache_dir, overlap=False, start=start_idx, end=end_idx)
    ov_path = _partial_path(cache_dir, overlap=True,  start=start_idx, end=end_idx)

    if no_path.exists() and ov_path.exists():
        logger.info(
            "Partial files for [%d, %d) already exist — skipping.",
            start_idx, end_idx,
        )
        return {"nooverlap": no_path, "overlap": ov_path}

    batch_raw = raw_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    actual_end = start_idx + len(batch_raw)

    print(f"\n--- Batch [{start_idx:,} – {actual_end:,}) | {len(batch_raw)} transcripts ---")

    # Preprocess this batch (fast: ~262 transcripts/sec)
    print("  Preprocessing...")
    t_pre = time.perf_counter()
    chunks_df = run_preprocessing_pipeline(batch_raw)
    pre_elapsed = time.perf_counter() - t_pre
    print(f"  Preprocessing done in {pre_elapsed:.1f}s — {len(chunks_df):,} total chunks")

    nooverlap_df = chunks_df[~chunks_df["overlap"]].copy().reset_index(drop=True)
    overlap_df   = chunks_df[chunks_df["overlap"]].copy().reset_index(drop=True)
    print(f"  Non-overlap: {len(nooverlap_df):,}  |  Overlap: {len(overlap_df):,}")

    # Load model if not provided
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_finbert_model()

    # Score non-overlap
    print(f"\n  Scoring non-overlap chunks (batch_size={batch_size})...")
    t0 = time.perf_counter()
    scored_no = score_chunks(nooverlap_df, model, tokenizer, device, batch_size)
    elapsed_no = time.perf_counter() - t0
    n_batches_no = (len(nooverlap_df) + batch_size - 1) // batch_size
    _partial_path(cache_dir, overlap=False, start=start_idx, end=actual_end).parent.mkdir(
        parents=True, exist_ok=True
    )
    no_path_actual = _partial_path(cache_dir, overlap=False, start=start_idx, end=actual_end)
    scored_no.to_parquet(no_path_actual, index=False)
    print(f"  Non-overlap done in {elapsed_no:.1f}s  → {no_path_actual.name}")

    # Score overlap
    print(f"\n  Scoring overlap chunks (batch_size={batch_size})...")
    t0 = time.perf_counter()
    scored_ov = score_chunks(overlap_df, model, tokenizer, device, batch_size)
    elapsed_ov = time.perf_counter() - t0
    ov_path_actual = _partial_path(cache_dir, overlap=True, start=start_idx, end=actual_end)
    scored_ov.to_parquet(ov_path_actual, index=False)
    print(f"  Overlap done in {elapsed_ov:.1f}s  → {ov_path_actual.name}")

    batch_elapsed = pre_elapsed + elapsed_no + elapsed_ov
    print(f"\n  Batch total: {batch_elapsed:.1f}s  ({batch_elapsed / 60:.1f} min)")

    return {"nooverlap": no_path_actual, "overlap": ov_path_actual}


def merge_partial_caches(cache_dir: Path, batch_size_for_sidecar: int = 32) -> dict[str, Path]:
    """Concatenate all batch partial files into the final cache parquets.

    Reads all ``nooverlap_*.parquet`` and ``overlap_*.parquet`` files from
    the partials subdirectory, sorts them by start index, checks for coverage
    gaps, and writes the merged result to the final cache file paths. A JSON
    provenance sidecar is written alongside each final parquet.

    Args:
        cache_dir: Root cache directory (data/cache/).
        batch_size_for_sidecar: Batch size to record in the sidecar metadata
            (informational only at merge time).

    Returns:
        Dict with 'nooverlap' and 'overlap' keys mapping to final cache Paths.

    Raises:
        FileNotFoundError: If no partial files are found for a category.
        ValueError: If gaps exist in the transcript index coverage.
    """
    final_paths: dict[str, Path] = {}

    for label, overlap_flag in (("nooverlap", False), ("overlap", True)):
        partials = sorted(
            _partials_dir(cache_dir).glob(f"{label}_*.parquet"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not partials:
            raise FileNotFoundError(
                f"No partial files found for '{label}' in {_partials_dir(cache_dir)}. "
                f"Run batches first."
            )

        # Check for gaps
        prev_end = 0
        for p in partials:
            parts = p.stem.split("_")
            start, end = int(parts[1]), int(parts[2])
            if start != prev_end:
                raise ValueError(
                    f"Coverage gap in {label} partials: expected start={prev_end}, "
                    f"got start={start} in {p.name}. "
                    f"Re-run missing batches before merging."
                )
            prev_end = end

        print(f"Merging {len(partials)} {label} partial files ({prev_end:,} transcripts)...")
        dfs = [pd.read_parquet(p) for p in tqdm(partials, desc=f"Loading {label} partials")]
        merged = pd.concat(dfs, ignore_index=True)

        final_path = cache_dir / (NOOVERLAP_CACHE_FILENAME if not overlap_flag else OVERLAP_CACHE_FILENAME)
        cache_scores(
            merged,
            final_path,
            overlap=overlap_flag,
            batch_size=batch_size_for_sidecar,
            avg_batch_time_s=0.0,  # not meaningful at merge time
        )
        print(f"  Merged {len(merged):,} chunks → {final_path.name}\n")
        final_paths[label] = final_path

    return final_paths


def print_batch_status(cache_dir: Path, total_transcripts: int = 18_755) -> None:
    """Print a human-readable progress report on completed and remaining batches.

    Args:
        cache_dir: Root cache directory (data/cache/).
        total_transcripts: Total number of transcripts in the dataset.
    """
    next_start = get_batch_progress(cache_dir)
    partials = list(_partials_dir(cache_dir).glob("nooverlap_*.parquet"))

    completed_transcripts = next_start
    remaining_transcripts = max(total_transcripts - next_start, 0)
    n_batches_done = len(partials)
    n_batches_remaining = (remaining_transcripts + BATCH_SIZE_TRANSCRIPTS - 1) // BATCH_SIZE_TRANSCRIPTS

    # Estimate time remaining using ~6.1s/transcript combined (0.055s/chunk * 110 chunks)
    secs_per_transcript = 6.1
    est_remaining_secs  = remaining_transcripts * secs_per_transcript

    final_no  = cache_dir / NOOVERLAP_CACHE_FILENAME
    final_ov  = cache_dir / OVERLAP_CACHE_FILENAME
    merged_no = final_no.exists()
    merged_ov = final_ov.exists()

    print(f"\n{'='*60}")
    print(f"  FinBERT inference progress")
    print(f"{'='*60}")
    print(f"  Transcripts processed : {completed_transcripts:,} / {total_transcripts:,}"
          f"  ({100 * completed_transcripts / total_transcripts:.1f}%)")
    print(f"  Batches completed     : {n_batches_done}")
    print(f"  Batches remaining     : {n_batches_remaining}")
    print(f"  Estimated time left   : ~{est_remaining_secs / 3600:.1f} hours")
    print(f"  Final cache merged    : nooverlap={'yes' if merged_no else 'no'}  "
          f"overlap={'yes' if merged_ov else 'no'}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 6. Single-shot orchestration (kept for small / test datasets)
# ---------------------------------------------------------------------------

def run_inference_pipeline(
    processed_df: pd.DataFrame,
    cache_dir: Path,
    *,
    force_rerun: bool = False,
    batch_size: Optional[int] = None,
) -> dict[str, Path]:
    """Run FinBERT on all preprocessed chunks and cache the results.

    Intended for small datasets or testing. For the full 18,755-transcript
    dataset use run_transcript_batch() in a loop (see __main__).

    Non-overlap and overlap chunks are scored and cached in separate files.
    If both cache files already exist, inference is skipped unless
    force_rerun=True.

    Args:
        processed_df: Per-chunk DataFrame from run_preprocessing_pipeline().
            Must contain 'chunk_text' and 'overlap' columns.
        cache_dir: Directory for .parquet and .json sidecar files.
        force_rerun: Re-run inference even when cache files are present.
        batch_size: Chunks per forward pass. Defaults to 32.

    Returns:
        Dict mapping 'nooverlap' and 'overlap' to their cache file Paths.
    """
    nooverlap_path = cache_dir / NOOVERLAP_CACHE_FILENAME
    overlap_path   = cache_dir / OVERLAP_CACHE_FILENAME
    cache_paths    = {"nooverlap": nooverlap_path, "overlap": overlap_path}

    if not force_rerun and nooverlap_path.exists() and overlap_path.exists():
        logger.info("Both cache files exist — skipping (use force_rerun=True to override).")
        return cache_paths

    if batch_size is None:
        batch_size = 32

    nooverlap_df = processed_df[~processed_df["overlap"]].copy().reset_index(drop=True)
    overlap_df   = processed_df[processed_df["overlap"]].copy().reset_index(drop=True)

    model, tokenizer, device = load_finbert_model()
    pipeline_t0 = time.perf_counter()

    if force_rerun or not nooverlap_path.exists():
        t0 = time.perf_counter()
        scored_no = score_chunks(nooverlap_df, model, tokenizer, device, batch_size)
        elapsed_no = time.perf_counter() - t0
        n_batches_no = (len(nooverlap_df) + batch_size - 1) // batch_size
        cache_scores(
            scored_no, nooverlap_path,
            overlap=False, batch_size=batch_size,
            avg_batch_time_s=elapsed_no / max(n_batches_no, 1),
        )

    if force_rerun or not overlap_path.exists():
        t0 = time.perf_counter()
        scored_ov = score_chunks(overlap_df, model, tokenizer, device, batch_size)
        elapsed_ov = time.perf_counter() - t0
        n_batches_ov = (len(overlap_df) + batch_size - 1) // batch_size
        cache_scores(
            scored_ov, overlap_path,
            overlap=True, batch_size=batch_size,
            avg_batch_time_s=elapsed_ov / max(n_batches_ov, 1),
        )

    total_elapsed = time.perf_counter() - pipeline_t0
    print(f"Total inference time: {total_elapsed:.1f}s  ({total_elapsed / 60:.1f} min)")
    return cache_paths


# ---------------------------------------------------------------------------
# CLI entry point — segmented inference
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FinBERT segmented inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/sentiment.py                  # run next ~3-hour batch (auto-resume)
  python src/sentiment.py --status         # show progress
  python src/sentiment.py --merge          # assemble final cache files from partials
  python src/sentiment.py --start 3500     # run batch starting at transcript index 3500
  python src/sentiment.py --batch-size 16  # override chunks-per-forward-pass
        """,
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print progress report and exit.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all completed partial files into final cache parquets.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="IDX",
        help="Transcript start index. Defaults to auto-detected next index.",
    )
    parser.add_argument(
        "--transcripts-per-batch",
        type=int,
        default=BATCH_SIZE_TRANSCRIPTS,
        metavar="N",
        help=f"Transcripts per batch (default: {BATCH_SIZE_TRANSCRIPTS} ≈ 3 hours).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Chunks per FinBERT forward pass (default: 32).",
    )
    args = parser.parse_args()

    from src.data_ingestion import load_transcripts

    # ------------------------------------------------------------------
    # --status: just show progress
    # ------------------------------------------------------------------
    if args.status:
        print_batch_status(config.CACHE_DIR)
        import sys; sys.exit(0)

    # ------------------------------------------------------------------
    # --merge: assemble final cache files
    # ------------------------------------------------------------------
    if args.merge:
        print_batch_status(config.CACHE_DIR)
        paths = merge_partial_caches(config.CACHE_DIR, batch_size_for_sidecar=args.batch_size)
        print("Merge complete.")
        for label, p in paths.items():
            df = pd.read_parquet(p)
            print(f"  {label}: {len(df):,} chunks → {p}")
        import sys; sys.exit(0)

    # ------------------------------------------------------------------
    # Default: run next batch
    # ------------------------------------------------------------------
    print("Loading raw transcripts...")
    raw_df = load_transcripts(config.RAW_DIR / "motley-fool-data.pkl")
    total_transcripts = len(raw_df)

    print_batch_status(config.CACHE_DIR, total_transcripts=total_transcripts)

    start_idx = args.start if args.start is not None else get_batch_progress(config.CACHE_DIR)
    end_idx   = min(start_idx + args.transcripts_per_batch, total_transcripts)

    if start_idx >= total_transcripts:
        print("All transcripts have been processed. Run --merge to assemble final caches.")
        import sys; sys.exit(0)

    print(f"Next batch: transcripts [{start_idx:,} – {end_idx:,})  "
          f"({end_idx - start_idx:,} transcripts)")
    print(f"Estimated batch time: ~{(end_idx - start_idx) * 6.1 / 3600:.1f} hours\n")

    # Load model once so it's not re-loaded between overlap/non-overlap passes
    model, tokenizer, device = load_finbert_model()
    print(f"Device: {device}\n")

    t_batch_start = time.perf_counter()
    run_transcript_batch(
        raw_df,
        start_idx=start_idx,
        end_idx=end_idx,
        cache_dir=config.CACHE_DIR,
        batch_size=args.batch_size,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    batch_wall = time.perf_counter() - t_batch_start

    # Show updated progress
    print_batch_status(config.CACHE_DIR, total_transcripts=total_transcripts)
    print(f"Batch finished in {batch_wall / 3600:.2f} hours. "
          f"Run again to continue, or --merge when all batches are done.")
