# Earnings Call NLP

Sentiment analysis of earnings call transcripts using FinBERT, covering 18,755 calls across 2,876 tickers from 2017–2023 (Motley Fool dataset).

## Pipeline overview

```
Raw transcripts (.pkl)
        │
        ▼
  data_ingestion.py   — load, validate, standardize columns
        │
        ▼
  preprocessing.py    — parse sections → tag speakers → chunk
        │              (produces non-overlap + overlap chunks per transcript)
        ▼
  sentiment.py        — FinBERT inference → parquet cache
        │
        ▼
  signal testing      — compare chunking strategies, build return signals
```

## Dataset

- **Source**: [Motley Fool Scraped Earnings Call Transcripts](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts) (Kaggle)
- **Scale**: 18,755 transcripts, 2,876 unique tickers, 2017–2023
- **Format**: pickle file containing a DataFrame with columns: `ticker`, `date`, `exchange`, `q` (fiscal quarter), `transcript`
- **Coverage**: ~54% of transcripts have an explicit Q&A section; the remaining ~46% are parsed using a heuristic analyst speaker boundary

## Setup

```bash
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the dataset (requires Kaggle API credentials in `~/.kaggle/kaggle.json`):

```bash
.venv/bin/kaggle datasets download \
  -d tpotterer/motley-fool-scraped-earnings-call-transcripts \
  -p data/raw --unzip
```

## Preprocessing

`src/preprocessing.py` transforms raw transcript text into per-chunk DataFrames ready for inference. Each transcript goes through four stages:

### 1. Section parsing

Transcripts are split into `prepared_remarks` and `qa_session`. Three detection strategies are applied in order:

1. Explicit `"Questions and Answers:"` header line (~54% of transcripts)
2. Fallback: first analyst speaker line (`"Name -- Firm -- Analyst"`) after a minimum number of prepared-remarks lines (~46% of transcripts)
3. No Q&A found: entire body goes to `prepared_remarks`

### 2. Text cleaning

Boilerplate is stripped: operator procedural phrases, legal safe-harbour sentences, recording/replay notices. Speaker attribution lines and financial content are preserved.

### 3. Speaker tagging

Speaker turns are identified by `"Name -- Title"` or `"Name -- Firm -- Analyst"` lines. Each speaker is assigned a coarse role label: `ceo`, `cfo`, `executive`, `ir`, `analyst`, or `operator`.

### 4. Chunking — two strategies

This is the key design decision. FinBERT has a hard 512-token limit, so transcripts must be split into chunks. Two strategies are produced for every transcript:

**Non-overlap (speaker-turn chunking)**
Each chunk corresponds to one speaker turn. Long turns are split at sentence boundaries to stay under 512 tokens. Chunks are independent and speaker-attributed — a CEO's prepared remarks are a separate chunk from the CFO's, and separate from analyst questions. This is the natural unit of earnings call discourse.

**Overlap (50%-stride sliding windows)**
All speaker turns within a section are concatenated into a single text, then chunked using a sliding window with 50% stride. Each chunk overlaps by half with the previous one. Speaker attribution is lost (chunks are labelled `"mixed"`), but sentiment that falls at a non-overlap chunk boundary is captured.

**Why both?** Sentiment in financial text is often expressed across sentence and turn boundaries. A CFO might hedge a strong statement in the following sentence, or a CEO might qualify remarks made earlier in the same section. The non-overlap strategy can split these in half — one chunk gets the positive signal, the next gets the hedge — and the sentiment averages out. The overlap strategy mitigates this by ensuring that every sentence appears in at least two chunks. Both strategies are scored and compared during signal testing to determine which produces more predictive sentiment signals. The overlap chunk count is typically ~70% of the non-overlap count (not double, because the 50% stride advances by half a window at a time and many turns are short enough to fit in a single chunk under either strategy).

## FinBERT inference

Model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) via HuggingFace Transformers. Three output probabilities per chunk: `positive_prob`, `negative_prob`, `neutral_prob` (sum to 1.0).

> **This step takes a long time — even on capable hardware.**
>
> The dataset produces ~2 million chunks across both chunking strategies. On Apple Silicon MPS (Mac Mini), each batch of 1,750 transcripts takes roughly 2.5–3 hours. The full 11-batch run is a **25–30 hour** wall-clock commitment. Plan accordingly: start the run before an overnight or a long away-from-keyboard window. The pipeline is fully resumable — partial results are written after each batch and the next run auto-detects where to continue.

Inference commands:

```bash
# Run next batch (~1,750 transcripts, auto-resumes from last completed batch)
PYTHONPATH=. .venv/bin/python3.12 src/sentiment.py

# Check progress
PYTHONPATH=. .venv/bin/python3.12 src/sentiment.py --status

# Force a specific start index
PYTHONPATH=. .venv/bin/python3.12 src/sentiment.py --start 3500

# After all 11 batches complete, assemble final cache files
PYTHONPATH=. .venv/bin/python3.12 src/sentiment.py --merge
```

Cache layout (`data/cache/`):

```
partials/
  nooverlap_000000_001750.parquet   ← batch partial files
  overlap_000000_001750.parquet
  ...
finbert_scores_nooverlap.parquet    ← final merged cache (after --merge)
finbert_scores_overlap.parquet
finbert_scores_nooverlap.json       ← provenance sidecar
finbert_scores_overlap.json
```

### Technical notes

- **Batch size**: 32 chunks per forward pass — MPS parallelism saturates quickly at FinBERT's sequence lengths; larger batches don't meaningfully increase throughput
- **Truncation**: chunks estimated at >512 tokens are truncated by the tokenizer; a warning is emitted at inference time. Tightening the chunking limit in preprocessing would eliminate these.
- **Hardware tested**: Mac Mini (Apple Silicon MPS), ~2.5–2.75 hours/batch; MacBook Air M4 (thermal throttling under sustained load), ~5.5 hours/batch
- **Score validation**: a 100-chunk sample was reviewed before committing to the full run — scores were qualitatively sensible (neutral ~67%, positive ~21%, negative ~12%; clearly positive/negative examples validated manually)
