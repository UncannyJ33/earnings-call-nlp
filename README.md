# Earnings Call NLP → Alpha Signal Testing

Extracting sentiment from earnings call transcripts using FinBERT and testing whether tone shifts predict post-earnings stock returns.

---

## The Question

The efficient market hypothesis says public information shouldn't predict returns — by the time a quarterly earnings call ends, any signal in management language should already be priced in. But there are reasons to expect a gap.

Earnings calls are unstructured. Analysts and algorithms are optimized to process numbers — EPS beats, revenue guidance, margin expansion. Language is harder. A CFO can report in-line numbers while conveying stress through hedged phrasing, increased negative language, or a noticeably defensive Q&A session. The behavioral finance literature suggests that markets incorporate qualitative signals more slowly than quantitative ones, particularly for smaller or less-covered companies where fewer analysts are listening for subtle tone shifts.

This project tests that idea empirically: does a quarter-over-quarter change in management tone — as measured by FinBERT sentiment scoring — predict abnormal stock returns in the 1, 3, or 5 trading days following the call? The answer is: yes, weakly, in a subset of sectors. The rest of this document explains what was tested, what the results show, and where the analysis falls short.

---

## Data & Methodology

### Dataset

The [Motley Fool Scraped Earnings Call Transcripts](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts) dataset from Kaggle contains 18,755 earnings call transcripts across 2,876 unique tickers, spanning 2017–2023. Each record includes the ticker, date, exchange, fiscal quarter, and the full transcript text. Post-earnings stock returns were computed using price data from yfinance; 20.1% of events (3,526 of 17,542) were excluded due to missing price data — primarily delisted, acquired, or renamed companies — which introduces survivorship bias toward larger firms.

### FinBERT Sentiment Scoring

[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (via HuggingFace Transformers) was used to score transcript sentiment. FinBERT outputs three probabilities per input — `positive_prob`, `negative_prob`, `neutral_prob` — that sum to 1.0. The score used throughout is **net sentiment**: `positive_prob − negative_prob`.

FinBERT has a hard 512-token limit. A typical earnings call transcript runs 8,000+ words (~10,000 tokens), so transcripts must be split into chunks before scoring.

### Chunking Strategy

Two strategies were implemented and compared:

**Speaker-turn chunking (primary)**: Each speaker turn in the transcript is treated as a separate chunk. Long turns are split at sentence boundaries to stay under 512 tokens. This preserves speaker attribution — CEO prepared remarks, CFO financial commentary, and analyst questions are scored independently. The result is 15–25 chunks per transcript for a typical call.

**Overlap chunking (comparison)**: All speaker turns within each section (prepared remarks, Q&A) are concatenated into a single text, then chunked using a 50%-stride sliding window. Every sentence appears in at least two chunks, which theoretically captures sentiment that falls at a non-overlap chunk boundary.

**Finding**: At the transcript level, the two strategies produce nearly identical signal. Spearman correlations between `sentiment_delta` and `CAR_3d` differed by less than 0.003 across all tested windows. This is expected: chunk boundary effects wash out when aggregating 15–25 chunk scores to a single transcript-level mean. The speaker-turn strategy was retained as primary because it preserves attribution and is more interpretable.

### Aggregation Strategies

Multiple aggregation statistics were computed per transcript:

- **Mean** (`net_sentiment`): overall tone across all chunks — the level signal
- **Standard deviation** (`sentiment_variance`): spread of chunk scores within a call — captures inconsistency and hedging
- **Max positive** (`max_positive`): the most optimistic single chunk — peak tone
- **Min positive** (`min_positive`): the least optimistic chunk — the worst moment in the call

Each captures a different aspect of the transcript. Mean is the workhorse for return prediction; variance was tested as a potential hedge/uncertainty proxy.

### Return Calculation: Cumulative Abnormal Return (CAR)

Post-earnings stock performance is measured as **Cumulative Abnormal Return (CAR)**: the stock's daily return minus the S&P 500 (SPY) return, summed over 1, 3, and 5 trading days following the call.

**Earnings window timing**: Getting the start of the return window right matters. Calls before 4:00 p.m. ET (morning or midday) use the earnings date itself as day 1 — the market is open and reacts immediately. Calls at or after 4:00 p.m. ET use the next trading day — the market is closed when the call happens, so the earliest reaction is the following morning. Call times are extracted directly from the dataset timestamps.

**Why 1, 3, and 5 days?** The 1-day window captures the immediate market reaction. The 3-day window is the standard post-earnings drift window used in academic literature. The 5-day window tests whether any signal persists into the following week — drift or reversal.

**Robustness**: Beta-adjusted CAR was computed as a robustness check. Each stock's beta is estimated on the 120 trading days prior to the earnings event (non-overlapping with the CAR window). Regression coefficients were essentially unchanged between market-adjusted and beta-adjusted specifications, suggesting the results are not driven by systematic risk mismeasurement.

---

## Key Features (Analytical Differentiators)

**Quarter-over-quarter sentiment delta** (`sentiment_delta`): The change in net sentiment from the prior earnings call to the current one, for the same company. This is the primary signal. Raw sentiment level carries substantial company-fixed effects — optimistic management teams are always positive, pessimistic ones are always negative. The delta removes that baseline and asks: *did tone improve or worsen this quarter?* A company that shifts from −0.1 to +0.1 is likely communicating something the prior quarter did not.

**Prepared remarks vs. Q&A divergence** (`qa_divergence`): Prepared remarks are scripted and optimized for message control. The Q&A session is live — analysts ask pointed questions and management must respond in real time. A large negative divergence (Q&A much more negative than prepared remarks) suggests that management's carefully constructed narrative isn't holding up under questioning. This is a proxy for management defensiveness.

**Multi-speaker analysis**: CEO sentiment, CFO sentiment, and analyst tone are scored separately. Management optimism from a CEO is different from the same language from a CFO (whose domain is financial controls and risk). Analyst tone during Q&A captures the sell-side's collective sentiment about the call in real time.

**Sector-level decomposition**: The signal is tested separately for each sector. In sectors where stock prices are driven by macro factors outside management's control (energy, utilities, basic materials), tone is unlikely to add information. In sectors where forward guidance and operational execution drive returns (technology, healthcare, industrials), tone is more likely to carry independent signal.

**Statistical rigor**: All correlations are reported with Bonferroni-corrected p-values (21 tests across 7 features × 3 windows). Regressions use HC3 heteroskedasticity-robust standard errors. Multicollinearity was detected and addressed: the full 7-feature model had VIFs of 56–91 for several features, so a reduced 3-feature model (`sentiment_delta`, `qa_divergence`, `analyst_tone`) was used for regression, retaining only features with distinct constructs. An out-of-sample test was run by training on events through 2021 and testing on 2022–2023.

---

## Results

### Correlations

All seven sentiment features show statistically significant Spearman correlations with all three CAR windows (all p < 0.001 after Bonferroni correction, n = 11,000–14,000 events). The magnitudes are modest:

| Feature | CAR_1d | CAR_3d | CAR_5d |
|---|---|---|---|
| Sentiment Delta (QoQ) | 0.128 | **0.133** | 0.125 |
| Net Sentiment | 0.117 | 0.107 | 0.093 |
| Analyst Tone | 0.111 | 0.103 | 0.091 |
| Q&A Divergence | −0.080 | −0.065 | −0.055 |

`sentiment_delta` consistently produces the strongest signal. Correlations decay slightly moving from 3-day to 5-day windows, suggesting the market incorporates tone information within a few days rather than drifting over the full week.

### Quantile Analysis

Events sorted into quartiles by `sentiment_delta` show a clean monotonic pattern across all four quartiles with no inversions — the staircase pattern that indicates a genuine ordinal relationship rather than a noise artifact driven by extreme observations:

| Quartile | Mean sentiment delta | Mean 3-day CAR |
|---|---|---|
| Q1 (most negative tone shift) | −0.093 | −1.66% |
| Q2 | −0.019 | +0.13% |
| Q3 | +0.025 | +0.66% |
| Q4 (most positive tone shift) | +0.100 | +1.60% |

**Q4−Q1 spread: +3.3 percentage points** (Welch t = 12.87, p < 0.001, n = 11,810).

![Quantile staircase](figures/signal_quantile_staircase.png)

### Sector Breakdown

The signal is concentrated in sectors where management language carries forward-looking information:

**Signal present** (spread significant at p < 0.05): Healthcare (+4.96%), Consumer Defensive (+3.69%), Industrials (+3.65%), Technology (+3.44%), Consumer Cyclical (+2.27%).

**Signal absent**: Energy, Utilities, Basic Materials, Financial Services, Real Estate, and Communication Services (a mixed sector spanning regulated legacy telecom and growth streaming/social whose two sub-groups likely offset each other).

The absence pattern is economically coherent. Energy and commodity stocks move with oil prices, not management tone. Utilities are rate-sensitive bond proxies where sentiment is largely irrelevant to valuation. Financial Services firms face compliance-constrained language that reduces FinBERT's discriminating power.

![Sector breakdown](figures/signal_sector_heatmap.png)

### Alpha Decay

The Q4−Q1 spread by year shows no evidence of monotonic decay:

| Year | Q4−Q1 spread | Significant |
|---|---|---|
| 2019 | +4.93% | Yes |
| 2020 | +3.48% | Yes |
| 2021 | +2.50% | Yes |
| 2022 | +4.10% | Yes |
| 2023 | +6.43% | No (n=80, partial year) |

The 2021 dip (lowest spread, +2.5%) coincides with the meme-stock and COVID-recovery regime — anomalous market conditions rather than structural signal decay. The signal recovers fully in 2022. 2017–2018 are excluded because `sentiment_delta` requires a prior-quarter call, leaving too few valid observations in those early years.

![Alpha decay](figures/signal_alpha_decay.png)

### Out-of-Sample Validation

Quantile thresholds were fixed using events through 2021 (training period) and applied without modification to 2022–2023 events (holdout). The Q4−Q1 spread in the holdout period is directionally consistent with the in-sample result, though narrower and with higher variance given the smaller holdout sample (~1,200 events). The signal does not appear to be a pure backtest artifact.

---

## Honest Caveats

**Post-earnings drift is regime-dependent.** The signal magnitude varies meaningfully by year. A strategy built on this signal would need to survive periods like 2021 where the spread compresses significantly.

**This is not a trading strategy.** The quantile spread of +3.3pp is a gross signal before transaction costs, market impact, or slippage. Real-world transaction costs for small-cap names (which dominate the dataset's long tail) would likely consume most or all of this spread.

**Survivorship bias.** 20.1% of events are missing price data, disproportionately from small-cap and eventually-delisted companies (median market cap $1.1B vs. $5.0B for the retained sample). The analysis overrepresents companies that survived 2017–2023.

**FinBERT domain mismatch.** FinBERT was trained on financial news articles and analyst reports, not earnings call transcripts. The model has no exposure to the specific register of earnings calls — operator instructions, safe-harbour language, and scripted Q&A responses. Preprocessing strips boilerplate, but a model fine-tuned on transcript data would likely perform better.

**No earnings surprise control.** Analyst consensus beat/miss data is not included in the dataset. Sentiment delta may partially proxy for a genuine earnings beat before it is fully priced, rather than capturing incremental information in *how* management talks.

**Correlations, not causation.** A Spearman r of 0.133 means tone explains roughly 1.8% of variance in 3-day CAR. The signal is real but small, and its economic mechanism is not definitively identified here.

---

## Deployment Considerations

Running this as a live signal would require:

**Transcript acquisition.** Motley Fool, Seeking Alpha, and S&P Global all offer transcript APIs. Transcripts are typically available within minutes of call completion. Processing latency for FinBERT inference on a single transcript is under 30 seconds on modern hardware.

**Model fine-tuning.** A model fine-tuned on labeled earnings call data would likely improve discriminating power. The FinBERT domain mismatch compresses scores toward neutral relative to a transcript-native model.

**Scale and licensing.** Processing 2,876 tickers quarterly is manageable. At institutional scale (Russell 3000+), transcript licensing costs from S&P or Refinitiv become a meaningful budget item.

**Signal combination.** A Spearman r of 0.13 is too weak to trade in isolation. The realistic application is as a feature in a multi-signal model that also incorporates earnings surprise, guidance revision, and price momentum.

---

## Technical Stack

Python 3.12 · [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) via HuggingFace Transformers · PyTorch (MPS acceleration on Apple Silicon) · pandas · numpy · scipy · statsmodels · yfinance · plotly · Streamlit

---

## Project Structure

```
earnings-call-nlp/
├── src/
│   ├── data_ingestion.py   — load and validate raw transcripts
│   ├── preprocessing.py    — section parsing, speaker tagging, chunking
│   ├── sentiment.py        — FinBERT batch inference, parquet caching
│   ├── features.py         — sentiment aggregation and feature engineering
│   ├── returns.py          — CAR calculation (market- and beta-adjusted)
│   └── signal_testing.py   — correlation, regression, quantile, OOS tests
├── notebooks/
│   ├── 01_eda.ipynb                — dataset overview and distributions
│   ├── 02_sentiment_analysis.ipynb — FinBERT scores, section/role breakdowns
│   ├── 03_signal_testing.ipynb     — correlations, regression, quantile analysis
│   └── 04_results_summary.ipynb   — executive summary and key charts
├── app/
│   └── streamlit_app.py    — interactive dashboard
├── tests/
│   └── test_returns.py     — pytest suite for CAR calculation logic
├── figures/                — exported chart PNGs (used in notebooks and README)
├── config.py               — all paths and parameters
└── requirements.txt
```

Data files and FinBERT inference cache are not included in the repository (large files). See **Running the Project** below.

---

## Running the Project

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd earnings-call-nlp
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Download the dataset (requires Kaggle API credentials in ~/.kaggle/kaggle.json)
.venv/bin/kaggle datasets download \
  -d tpotterer/motley-fool-scraped-earnings-call-transcripts \
  -p data/raw --unzip

# 3. Run FinBERT inference (25–30 hours on Apple Silicon MPS; auto-resumes)
PYTHONPATH=. .venv/bin/python3.12 src/sentiment.py
# After all batches complete:
PYTHONPATH=. .venv/bin/python3.12 src/sentiment.py --merge

# 4. Build features and compute returns
PYTHONPATH=. .venv/bin/python3.12 src/features.py
PYTHONPATH=. .venv/bin/python3.12 src/returns.py

# 5. Run notebooks in order (01 → 04) via Jupyter
.venv/bin/jupyter notebook

# 6. Launch the interactive dashboard
PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app.py

# 7. Run tests
PYTHONPATH=. .venv/bin/pytest tests/
```

The FinBERT inference step is the bottleneck — approximately 25–30 hours on Apple Silicon MPS. The pipeline is fully resumable: partial results are written after each batch of ~1,750 transcripts, and subsequent runs auto-detect where to continue.

---

## Acknowledgments

Built with assistance from Claude Code for pipeline scaffolding and iteration.
