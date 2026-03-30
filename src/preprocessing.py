"""
src/preprocessing.py — Transcript parsing and preprocessing pipeline.

Pipeline stages:
  1. parse_sections()      — split transcript into prepared_remarks / qa_session
  2. clean_text()          — remove boilerplate, fix whitespace
  3. tag_speakers()        — identify speaker turns with name + role
  4. tokenize_for_finbert() — chunk turns for FinBERT (≤512 tokens)
  5. run_preprocessing_pipeline() — chain all stages over a full DataFrame

Transcript format (from data exploration):
  - Every transcript starts with a literal "Prepared Remarks:" header line.
  - Q&A starts with "Questions and Answers:" (~54% of transcripts) or is
    embedded without a header (~46%). For the latter, the first analyst
    speaker line is used as the Q&A boundary.
  - Speaker attribution lines follow the pattern:
      "Name -- Title"                         (company speaker)
      "Name -- Firm -- Analyst"               (sell-side analyst)
      "Operator"                              (bare word, no dashes)
  - Transcripts end with a footer:
      "Duration: N minutes\nCall participants:\n..."
      "More TICKER analysis\nAll earnings call transcripts"
"""

import re
from typing import TypedDict

import pandas as pd
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class SpeakerTurn(TypedDict):
    speaker: str
    role: str       # ceo | cfo | executive | ir | analyst | operator | unknown
    text: str


class Chunk(TypedDict):
    source_section: str   # prepared_remarks | qa
    speaker: str
    role: str
    chunk_index: int
    chunk_text: str
    chunk_token_count: int
    overlap: bool


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

# Section header lines (exact match on stripped line)
_RE_PREPARED_REMARKS = re.compile(r"^Prepared Remarks:$", re.IGNORECASE)
_RE_QA_HEADER = re.compile(
    r"^(Questions and Answers:|Questions & Answers:|Question-and-Answer Session|Questions and Answers)$",
    re.IGNORECASE,
)

# Speaker attribution line: "Name -- Title" or "Name -- Firm -- Analyst"
# Must have at least one ' -- ' and be short (< 120 chars, no leading whitespace)
_RE_SPEAKER_LINE = re.compile(r"^([^-\n][^\n]*?) -- (.+)$")

# Bare "Operator" line
_RE_OPERATOR_LINE = re.compile(r"^Operator$", re.IGNORECASE)

# First-question signal used as fallback Q&A boundary
_RE_FIRST_QUESTION = re.compile(
    r"^(Our first question|The first question|We will now begin|"
    r"We now open|We'll now take|We'll now open|"
    r"\[Operator Instructions\].*question)",
    re.IGNORECASE,
)

# Footer markers — everything from here on is metadata, not transcript content
_RE_FOOTER = re.compile(
    r"^(Duration:\s*\d+|Call participants:|More \w+ analysis|"
    r"All earnings call transcripts|\[Operator Closing Remarks\])",
    re.IGNORECASE,
)

# Boilerplate patterns for clean_text()
_BOILERPLATE_PATTERNS = [
    # Operator procedural instructions
    re.compile(
        r"\[Operator Instructions\]", re.IGNORECASE
    ),
    re.compile(
        r"(?:your|the) (?:line is (?:now )?open|phone line is open)",
        re.IGNORECASE,
    ),
    re.compile(
        r"please (?:go ahead|proceed|ask your question)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:thank you for (?:standing by|holding|joining)|"
        r"ladies and gentlemen,? (?:thank you|good (?:morning|afternoon|evening)))",
        re.IGNORECASE,
    ),
    # Legal safe-harbour boilerplate (long sentence mentioning forward-looking)
    re.compile(
        r"(?:this (?:call|conference|discussion) (?:may |will )?contain[s]? "
        r"forward-looking statements?[^.]*\.)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:intended to qualify for the safe harbor[^.]*\.)",
        re.IGNORECASE,
    ),
    # Recording notices
    re.compile(
        r"(?:this (?:call|conference) is being recorded[^.]*\.)",
        re.IGNORECASE,
    ),
    # Webcast / replay boilerplate
    re.compile(
        r"(?:a (?:webcast |audio )?replay of this (?:call|conference)[^.]*\.)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

# Keywords that identify executive vs. IR vs. analyst roles from title strings
_EXEC_CEO_KW = re.compile(
    r"\b(chief executive officer|co-?ceo|co-?founder.{0,30}ceo|"
    r"president.{0,40}ceo|ceo.{0,40}president|chairman.{0,40}ceo)\b",
    re.IGNORECASE,
)
_EXEC_CFO_KW = re.compile(
    r"\b(chief financial officer|co-?cfo|evp.{0,20}cfo|svp.{0,20}cfo)\b",
    re.IGNORECASE,
)
_IR_KW = re.compile(
    r"\b(investor relations|ir |head of ir|vp.{0,10}ir|"
    r"director.{0,20}investor|vice president.{0,20}investor)\b",
    re.IGNORECASE,
)
_ANALYST_KW = re.compile(r"\banalyst\b", re.IGNORECASE)


def _classify_role(title_or_firm: str, third_field: str | None = None) -> str:
    """Map a speaker's title/firm string to a coarse role label.

    Args:
        title_or_firm: The second ' -- ' field from the speaker line.
        third_field:   The optional third field (present for analysts:
                       "Name -- Firm -- Analyst").

    Returns:
        One of: 'ceo' | 'cfo' | 'executive' | 'ir' | 'analyst' | 'operator'
    """
    # Three-field lines: the third field is almost always "Analyst"
    if third_field and _ANALYST_KW.search(third_field):
        return "analyst"

    combined = f"{title_or_firm} {third_field or ''}"

    if _EXEC_CEO_KW.search(combined):
        return "ceo"
    if _EXEC_CFO_KW.search(combined):
        return "cfo"
    if _IR_KW.search(combined):
        return "ir"
    if _ANALYST_KW.search(combined):
        return "analyst"
    # Any other named title is a generic executive
    return "executive"


# ---------------------------------------------------------------------------
# 1. parse_sections
# ---------------------------------------------------------------------------

def parse_sections(transcript_text: str) -> dict[str, str]:
    """Split a transcript into 'prepared_remarks' and 'qa_session' sections.

    Detection strategy (in priority order):
      1. Explicit "Questions and Answers:" header line → definitive split.
      2. Explicit "Question-and-Answer Session" header line → definitive split.
      3. First analyst speaker turn line ("Name -- Firm -- Analyst") after
         the prepared remarks block → heuristic split for the ~46% of
         transcripts that lack the explicit header.
      4. No Q&A found → entire body goes to prepared_remarks, qa_session = "".

    The "Duration: N minutes / Call participants: / More X analysis" footer
    is stripped from both sections.

    Args:
        transcript_text: Raw transcript string as stored in the dataset.

    Returns:
        dict with keys 'prepared_remarks' and 'qa_session', both str.
    """
    lines = transcript_text.split("\n")

    # Strip trailing footer lines
    footer_start = len(lines)
    for i, line in enumerate(lines):
        if _RE_FOOTER.match(line.strip()):
            footer_start = i
            break
    lines = lines[:footer_start]

    # Skip leading "Prepared Remarks:" header line if present
    body_start = 0
    if lines and _RE_PREPARED_REMARKS.match(lines[0].strip()):
        body_start = 1

    # --- Strategy 1 & 2: look for explicit Q&A header line ---
    qa_line_idx = None
    for i in range(body_start, len(lines)):
        if _RE_QA_HEADER.match(lines[i].strip()):
            qa_line_idx = i
            break

    if qa_line_idx is not None:
        prepared = "\n".join(lines[body_start:qa_line_idx]).strip()
        qa = "\n".join(lines[qa_line_idx + 1:]).strip()
        return {"prepared_remarks": prepared, "qa_session": qa}

    # --- Strategy 3: first analyst speaker line as fallback boundary ---
    # We only look after a minimum number of prepared-remarks lines to avoid
    # false positives on the very first analyst mentioned in an intro.
    MIN_PREPARED_LINES = 10
    for i in range(body_start + MIN_PREPARED_LINES, len(lines)):
        line = lines[i].strip()
        m = _RE_SPEAKER_LINE.match(line)
        if m:
            parts = [p.strip() for p in line.split(" -- ")]
            # Analyst lines have exactly 3 parts: Name -- Firm -- Analyst
            if len(parts) == 3 and _ANALYST_KW.search(parts[2]):
                prepared = "\n".join(lines[body_start:i]).strip()
                qa = "\n".join(lines[i:]).strip()
                return {"prepared_remarks": prepared, "qa_session": qa}

    # --- Strategy 4: no Q&A found ---
    prepared = "\n".join(lines[body_start:]).strip()
    return {"prepared_remarks": prepared, "qa_session": ""}


# ---------------------------------------------------------------------------
# 2. clean_text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove boilerplate and normalise whitespace from a transcript section.

    Preserves:
      - Speaker attribution lines ("Name -- Title")
      - Financial numbers and terminology
      - Sentence content

    Removes:
      - Operator procedural phrases ("[Operator Instructions]", "your line is open")
      - Legal safe-harbour boilerplate sentences
      - Recording / replay notices
      - Excessive blank lines

    Args:
        text: A section string (prepared_remarks or qa_session).

    Returns:
        Cleaned text string.
    """
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Collapse runs of whitespace within lines, preserve line breaks
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Collapse multiple internal spaces
        line = re.sub(r"  +", " ", line)
        cleaned_lines.append(line)

    # Remove runs of more than one blank line
    result_lines: list[str] = []
    prev_blank = False
    for line in cleaned_lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank

    return "\n".join(result_lines).strip()


# ---------------------------------------------------------------------------
# 3. tag_speakers
# ---------------------------------------------------------------------------

def tag_speakers(section_text: str, source_section: str = "unknown") -> list[SpeakerTurn]:
    """Identify speaker turns and assign role labels.

    A new speaker turn begins when a line matches either:
      - "Name -- Title" or "Name -- Firm -- Analyst"  (named speaker)
      - "Operator"                                     (bare operator line)

    Everything between consecutive speaker lines is that speaker's text.
    Empty turns (no text content after stripping boilerplate) are dropped.

    Args:
        section_text: Cleaned text for one section (prepared_remarks / qa).
        source_section: Label for the section, used only for debugging.

    Returns:
        List of SpeakerTurn dicts: {speaker, role, text}.
    """
    lines = section_text.split("\n")

    turns: list[SpeakerTurn] = []
    current_speaker = "Unknown"
    current_role = "unknown"
    current_lines: list[str] = []

    def flush_turn() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            turns.append(
                SpeakerTurn(
                    speaker=current_speaker,
                    role=current_role,
                    text=text,
                )
            )

    for line in lines:
        stripped = line.strip()

        # Check for bare "Operator" line
        if _RE_OPERATOR_LINE.match(stripped):
            flush_turn()
            current_lines = []
            current_speaker = "Operator"
            current_role = "operator"
            continue

        # Check for "Name -- Title [-- Analyst]" line
        m = _RE_SPEAKER_LINE.match(stripped)
        if m and len(stripped) < 120:
            parts = [p.strip() for p in stripped.split(" -- ")]
            name = parts[0]
            title_or_firm = parts[1] if len(parts) > 1 else ""
            third = parts[2] if len(parts) > 2 else None

            flush_turn()
            current_lines = []
            current_speaker = name
            current_role = _classify_role(title_or_firm, third)
            continue

        # Regular content line
        current_lines.append(line)

    flush_turn()
    return turns


# ---------------------------------------------------------------------------
# 4. tokenize_for_finbert
# ---------------------------------------------------------------------------

# Approximate tokens per character for English text (used for fast estimation
# before loading the actual tokeniser). FinBERT uses WordPiece; 0.22 is a
# conservative estimate derived from typical financial prose.
_CHARS_PER_TOKEN_EST = 4.5


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate without loading a tokeniser.

    Args:
        text: Input string.

    Returns:
        Estimated token count (int).
    """
    return max(1, round(len(text) / _CHARS_PER_TOKEN_EST))


def _split_at_sentence_boundaries(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks of at most max_tokens, breaking at sentences.

    Args:
        text: The text to split.
        max_tokens: Maximum estimated tokens per chunk.

    Returns:
        List of text chunks.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c for c in chunks if c.strip()]


def tokenize_for_finbert(
    speaker_turns: list[SpeakerTurn],
    source_section: str,
    max_tokens: int = 512,
) -> tuple[list[Chunk], list[Chunk]]:
    """Chunk speaker turns into FinBERT-compatible segments.

    Two strategies are returned:
      - Non-overlapping: one chunk per speaker turn; long turns split at
        sentence boundaries. Each resulting chunk is independent.
      - Overlapping (50%): sentence-boundary chunks with 50% token overlap
        across the entire section text. Useful for capturing sentiment that
        spans turn boundaries.

    Args:
        speaker_turns: Output of tag_speakers() for one section.
        source_section: 'prepared_remarks' or 'qa'.
        max_tokens: Maximum tokens per chunk (default 512 per FinBERT limit).

    Returns:
        Tuple of (non_overlap_chunks, overlap_chunks), each a list of Chunk
        dicts with keys: source_section, speaker, role, chunk_index,
        chunk_text, chunk_token_count, overlap.
    """
    non_overlap: list[Chunk] = []
    chunk_idx = 0

    # --- Non-overlapping: speaker-turn chunking ---
    for turn in speaker_turns:
        estimated = _estimate_tokens(turn["text"])
        if estimated <= max_tokens:
            sub_chunks = [turn["text"]]
        else:
            sub_chunks = _split_at_sentence_boundaries(turn["text"], max_tokens)

        for sub in sub_chunks:
            if not sub.strip():
                continue
            non_overlap.append(
                Chunk(
                    source_section=source_section,
                    speaker=turn["speaker"],
                    role=turn["role"],
                    chunk_index=chunk_idx,
                    chunk_text=sub.strip(),
                    chunk_token_count=_estimate_tokens(sub),
                    overlap=False,
                )
            )
            chunk_idx += 1

    # --- Overlapping: 50% stride over the full section text ---
    # Collect all sentences from all turns into a single flat list.
    all_text = " ".join(turn["text"] for turn in speaker_turns)
    all_sentences = re.split(r"(?<=[.!?])\s+", all_text.strip())

    overlap_chunks: list[Chunk] = []
    overlap_idx = 0
    i = 0

    while i < len(all_sentences):
        window_sentences: list[str] = []
        window_tokens = 0

        j = i
        while j < len(all_sentences):
            st = _estimate_tokens(all_sentences[j])
            if window_tokens + st > max_tokens and window_sentences:
                break
            window_sentences.append(all_sentences[j])
            window_tokens += st
            j += 1

        chunk_text = " ".join(window_sentences).strip()
        if chunk_text:
            overlap_chunks.append(
                Chunk(
                    source_section=source_section,
                    speaker="mixed",       # overlap chunks span speaker turns
                    role="mixed",
                    chunk_index=overlap_idx,
                    chunk_text=chunk_text,
                    chunk_token_count=window_tokens,
                    overlap=True,
                )
            )
            overlap_idx += 1

        # Advance by ~50% of the window
        advance = max(1, len(window_sentences) // 2)
        i += advance

    return non_overlap, overlap_chunks


# ---------------------------------------------------------------------------
# 5. run_preprocessing_pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline over a transcript DataFrame.

    Steps per transcript:
      1. Deduplicate: for (ticker, date) groups keep the longest transcript.
      2. parse_sections() → prepared_remarks + qa_session text.
      3. clean_text() each section.
      4. tag_speakers() each section.
      5. tokenize_for_finbert() → non-overlap and overlap chunks.
      6. Assemble into a flat per-chunk DataFrame.

    Args:
        df: DataFrame from data_ingestion.load_transcripts(), containing at
            minimum: ticker, date, fiscal_quarter, exchange, transcript_text.

    Returns:
        DataFrame with one row per chunk and columns:
          ticker, date, fiscal_quarter, exchange,
          source_section, speaker, role,
          chunk_text, chunk_index, chunk_token_count, overlap
    """
    # --- Step 1: deduplicate — keep longest transcript per (ticker, date) ---
    df = df.copy()
    df["_text_len"] = df["transcript_text"].str.len()
    df = (
        df.sort_values("_text_len", ascending=False)
        .drop_duplicates(subset=["ticker", "date"], keep="first")
        .drop(columns=["_text_len"])
        .reset_index(drop=True)
    )

    records: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        sections = parse_sections(row["transcript_text"])

        for section_name in ("prepared_remarks", "qa_session"):
            section_text = sections[section_name]
            if not section_text.strip():
                continue

            section_text = clean_text(section_text)
            turns = tag_speakers(section_text, source_section=section_name)

            if not turns:
                continue

            non_overlap, overlap = tokenize_for_finbert(
                turns,
                source_section=section_name,
                max_tokens=config.MAX_TOKENS,
            )

            base_meta = {
                "ticker": row["ticker"],
                "date": row["date"],
                "fiscal_quarter": row["fiscal_quarter"],
                "exchange": row["exchange"],
            }

            for chunk in non_overlap + overlap:
                records.append({**base_meta, **chunk})

    result_df = pd.DataFrame(records)

    # Ensure consistent column ordering
    col_order = [
        "ticker", "date", "fiscal_quarter", "exchange",
        "source_section", "speaker", "role",
        "chunk_text", "chunk_index", "chunk_token_count", "overlap",
    ]
    return result_df[col_order]


# ---------------------------------------------------------------------------
# Smoke-test / section boundary + speaker examples when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    import pickle
    from src.data_ingestion import load_transcripts, validate_data

    print("Loading transcripts...")
    df = load_transcripts(config.RAW_DIR / "motley-fool-data.pkl")

    # -----------------------------------------------------------------------
    # Show 5 section boundary examples
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SECTION BOUNDARY EXAMPLES (5 samples)")
    print("=" * 60)

    # Pick 2 with explicit header, 2 with analyst-fallback, 1 with no Q&A
    explicit_qa = df[df["transcript_text"].str.contains(
        "Questions and Answers:", regex=False
    )].sample(2, random_state=1)

    no_header_qa = df[~df["transcript_text"].str.contains(
        "Questions and Answers:|Question-and-Answer Session", regex=True
    )].sample(2, random_state=2)

    no_qa_at_all = df[~df["transcript_text"].str.contains(
        r"-- Analyst", regex=True
    )].sample(1, random_state=3)

    for label, subset in [
        ("EXPLICIT Q&A HEADER", explicit_qa),
        ("ANALYST-FALLBACK BOUNDARY", no_header_qa),
        ("NO Q&A", no_qa_at_all),
    ]:
        for _, row in subset.iterrows():
            sections = parse_sections(row["transcript_text"])
            pr_tail = sections["prepared_remarks"][-200:].replace("\n", " | ")
            qa_head = sections["qa_session"][:200].replace("\n", " | ") if sections["qa_session"] else "(empty)"
            print(f"\n[{label}]  {row['ticker']}  {row['date'].date()}")
            print(f"  Prepared tail : ...{pr_tail}")
            print(f"  QA head       : {qa_head}")
            print(f"  Prepared len  : {len(sections['prepared_remarks'])} chars")
            print(f"  QA len        : {len(sections['qa_session'])} chars")

    # -----------------------------------------------------------------------
    # Show 5 speaker-tagged examples
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  SPEAKER-TAGGED EXAMPLES (first 5 turns of 2 transcripts)")
    print("=" * 60)

    sample_rows = df.sample(2, random_state=10)
    for _, row in sample_rows.iterrows():
        sections = parse_sections(row["transcript_text"])
        cleaned_pr = clean_text(sections["prepared_remarks"])
        turns = tag_speakers(cleaned_pr, "prepared_remarks")
        print(f"\n  {row['ticker']}  {row['date'].date()}  — first 5 turns of prepared_remarks")
        for i, turn in enumerate(turns[:5]):
            print(f"    [{i}] speaker={turn['speaker']!r}  role={turn['role']}")
            print(f"         text={repr(turn['text'][:120])}")
