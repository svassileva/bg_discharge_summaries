#!/usr/bin/env python3
"""
Cross-lingual entity matching using SapBERT cosine similarity.

Pipeline
--------
1. Read sentence alignment from ``sent_pairs_*.tsv`` files produced by
   ``translation_alignment.py`` (default: results/alignment/sentence_pairs/).
   Each row is one Gale-Church bead: a paired EN / BG sentence group.
2. Load raw EN and BG NER TSV files (``--src-ner-tsv`` /
   ``--hyp-ner-tsv``).  These are the authoritative entity source; no
   previously-aligned entity file is needed or used.
3. Assign each entity to its sentence bead by substring matching: the entity
   surface text is looked up in each bead's source / hypothesis text.  When
   the same text appears in multiple beads, the entity's ``start_span``
   (character offset in the original document) is used together with cumulative
   bead text lengths to select the most likely bead.  Entities that cannot be
   matched to any bead are placed in an unassigned bucket (bead_idx = "").
4. Collect all unique entity surface texts for batched SapBERT encoding.
5. Encode with cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR (XLM-R).
   For BG entities, the ``translated_text`` column (English translation produced
   by ``translate_entities.py``) is preferred over the raw Cyrillic text because
   SapBERT was trained mostly on English UMLS names.
6. Within each bead, greedily match EN↔BG entity pairs by cosine similarity
   ≥ --threshold (default 0.85); leave remaining entities unmatched.
7. Write output TSV with matched pairs and unmatched singleton rows.

Inputs
------
--sentence-pairs-dir  Directory with sent_pairs_*.tsv files produced by
                      translation_alignment.py
                      (default: results/alignment/sentence_pairs).
--src-ner-tsv         Raw EN NER TSV, e.g.
                      outputs/gliner_moe_entities_en_corpus_full.tsv
--hyp-ner-tsv         Raw BG NER TSV, e.g.
                      outputs/gliner_moe_entities_bg_corpus_full.tsv
--hyp-suffix          Suffix inserted before the extension to derive the BG
                      filename from the EN filename (default: ``_BGgpt``).

Output columns
--------------
filename, bead_idx, bead_src, bead_hyp,
src_entity_label, src_start_span, src_end_span, src_text,
hyp_entity_label, hyp_start_span, hyp_end_span, hyp_text, hyp_translated_text,
sapbert_similarity, match_type

match_type is "sapbert" for matched pairs and "none" for unmatched singletons.

Usage
-----
    python cross_lingual_entity_matching.py \\
        --src-ner-tsv outputs/gliner_moe_entities_en_corpus_full.tsv \\
        --hyp-ner-tsv outputs/gliner_moe_entities_bg_corpus_full.tsv \\
        --output-tsv results/sapbert_entity_matches.tsv

    # Restrict matches to same entity type:
    python cross_lingual_entity_matching.py \\
        --src-ner-tsv outputs/gliner_moe_entities_en_corpus_full.tsv \\
        --hyp-ner-tsv outputs/gliner_moe_entities_bg_corpus_full.tsv \\
        --output-tsv results/sapbert_entity_matches_same_label.tsv \\
        --require-same-label


"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sapbert_entity_linking import SAPBERT_MODEL, encode_texts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLD_DEFAULT = 0.85

OUTPUT_COLS = [
    "filename",
    "bead_idx",
    "bead_src",
    "bead_hyp",
    "src_entity_label",
    "src_start_span",
    "src_end_span",
    "src_text",
    "hyp_entity_label",
    "hyp_start_span",
    "hyp_end_span",
    "hyp_text",
    "hyp_translated_text",
    "sapbert_similarity",
    "match_type",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_text_for_row(row: dict, prefer_translated: bool = False) -> str:
    """Return the surface text to use for SapBERT encoding.

    When *prefer_translated* is True (used for BG entities), the
    ``translated_text`` column (English translation from ``translate_entities.py``)
    is preferred; falls back to the raw ``text`` field when absent.
    """
    if prefer_translated:
        t = row.get("translated_text", "").strip()
        if t:
            return t
    return row.get("text", "").strip()


def load_ner_by_doc(tsv_path: Path) -> dict[str, list[dict]]:
    """Load a raw NER TSV into a ``{filename → [rows]}`` mapping.

    Compatible with both the EN NER TSV (columns: ``filename``, ``entity_label``,
    ``entity_kind``, ``start_span``, ``end_span``, ``text``) and the BG NER TSV
    (same columns).
    """
    data: dict[str, list[dict]] = defaultdict(list)
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fname = row.get("filename", "").strip()
            if fname:
                data[fname].append(row)
    return dict(data)


def _make_hyp_filename(src_filename: str, suffix: str) -> str:
    """Derive the BG filename from the EN filename by inserting *suffix* before
    the file extension.

    Example::
        _make_hyp_filename("cardiovascular_0.txt", "_BGgpt")
        # → "cardiovascular_0_BGgpt.txt"
    """
    p = Path(src_filename)
    return str(p.with_name(p.stem + suffix + p.suffix))


def load_sentence_pairs(tsv_path: Path) -> list[tuple[str, str]]:
    """Load a ``sent_pairs_*.tsv`` file into an ordered list of (src, hyp) pairs.

    Each row represents one Gale-Church sentence bead.  The row index (0-based)
    is used as the bead index throughout the rest of the pipeline.
    """
    pairs: list[tuple[str, str]] = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            src = row.get("source", "").strip()
            hyp = row.get("hypothesis", "").strip()
            pairs.append((src, hyp))
    return pairs


def group_entities_by_bead_text(
    src_entities: list[dict],
    hyp_entities: list[dict],
    sentence_pairs: list[tuple[str, str]],
) -> dict[str, tuple[list[dict], list[dict], str, str]]:
    """Assign raw NER/EL entities to sentence beads by text substring matching.

    For each entity, the pipeline checks every bead's source / hypothesis text
    for a substring match with the entity surface text.  When the same text
    appears in multiple beads, the entity's ``start_span`` (character offset in
    the original document) is compared against cumulative bead text lengths to
    select the most likely bead — this is an approximation that works well when
    the bead texts cover most of the document.  Entities with no bead match are
    collected in an unassigned bucket keyed by ``""``.

    Parameters
    ----------
    src_entities:
        EN entity rows (each dict has at least ``text``, ``entity_label``,
        ``start_span``, ``end_span``).
    hyp_entities:
        BG entity rows (same fields plus optional ``translated_text``).
    sentence_pairs:
        Ordered list of ``(src_text, hyp_text)`` beads as returned by
        :func:`load_sentence_pairs`.

    Returns
    -------
    Ordered dict keyed by bead_idx string (``"0"``, ``"1"``, …, and ``""`` for
    unassigned).  Each value is
    ``(en_entities, bg_entities, bead_src_text, bead_hyp_text)``.
    """
    # Cumulative text lengths — used to estimate each bead's approximate
    # character position in the original document for disambiguation.
    src_cumlen: list[int] = [0]
    hyp_cumlen: list[int] = [0]
    for src_t, hyp_t in sentence_pairs:
        src_cumlen.append(src_cumlen[-1] + len(src_t))
        hyp_cumlen.append(hyp_cumlen[-1] + len(hyp_t))

    bead_en: dict[int, list[dict]] = defaultdict(list)
    bead_bg: dict[int, list[dict]] = defaultdict(list)
    unassigned_en: list[dict] = []
    unassigned_bg: list[dict] = []

    def _assign(
        entities: list[dict],
        bead_texts_col: int,          # 0 = src, 1 = hyp
        cumlen: list[int],
        target_en: dict[int, list[dict]],
        unassigned: list[dict],
    ) -> None:
        for ent in entities:
            text = ent.get("text", "").strip()
            if not text:
                continue
            try:
                start = int(ent.get("start_span", ""))
            except (ValueError, TypeError):
                start = -1

            matches = [
                i for i, pair in enumerate(sentence_pairs)
                if text in pair[bead_texts_col]
            ]

            if not matches:
                unassigned.append(ent)
                continue

            if len(matches) == 1 or start < 0:
                bead = matches[0]
            else:
                bead = min(matches, key=lambda i: abs(cumlen[i] - start))

            rec: dict = {
                "entity_label": ent.get("entity_label", ""),
                "start_span":   ent.get("start_span", ""),
                "end_span":     ent.get("end_span", ""),
                "text":         text,
            }
            if bead_texts_col == 1:  # BG side — carry translated_text if present
                rec["translated_text"] = ent.get("translated_text", "")
            target_en[bead].append(rec)

    _assign(src_entities, 0, src_cumlen, bead_en, unassigned_en)
    _assign(hyp_entities, 1, hyp_cumlen, bead_bg, unassigned_bg)

    # Build ordered result dict.
    all_numeric = sorted(set(bead_en) | set(bead_bg))
    result: dict[str, tuple[list[dict], list[dict], str, str]] = {
        str(i): (
            bead_en.get(i, []),
            bead_bg.get(i, []),
            sentence_pairs[i][0],
            sentence_pairs[i][1],
        )
        for i in all_numeric
        if 0 <= i < len(sentence_pairs)
    }

    # Unassigned bucket.
    if unassigned_en or unassigned_bg:
        result[""] = (
            [
                {
                    "entity_label": e.get("entity_label", ""),
                    "start_span":   e.get("start_span", ""),
                    "end_span":     e.get("end_span", ""),
                    "text":         e.get("text", "").strip(),
                }
                for e in unassigned_en
            ],
            [
                {
                    "entity_label":  e.get("entity_label", ""),
                    "start_span":    e.get("start_span", ""),
                    "end_span":      e.get("end_span", ""),
                    "text":          e.get("text", "").strip(),
                    "translated_text": e.get("translated_text", ""),
                }
                for e in unassigned_bg
            ],
            "",
            "",
        )

    return result


def collect_unique_texts(
    src_ner_tsv: Path,
    hyp_ner_tsv: Path,
) -> list[str]:
    """Collect all unique entity surface texts from the raw NER TSV files.

    For the EN TSV the ``text`` column is used.  For the BG TSV, ``translated_text``
    is preferred over ``text`` when present (matching the encoding preference).
    """
    unique: set[str] = set()

    with open(src_ner_tsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row.get("text", "").strip()
            if t:
                unique.add(t)

    with open(hyp_ner_tsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row.get("translated_text", "").strip() or row.get("text", "").strip()
            if t:
                unique.add(t)

    return sorted(unique)


def greedy_match(
    en_rows: list[dict],
    bg_rows: list[dict],
    en_embs: np.ndarray,
    bg_embs: np.ndarray,
    threshold: float,
    require_same_label: bool,
) -> list[tuple[Optional[dict], Optional[dict], float]]:
    """Greedy bipartite entity matching by cosine similarity.

    Embeddings are assumed to be L2-normalised, so the dot product equals the
    cosine similarity.  Pairs are sorted by similarity (highest first) and
    matched greedily: once an entity is matched it cannot be re-used.  Pairs
    below *threshold* are not matched.

    Parameters
    ----------
    en_rows / bg_rows:
        Entity rows from the EN / BG side (dicts with at least ``text`` and
        ``entity_label`` keys).
    en_embs / bg_embs:
        Corresponding L2-normalised embeddings, shape (n_en, dim) / (n_bg, dim).
    threshold:
        Minimum cosine similarity for a pair to be considered a match.
    require_same_label:
        When True, only EN/BG pairs that share the same ``entity_label`` are
        candidates for matching.

    Returns
    -------
    List of ``(en_row_or_None, bg_row_or_None, similarity)`` tuples.
    Unmatched entities appear as singletons with the other side as ``None``
    and similarity ``0.0``.
    """
    if not en_rows or not bg_rows:
        return (
            [(r, None, 0.0) for r in en_rows]
            + [(None, r, 0.0) for r in bg_rows]
        )

    # Cosine similarity matrix, shape (n_en, n_bg).
    sim: np.ndarray = en_embs @ bg_embs.T

    if require_same_label:
        for i, er in enumerate(en_rows):
            for j, br in enumerate(bg_rows):
                if (er.get("entity_label", "").lower()
                        != br.get("entity_label", "").lower()):
                    sim[i, j] = -1.0

    # Collect all candidate pairs above the threshold, sorted by score desc.
    candidates: list[tuple[float, int, int]] = []
    n_en, n_bg = len(en_rows), len(bg_rows)
    for i in range(n_en):
        for j in range(n_bg):
            if sim[i, j] >= threshold:
                candidates.append((float(sim[i, j]), i, j))
    candidates.sort(reverse=True)

    matched_en: set[int] = set()
    matched_bg: set[int] = set()
    pairs: list[tuple[Optional[dict], Optional[dict], float]] = []

    for score, i, j in candidates:
        if i in matched_en or j in matched_bg:
            continue
        pairs.append((en_rows[i], bg_rows[j], score))
        matched_en.add(i)
        matched_bg.add(j)

    # Unmatched singletons.
    for i, r in enumerate(en_rows):
        if i not in matched_en:
            pairs.append((r, None, 0.0))
    for j, r in enumerate(bg_rows):
        if j not in matched_bg:
            pairs.append((None, r, 0.0))

    return pairs


def _match_entity_group(
    en_rows: list[dict],
    bg_rows: list[dict],
    all_embs: np.ndarray,
    text2idx: dict[str, int],
    threshold: float,
    require_same_label: bool,
    _debug_label: str = "",
) -> list[tuple[Optional[dict], Optional[dict], float]]:
    """Match one group of EN and BG entities using pre-computed embeddings."""
    en_texts = [_encode_text_for_row(r, prefer_translated=False) for r in en_rows]
    bg_texts = [_encode_text_for_row(r, prefer_translated=True) for r in bg_rows]

    missing_en = [t for t in en_texts if t and t not in text2idx]
    missing_bg = [t for t in bg_texts if t and t not in text2idx]
    if missing_en:
        log.debug("[%s] EN texts NOT in text2idx (%d): %s", _debug_label, len(missing_en), missing_en[:5])
    if missing_bg:
        log.debug("[%s] BG texts NOT in text2idx (%d): %s", _debug_label, len(missing_bg), missing_bg[:5])

    valid_en = [(r, t) for r, t in zip(en_rows, en_texts) if t and t in text2idx]
    valid_bg = [(r, t) for r, t in zip(bg_rows, bg_texts) if t and t in text2idx]

    # Entities whose text is empty or absent from text2idx cannot be encoded;
    # collect them separately so they still appear in the output as unmatched singletons.
    invalid_en = [r for r, t in zip(en_rows, en_texts) if not t or t not in text2idx]
    invalid_bg = [r for r, t in zip(bg_rows, bg_texts) if not t or t not in text2idx]

    log.debug(
        "[%s] valid_en=%d/%d  valid_bg=%d/%d  (threshold=%.2f)",
        _debug_label, len(valid_en), len(en_rows), len(valid_bg), len(bg_rows), threshold,
    )

    en_valid_rows = [r for r, _ in valid_en]
    bg_valid_rows = [r for r, _ in valid_bg]

    # Rows that cannot be encoded are emitted unconditionally as unmatched singletons.
    unmatched_invalid: list[tuple[Optional[dict], Optional[dict], float]] = (
        [(r, None, 0.0) for r in invalid_en]
        + [(None, r, 0.0) for r in invalid_bg]
    )

    if not valid_en and not valid_bg:
        log.debug("[%s] No valid entities on either side — emitting only invalid singletons.", _debug_label)
        return unmatched_invalid

    if valid_en and valid_bg:
        en_mat = np.array([all_embs[text2idx[t]] for _, t in valid_en])
        bg_mat = np.array([all_embs[text2idx[t]] for _, t in valid_bg])

        # Log similarity matrix to diagnose threshold issues.
        sim = en_mat @ bg_mat.T
        log.debug(
            "[%s] Similarity matrix %s: max=%.4f  mean=%.4f  min=%.4f  (threshold=%.2f)",
            _debug_label, sim.shape,
            float(sim.max()), float(sim.mean()), float(sim.min()),
            threshold,
        )
        if log.isEnabledFor(logging.DEBUG) and sim.size <= 100:
            en_labels = [r.get('text','')[:20] for r in en_valid_rows]
            bg_labels = [_encode_text_for_row(r, prefer_translated=True)[:20] for r in bg_valid_rows]
            for i, el in enumerate(en_labels):
                for j, bl in enumerate(bg_labels):
                    log.debug("  sim('%s', '%s') = %.4f %s",
                              el, bl, sim[i, j],
                              "<MATCH>" if sim[i, j] >= threshold else "")

        return greedy_match(
            en_valid_rows, bg_valid_rows,
            en_mat, bg_mat,
            threshold, require_same_label,
        ) + unmatched_invalid

    # Only one side has encodable entities — emit all as unmatched singletons.
    return (
        [(r, None, 0.0) for r in en_valid_rows]
        + [(None, r, 0.0) for r in bg_valid_rows]
        + unmatched_invalid
    )


def _make_output_row(
    filename: str,
    bead_idx,
    bead_src: str,
    bead_hyp: str,
    en_row: Optional[dict],
    bg_row: Optional[dict],
    score: float,
) -> dict:
    """Build one output TSV row from a matched or singleton entity pair."""
    rec: dict = {
        "filename":   filename,
        "bead_idx":   bead_idx if bead_idx is not None else "",
        "bead_src":   bead_src,
        "bead_hyp":   bead_hyp,
        "src_entity_label": "",
        "src_start_span":   "",
        "src_end_span":     "",
        "src_text":         "",
        "hyp_entity_label": "",
        "hyp_start_span":   "",
        "hyp_end_span":     "",
        "hyp_text":         "",
        "hyp_translated_text": "",
        "sapbert_similarity": f"{score:.4f}" if (en_row and bg_row) else "",
        "match_type": "sapbert" if (en_row and bg_row) else "none",
    }
    if en_row is not None:
        rec["src_entity_label"] = en_row.get("entity_label", "")
        rec["src_start_span"]   = en_row.get("start_span", "")
        rec["src_end_span"]     = en_row.get("end_span", "")
        rec["src_text"]         = en_row.get("text", "")
    if bg_row is not None:
        rec["hyp_entity_label"]    = bg_row.get("entity_label", "")
        rec["hyp_start_span"]      = bg_row.get("start_span", "")
        rec["hyp_end_span"]        = bg_row.get("end_span", "")
        rec["hyp_text"]            = bg_row.get("text", "")
        rec["hyp_translated_text"] = bg_row.get("translated_text", "")
    return rec


# ---------------------------------------------------------------------------
# Core processing function (callable from CLI or other Python code)
# ---------------------------------------------------------------------------


def run(
    sentence_pairs_dir: str = "results/alignment/sentence_pairs",
    src_ner_tsv: str = "outputs/gliner_moe_entities_en_corpus_full.tsv",
    hyp_ner_tsv: str = "outputs/gliner_moe_entities_bg_corpus_full.tsv",
    output_tsv: str = "results/sapbert_entity_matches.tsv",
    threshold: float = THRESHOLD_DEFAULT,
    require_same_label: bool = False,
    sapbert_model: str = SAPBERT_MODEL,
    batch_size: int = 128,
    max_length: int = 25,
    max_docs: Optional[int] = None,
    hyp_suffix: str = "_BGgpt",
) -> dict[str, int]:
    """Match EN/BG entities in sentence-aligned beads using SapBERT similarity.

    All parameters mirror the CLI arguments and carry the same defaults.

    Parameters
    ----------
    sentence_pairs_dir:
        Directory with ``sent_pairs_*.tsv`` files produced by
        ``translation_alignment.py``.  Each file provides the Gale-Church
        sentence alignment for one document pair.
    src_ner_tsv:
        Path to the raw EN NER TSV.
    hyp_ner_tsv:
        Path to the raw BG NER TSV.
    output_tsv:
        Path for the output TSV file.
    threshold:
        Minimum cosine similarity to consider two entities a match.
    require_same_label:
        When True, only entities with the same ``entity_label`` are matched.
    sapbert_model:
        HuggingFace model identifier for SapBERT.
    batch_size:
        Encoding batch size.
    max_length:
        Maximum token length passed to the tokeniser (entity spans are short).
    max_docs:
        Limit processing to the first *max_docs* documents (``None`` = all).
    hyp_suffix:
        Suffix used to derive the BG filename from the EN filename when looking
        up entities in *hyp_ner_tsv* (default ``"_BGgpt"``).

    Returns
    -------
    dict with keys ``matched``, ``unmatched_en``, ``unmatched_bg``.
    """
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Discover sent_pairs TSV files
    # ------------------------------------------------------------------
    pairs_dir = Path(sentence_pairs_dir)
    sent_files = sorted(pairs_dir.glob("sent_pairs_*.tsv"))
    if max_docs is not None:
        sent_files = sent_files[:max_docs]
    if not sent_files:
        raise FileNotFoundError(
            f"No sent_pairs_*.tsv files found in: {pairs_dir}"
        )
    print(f"Found {len(sent_files)} sent_pairs TSV files in: {pairs_dir}")
    log.debug("First 5 sent_files: %s", [f.name for f in sent_files[:5]])

    # ------------------------------------------------------------------
    # First pass: collect all unique entity texts (no model loaded yet)
    # ------------------------------------------------------------------
    print("Collecting unique entity texts for batched encoding...")
    unique_text_list = collect_unique_texts(
        Path(src_ner_tsv), Path(hyp_ner_tsv)
    )
    print(f"Unique entity texts: {len(unique_text_list)}")

    # ------------------------------------------------------------------
    # Load SapBERT model and encode all texts in one pass
    # ------------------------------------------------------------------
    print(f"Loading SapBERT model: {sapbert_model}")
    tokenizer = AutoTokenizer.from_pretrained(sapbert_model)
    model = AutoModel.from_pretrained(sapbert_model).to(device)

    print("Encoding entity texts with SapBERT...")
    all_embs = encode_texts(
        unique_text_list,
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    text2idx: dict[str, int] = {t: i for i, t in enumerate(unique_text_list)}
    print(f"Encoded {all_embs.shape[0]} texts into {all_embs.shape[1]}-dim embeddings.")

    # ------------------------------------------------------------------
    # Pre-load all raw entity data
    # ------------------------------------------------------------------
    print(f"Loading raw EN NER entities from: {src_ner_tsv}")
    src_ner_all = load_ner_by_doc(Path(src_ner_tsv))
    print(f"  EN NER: {len(src_ner_all)} documents, "
          f"{sum(len(v) for v in src_ner_all.values())} total entities")
    log.debug("EN NER doc keys (first 3): %s", list(src_ner_all)[:3])

    print(f"Loading raw BG NER entities from: {hyp_ner_tsv}")
    hyp_ner_all = load_ner_by_doc(Path(hyp_ner_tsv))
    print(f"  BG NER: {len(hyp_ner_all)} documents, "
          f"{sum(len(v) for v in hyp_ner_all.values())} total entities")
    log.debug("BG NER doc keys (first 3): %s", list(hyp_ner_all)[:3])

    # Warn if BG rows have no translated_text (SapBERT performs better on EN text)
    _sample_bg = next(iter(hyp_ner_all.values()), [])
    _has_translated = any(r.get("translated_text", "").strip() for r in _sample_bg[:10])
    if not _has_translated:
        log.warning(
            "BG NER TSV appears to have NO 'translated_text' column (or all empty). "
            "SapBERT is trained on English; matching quality will be reduced. "
            "Consider using the el_results_*_bg_corpus_full.tsv file which has "
            "English translations."
        )

    # Detect same-file usage early and warn about expected failure mode.
    if Path(src_ner_tsv).resolve() == Path(hyp_ner_tsv).resolve():
        log.warning(
            "src_ner_tsv and hyp_ner_tsv point to the SAME FILE (%s). "
            "With hyp_suffix=%r the derived hyp filename will NOT be found in "
            "the NER dict (keys lack the suffix), so raw_hyp will be empty for "
            "every document. Even with hyp_suffix='' the EN entity texts will "
            "NOT be substrings of the Bulgarian hypothesis bead texts, causing "
            "all hyp entities to land in the unassigned bucket. "
            "Use separate EN/BG NER files for meaningful cross-lingual matching.",
            src_ner_tsv, hyp_suffix,
        )

    # ------------------------------------------------------------------
    # Matching pass
    # ------------------------------------------------------------------
    out_path = Path(output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_matched = 0
    n_unmatched_en = 0
    n_unmatched_bg = 0

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=OUTPUT_COLS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()

        for doc_idx, sent_path in enumerate(sent_files):
            # Derive the source filename from the TSV name:
            # sent_pairs_<filename>.tsv  →  <filename>
            filename = sent_path.name[len("sent_pairs_"):-len(".tsv")]

            sentence_pairs = load_sentence_pairs(sent_path)
            if not sentence_pairs:
                log.warning("[%s] No sentence pairs loaded — skipping.", filename)
                continue

            # Load raw entities for this document.
            # NER TSVs may omit the file extension (e.g. "doc_0" instead of
            # "doc_0.txt").  Try the full filename first; fall back to the stem
            # (extension stripped) so that files like
            # predictions_*_en_ner.tsv (no extension in rows) and the classic
            # gliner_moe_entities_*_corpus_full.tsv (rows include .txt) both work.
            src_fname = filename
            raw_src = src_ner_all.get(src_fname, [])
            if not raw_src and Path(src_fname).suffix:
                _src_stem = Path(src_fname).stem
                if _src_stem in src_ner_all:
                    src_fname = _src_stem
                    raw_src = src_ner_all[src_fname]
                    log.debug("[%s] EN lookup: using stem key '%s'", filename, src_fname)

            hyp_fname = _make_hyp_filename(src_fname, hyp_suffix)
            raw_hyp = hyp_ner_all.get(hyp_fname, [])
            if not raw_hyp and Path(hyp_fname).suffix:
                _hyp_stem = Path(hyp_fname).stem
                if _hyp_stem in hyp_ner_all:
                    hyp_fname = _hyp_stem
                    raw_hyp = hyp_ner_all[hyp_fname]
                    log.debug("[%s] BG lookup: using stem key '%s'", filename, hyp_fname)

            if not raw_src:
                log.warning(
                    "[%s] EN entities NOT FOUND. src_fname='%s'. "
                    "Available keys sample: %s",
                    filename, src_fname, list(src_ner_all)[:3],
                )
            if not raw_hyp:
                # Check whether the key WITHOUT the suffix exists — that would
                # mean the caller passed the wrong file or wrong --hyp-suffix.
                _suffix_present_in_src = src_fname in hyp_ner_all
                _hint = (
                    f" (key '{src_fname}' EXISTS in hyp dict without suffix — "
                    "try --hyp-suffix \"\")"
                    if _suffix_present_in_src else ""
                )
                log.warning(
                    "[%s] BG entities NOT FOUND. hyp_fname='%s'.%s "
                    "Available keys sample: %s",
                    filename, hyp_fname, _hint, list(hyp_ner_all)[:3],
                )

            log.debug(
                "[%s] sentence_pairs=%d  raw_src=%d  raw_hyp=%d",
                filename, len(sentence_pairs), len(raw_src), len(raw_hyp),
            )

            # Assign entities to sentence beads by text matching.
            beads = group_entities_by_bead_text(raw_src, raw_hyp, sentence_pairs)

            # Log bead assignment summary.
            total_en_assigned = sum(len(v[0]) for k, v in beads.items() if k != "")
            total_bg_assigned = sum(len(v[1]) for k, v in beads.items() if k != "")
            unassigned_en = len(beads.get("", ([], []))[0])
            unassigned_bg = len(beads.get("", ([], []))[1])
            log.debug(
                "[%s] Bead assignment: EN assigned=%d unassigned=%d | "
                "BG assigned=%d unassigned=%d | beads with entities=%d",
                filename,
                total_en_assigned, unassigned_en,
                total_bg_assigned, unassigned_bg,
                sum(1 for k, v in beads.items() if v[0] or v[1]),
            )
            if raw_hyp and total_bg_assigned == 0 and unassigned_bg > 0:
                _sample_miss = [e.get("text","")[:30] for e in raw_hyp[:3]]
                log.warning(
                    "[%s] ALL %d BG entities failed bead assignment "
                    "(entity text not found as substring in any hypothesis bead). "
                    "Sample missing texts: %s. "
                    "This usually means the hyp NER was run on a different text "
                    "than what was sentence-aligned, or the hypothesis language "
                    "does not match the entity texts.",
                    filename, unassigned_bg, _sample_miss,
                )

            for bead_idx, (en_ents, bg_ents, bead_src, bead_hyp) in beads.items():
                if not en_ents and not bg_ents:
                    continue

                pairs = _match_entity_group(
                    en_ents, bg_ents,
                    all_embs, text2idx,
                    threshold, require_same_label,
                    _debug_label=f"{filename}:bead{bead_idx}",
                )

                for en_r, bg_r, score in pairs:
                    writer.writerow(_make_output_row(
                        filename,
                        bead_idx if bead_idx else None,
                        bead_src,
                        bead_hyp,
                        en_r, bg_r, score,
                    ))
                    if en_r and bg_r:
                        n_matched += 1
                    elif en_r:
                        n_unmatched_en += 1
                    else:
                        n_unmatched_bg += 1

            if (doc_idx + 1) % 10 == 0:
                print(f"  Processed {doc_idx + 1}/{len(sent_files)} documents...")

    print(f"\nOutput written to: {out_path}")
    print(f"  Matched EN↔BG pairs :  {n_matched}")
    print(f"  Unmatched EN entities: {n_unmatched_en}")
    print(f"  Unmatched BG entities: {n_unmatched_bg}")

    if n_matched == 0:
        log.warning(
            "ZERO pairs matched. Likely causes:\n"
            "  1. WRONG HYP FILE / SUFFIX: hyp_fname lookup returned empty for "
            "all docs (check WARNING lines above for '[doc] BG entities NOT FOUND').\n"
            "  2. BEAD ASSIGNMENT FAILURE: all hyp entities are unassigned because "
            "entity text is not a substring of the hypothesis bead text "
            "(check 'ALL BG entities failed bead assignment' warnings above).\n"
            "  3. THRESHOLD TOO HIGH: SapBERT similarity for cross-lingual "
            "EN↔Cyrillic pairs is typically 0.6–0.8; the default threshold of "
            "%.2f may be too strict. Try --threshold 0.70 or lower.\n"
            "  4. NO TRANSLATED TEXT: BG entities are encoded as raw Cyrillic. "
            "Pass el_results_*_bg_corpus_full.tsv as --hyp-ner-tsv to use "
            "English translations for SapBERT encoding.",
            threshold,
        )

    return {
        "matched": n_matched,
        "unmatched_en": n_unmatched_en,
        "unmatched_bg": n_unmatched_bg,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Match EN/BG entities within Gale-Church sentence beads using "
            "SapBERT cosine similarity.  Sentence alignment is taken from "
            "sent_pairs_*.tsv files (translation_alignment.py output); "
            "entities are loaded fresh from raw NER TSV files."
        )
    )
    # --- Sentence alignment input ---
    ap.add_argument(
        "--sentence-pairs-dir",
        default="results/alignment/sentence_pairs",
        help=(
            "Directory with sent_pairs_*.tsv files produced by "
            "translation_alignment.py "
            "(default: results/alignment/sentence_pairs)."
        ),
    )
    # --- Raw entity inputs ---
    ap.add_argument(
        "--src-ner-tsv",
        dest="src_ner_tsv",
        default="outputs/gliner_moe_entities_en_corpus_full.tsv",
        help="Raw EN NER TSV (filename, entity_label, entity_kind, start_span, end_span, text).",
    )
    ap.add_argument(
        "--hyp-ner-tsv",
        dest="hyp_ner_tsv",
        default="outputs/gliner_moe_entities_bg_corpus_full.tsv",
        help="Raw BG NER TSV (same columns, plus optional ``translated_text`` from translate_entities.py).",
    )
    ap.add_argument(
        "--hyp-suffix",
        default="_BGgpt",
        help=(
            "Suffix inserted before the file extension to derive the BG filename "
            "from the EN filename when looking up entities in --hyp-ner-tsv "
            "(default: '_BGgpt')."
        ),
    )
    # --- Output ---
    ap.add_argument(
        "--output-tsv",
        default="results/sapbert_entity_matches.tsv",
        help="Output TSV path.",
    )
    # --- Matching parameters ---
    ap.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help=f"Cosine similarity threshold for matching (default: {THRESHOLD_DEFAULT}).",
    )
    ap.add_argument(
        "--require-same-label",
        action="store_true",
        help="Only match entities that share the same entity_label.",
    )
    # --- Encoding parameters ---
    ap.add_argument(
        "--sapbert-model",
        default=SAPBERT_MODEL,
        help="HuggingFace model identifier for SapBERT.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Encoding batch size.",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=25,
        help="Max token length for SapBERT encoding (entity spans are short).",
    )
    # --- Misc ---
    ap.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Process only the first N documents (useful for testing).",
    )
    args = ap.parse_args()
    run(
        sentence_pairs_dir=args.sentence_pairs_dir,
        src_ner_tsv=args.src_ner_tsv,
        hyp_ner_tsv=args.hyp_ner_tsv,
        output_tsv=args.output_tsv,
        threshold=args.threshold,
        require_same_label=args.require_same_label,
        sapbert_model=args.sapbert_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_docs=args.max_docs,
        hyp_suffix=args.hyp_suffix,
    )


if __name__ == "__main__":
    main()
