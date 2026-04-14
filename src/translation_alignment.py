#!/usr/bin/env python3
"""
Sentence splitting, sentence alignment, entity linking alignment, and
relation alignment for cross-lingual translation evaluation.

Provides:
  - Sentence splitting (spaCy multilingual pipeline)
  - Gale-Church sentence alignment
  - UMLS entity-linking alignment across aligned sentence beads
  - Relation-extraction alignment across aligned sentence beads
  - :func:`align_document_pair` — the main entry point that runs all alignment
    steps for a single source/hypothesis document pair and writes intermediate
    TSV files (sentence_pairs, aligned_entities, aligned_relations) as side
    effects.
"""

import csv
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import pandas as pd
import spacy
from nltk.translate import gale_church

# ---------------------------------------------------------------------------
# Translation system prompt
# ---------------------------------------------------------------------------
_TRANSLATION_PROMPT = (
    "You are a professional medical translator. "
    "Translate the following English hospital discharge summary to Bulgarian. "
    "Preserve all medical terminology, structure, section headings, and formatting "
    "exactly as they appear in the source. "
    "Output only the Bulgarian translation, without any additional commentary."
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_texts(input_dir: str, max_docs: Optional[int] = None) -> list[tuple[str, str]]:
    """Return sorted list of (relative_path, text) for every .txt file.

    If *max_docs* is given, only the first *max_docs* files (by sorted path)
    are read from disk.
    """
    root = Path(input_dir)
    paths = sorted(str(p.relative_to(root)) for p in root.rglob("*.txt"))
    if max_docs is not None:
        paths = paths[:max_docs]
    return [(rel, (root / rel).read_text(encoding="utf-8")) for rel in paths]


def _hyp_rel(rel: str, suffix: str = "_BGgpt") -> str:
    """
    Insert *suffix* before the file extension of a relative path.

    Examples
    --------
    >>> _hyp_rel("cardio/report.txt")
    'cardio/report_BGgpt.txt'
    >>> _hyp_rel("report.txt")
    'report_BGgpt.txt'
    """
    p = Path(rel)
    return str(p.with_name(p.stem + suffix + p.suffix))


def load_translations(
    trans_dir: str, texts: list[tuple[str, str]]
) -> dict[str, str]:
    """
    Load pre-translated files keyed by their *source* relative path.

    The translation for source file ``rel`` (e.g. ``folder/note.txt``) is
    expected to live at ``_hyp_rel(rel)`` inside *trans_dir*
    (e.g. ``folder/note_BGgpt.txt``).  If that is not found the original
    name is tried as a fallback so the function works with both naming
    schemes.
    """
    root = Path(trans_dir)
    result: dict[str, str] = {}
    for rel, _ in texts:
        # Primary: <stem>_BGgpt<ext>
        hyp_path = root / _hyp_rel(rel)
        if hyp_path.exists():
            result[rel] = hyp_path.read_text(encoding="utf-8")
            continue
        # Fallback: same name as source
        path = root / rel
        if path.exists():
            result[rel] = path.read_text(encoding="utf-8")
        else:
            print(f"  [warn] translation not found: {hyp_path} (also tried {path})")
    return result


def save_translation(text: str, rel_path: str, save_dir: str) -> None:
    out = Path(save_dir) / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Translation via OpenAI
# ---------------------------------------------------------------------------

def translate_text(text: str, client, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _TRANSLATION_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Common abbreviations that must never trigger a sentence boundary.
_BG_ABBREVS: set[str] = {
    "Dr",   # Doctor (English)
    "Mr",   # Mister (English)
    "Ms",   # Miss/Ms (English)
    "Mrs",  # Missus (English)
    "Prof", # Professor (English) — also covers Bulgarian проф
    "г",    # година / year — '2024 г.'
    "гр",   # град / city
    "ул",   # улица / street
    "бул",  # булевард / boulevard
    "пл",   # площад / square
    "обл",  # област / region
    "общ",  # община / municipality
    "д-р",  # доктор
    "проф", # професор
    "доц",  # доцент
    "чл",   # член / member
    "стр",  # страница / page
    "тел",  # телефон / phone
    "кв",   # квартал / quarter
    "ж.к",  # жилищен комплекс
    "ет",   # етаж / floor
    "ап",   # апартамент
    "бр",   # брой / number/count
    "лв",   # лева (currency)
    "и др", # и други / etc.
    "и т.н",  # така нататък / etc.    
}

# Module-level spaCy model — loaded lazily on first call.
_spacy_nlp: "spacy.language.Language | None" = None


def _get_spacy_nlp() -> "spacy.language.Language":
    """Load (once) the spaCy MultiLanguage sentencizer pipeline."""
    global _spacy_nlp
    if _spacy_nlp is None:
        # xx_sent_ud_sm is the multilingual sentence-segmentation model.
        # Falls back to a blank multilingual pipeline + sentencizer when the
        # model is not installed so the code never hard-crashes at import time.
        try:
            nlp = spacy.load("xx_sent_ud_sm", exclude=["ner", "lemmatizer"])
        except OSError:
            nlp = spacy.blank("xx")
            nlp.add_pipe("sentencizer")
        # Register Bulgarian abbreviations so the tokenizer never treats
        # the period after them as a sentence-ending full stop.
        for abbrev in _BG_ABBREVS:
            nlp.tokenizer.rules.pop(abbrev + ".", None)  # remove any existing rule
            # Mark as abbreviation in the special-cases table.
            from spacy.symbols import ORTH
            nlp.tokenizer.add_special_case(abbrev + ".", [{ORTH: abbrev + "."}])
        _spacy_nlp = nlp
    return _spacy_nlp


def split_sentences(text: str) -> list[str]:
    """Sentence-split *text* using the spaCy MultiLanguage pipeline.

    Each non-empty line is first treated as a hard boundary: the text is split
    on newlines and each line is then further segmented by spaCy.  This ensures
    that both English and Bulgarian structural line breaks (section headings,
    list items, etc.) always produce separate sentences.

    Common Bulgarian abbreviations (г., гр., ул., бул., …) are registered as
    special-case tokens so the period after them does not trigger a sentence
    boundary within a line.
    """
    nlp = _get_spacy_nlp()
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        doc = nlp(line)
        sentences.extend(sent.text.strip() for sent in doc.sents if sent.text.strip())
    return sentences


def split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Like split_sentences but also returns (start_char, end_char) byte offsets
    in the *original* text for each sentence.

    Offsets can be used to filter entity-linking / relation-extraction rows
    whose span fields reference character positions in the same text.
    """
    nlp = _get_spacy_nlp()
    result: list[tuple[str, int, int]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_text = raw_line.rstrip("\r\n")
        stripped = line_text.strip()
        left_pad = len(line_text) - len(line_text.lstrip())
        if stripped:
            doc = nlp(stripped)
            for sent in doc.sents:
                s = sent.text.strip()
                if s:
                    abs_start = offset + left_pad + sent.start_char
                    abs_end   = offset + left_pad + sent.end_char
                    result.append((s, abs_start, abs_end))
        offset += len(raw_line)
    return result


def align_sentences(
    src_sents: list[str], hyp_sents: list[str]
) -> list[tuple[str, str]]:
    """Align source and hypothesis sentences using the Gale-Church algorithm.

    Multi-sentence beads are joined with a single space so the result is always
    a flat list of (source_text, hypothesis_text) string pairs ready for
    BERTScore / LaBSE evaluation.  Beads with an empty side (0-N or N-0
    deletions/insertions) are skipped.
    """
    if not src_sents or not hyp_sents:
        return []
    alignment: list[tuple[int, int]] = gale_church.align_blocks(
        [len(s) for s in src_sents],
        [len(s) for s in hyp_sents],
    )
    if not alignment:
        return []

    pairs: list[tuple[str, str]] = []

    def _flush(bead_pairs: list[tuple[int, int]]) -> None:
        src_indices = sorted({p[0] for p in bead_pairs})
        hyp_indices = sorted({p[1] for p in bead_pairs})
        src_group = " ".join(src_sents[i] for i in src_indices)
        hyp_group = " ".join(hyp_sents[j] for j in hyp_indices)
        if src_group and hyp_group:
            pairs.append((src_group, hyp_group))

    bead: list[tuple[int, int]] = [alignment[0]]
    for i in range(1, len(alignment)):
        prev_src, prev_hyp = alignment[i - 1]
        curr_src, curr_hyp = alignment[i]
        if curr_src != prev_src and curr_hyp != prev_hyp:
            _flush(bead)
            bead = [alignment[i]]
        else:
            bead.append(alignment[i])
    _flush(bead)

    return pairs


# Each bead: (src_text, hyp_text, src_start, src_end, hyp_start, hyp_end)
_Bead = tuple[str, str, int, int, int, int]


def align_sentences_with_spans(
    src_sents: list[tuple[str, int, int]],
    hyp_sents: list[tuple[str, int, int]],
) -> list[_Bead]:
    """Gale-Church alignment over span-aware sentence lists.

    *src_sents* / *hyp_sents* are lists of (text, start_char, end_char) as
    returned by :func:`split_sentences_with_spans`.

    Returns one ``_Bead`` per aligned group:
    ``(src_text, hyp_text, src_start, src_end, hyp_start, hyp_end)``
    where the span pair covers every sentence in that bead.
    Beads with an empty side are skipped.
    """
    if not src_sents or not hyp_sents:
        return []
    src_texts = [s[0] for s in src_sents]
    hyp_texts = [s[0] for s in hyp_sents]
    alignment: list[tuple[int, int]] = gale_church.align_blocks(
        [len(s) for s in src_texts],
        [len(s) for s in hyp_texts],
    )
    if not alignment:
        return []

    result: list[_Bead] = []

    def _flush(bead_pairs: list[tuple[int, int]]) -> None:
        src_indices = sorted({p[0] for p in bead_pairs})
        hyp_indices = sorted({p[1] for p in bead_pairs})
        src_group = " ".join(src_texts[i] for i in src_indices)
        hyp_group = " ".join(hyp_texts[j] for j in hyp_indices)
        if src_group and hyp_group:
            result.append((
                src_group, hyp_group,
                src_sents[src_indices[0]][1], src_sents[src_indices[-1]][2],
                hyp_sents[hyp_indices[0]][1], hyp_sents[hyp_indices[-1]][2],
            ))

    bead: list[tuple[int, int]] = [alignment[0]]
    for i in range(1, len(alignment)):
        prev_src, prev_hyp = alignment[i - 1]
        curr_src, curr_hyp = alignment[i]
        if curr_src != prev_src and curr_hyp != prev_hyp:
            _flush(bead)
            bead = [alignment[i]]
        else:
            bead.append(alignment[i])
    _flush(bead)

    return result


# ---------------------------------------------------------------------------
# KG / EL / RE data loading and helpers
# ---------------------------------------------------------------------------

def load_el_tsv(path: str) -> dict[str, list[dict]]:
    """Load entity linking TSV → dict mapping filename → list of entity rows."""
    data: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fname = row.get("filename", "").strip()
            data.setdefault(fname, []).append(row)
    return data


def load_re_tsv(path: str) -> dict[str, list[dict]]:
    """Load relation extraction TSV → dict mapping filename → list of relation rows."""
    data: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fname = row.get("filename", "").strip()
            data.setdefault(fname, []).append(row)
    return data


def _norm_span(v: str) -> str:
    """Normalise a span string to a plain integer string, e.g. '042' → '42'."""
    try:
        return str(int(v.strip()))
    except (ValueError, AttributeError):
        return v.strip()


def _span_cui_map(el_rows: list[dict]) -> dict[tuple[str, str], str]:
    """Build a (start_span, end_span) → top-1 UMLS CUI map for one document.

    Span keys are normalised to plain integer strings so they match the values
    written by fhir_relation_extraction.py (which stores spans as integers).
    """
    m: dict[tuple[str, str], str] = {}
    for row in el_rows:
        start = _norm_span(row.get("start_span", ""))
        end   = _norm_span(row.get("end_span",   ""))
        cui   = row.get("umls_cui_1", "").strip()
        if start and end and cui:
            m[(start, end)] = cui
    return m


def _entity_cui_set(el_rows: list[dict]) -> set[str]:
    """Return the set of top-1 UMLS CUIs mentioned in entity linking rows."""
    return {
        row["umls_cui_1"].strip()
        for row in el_rows
        if row.get("umls_cui_1", "").strip()
    }


def resolve_relations(
    re_data: dict[str, list[dict]],
    el_data: dict[str, list[dict]],
) -> dict[str, set[tuple[str, str, str]]]:
    """
    Pre-resolve RE rows to (subject_cui, relation, object_cui) triple sets
    by looking up each relation's subject and object span positions in the
    EL span→CUI map built from *el_data*.

    This lets a single canonical set of CUI triples represent every document's
    relations regardless of surface language, so English and Bulgarian triples
    can be compared directly by set intersection.

    Documents present in *re_data* but absent (or empty) in *el_data* resolve
    their spans against an empty map, producing an empty triple set.
    """
    resolved: dict[str, set[tuple[str, str, str]]] = {}
    for fname, re_rows in re_data.items():
        span_cui = _span_cui_map(el_data.get(fname, []))
        resolved[fname] = _relation_triple_set(re_rows, span_cui)
    return resolved


def _span_text_map(el_rows: list[dict]) -> dict[tuple[str, str], str]:
    """Build (start_span, end_span) → entity text for one document.
    Prefers ``translated_text`` over raw ``text`` (same logic as _entity_text_list).
    """
    m: dict[tuple[str, str], str] = {}
    for row in el_rows:
        start = _norm_span(row.get("start_span", ""))
        end   = _norm_span(row.get("end_span",   ""))
        t = row.get("translated_text", "").strip() or row.get("text", "").strip()
        if start and end and t:
            m[(start, end)] = t
    return m


def _entity_text_list(el_rows: list[dict]) -> list[str]:
    """
    Extract entity surface texts for embedding comparison.
    Prefers ``translated_text`` (back-translation produced by
    sapbert_entity_linking.py --translate) over the raw ``text`` field so that
    an EN↔EN BERTScore is computed when available.
    """
    texts: list[str] = []
    for row in el_rows:
        t = row.get("translated_text", "").strip() or row.get("text", "").strip()
        if t:
            texts.append(t)
    return texts


def _relation_text_list(
    re_rows: list[dict],
    span_text: dict[tuple[str, str], str],
) -> list[str]:
    """
    Build text representations of relations for embedding comparison.
    Format: ``"<subject_text> [<relation_type>] <object_text>"``
    Entity texts are resolved from *span_text* (preferring back-translated text).
    Triples whose subject or object span has no text are skipped.
    """
    texts: list[str] = []
    for row in re_rows:
        subj_text = span_text.get((
            _norm_span(row.get("subject_start", "")),
            _norm_span(row.get("subject_end",   "")),
        ))
        obj_text = span_text.get((
            _norm_span(row.get("object_start", "")),
            _norm_span(row.get("object_end",   "")),
        ))
        rel = row.get("relation", "").strip()
        if subj_text and obj_text and rel:
            texts.append(f"{subj_text} [{rel}] {obj_text}")
    return texts


def _relation_triple_set(
    re_rows: list[dict],
    span_cui: dict[tuple[str, str], str],
) -> set[tuple[str, str, str]]:
    """
    Build a set of (subject_cui, relation, object_cui) triples.
    Triples whose subject or object span cannot be resolved to a CUI are skipped.
    """
    triples: set[tuple[str, str, str]] = set()
    for row in re_rows:
        subj_cui = span_cui.get((
            _norm_span(row.get("subject_start", "")),
            _norm_span(row.get("subject_end",   "")),
        ))
        obj_cui = span_cui.get((
            _norm_span(row.get("object_start", "")),
            _norm_span(row.get("object_end",   "")),
        ))
        rel = row.get("relation", "").strip()
        if subj_cui and obj_cui and rel:
            triples.add((subj_cui, rel, obj_cui))
    return triples


def _filter_el_by_span(el_rows: list[dict], start: int, end: int) -> list[dict]:
    """Return EL rows whose entity start span falls within [start, end).

    Using start-based containment (rather than requiring the full span to fit)
    ensures that entities straddling a sentence boundary are still captured by
    the bead whose span includes the entity's start position.
    """
    out = []
    for row in el_rows:
        try:
            s = int(row.get("start_span", ""))
        except (ValueError, TypeError):
            continue
        if s >= start and s < end:
            out.append(row)
    return out


def _filter_re_by_span(re_rows: list[dict], start: int, end: int) -> list[dict]:
    """Return RE rows whose subject AND object start spans both fall within [start, end).

    Using start-based containment mirrors the logic in _filter_el_by_span so that
    relations whose argument spans straddle a sentence boundary are still assigned
    to the bead containing each argument's start position.
    """
    out = []
    for row in re_rows:
        try:
            ss = int(row.get("subject_start", ""))
            os_ = int(row.get("object_start", ""))
        except (ValueError, TypeError):
            continue
        if ss >= start and ss < end and os_ >= start and os_ < end:
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Relation alignment within a bead
# ---------------------------------------------------------------------------

def _match_el_rows(
    src_rows: list[dict],
    hyp_rows: list[dict],
) -> list[tuple[dict | None, dict | None, str]]:
    """Match source and hypothesis EL rows using CUI then text, enforcing entity type.

    Pass 1 — **CUI**: key = ``(label, "cui:<umls_cui_1>")``.  Both sides must
        carry a non-empty ``umls_cui_1`` and the same entity label.  Match type
        recorded as ``"CUI"``.

    Pass 2 — **Text**: key = ``(label, "text:<en_text>")``.  Uses
        ``translated_text`` (back-translation) when present, otherwise ``text``.
        Both sides must have non-empty text and the same entity label.  Match
        type recorded as ``"text"``.

    Rows that cannot be matched by CUI or text are emitted as
    ``(row, None, "none")`` / ``(None, row, "none")`` singletons.
    """
    from collections import defaultdict

    def _norm(v: str) -> str:
        return v.strip().lower()

    def _cui_key(row: dict) -> tuple[str, str] | None:
        cui = row.get("umls_cui_1", "").strip()
        if cui:
            return (_norm(row.get("label", "")), f"cui:{cui}")
        return None

    def _text_key(row: dict) -> tuple[str, str] | None:
        txt = (row.get("translated_text", "").strip()
               or row.get("text", "").strip())
        if txt:
            return (_norm(row.get("label", "")), f"text:{txt.lower()}")
        return None

    def _run_pass(
        candidates_src: list[dict],
        candidates_hyp: list[dict],
        key_fn,
    ) -> tuple[list[tuple[dict | None, dict | None]], list[dict], list[dict]]:
        src_by_key: dict[tuple, list[dict]] = defaultdict(list)
        hyp_by_key: dict[tuple, list[dict]] = defaultdict(list)

        for r in candidates_src:
            k = key_fn(r)
            if k is not None:
                src_by_key[k].append(r)

        for r in candidates_hyp:
            k = key_fn(r)
            if k is not None:
                hyp_by_key[k].append(r)

        match_keys = sorted(set(src_by_key) & set(hyp_by_key))

        out_pairs: list[tuple[dict | None, dict | None]] = []
        matched_src_ids: set[int] = set()
        matched_hyp_ids: set[int] = set()

        for key in match_keys:
            for s, h in zip_longest(src_by_key.get(key, []), hyp_by_key.get(key, [])):
                out_pairs.append((s, h))
                if s is not None:
                    matched_src_ids.add(id(s))
                if h is not None:
                    matched_hyp_ids.add(id(h))

        leftover_src = [r for r in candidates_src if id(r) not in matched_src_ids]
        leftover_hyp = [r for r in candidates_hyp if id(r) not in matched_hyp_ids]
        return out_pairs, leftover_src, leftover_hyp

    pairs: list[tuple[dict | None, dict | None, str]] = []

    p1, rem_src, rem_hyp = _run_pass(src_rows, hyp_rows, _cui_key)
    pairs.extend((s, h, "CUI") for s, h in p1)
    p2, rem_src, rem_hyp = _run_pass(rem_src,  rem_hyp,  _text_key)
    pairs.extend((s, h, "text") for s, h in p2)

    # Remaining rows have no CUI or text match — emit as unmatched singletons.
    for r in rem_src:
        pairs.append((r, None, "none"))
    for r in rem_hyp:
        pairs.append((None, r, "none"))

    return pairs


def _match_re_rows(
    src_rows: list[dict],
    hyp_rows: list[dict],
    src_span_cui:  dict[tuple[str, str], str] | None = None,
    hyp_span_cui:  dict[tuple[str, str], str] | None = None,
    src_span_text: dict[tuple[str, str], str] | None = None,
    hyp_span_text: dict[tuple[str, str], str] | None = None,
) -> list[tuple[dict | None, dict | None]]:
    """Match source and hypothesis RE rows using a cascaded two-pass strategy.

    Both passes require the ``relation`` field to match exactly between src and
    hyp (it is part of every key), so rows with different relation types are
    never paired.

    Pass 1 — **CUI** (both subject *and* object must have a CUI on each side):
        Key = ``(relation, "cui:<subj_CUI>", "cui:<obj_CUI>")``.
        Only pairs where *both* sides resolve their subject and object spans to
        a CUI are eligible.  This is the highest-precision tier.

    Pass 2 — **English text** (rows unmatched after pass 1):
        Key = ``(relation, "text:<subj_en>", "text:<obj_en>")``.
        For source rows the text is the original English surface form; for
        hypothesis rows it is the ``translated_text`` back-translation stored
        by ``sapbert_entity_linking.py``, making both sides comparable in
        English.  Only pairs where both spans have text on both sides are
        eligible.  This tier recovers from incorrect CUIs.

    Rows that cannot be matched by CUI or text are emitted as
    ``(row, None)`` / ``(None, row)`` singletons.

    Within each pass, rows sharing the same key are paired positionally (first
    with first, etc.) and any surplus rows from the longer group are carried
    forward as unmatched.
    """
    from collections import defaultdict

    def _norm(v: str) -> str:
        return v.strip().lower()

    def _span(row: dict, start_f: str, end_f: str) -> tuple[str, str]:
        return (_norm_span(row.get(start_f, "")), _norm_span(row.get(end_f, "")))

    def _cui_key(row: dict, span_cui: dict[tuple[str, str], str] | None) -> tuple[str, str, str] | None:
        """Return CUI-based key only when *both* subject and object have a CUI."""
        if not span_cui:
            return None
        sc = span_cui.get(_span(row, "subject_start", "subject_end"), "")
        oc = span_cui.get(_span(row, "object_start",  "object_end"),  "")
        if sc and oc:
            return (_norm(row.get("relation", "")), f"cui:{sc}", f"cui:{oc}")
        return None

    def _text_key(row: dict, span_text: dict[tuple[str, str], str] | None) -> tuple[str, str, str] | None:
        """Return text-based key only when *both* subject and object have text."""
        if not span_text:
            return None
        st = span_text.get(_span(row, "subject_start", "subject_end"), "").strip().lower()
        ot = span_text.get(_span(row, "object_start",  "object_end"),  "").strip().lower()
        if st and ot:
            return (_norm(row.get("relation", "")), f"text:{st}", f"text:{ot}")
        return None

    def _run_pass(
        candidates_src: list[dict],
        candidates_hyp: list[dict],
        key_fn_src,
        key_fn_hyp,
        require_both: bool,
    ) -> tuple[list[tuple[dict | None, dict | None]], list[dict], list[dict]]:
        """Group by key, pair positionally, return (pairs, unmatched_src, unmatched_hyp)."""
        src_by_key: dict[tuple, list[dict]] = defaultdict(list)
        hyp_by_key: dict[tuple, list[dict]] = defaultdict(list)

        for r in candidates_src:
            k = key_fn_src(r)
            if k is not None:
                src_by_key[k].append(r)

        for r in candidates_hyp:
            k = key_fn_hyp(r)
            if k is not None:
                hyp_by_key[k].append(r)

        # Only pair keys that appear on *both* sides when require_both=True
        if require_both:
            match_keys = sorted(set(src_by_key) & set(hyp_by_key))
        else:
            match_keys = sorted(set(src_by_key) | set(hyp_by_key))

        out_pairs: list[tuple[dict | None, dict | None]] = []
        matched_src_ids: set[int] = set()
        matched_hyp_ids: set[int] = set()

        for key in match_keys:
            sg = src_by_key.get(key, [])
            hg = hyp_by_key.get(key, [])
            for s, h in zip_longest(sg, hg):
                out_pairs.append((s, h))
                if s is not None:
                    matched_src_ids.add(id(s))
                if h is not None:
                    matched_hyp_ids.add(id(h))

        # Rows with a key that didn't match the other side + rows with no key
        leftover_src = [r for r in candidates_src if id(r) not in matched_src_ids]
        leftover_hyp = [r for r in candidates_hyp if id(r) not in matched_hyp_ids]
        return out_pairs, leftover_src, leftover_hyp

    pairs: list[tuple[dict | None, dict | None, str, str, str]] = []

    # Pass 1: CUI — subject=CUI, object=CUI, relation=text
    p1, rem_src, rem_hyp = _run_pass(
        src_rows, hyp_rows,
        lambda r: _cui_key(r, src_span_cui),
        lambda r: _cui_key(r, hyp_span_cui),
        require_both=True,
    )
    pairs.extend((s, h, "CUI", "CUI", "text") for s, h in p1)

    # Pass 2: text — subject=text, object=text, relation=text
    p2, rem_src, rem_hyp = _run_pass(
        rem_src, rem_hyp,
        lambda r: _text_key(r, src_span_text),
        lambda r: _text_key(r, hyp_span_text),
        require_both=True,
    )
    pairs.extend((s, h, "text", "text", "text") for s, h in p2)

    # Remaining rows have no CUI or text match with same relation — emit as singletons.
    for r in rem_src:
        pairs.append((r, None, "none", "none", "none"))
    for r in rem_hyp:
        pairs.append((None, r, "none", "none", "none"))

    return pairs


# ---------------------------------------------------------------------------
# Document alignment result
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Holds all alignment outputs for one source/hypothesis document pair.

    Attributes
    ----------
    beads:
        List of aligned sentence groups as ``_Bead`` tuples
        ``(src_text, hyp_text, src_start, src_end, hyp_start, hyp_end)``.
    src_spans / hyp_spans:
        Per-sentence ``(text, start_char, end_char)`` lists as produced by
        :func:`split_sentences_with_spans`.
    agg_src_el / agg_hyp_el:
        Entity-linking rows aggregated from beads (or the full document lists
        when no beads were produced).
    agg_src_re / agg_hyp_re:
        Relation-extraction rows aggregated from beads.
    agg_src_triples / agg_hyp_triples:
        Pre-resolved ``(subject_cui, relation, object_cui)`` triple sets, or
        ``None`` when RE data is unavailable.
    entity_records / relation_records:
        Per-bead matched EL/RE pairs written to the aligned TSV files.
    """
    beads: list
    src_spans: list
    hyp_spans: list
    agg_src_el: list
    agg_hyp_el: list
    agg_src_re: list
    agg_hyp_re: list
    agg_src_triples: Optional[set]
    agg_hyp_triples: Optional[set]
    entity_records: list
    relation_records: list


def align_document_pair(
    source: str,
    hypothesis: str,
    filename: str,
    base_path,
    src_el: list[dict] | None = None,
    hyp_el: list[dict] | None = None,
    src_triples: set[tuple[str, str, str]] | None = None,
    hyp_triples: set[tuple[str, str, str]] | None = None,
    src_re: list[dict] | None = None,
    hyp_re: list[dict] | None = None,
) -> AlignmentResult:
    """Run all alignment steps for one source/hypothesis document pair.

    1. Splits both texts into sentences with character-offset spans.
    2. Aligns sentence pairs via the Gale-Church algorithm.
    3. For each aligned bead, filters EL and RE rows to the bead's span and
       matches them with the cascaded CUI / text / label strategy.
    4. Aggregates the filtered rows and re-resolves CUI triple sets.
    5. Writes intermediate TSV files as side effects:
       - ``<base_path>/sentence_pairs/sent_pairs_<filename>.tsv``
       - ``<base_path>/aligned_entities/aligned_entities_<filename>.tsv``
       - ``<base_path>/aligned_relations/aligned_relations_<filename>.tsv``

    Parameters
    ----------
    source / hypothesis:
        Full document texts for the source (English) and hypothesis (Bulgarian).
    filename:
        Short identifier used in output file names (no directory component).
    base_path:
        Root directory for all output TSV files.
    src_el / hyp_el:
        Entity-linking rows (from :func:`load_el_tsv`) for the document, or
        ``None`` to skip KG alignment entirely.
    src_triples / hyp_triples:
        Pre-resolved CUI triple sets used as fallback when there are no aligned
        beads.  Pass ``None`` when RE data is unavailable.
    src_re / hyp_re:
        Relation-extraction rows (from :func:`load_re_tsv`) for the document,
        or ``None`` when RE data is unavailable.

    Returns
    -------
    :class:`AlignmentResult`
        All alignment outputs needed for downstream metric computation.
    """
    base_path = Path(base_path)

    # --- Sentence splitting ---
    src_spans = split_sentences_with_spans(source)
    hyp_spans = split_sentences_with_spans(hypothesis)
    beads = align_sentences_with_spans(src_spans, hyp_spans)

    # --- Write sentence pairs TSV ---
    if beads:
        df_sent_pairs = pd.DataFrame(
            [(b[0], b[1]) for b in beads], columns=["source", "hypothesis"]
        )
        sent_dir = base_path / "sentence_pairs"
        sent_dir.mkdir(parents=True, exist_ok=True)
        df_sent_pairs.to_csv(
            sent_dir / f"sent_pairs_{filename}.tsv", index=False, sep="\t"
        )  # debug

    # --- EL / RE alignment ---
    entity_records: list[dict] = []
    relation_records: list[dict] = []
    agg_src_el: list[dict] = []
    agg_hyp_el: list[dict] = []
    agg_src_re: list[dict] = []
    agg_hyp_re: list[dict] = []
    agg_src_triples: set[tuple[str, str, str]] | None = None
    agg_hyp_triples: set[tuple[str, str, str]] | None = None

    if src_el is not None and hyp_el is not None:
        if beads:
            for bead_idx, (s_txt, h_txt, s_start, s_end, h_start, h_end) in enumerate(beads):
                bead_meta = {"_bead_idx": bead_idx, "_bead_src": s_txt, "_bead_hyp": h_txt}
                bead_src_el = _filter_el_by_span(src_el, s_start, s_end)
                bead_hyp_el = _filter_el_by_span(hyp_el, h_start, h_end)
                agg_src_el.extend(bead_src_el)
                agg_hyp_el.extend(bead_hyp_el)
                for src_r, hyp_r, match_type in _match_el_rows(bead_src_el, bead_hyp_el):
                    rec = {**bead_meta, "match_type": match_type}
                    if src_r is not None:
                        for k, v in src_r.items():
                            rec[f"src_{k}"] = v
                    if hyp_r is not None:
                        for k, v in hyp_r.items():
                            rec[f"hyp_{k}"] = v
                        translated = (hyp_r.get("translated_text", "").strip()
                                      or hyp_r.get("text", "").strip())
                        if translated:
                            rec["hyp_text"] = translated
                    entity_records.append(rec)
                bead_src_re = _filter_re_by_span(src_re, s_start, s_end) if src_re is not None else []
                bead_hyp_re = _filter_re_by_span(hyp_re, h_start, h_end) if hyp_re is not None else []
                agg_src_re.extend(bead_src_re)
                agg_hyp_re.extend(bead_hyp_re)
                bead_src_cui  = _span_cui_map(bead_src_el)
                bead_hyp_cui  = _span_cui_map(bead_hyp_el)
                bead_src_text = _span_text_map(bead_src_el)
                bead_hyp_text = _span_text_map(bead_hyp_el)
                for src_r, hyp_r, subj_mt, obj_mt, rel_mt in _match_re_rows(
                    bead_src_re, bead_hyp_re,
                    bead_src_cui, bead_hyp_cui,
                    bead_src_text, bead_hyp_text,
                ):
                    rec = {
                        **bead_meta,
                        "subject_match_type": subj_mt,
                        "object_match_type": obj_mt,
                        "relation_match_type": rel_mt,
                    }
                    if src_r is not None:
                        for k, v in src_r.items():
                            rec[f"src_{k}"] = v
                        src_subj_cui = bead_src_cui.get((
                            _norm_span(src_r.get("subject_start", "")),
                            _norm_span(src_r.get("subject_end", "")),
                        ), "")
                        src_obj_cui = bead_src_cui.get((
                            _norm_span(src_r.get("object_start", "")),
                            _norm_span(src_r.get("object_end", "")),
                        ), "")
                        if src_subj_cui:
                            rec["src_subject_cui"] = src_subj_cui
                        if src_obj_cui:
                            rec["src_object_cui"] = src_obj_cui
                    if hyp_r is not None:
                        for k, v in hyp_r.items():
                            rec[f"hyp_{k}"] = v
                        subj_translated = bead_hyp_text.get((
                            _norm_span(hyp_r.get("subject_start", "")),
                            _norm_span(hyp_r.get("subject_end", "")),
                        ), "")
                        obj_translated = bead_hyp_text.get((
                            _norm_span(hyp_r.get("object_start", "")),
                            _norm_span(hyp_r.get("object_end", "")),
                        ), "")
                        if subj_translated:
                            rec["hyp_subject_text"] = subj_translated
                        if obj_translated:
                            rec["hyp_object_text"] = obj_translated
                        hyp_subj_cui = bead_hyp_cui.get((
                            _norm_span(hyp_r.get("subject_start", "")),
                            _norm_span(hyp_r.get("subject_end", "")),
                        ), "")
                        hyp_obj_cui = bead_hyp_cui.get((
                            _norm_span(hyp_r.get("object_start", "")),
                            _norm_span(hyp_r.get("object_end", "")),
                        ), "")
                        if hyp_subj_cui:
                            rec["hyp_subject_cui"] = hyp_subj_cui
                        if hyp_obj_cui:
                            rec["hyp_object_cui"] = hyp_obj_cui
                    relation_records.append(rec)
            # Re-resolve CUI triples from the filtered RE rows using the
            # filtered EL span maps (language-neutral comparison still holds).
            if agg_src_re and agg_hyp_re:
                agg_src_triples = _relation_triple_set(
                    agg_src_re, _span_cui_map(agg_src_el)
                )
                agg_hyp_triples = _relation_triple_set(
                    agg_hyp_re, _span_cui_map(agg_hyp_el)
                )
        else:
            agg_src_el = src_el
            agg_hyp_el = hyp_el
            agg_src_re = src_re or []
            agg_hyp_re = hyp_re or []
            agg_src_triples = src_triples
            agg_hyp_triples = hyp_triples

        # Span→translated_text and span→CUI maps for the full documents
        # (used for matching unaligned RE rows whose bead maps are not available).
        doc_src_text: dict[tuple[str, str], str] = _span_text_map(src_el)
        doc_hyp_text: dict[tuple[str, str], str] = _span_text_map(hyp_el)
        doc_src_cui:  dict[tuple[str, str], str] = _span_cui_map(src_el)
        doc_hyp_cui:  dict[tuple[str, str], str] = _span_cui_map(hyp_el)

        # --- Write aligned entity / relation TSVs ---
        # Rows that were NOT covered by any aligned bead are appended with
        # _bead_idx = None so they are still visible in the output file.
        if beads:
            aligned_src_el_ids = {id(r) for r in agg_src_el}
            aligned_hyp_el_ids = {id(r) for r in agg_hyp_el}
            aligned_src_re_ids = {id(r) for r in agg_src_re}
            aligned_hyp_re_ids = {id(r) for r in agg_hyp_re}
            unaligned_meta = {"_bead_idx": None, "_bead_src": None, "_bead_hyp": None}
            unaligned_src_el = [r for r in src_el if id(r) not in aligned_src_el_ids]
            unaligned_hyp_el = [r for r in hyp_el if id(r) not in aligned_hyp_el_ids]
            for src_r in unaligned_src_el:
                rec = {**unaligned_meta, "match_type": "none"}
                for k, v in src_r.items():
                    rec[f"src_{k}"] = v
                entity_records.append(rec)
            for hyp_r in unaligned_hyp_el:
                rec = {**unaligned_meta, "match_type": "none"}
                for k, v in hyp_r.items():
                    rec[f"hyp_{k}"] = v
                translated = (hyp_r.get("translated_text", "").strip()
                              or hyp_r.get("text", "").strip())
                if translated:
                    rec["hyp_text"] = translated
                entity_records.append(rec)
            unaligned_src_re = [r for r in (src_re or []) if id(r) not in aligned_src_re_ids]
            unaligned_hyp_re = [r for r in (hyp_re or []) if id(r) not in aligned_hyp_re_ids]
            for src_r, hyp_r, subj_mt, obj_mt, rel_mt in _match_re_rows(
                unaligned_src_re, unaligned_hyp_re,
                doc_src_cui, doc_hyp_cui,
                doc_src_text, doc_hyp_text,
            ):
                rec = {
                    **unaligned_meta,
                    "subject_match_type": subj_mt,
                    "object_match_type": obj_mt,
                    "relation_match_type": rel_mt,
                }
                if src_r is not None:
                    for k, v in src_r.items():
                        rec[f"src_{k}"] = v
                    src_subj_cui = doc_src_cui.get((
                        _norm_span(src_r.get("subject_start", "")),
                        _norm_span(src_r.get("subject_end", "")),
                    ), "")
                    src_obj_cui = doc_src_cui.get((
                        _norm_span(src_r.get("object_start", "")),
                        _norm_span(src_r.get("object_end", "")),
                    ), "")
                    if src_subj_cui:
                        rec["src_subject_cui"] = src_subj_cui
                    if src_obj_cui:
                        rec["src_object_cui"] = src_obj_cui
                if hyp_r is not None:
                    for k, v in hyp_r.items():
                        rec[f"hyp_{k}"] = v
                    subj_translated = doc_hyp_text.get((
                        _norm_span(hyp_r.get("subject_start", "")),
                        _norm_span(hyp_r.get("subject_end", "")),
                    ), "")
                    obj_translated = doc_hyp_text.get((
                        _norm_span(hyp_r.get("object_start", "")),
                        _norm_span(hyp_r.get("object_end", "")),
                    ), "")
                    if subj_translated:
                        rec["hyp_subject_text"] = subj_translated
                    if obj_translated:
                        rec["hyp_object_text"] = obj_translated
                    hyp_subj_cui = doc_hyp_cui.get((
                        _norm_span(hyp_r.get("subject_start", "")),
                        _norm_span(hyp_r.get("subject_end", "")),
                    ), "")
                    hyp_obj_cui = doc_hyp_cui.get((
                        _norm_span(hyp_r.get("object_start", "")),
                        _norm_span(hyp_r.get("object_end", "")),
                    ), "")
                    if hyp_subj_cui:
                        rec["hyp_subject_cui"] = hyp_subj_cui
                    if hyp_obj_cui:
                        rec["hyp_object_cui"] = hyp_obj_cui
                relation_records.append(rec)

        if entity_records:
            ent_dir = base_path / "aligned_entities"
            ent_dir.mkdir(parents=True, exist_ok=True)
            ent_df = pd.DataFrame(entity_records)
            meta_cols = ["_bead_idx", "_bead_src", "_bead_hyp", "match_type"]
            src_cols = sorted(c for c in ent_df.columns if c.startswith("src_"))
            hyp_cols = sorted(c for c in ent_df.columns if c.startswith("hyp_"))
            other_cols = [c for c in ent_df.columns if c not in set(meta_cols + src_cols + hyp_cols)]
            ent_df[meta_cols + src_cols + hyp_cols + other_cols].to_csv(
                ent_dir / f"aligned_entities_{filename}.tsv", index=False, sep="\t"
            )
        if relation_records:
            rel_dir = base_path / "aligned_relations"
            rel_dir.mkdir(parents=True, exist_ok=True)
            rel_df = pd.DataFrame(relation_records)
            meta_cols = ["_bead_idx", "_bead_src", "_bead_hyp",
                         "subject_match_type", "object_match_type", "relation_match_type"]
            src_cols = sorted(c for c in rel_df.columns if c.startswith("src_"))
            hyp_cols = sorted(c for c in rel_df.columns if c.startswith("hyp_"))
            other_cols = [c for c in rel_df.columns if c not in set(meta_cols + src_cols + hyp_cols)]
            rel_df[meta_cols + src_cols + hyp_cols + other_cols].to_csv(
                rel_dir / f"aligned_relations_{filename}.tsv", index=False, sep="\t"
            )

    return AlignmentResult(
        beads=beads,
        src_spans=src_spans,
        hyp_spans=hyp_spans,
        agg_src_el=agg_src_el,
        agg_hyp_el=agg_hyp_el,
        agg_src_re=agg_src_re,
        agg_hyp_re=agg_hyp_re,
        agg_src_triples=agg_src_triples,
        agg_hyp_triples=agg_hyp_triples,
        entity_records=entity_records,
        relation_records=relation_records,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run alignment for a corpus of source/hypothesis document pairs.

    Produces per-document sentence-pair, aligned-entity, and aligned-relation
    TSV files under the output directory.  Optionally accepts EL and RE TSV
    files for entity/relation-level alignment.

    Usage examples
    --------------
    # Sentence alignment only (no KG data):
    python translation_alignment.py texts_top_1 results/alignment \\
        --translations-dir translations/gpt41_top1

    # With entity linking and relation extraction:
    python translation_alignment.py texts_top_1 results/alignment \\
        --translations-dir translations/gpt41_top1 \\
        --el-src el_src.tsv --el-hyp el_hyp.tsv \\
        --rel-src re_src.tsv --rel-hyp re_hyp.tsv
    """
    import argparse
    import sys

    # ------------------------------------------------------------------
    # Default values — edit these when running without CLI arguments
    # ------------------------------------------------------------------
    DEFAULT_INPUT_DIR        = "texts_top_1"
    DEFAULT_OUTPUT_DIR       = "results/alignment_top1"
    DEFAULT_TRANSLATIONS_DIR = "translations/gpt41_top1"
    DEFAULT_EL_SRC           = None
    DEFAULT_EL_HYP           = None
    DEFAULT_REL_SRC          = None
    DEFAULT_REL_HYP          = None
    DEFAULT_MAX_DOCS         = None
    # ------------------------------------------------------------------

    if not sys.argv[1:]:
        # No CLI arguments: run directly with the defaults above
        _run_alignment(
            input_dir=DEFAULT_INPUT_DIR,
            output_dir=DEFAULT_OUTPUT_DIR,
            translations_dir=DEFAULT_TRANSLATIONS_DIR,
            el_src=DEFAULT_EL_SRC,
            el_hyp=DEFAULT_EL_HYP,
            rel_src=DEFAULT_REL_SRC,
            rel_hyp=DEFAULT_REL_HYP,
            max_docs=DEFAULT_MAX_DOCS,
        )
        return

    parser = argparse.ArgumentParser(
        description="Align EN source and BG hypothesis document pairs and write "
                    "sentence-pair / entity / relation TSV files."
    )
    parser.add_argument("input_dir",  help="Directory with English .txt source files")
    parser.add_argument("output_dir", help="Root directory for output TSV files")
    parser.add_argument(
        "--translations-dir",
        metavar="DIR",
        required=True,
        help="Directory containing pre-translated Bulgarian .txt files "
             "(mirroring input_dir structure)",
    )
    parser.add_argument(
        "--el-src",
        metavar="TSV",
        default=None,
        help="Entity linking TSV for the English source texts.",
    )
    parser.add_argument(
        "--el-hyp",
        metavar="TSV",
        default=None,
        help="Entity linking TSV for the Bulgarian hypothesis translations.",
    )
    parser.add_argument(
        "--rel-src",
        metavar="TSV",
        default=None,
        help="Relation extraction TSV for the English source texts.",
    )
    parser.add_argument(
        "--rel-hyp",
        metavar="TSV",
        default=None,
        help="Relation extraction TSV for the Bulgarian hypothesis translations.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N document pairs (useful for quick testing).",
    )

    args = parser.parse_args()
    _run_alignment(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        translations_dir=args.translations_dir,
        el_src=args.el_src,
        el_hyp=args.el_hyp,
        rel_src=args.rel_src,
        rel_hyp=args.rel_hyp,
        max_docs=args.max_docs,
    )


def _run_alignment(
    input_dir: str,
    output_dir: str,
    translations_dir: str,
    el_src: Optional[str] = None,
    el_hyp: Optional[str] = None,
    rel_src: Optional[str] = None,
    rel_hyp: Optional[str] = None,
    max_docs: Optional[int] = None,
) -> None:
    """Core alignment loop called by :func:`main`."""
    from tqdm import tqdm

    texts = load_texts(input_dir, max_docs)
    print(f"Source texts: {len(texts)} from '{input_dir}'" +
          (f" (limited to {max_docs})" if max_docs is not None else ""))

    translations = load_translations(translations_dir, texts)
    print(f"Translations: {len(translations)} from '{translations_dir}'")

    eval_pairs = [(rel, txt) for rel, txt in texts if rel in translations]
    print(f"Pairs to align: {len(eval_pairs)}")

    use_kg = bool(el_src and el_hyp)
    use_re = False
    src_el_data: dict[str, list[dict]] = {}
    hyp_el_data: dict[str, list[dict]] = {}
    src_triples_data: dict[str, set[tuple[str, str, str]]] = {}
    hyp_triples_data: dict[str, set[tuple[str, str, str]]] = {}
    src_re_data: dict[str, list[dict]] = {}
    hyp_re_data: dict[str, list[dict]] = {}

    if use_kg:
        print(f"Loading EL (src) from: {el_src}")
        src_el_data = load_el_tsv(el_src)
        print(f"Loading EL (hyp) from: {el_hyp}")
        hyp_el_data = load_el_tsv(el_hyp)
        if rel_src and rel_hyp:
            print(f"Loading RE (src) from: {rel_src}")
            src_re_data = load_re_tsv(rel_src)
            print(f"Loading RE (hyp) from: {rel_hyp}")
            hyp_re_data = load_re_tsv(rel_hyp)
            print("Resolving RE spans to UMLS CUIs from EL files …")
            src_triples_data = resolve_relations(src_re_data, src_el_data)
            hyp_triples_data = resolve_relations(hyp_re_data, hyp_el_data)
            use_re = True
        elif rel_src or rel_hyp:
            print("[warn] Both --rel-src and --rel-hyp are needed for relation alignment; skipping.")
    elif el_src or el_hyp:
        print("[warn] Both --el-src and --el-hyp are needed for entity alignment; skipping.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Aligning …")
    for rel, src_text in tqdm(eval_pairs, unit="doc"):
        hyp_text = translations[rel]
        hyp_key  = _hyp_rel(rel)

        hyp_el_rows = (hyp_el_data.get(hyp_key) or hyp_el_data.get(rel)) if use_kg else None
        src_triples = src_triples_data.get(rel) if use_re else None
        hyp_triples = (
            (hyp_triples_data.get(hyp_key) or hyp_triples_data.get(rel)) if use_re else None
        )
        src_re_rows = src_re_data.get(rel) if use_re else None
        hyp_re_rows = (
            (hyp_re_data.get(hyp_key) or hyp_re_data.get(rel)) if use_re else None
        )

        filename = rel[rel.rindex('/') + 1:]
        align_document_pair(
            src_text, hyp_text, filename, out_path,
            src_el=src_el_data.get(rel) if use_kg else None,
            hyp_el=hyp_el_rows,
            src_triples=src_triples,
            hyp_triples=hyp_triples,
            src_re=src_re_rows,
            hyp_re=hyp_re_rows,
        )

    print(f"\nAlignment outputs written to '{output_dir}'")


if __name__ == "__main__":
    main()
