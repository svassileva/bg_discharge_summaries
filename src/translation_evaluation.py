#!/usr/bin/env python3
"""
Evaluate English → Bulgarian machine translation of discharge summaries.

Two evaluation axes:
  1. Cross-lingual BERTScore  — uses a multilingual BERT variant (mBERT or XLM-R)
     to compare token-level embeddings between the English source and the Bulgarian
     hypothesis in a shared multilingual space.
  2. LaBSE cosine similarity  — Language-agnostic BERT Sentence Embeddings produce
     a single vector per text regardless of language; cosine distance captures
     document-level semantic preservation.

Both metrics are run at document level and at sentence level (aligned by position).

Sentence splitting, Gale-Church alignment, entity-linking alignment, and
relation-extraction alignment are handled by :mod:`translation_alignment`.

#pip install bert-score sentence-transformers openai tqdm


Usage examples
--------------
# Translate on-the-fly with GPT-4.1 and save translations:
python translation_evaluation.py texts_top_1 results/eval_top1.tsv \\
    --translate --openai-model gpt-4.1 \\
    --save-translations translations/gpt41_top1

# Evaluate pre-existing Bulgarian translations:
python translation_evaluation.py texts_top_30 results/eval_top30.tsv \\
    --translations-dir translations/gpt41_top30

# Use XLM-R for BERTScore instead of mBERT:
python translation_evaluation.py texts_top_1 results/eval_xlmr.tsv \\
    --translate --bertscore-model xlm-roberta-base

Requirements
------------
    pip install bert-score sentence-transformers openai tqdm numpy torch
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from translation_alignment import (
    _entity_cui_set,
    _entity_text_list,
    _hyp_rel,
    _relation_text_list,
    _span_text_map,
    align_document_pair,
    load_el_tsv,
    load_re_tsv,
    load_texts,
    load_translations,
    resolve_relations,
    save_translation,
    translate_text,
)


# ---------------------------------------------------------------------------
# BERTScore (cross-lingual)
# ---------------------------------------------------------------------------

def compute_bertscore(
    scorer: BERTScorer,
    sources: list[str],
    hypotheses: list[str],
) -> tuple[list[float], list[float], list[float]]:
    """
    Cross-lingual BERTScore: hypotheses (BG) vs. sources (EN) as pseudo-references.
    Returns (precision, recall, F1) lists.
    """
    P, R, F1 = scorer.score(hypotheses, sources)
    return P.tolist(), R.tolist(), F1.tolist()


# ---------------------------------------------------------------------------
# LaBSE similarity
# ---------------------------------------------------------------------------

def compute_labse(
    model: SentenceTransformer,
    sources: list[str],
    hypotheses: list[str],
    batch_size: int = 8,
) -> list[float]:
    """
    Encode both lists with LaBSE (L2-normalised) and return per-pair cosine scores.
    Since embeddings are normalised the dot product equals cosine similarity.
    """
    src_emb = model.encode(
        sources, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    hyp_emb = model.encode(
        hypotheses, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return [float(np.dot(s, h)) for s, h in zip(src_emb, hyp_emb)]


# ---------------------------------------------------------------------------
# KGBERTScore — entity- and relation-level knowledge graph consistency
# ---------------------------------------------------------------------------

def _prf(ref: set, pred: set) -> tuple[float, float, float]:
    """Compute precision / recall / F1 with *ref* as reference and *pred* as prediction."""
    precision = len(ref & pred) / len(pred) if pred else 0.0
    recall    = len(ref & pred) / len(ref)  if ref  else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_kg_embed_overlap(
    scorer: BERTScorer,
    src_el: list[dict],
    hyp_el: list[dict],
    threshold: float = 0.85,
) -> dict:
    """
    Entity-level BERTScore similarity between source and hypothesis entity texts.

    For every hypothesis entity surface form (preferring ``translated_text`` as
    back-translation when available) the maximum BERTScore F1 against all source
    entity texts is computed.  An entity is counted as *matched* when this
    maximum exceeds ``threshold``.  Symmetric treatment of source entities gives
    recall.

    All pairs are evaluated in a single BERTScorer batch for efficiency.

    Returns
    -------
    dict with keys ``kg_embed_precision``, ``kg_embed_recall``, ``kg_embed_f1``
    (or ``None`` when either side has no entity texts).
    """
    src_texts = _entity_text_list(src_el)
    hyp_texts = _entity_text_list(hyp_el)

    if not src_texts or not hyp_texts:
        return {
            "kg_embed_precision": None,
            "kg_embed_recall":    None,
            "kg_embed_f1":        None,
        }

    n_src = len(src_texts)
    n_hyp = len(hyp_texts)

    # Build flat cands/refs lists covering all (hyp, src) pairs in one batch
    cands = [h for h in hyp_texts for _ in src_texts]   # each hyp repeated n_src times
    refs  = src_texts * n_hyp                             # src cycle repeated n_hyp times

    _, _, F1 = scorer.score(cands, refs)
    F1_mat = F1.reshape(n_hyp, n_src)                    # (hyp × src)

    # Precision: fraction of hyp entities with a close src match
    hyp_matched = int((F1_mat.max(dim=1).values >= threshold).sum().item())
    # Recall: fraction of src entities with a close hyp match
    src_covered = int((F1_mat.max(dim=0).values >= threshold).sum().item())

    precision = hyp_matched / n_hyp
    recall    = src_covered / n_src
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "kg_embed_precision": round(precision, 5),
        "kg_embed_recall":    round(recall, 5),
        "kg_embed_f1":        round(f1, 5),
    }


def compute_kg_rel_embed_overlap(
    scorer: BERTScorer,
    src_re: list[dict],
    hyp_re: list[dict],
    src_el: list[dict],
    hyp_el: list[dict],
    threshold: float = 0.85,
) -> dict:
    """
    Relation-level BERTScore similarity using surface/back-translated entity texts.

    Each relation is represented as ``"<subject_text> [<relation_type>] <object_text>"``
    where entity texts come from EL rows (preferring ``translated_text`` as
    back-translation when available, enabling EN↔EN BERTScore even for
    hypothesis-side relations).  BERTScore F1 is computed for every
    (hyp_relation, src_relation) pair; a relation is counted as *matched* when
    the maximum F1 against all source relations exceeds ``threshold``.

    Returns
    -------
    dict with keys ``kg_rel_embed_precision``, ``kg_rel_embed_recall``,
    ``kg_rel_embed_f1`` (or ``None`` when either side has no representable relations).
    """
    src_span_text = _span_text_map(src_el)
    hyp_span_text = _span_text_map(hyp_el)
    src_texts = _relation_text_list(src_re, src_span_text)
    hyp_texts = _relation_text_list(hyp_re, hyp_span_text)

    if not src_texts or not hyp_texts:
        return {
            "kg_rel_embed_precision": None,
            "kg_rel_embed_recall":    None,
            "kg_rel_embed_f1":        None,
        }

    n_src = len(src_texts)
    n_hyp = len(hyp_texts)

    cands = [h for h in hyp_texts for _ in src_texts]
    refs  = src_texts * n_hyp

    _, _, F1 = scorer.score(cands, refs)
    F1_mat = F1.reshape(n_hyp, n_src)

    hyp_matched = int((F1_mat.max(dim=1).values >= threshold).sum().item())
    src_covered = int((F1_mat.max(dim=0).values >= threshold).sum().item())

    precision = hyp_matched / n_hyp
    recall    = src_covered / n_src
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "kg_rel_embed_precision": round(precision, 5),
        "kg_rel_embed_recall":    round(recall, 5),
        "kg_rel_embed_f1":        round(f1, 5),
    }


def compute_kgbertscore(
    src_el: list[dict],
    hyp_el: list[dict],
    src_triples: set[tuple[str, str, str]] | None,
    hyp_triples: set[tuple[str, str, str]] | None,
    scorer: BERTScorer | None = None,
    kg_embed_threshold: float = 0.85,
    src_re: list[dict] | None = None,
    hyp_re: list[dict] | None = None,
) -> dict:
    """
    Compute KGBERTScore for one document pair.

    Entity-level (UMLS CUI set overlap, top-1 linking per entity):
      precision  = |hyp_cuis ∩ src_cuis| / |hyp_cuis|
      recall     = |hyp_cuis ∩ src_cuis| / |src_cuis|
      f1         = harmonic mean

    Relation-level ((subject_CUI, relation_type, object_CUI) triple set overlap):
      Same P/R/F1 formulation over the pre-resolved CUI triple sets.
      Triples are resolved from RE spans via EL files before this call, so
      subject/object positions are already mapped to language-neutral CUIs,
      allowing direct set comparison between English and translated triples.
    """
    result: dict = {}

    # --- Entity level ---
    src_cuis = _entity_cui_set(src_el)
    hyp_cuis = _entity_cui_set(hyp_el)
    ep, er, ef = _prf(src_cuis, hyp_cuis)
    result["kg_entity_precision"] = round(ep, 5)
    result["kg_entity_recall"]    = round(er, 5)
    result["kg_entity_f1"]        = round(ef, 5)
    result["kg_n_src_entities"]   = len(src_cuis)
    result["kg_n_hyp_entities"]   = len(hyp_cuis)

    # --- Relation level (pre-resolved CUI triples) ---
    if src_triples is not None and hyp_triples is not None:
        rp, rr, rf = _prf(src_triples, hyp_triples)
        result["kg_rel_precision"]   = round(rp, 5)
        result["kg_rel_recall"]      = round(rr, 5)
        result["kg_rel_f1"]          = round(rf, 5)
        result["kg_n_src_relations"] = len(src_triples)
        result["kg_n_hyp_relations"] = len(hyp_triples)
    else:
        for k in (
            "kg_rel_precision", "kg_rel_recall", "kg_rel_f1",
            "kg_n_src_relations", "kg_n_hyp_relations",
        ):
            result[k] = None

    # --- Entity embedding level (BERTScore on surface text / back-translation) ---
    if scorer is not None:
        embed = compute_kg_embed_overlap(scorer, src_el, hyp_el, kg_embed_threshold)
        result.update(embed)
    else:
        result["kg_embed_precision"] = None
        result["kg_embed_recall"]    = None
        result["kg_embed_f1"]        = None

    # --- Relation embedding level (BERTScore on relation text / back-translation) ---
    if scorer is not None and src_re is not None and hyp_re is not None:
        rel_embed = compute_kg_rel_embed_overlap(
            scorer, src_re, hyp_re, src_el, hyp_el, kg_embed_threshold
        )
        result.update(rel_embed)
    else:
        result["kg_rel_embed_precision"] = None
        result["kg_rel_embed_recall"]    = None
        result["kg_rel_embed_f1"]        = None

    return result


# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    source: str,
    hypothesis: str,
    bertscore: BERTScorer,
    labse: SentenceTransformer,
    filename: str,
    base_path: str,
    src_el: list[dict] | None = None,
    hyp_el: list[dict] | None = None,
    src_triples: set[tuple[str, str, str]] | None = None,
    hyp_triples: set[tuple[str, str, str]] | None = None,
    kg_embed_threshold: float = 0.85,
    src_re: list[dict] | None = None,
    hyp_re: list[dict] | None = None,
) -> dict:
    row: dict = {}

    # --- Document level ---
    P, R, F1 = compute_bertscore(bertscore, [source], [hypothesis])
    labse_scores = compute_labse(labse, [source], [hypothesis])

    row["doc_bs_precision"] = round(P[0], 5)
    row["doc_bs_recall"]    = round(R[0], 5)
    row["doc_bs_f1"]        = round(F1[0], 5)
    row["doc_labse"]        = round(labse_scores[0], 5)

    # --- Alignment (sentence splitting + Gale-Church + EL/RE alignment) ---
    alignment = align_document_pair(
        source, hypothesis, filename, base_path,
        src_el=src_el, hyp_el=hyp_el,
        src_triples=src_triples, hyp_triples=hyp_triples,
        src_re=src_re, hyp_re=hyp_re,
    )
    beads = alignment.beads
    n = len(beads)

    # --- Sentence level metrics ---
    if n > 0:
        aligned_src = [b[0] for b in beads]
        aligned_hyp = [b[1] for b in beads]

        sP, sR, sF1 = compute_bertscore(bertscore, aligned_src, aligned_hyp)
        s_labse     = compute_labse(labse, aligned_src, aligned_hyp)

        row["sent_bs_precision_mean"] = round(float(np.mean(sP)), 5)
        row["sent_bs_recall_mean"]    = round(float(np.mean(sR)), 5)
        row["sent_bs_f1_mean"]        = round(float(np.mean(sF1)), 5)
        row["sent_labse_mean"]        = round(float(np.mean(s_labse)), 5)
        row["sent_labse_std"]         = round(float(np.std(s_labse)), 5)
        row["sent_labse_low_frac"]    = round(float(np.mean([s < 0.75 for s in s_labse])), 5)
        row["n_src_sentences"]        = len(alignment.src_spans)
        row["n_hyp_sentences"]        = len(alignment.hyp_spans)
        row["n_aligned_sentences"]    = n
    else:
        for k in (
            "sent_bs_precision_mean", "sent_bs_recall_mean", "sent_bs_f1_mean",
            "sent_labse_mean", "sent_labse_std", "sent_labse_low_frac",
            "n_src_sentences", "n_hyp_sentences", "n_aligned_sentences",
        ):
            row[k] = None

    # --- KGBERTScore (computed on alignment-filtered rows) ---
    if src_el is not None and hyp_el is not None:
        kg = compute_kgbertscore(
            alignment.agg_src_el, alignment.agg_hyp_el,
            alignment.agg_src_triples, alignment.agg_hyp_triples,
            scorer=bertscore,
            kg_embed_threshold=kg_embed_threshold,
            src_re=alignment.agg_src_re or None,
            hyp_re=alignment.agg_hyp_re or None,
        )
        row.update(kg)
    else:
        for k in (
            "kg_entity_precision", "kg_entity_recall", "kg_entity_f1",
            "kg_n_src_entities", "kg_n_hyp_entities",
            "kg_rel_precision", "kg_rel_recall", "kg_rel_f1",
            "kg_n_src_relations", "kg_n_hyp_relations",
            "kg_embed_precision", "kg_embed_recall", "kg_embed_f1",
            "kg_rel_embed_precision", "kg_rel_embed_recall", "kg_rel_embed_f1",
        ):
            row[k] = None

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_FIELDNAMES = [
    "file",
    "doc_bs_precision", "doc_bs_recall", "doc_bs_f1",
    "doc_labse",
    "sent_bs_precision_mean", "sent_bs_recall_mean", "sent_bs_f1_mean",
    "sent_labse_mean", "sent_labse_std", "sent_labse_low_frac",
    "n_src_sentences", "n_hyp_sentences", "n_aligned_sentences",
    # KGBERTScore
    "kg_entity_precision", "kg_entity_recall", "kg_entity_f1",
    "kg_n_src_entities", "kg_n_hyp_entities",
    "kg_rel_precision", "kg_rel_recall", "kg_rel_f1",
    "kg_n_src_relations", "kg_n_hyp_relations",
    # KGBERTScore — embedding-based entity match
    "kg_embed_precision", "kg_embed_recall", "kg_embed_f1",
    # KGBERTScore — embedding-based relation match
    "kg_rel_embed_precision", "kg_rel_embed_recall", "kg_rel_embed_f1",
]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process(
    input_dir: str,
    output_tsv: str,
    translations_dir: Optional[str] = None,
    translate: bool = False,
    openai_model: str = "gpt-4.1",
    bertscore_model: str = "bert-base-multilingual-cased",
    labse_model: str = "sentence-transformers/LaBSE",
    save_translations: Optional[str] = None,
    device: Optional[str] = None,
    el_src: Optional[str] = None,
    el_hyp: Optional[str] = None,
    rel_src: Optional[str] = None,
    rel_hyp: Optional[str] = None,
    kg_embed_threshold: float = 0.85,
    max_docs: Optional[int] = None,
) -> None:
    """Run the full evaluation pipeline with explicitly supplied parameters."""

    resolved_device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {resolved_device}")

    # --- Load source texts ---
    texts = load_texts(input_dir, max_docs)
    print(f"Source texts loaded: {len(texts)} from '{input_dir}'" +
          (f" (limited to {max_docs})" if max_docs is not None else ""))

    # --- Obtain translations ---
    translations: dict[str, str]

    if translations_dir:
        translations = load_translations(translations_dir, texts)
        print(f"Translations loaded: {len(translations)} from '{translations_dir}'")
    elif translate:
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")

        client = OpenAI(api_key=api_key)
        translations = {}
        print(f"Translating {len(texts)} texts with '{openai_model}' …")
        for rel, src_text in tqdm(texts, unit="doc"):
            bg_text = translate_text(src_text, client, openai_model)
            translations[rel] = bg_text
            if save_translations:
                save_translation(bg_text, rel, save_translations)
    else:
        raise ValueError("Provide either translations_dir or set translate=True.")

    # Filter to texts that have a translation
    eval_pairs = [(rel, txt) for rel, txt in texts if rel in translations]
    print(f"Pairs to evaluate: {len(eval_pairs)}")

    # --- Load KG data (optional) ---
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
            # Resolve subject/object spans to UMLS CUIs using the EL span maps so
            # that triples are represented as (subject_cui, relation, object_cui).
            # CUIs are language-neutral, enabling direct set comparison between
            # English source triples and translated hypothesis triples.
            print("Resolving RE spans to UMLS CUIs from EL files …")
            src_triples_data = resolve_relations(src_re_data, src_el_data)
            hyp_triples_data = resolve_relations(hyp_re_data, hyp_el_data)
            use_re = True
            print(f"  src documents with triples: "
                  f"{sum(1 for t in src_triples_data.values() if t)} / "
                  f"{len(src_triples_data)}")
            print(f"  hyp documents with triples: "
                  f"{sum(1 for t in hyp_triples_data.values() if t)} / "
                  f"{len(hyp_triples_data)}")
        elif rel_src or rel_hyp:
            print("[warn] Both rel_src and rel_hyp are needed for relation scoring; skipping.")
    else:
        if el_src or el_hyp:
            print("[warn] Both el_src and el_hyp are needed for KGBERTScore; skipping.")

    # --- Load models ---
    print(f"Loading BERTScorer  ({bertscore_model}) …")
    bertscore = BERTScorer(
        model_type=bertscore_model,
        lang=None,               # language auto-detected — required for cross-lingual use
        device=resolved_device,
        rescale_with_baseline=False,
    )

    print(f"Loading LaBSE       ({labse_model}) …")
    labse = SentenceTransformer(labse_model, device=resolved_device)

    # --- Evaluate ---
    rows: list[dict] = []
    all_results: dict[str, dict] = {}

    print("Evaluating …")
    for rel, src_text in tqdm(eval_pairs, unit="doc"):
        hyp_text = translations[rel]
        # EL/RE TSVs for the hypothesis may use either the _BGgpt-suffixed filename
        # (when NER was run directly on translated texts) or the original source
        # filename (when the RE/EL pipeline reused the source NER output).
        # Try the suffixed key first; fall back to the source key.
        hyp_key = _hyp_rel(rel)
        hyp_el_rows = (hyp_el_data.get(hyp_key) or hyp_el_data.get(rel)) if use_kg else None
        # CUI triples are pre-resolved: try the suffixed hypothesis key first,
        # then fall back to the source key (same as the EL/RE lookup above).
        hyp_triples = (
            (hyp_triples_data.get(hyp_key) or hyp_triples_data.get(rel))
            if use_re else None
        )
        src_triples = src_triples_data.get(rel) if use_re else None

        src_re_rows = src_re_data.get(rel) if use_re else None
        hyp_re_rows = (
            (hyp_re_data.get(hyp_key) or hyp_re_data.get(rel)) if use_re else None
        )

        if use_kg and hyp_el_rows is None:
            print(f"  [warn] EL (hyp) not found for '{rel}' "
                  f"(tried '{hyp_key}' and '{rel}')")
        if use_re and hyp_triples is None:
            print(f"  [warn] RE triples (hyp) not found for '{rel}' "
                  f"(tried '{hyp_key}' and '{rel}')")
        result = evaluate_pair(
            src_text, hyp_text, bertscore, labse, filename=rel[rel.rindex('/') + 1:], base_path=Path(output_tsv).parent,
            src_el=src_el_data.get(rel) if use_kg else None,
            hyp_el=hyp_el_rows,
            src_triples=src_triples,
            hyp_triples=hyp_triples,
            kg_embed_threshold=kg_embed_threshold,
            src_re=src_re_rows,
            hyp_re=hyp_re_rows,
        )
        result["file"] = rel
        rows.append(result)
        all_results[rel] = result

    # --- Aggregate summary ---
    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 5) if vals else None

    summary = {
        "n_documents":           len(rows),
        "bertscore_model":       bertscore_model,
        "labse_model":           labse_model,
        "mean_doc_bs_f1":        _mean("doc_bs_f1"),
        "mean_doc_bs_precision": _mean("doc_bs_precision"),
        "mean_doc_bs_recall":    _mean("doc_bs_recall"),
        "mean_doc_labse":        _mean("doc_labse"),
        "mean_sent_bs_f1":       _mean("sent_bs_f1_mean"),
        "mean_sent_labse":       _mean("sent_labse_mean"),
        "mean_sent_labse_low_frac": _mean("sent_labse_low_frac"),
        # KGBERTScore
        "mean_kg_entity_f1":        _mean("kg_entity_f1"),
        "mean_kg_entity_precision": _mean("kg_entity_precision"),
        "mean_kg_entity_recall":    _mean("kg_entity_recall"),
        "mean_kg_rel_f1":           _mean("kg_rel_f1"),
        "mean_kg_rel_precision":    _mean("kg_rel_precision"),
        "mean_kg_rel_recall":       _mean("kg_rel_recall"),
        # KGBERTScore — embedding-based entity match
        "kg_embed_threshold":      kg_embed_threshold,
        "mean_kg_embed_f1":        _mean("kg_embed_f1"),
        "mean_kg_embed_precision": _mean("kg_embed_precision"),
        "mean_kg_embed_recall":    _mean("kg_embed_recall"),
        # KGBERTScore — embedding-based relation match
        "mean_kg_rel_embed_f1":        _mean("kg_rel_embed_f1"),
        "mean_kg_rel_embed_precision": _mean("kg_rel_embed_precision"),
        "mean_kg_rel_embed_recall":    _mean("kg_rel_embed_recall"),
    }

    # --- Write TSV ---
    out_tsv = Path(output_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=_FIELDNAMES, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTSV  → {out_tsv}")

    # --- Write JSON (summary + per-document) ---
    out_json = out_tsv.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "documents": all_results}, f, indent=2, ensure_ascii=False)
    print(f"JSON → {out_json}")

    # --- Console summary ---
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    # ------------------------------------------------------------------
    # Default values — edit these when running without CLI arguments
    # ------------------------------------------------------------------
    DEFAULT_INPUT_DIR        = "texts_top_1"
    DEFAULT_OUTPUT_TSV       = "results/eval_top1.tsv"
    DEFAULT_TRANSLATIONS_DIR = "translations/gpt41_top1"   # set to None to use --translate
    DEFAULT_TRANSLATE        = False                         # True → call OpenAI on-the-fly
    DEFAULT_OPENAI_MODEL     = "gpt-4.1"
    DEFAULT_BERTSCORE_MODEL  = "bert-base-multilingual-cased"
    DEFAULT_LABSE_MODEL      = "sentence-transformers/LaBSE"
    DEFAULT_SAVE_TRANSLATIONS = None
    DEFAULT_DEVICE           = None                          # None → auto-detect
    DEFAULT_EL_SRC           = None
    DEFAULT_EL_HYP           = None
    DEFAULT_REL_SRC          = None
    DEFAULT_REL_HYP          = None
    DEFAULT_KG_EMBED_THRESHOLD = 0.85
    DEFAULT_MAX_DOCS         = None                          # None → all documents
    # ------------------------------------------------------------------

    if not sys.argv[1:]:
        # No CLI arguments: run directly with the defaults above
        process(
            input_dir=DEFAULT_INPUT_DIR,
            output_tsv=DEFAULT_OUTPUT_TSV,
            translations_dir=DEFAULT_TRANSLATIONS_DIR,
            translate=DEFAULT_TRANSLATE,
            openai_model=DEFAULT_OPENAI_MODEL,
            bertscore_model=DEFAULT_BERTSCORE_MODEL,
            labse_model=DEFAULT_LABSE_MODEL,
            save_translations=DEFAULT_SAVE_TRANSLATIONS,
            device=DEFAULT_DEVICE,
            el_src=DEFAULT_EL_SRC,
            el_hyp=DEFAULT_EL_HYP,
            rel_src=DEFAULT_REL_SRC,
            rel_hyp=DEFAULT_REL_HYP,
            kg_embed_threshold=DEFAULT_KG_EMBED_THRESHOLD,
            max_docs=DEFAULT_MAX_DOCS,
        )
        return

    # CLI path
    parser = argparse.ArgumentParser(
        description="Evaluate EN→BG translation of discharge summaries"
    )
    parser.add_argument("input_dir",  help="Directory with English .txt source files")
    parser.add_argument("output_tsv", help="Output TSV results file")

    trans_group = parser.add_mutually_exclusive_group(required=True)
    trans_group.add_argument(
        "--translations-dir",
        metavar="DIR",
        help="Directory containing pre-translated Bulgarian .txt files "
             "(mirroring input_dir structure)",
    )
    trans_group.add_argument(
        "--translate",
        action="store_true",
        help="Translate on-the-fly using the OpenAI API (requires OPENAI_API_KEY)",
    )

    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model to use for translation (default: {DEFAULT_OPENAI_MODEL})",
    )
    parser.add_argument(
        "--bertscore-model",
        default=DEFAULT_BERTSCORE_MODEL,
        help=f"HuggingFace model for cross-lingual BERTScore "
             f"(default: {DEFAULT_BERTSCORE_MODEL}; alternative: xlm-roberta-base)",
    )
    parser.add_argument(
        "--labse-model",
        default=DEFAULT_LABSE_MODEL,
        help=f"SentenceTransformer model for LaBSE similarity "
             f"(default: {DEFAULT_LABSE_MODEL})",
    )
    parser.add_argument(
        "--save-translations",
        metavar="DIR",
        default=None,
        help="Save generated Bulgarian translations here (only used with --translate)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. 'cuda' or 'cpu'. Auto-detected when omitted.",
    )

    kg_group = parser.add_argument_group(
        "KGBERTScore",
        "Entity- and relation-level translation consistency via UMLS CUI overlap. "
        "--el-src and --el-hyp are required to enable this metric; "
        "--rel-src and --rel-hyp additionally enable relation-level scoring.",
    )
    kg_group.add_argument(
        "--el-src",
        metavar="TSV",
        default=None,
        help="Entity linking TSV (sapbert_entity_linking.py output) for the "
             "English source texts.",
    )
    kg_group.add_argument(
        "--el-hyp",
        metavar="TSV",
        default=None,
        help="Entity linking TSV for the Bulgarian hypothesis translations.",
    )
    kg_group.add_argument(
        "--rel-src",
        metavar="TSV",
        default=None,
        help="Relation extraction TSV (fhir_relation_extraction.py output) for "
             "the English source texts.",
    )
    kg_group.add_argument(
        "--rel-hyp",
        metavar="TSV",
        default=None,
        help="Relation extraction TSV for the Bulgarian hypothesis translations.",
    )
    kg_group.add_argument(
        "--kg-embed-threshold",
        type=float,
        default=DEFAULT_KG_EMBED_THRESHOLD,
        metavar="FLOAT",
        help=f"BERTScore F1 threshold for the embedding-based entity comparison "
             f"(default: {DEFAULT_KG_EMBED_THRESHOLD}).",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N document pairs (useful for quick testing).",
    )

    args = parser.parse_args()

    process(
        input_dir=args.input_dir,
        output_tsv=args.output_tsv,
        translations_dir=args.translations_dir,
        translate=args.translate,
        openai_model=args.openai_model,
        bertscore_model=args.bertscore_model,
        labse_model=args.labse_model,
        save_translations=args.save_translations,
        device=args.device,
        el_src=args.el_src,
        el_hyp=args.el_hyp,
        rel_src=args.rel_src,
        rel_hyp=args.rel_hyp,
        kg_embed_threshold=args.kg_embed_threshold,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
