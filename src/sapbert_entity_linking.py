"""
Entity Linking to UMLS using cross-lingual SapBERT (XLM-R based).

Pipeline
--------
1. Load NER entities from a TSV produced by gliner_moe_ner.py
   (columns: filename, entity_label, start_span, end_span, text).
2. Optionally translate entity text BG→EN via Helsinki-NLP/opus-mt-bg-en.
3. Encode the (possibly translated) entity strings with
   cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR.
4. Build (or load cached) UMLS concept embeddings over candidate names.
5. Retrieve top-k UMLS concepts per entity using cosine similarity
   (faiss if available, numpy fallback).
6. Write results to a TSV.

UMLS candidates file
--------------------
Provide a two-column, tab-separated file (no header):
    CUI<TAB>preferred_name

Can be generated from MRCONSO.RRF (requires a UMLS licence):
    # English names only
    awk -F'|' '$2=="ENG"{print $1"\t"$15}' MRCONSO.RRF | sort -u > umls_eng.tsv
    # All languages (for cross-lingual linking)
    awk -F'|' '{print $1"\t"$15}' MRCONSO.RRF | sort -u > umls_all.tsv

Per-type UMLS subsets (optional, reduces search space)
------------------------------------------------------
Use --type-candidates to map entity labels to dedicated subset TSVs:

    --type-candidates "disease=umls_diseases.tsv,drug=umls_drugs.tsv"

For each entity label listed, the corresponding TSV is searched instead of
(or in addition to) the global --candidates-tsv.  Entity types not in the
mapping fall back to the global --candidates-tsv.

Suggested UMLS semantic-type filters per entity label
(apply to MRCONSO.RRF via MRSTY.RRF before running):

    disease / disorder  → T020,T047,T048,T191  (Acquired Abnormality,
                          Disease or Syndrome, Mental/Behavioral Dysfunction,
                          Neoplastic Process)
    finding / symptom   → T033,T034,T184
    drug / medication   → T109,T121,T195  (Organic Chemical,
                          Pharmacologic Substance, Antibiotic)
    gene / protein      → T028,T116,T123,T126
    anatomy             → T017,T023,T029,T030
    procedure           → T058,T059,T060,T061

Install
-------
    pip install transformers torch sentencepiece sacremoses
    pip install faiss-cpu   # optional but recommended for large concept sets

Usage
-----
    # Without translation (entity text may be in any language SapBERT supports)
    python sapbert_entity_linking.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --candidates-tsv umls_eng.tsv \\
        --output-tsv el_results.tsv

    # With per-type subsets and BG→EN translation
    python sapbert_entity_linking.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --candidates-tsv umls_eng.tsv \\
        --type-candidates "disease=umls_diseases.tsv,drug=umls_drugs.tsv" \\
        --output-tsv el_results_translated.tsv \\
        --translate
"""

import argparse
import csv
import os
import pickle
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer

# ---------------------------------------------------------------------------
# Default model identifiers
# ---------------------------------------------------------------------------
SAPBERT_MODEL = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-bg-en"

# Pre-built --type-candidates string covering all umls_kb/ subsets.
# Each label matches the entity_label values produced by gliner_moe_ner.py.
TYPE_CANDIDATES_DEFAULT = (
    "adverse drug reaction=umls_kb/kb_adverse_drug_reaction.tsv,"
    "allergy=umls_kb/kb_allergy.tsv,"
    "anatomical structure=umls_kb/kb_anatomical_structure.tsv,"
    "body site=umls_kb/kb_body_site.tsv,"
    "clinical sign=umls_kb/kb_clinical_sign.tsv,"
    "comorbidity=umls_kb/kb_comorbidity.tsv,"
    "diagnostic procedure=umls_kb/kb_diagnostic_procedure.tsv,"
    "differential diagnosis=umls_kb/kb_differential_diagnosis.tsv,"
    "disease=umls_kb/kb_disease.tsv,"
    "disorder=umls_kb/kb_disorder.tsv,"
    "drug allergy=umls_kb/kb_drug_allergy.tsv,"
    "drug class=umls_kb/kb_drug_class.tsv,"
    "drug name=umls_kb/kb_drug_name.tsv,"
    "family history=umls_kb/kb_family_history.tsv,"
    "hospital department=umls_kb/kb_hospital_department.tsv,"
    "imaging finding=umls_kb/kb_imaging_finding.tsv,"
    "injury=umls_kb/kb_injury.tsv,"
    "laboratory test=umls_kb/kb_laboratory_test.tsv,"
    "medical device=umls_kb/kb_medical_device.tsv,"
    "medical procedure=umls_kb/kb_medical_procedure.tsv,"
    "medical specialty=umls_kb/kb_medical_specialty.tsv,"
    "medication=umls_kb/kb_medication.tsv,"
    "medication indication=umls_kb/kb_medication_indication.tsv,"
    "primary diagnosis=umls_kb/kb_primary_diagnosis.tsv,"
    "route of administration=umls_kb/kb_route_of_administration.tsv,"
    "secondary diagnosis=umls_kb/kb_secondary_diagnosis.tsv,"
    "surgical procedure=umls_kb/kb_surgical_procedure.tsv,"
    "symptom=umls_kb/kb_symptom.tsv,"
    "therapeutic procedure=umls_kb/kb_therapeutic_procedure.tsv,"
    "vital sign=umls_kb/kb_vital_sign.tsv"
)

# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

def load_abbrev_dict(path: str) -> dict[str, str]:
    """Load an abbreviation dictionary from a two-column TSV (abbrev<TAB>expansion).

    Lines starting with '#' are treated as comments and ignored.
    Abbreviations are stored as-is (case-sensitive) to preserve medical
    capitalisation conventions (e.g. "АХ" vs "ах").
    """
    abbrev: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0]:
                abbrev[parts[0].strip()] = parts[1].strip()
    print(f"Loaded {len(abbrev)} abbreviation entries from: {path}")
    return abbrev


def expand_abbreviations(texts: list[str], abbrev: dict[str, str]) -> list[str]:
    """Replace abbreviations with their expansions in every text string.

    Replacement is whole-token: an abbreviation only matches when it is
    surrounded by word boundaries (Unicode-aware).  Longer abbreviations
    are tried first so that "АГ" does not shadow "АГК" when both are
    present in the dictionary.
    """
    if not abbrev:
        return texts

    # Build a single regex that tries longer keys first.
    sorted_keys = sorted(abbrev, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted_keys) + r")(?!\w)"
    )

    expanded = [pattern.sub(lambda m: abbrev[m.group(0)], t) for t in texts]
    changed = sum(1 for o, e in zip(texts, expanded) if o != e)
    print(f"Abbreviation expansion: {changed}/{len(texts)} entity strings modified.")
    return expanded


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over non-padding positions."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 128,
    max_length: int = 25,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Encode a list of strings into L2-normalised float32 embeddings."""
    all_embs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            emb = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Translation helpers (BG → EN)
# ---------------------------------------------------------------------------

def build_translator(device: torch.device) -> tuple[MarianTokenizer, MarianMTModel]:
    print(f"Loading translation model: {TRANSLATION_MODEL}")
    tok = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    mdl = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(device)
    return tok, mdl


def translate_bg_to_en(
    texts: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Translate a list of Bulgarian strings to English."""
    results: list[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            translated_ids = model.generate(**inputs)
            results.extend(tokenizer.batch_decode(translated_ids, skip_special_tokens=True))
    return results


# ---------------------------------------------------------------------------
# UMLS concept index
# ---------------------------------------------------------------------------

def _load_candidates(candidates_tsv: str) -> tuple[list[str], list[str]]:
    """Read CUI-name pairs from a two-column TSV."""
    cuis: list[str] = []
    names: list[str] = []
    with open(candidates_tsv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            cuis.append(row[0].strip())
            names.append(row[1].strip())
    return cuis, names


def build_or_load_index(
    candidates_tsv: str,
    cache_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    device: torch.device,
    label: str = "global",
) -> tuple[list[str], list[str], np.ndarray]:
    """Return (cuis, names, L2-normalised embeddings), loading from cache if available."""
    if os.path.isfile(cache_path):
        print(f"[{label}] Loading UMLS index from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cuis, names, embs = pickle.load(f)
        print(f"  {len(names):,} concept embeddings loaded.")
        return cuis, names, embs

    print(f"[{label}] Building UMLS embedding index from: {candidates_tsv}")
    cuis, names = _load_candidates(candidates_tsv)
    print(f"  {len(names):,} concept names loaded.")

    embs = encode_texts(names, tokenizer, model, batch_size=batch_size, device=device)

    print(f"  Saving index cache to: {cache_path}")
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((cuis, names, embs), f)

    return cuis, names, embs


def _cache_path_for_type(base_cache_path: str, label: str) -> str:
    """Derive a per-entity-type cache path from the base cache path."""
    root, ext = os.path.splitext(base_cache_path)
    safe_label = label.replace(" ", "_").replace("/", "-")
    return f"{root}__{safe_label}{ext}"


def build_type_indices(
    type_candidates: dict[str, str],
    base_cache_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    device: torch.device,
) -> dict[str, tuple[list[str], list[str], np.ndarray]]:
    """
    Build or load a UMLS index for each entity type in *type_candidates*.

    Returns a dict mapping entity_label → (cuis, names, embeddings).
    """
    indices: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    for label, tsv_path in type_candidates.items():
        cache = _cache_path_for_type(base_cache_path, label)
        indices[label] = build_or_load_index(
            tsv_path, cache, tokenizer, model, batch_size, device, label=label
        )
    return indices


# ---------------------------------------------------------------------------
# Cosine similarity retrieval
# ---------------------------------------------------------------------------

def _retrieve_faiss(
    query_embs: np.ndarray,
    index_embs: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve via faiss flat IP index (vectors are already L2-normalised)."""
    import faiss  # type: ignore

    dim = index_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(index_embs)
    scores, indices = index.search(query_embs, top_k)
    return scores, indices


def _retrieve_numpy(
    query_embs: np.ndarray,
    index_embs: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve via numpy dot-product (cosine sim since vectors are L2-normalised)."""
    scores = query_embs @ index_embs.T  # (Q, C)
    k = min(top_k, scores.shape[1])
    # argpartition gives unsorted top-k; argsort within those k
    part_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, part_idx, axis=1)
    sort_order = np.argsort(-top_scores, axis=1)
    sorted_idx = np.take_along_axis(part_idx, sort_order, axis=1)
    sorted_scores = np.take_along_axis(top_scores, sort_order, axis=1)
    return sorted_scores, sorted_idx


def retrieve_top_k(
    query_embs: np.ndarray,
    index_embs: np.ndarray,
    cuis: list[str],
    names: list[str],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[list[dict]]:
    """
    For each query embedding return a ranked list of up to top_k dicts with
    keys: cui, name, score.  Candidates with cosine similarity below
    *threshold* are omitted.
    """
    try:
        scores, indices = _retrieve_faiss(query_embs, index_embs, top_k)
        print("  Using faiss for similarity search.")
    except ImportError:
        print("  faiss not found; falling back to numpy similarity search.")
        scores, indices = _retrieve_numpy(query_embs, index_embs, top_k)

    results: list[list[dict]] = []
    for q in range(len(query_embs)):
        row = [
            {"cui": cuis[idx], "name": names[idx], "score": float(scores[q, k])}
            for k, idx in enumerate(indices[q])
            if float(scores[q, k]) >= threshold
        ]
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_ner_tsv(ner_tsv: str) -> list[dict]:
    with open(ner_tsv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_results(
    entities: list[dict],
    linked: list[list[dict]],
    expanded: list[str] | None,
    translated: list[str] | None,
    output_tsv: str,
    top_k: int,
) -> None:
    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        header = ["filename", "entity_label", "start_span", "end_span", "text"]
        if expanded is not None:
            header.append("expanded_text")
        if translated is not None:
            header.append("translated_text")
        for k in range(1, top_k + 1):
            header += [f"umls_cui_{k}", f"umls_name_{k}", f"score_{k}"]
        writer.writerow(header)

        for i, (ent, candidates) in enumerate(zip(entities, linked)):
            row = [
                ent.get("filename", ""),
                ent.get("entity_label", ""),
                ent.get("start_span", ""),
                ent.get("end_span", ""),
                ent.get("text", ""),
            ]
            if expanded is not None:
                row.append(expanded[i])
            if translated is not None:
                row.append(translated[i])
            for k_idx in range(top_k):
                if k_idx < len(candidates):
                    c = candidates[k_idx]
                    row += [c["cui"], c["name"], f"{c['score']:.4f}"]
                else:
                    row += ["", "", ""]
            writer.writerow(row)

    print(f"Results written to: {output_tsv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link NER entities to UMLS concepts using cross-lingual SapBERT."
    )
    parser.add_argument(
        "--ner-tsv",
        required=True,
        help="NER output TSV produced by gliner_moe_ner.py.",
    )
    parser.add_argument(
        "--candidates-tsv",
        required=True,
        help="UMLS candidates TSV: CUI<TAB>concept_name (no header). Used as fallback for entity types not covered by --type-candidates.",
    )
    parser.add_argument(
        "--type-candidates",
        default=TYPE_CANDIDATES_DEFAULT,
        help=(
            "Comma-separated per-entity-type UMLS subset TSVs. "
            "Format: label1=path1.tsv,label2=path2.tsv  "
            "(labels are matched case-insensitively against entity_label column). "
            "Entity types not listed fall back to --candidates-tsv."
        ),
    )
    parser.add_argument(
        "--output-tsv",
        default="el_results.tsv",
        help="Output TSV path. Default: el_results.tsv",
    )
    parser.add_argument(
        "--cache-path",
        default="output/umls_emb_cache.pkl",
        help="Pickle cache for UMLS embeddings (rebuilt automatically if missing). Default: output/umls_emb_cache.pkl",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top UMLS candidates returned per entity. Default: 5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tokenisation/encoding batch size. Default: 128",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=25,
        help="Max token length for entity/concept name encoding. Default: 25",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate entity text from Bulgarian to English before linking.",
    )
    parser.add_argument(
        "--abbrev-dict",
        default=None,
        help=(
            "Path to a TSV file with abbreviation expansions "
            "(abbreviation<TAB>expansion, one per line, '#' lines ignored). "
            "Expansions are applied to entity text in its source language "
            "before translation (if --translate) and before SapBERT encoding."
        ),
    )
    parser.add_argument(
        "--sapbert-model",
        default=SAPBERT_MODEL,
        help=f"HuggingFace SapBERT model name or local path. Default: {SAPBERT_MODEL}",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Minimum cosine similarity score for a UMLS candidate to be kept. Default: 0.85",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load entities from NER output
    entities = read_ner_tsv(args.ner_tsv)
    if not entities:
        print("No entities found in NER TSV. Exiting.")
        return
    raw_texts = [e.get("text", "") for e in entities]
    print(f"Loaded {len(entities)} entities from: {args.ner_tsv}")

    # Abbreviation expansion (source-language, before translation/encoding)
    abbrev_dict: dict[str, str] = {}
    if args.abbrev_dict:
        abbrev_dict = load_abbrev_dict(args.abbrev_dict)
        raw_texts = expand_abbreviations(raw_texts, abbrev_dict)

    # Optional BG→EN translation
    translated_texts: list[str] | None = None
    if args.translate:
        tr_tok, tr_mdl = build_translator(device)
        print(f"Translating {len(raw_texts)} entity string(s) …")
        translated_texts = translate_bg_to_en(
            raw_texts, tr_tok, tr_mdl, batch_size=64, device=device
        )
        query_texts = translated_texts
    else:
        query_texts = raw_texts

    # Load SapBERT
    print(f"Loading SapBERT: {args.sapbert_model}")
    sap_tok = AutoTokenizer.from_pretrained(args.sapbert_model)
    sap_mdl = AutoModel.from_pretrained(args.sapbert_model).to(device)

    # Encode entity queries
    print(f"Encoding {len(query_texts)} entity queries …")
    query_embs = encode_texts(
        query_texts, sap_tok, sap_mdl,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # Parse per-type candidates mapping
    type_candidates: dict[str, str] = {}
    if args.type_candidates:
        for item in args.type_candidates.split(","):
            item = item.strip()
            if "=" not in item:
                raise ValueError(
                    f"--type-candidates entry must be label=path, got: {item!r}"
                )
            lbl, path = item.split("=", 1)
            type_candidates[lbl.strip().lower()] = path.strip()
        print(
            f"Per-type UMLS subsets defined for: {', '.join(sorted(type_candidates))}"
        )

    # Build or load global UMLS concept index (used as fallback)
    global_cuis, global_names, global_index_embs = build_or_load_index(
        args.candidates_tsv,
        args.cache_path,
        sap_tok,
        sap_mdl,
        args.batch_size,
        device,
        label="global",
    )

    # Build or load per-type indices (only the types actually present in data)
    present_types = {e.get("entity_label", "").lower() for e in entities}
    active_type_candidates = {
        lbl: path
        for lbl, path in type_candidates.items()
        if lbl in present_types
    }
    type_indices = build_type_indices(
        active_type_candidates,
        args.cache_path,
        sap_tok,
        sap_mdl,
        args.batch_size,
        device,
    )

    # Retrieve top-k UMLS concepts, routing each entity to its type-specific index
    print(f"Retrieving top-{args.top_k} UMLS concepts per entity …")
    linked: list[list[dict]] = [[] for _ in entities]

    # Group entity indices by their index key (type label or "__global__")
    groups: dict[str, list[int]] = {}
    for i, ent in enumerate(entities):
        key = ent.get("entity_label", "").lower()
        if key not in type_indices:
            key = "__global__"
        groups.setdefault(key, []).append(i)

    for key, idxs in groups.items():
        if key == "__global__":
            cuis, names, index_embs = global_cuis, global_names, global_index_embs
            label_str = "global fallback"
        else:
            cuis, names, index_embs = type_indices[key]
            label_str = key
        group_query_embs = query_embs[np.array(idxs)]
        print(
            f"  [{label_str}] {len(idxs)} entities × {len(names):,} candidates"
        )
        group_results = retrieve_top_k(
            group_query_embs, index_embs, cuis, names,
            top_k=args.top_k, threshold=args.similarity_threshold,
        )
        for list_pos, orig_idx in enumerate(idxs):
            linked[orig_idx] = group_results[list_pos]

    # Write output
    write_results(
        entities, linked,
        expanded=raw_texts if abbrev_dict else None,
        translated=translated_texts,
        output_tsv=args.output_tsv,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
