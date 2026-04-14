"""
FHIR Relation Extraction using zero-shot NLI (MoritzLaurer/xlm-v-base-mnli-xnli).

For each pair of entities extracted by the GLiNER NER pipeline this script:
  1. Filters candidate pairs by permitted (subject_type, object_type) per FHIR relation.
  2. Builds a language-specific hypothesis string from the entity surface forms.
  3. Runs the hypothesis against the containing sentence context through the NLI model.
  4. Assigns the relation with the highest entailment score above *threshold*,
     or records no row if no relation clears the threshold.

Language is detected automatically per file (langdetect) or forced via --lang.

Usage:
    python fhir_relation_extraction.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --input-dir texts_top_30 \\
        --output-tsv fhir_relations.tsv \\
        [--threshold 0.5] [--lang en|bg|auto] \\
        [--batch-size 16] [--device cpu|cuda] \\
        [--max-distance 500]

Output TSV columns:
    filename  subject_text  subject_label  subject_start  subject_end
    relation  object_text   object_label   object_start   object_end
    score     lang
"""

import os
import re
import csv
import glob
import argparse
import collections
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------------------------- #
# FHIR relation definitions                                                     #
# --------------------------------------------------------------------------- #

RELATIONS: list[str] = [
    "has_finding",
    "has_location",
    "medication_for",
    "has_dosage",
    "has_route",
    "has_frequency",
    "has_duration",
    "treated_with",
    "diagnosed_by",
    "adverse_reaction_to",
    "test_result_of",
]

# For each relation: which entity labels may act as subject / object.
# Labels match those produced by gliner_moe_ner.py.
RELATION_ENTITY_TYPES: dict[str, dict[str, set[str]]] = {
    # A diagnosis / disease presents with a clinical finding.
    "has_finding": {
        "subject": {
            "primary diagnosis", "secondary diagnosis", "comorbidity",
            "differential diagnosis", "disease", "disorder", "injury",
        },
        "object": {
            "symptom", "clinical sign", "vital sign",
            "laboratory result", "imaging finding",
            "body temperature", "blood pressure",
            "heart rate", "oxygen saturation",
        },
    },
    # A clinical entity is located at / involves a body site.
    "has_location": {
        "subject": {
            "disease", "disorder", "injury", "symptom", "clinical sign",
            "imaging finding", "medical procedure", "surgical procedure",
            "diagnostic procedure", "therapeutic procedure",
        },
        "object": {"body site", "anatomical structure"},
    },
    # A medication is prescribed / indicated for a condition.
    "medication_for": {
        "subject": {"medication", "drug name", "drug class"},
        "object": {
            "primary diagnosis", "secondary diagnosis", "comorbidity",
            "disease", "disorder", "medication indication",
        },
    },
    # A medication is administered at a specific dose.
    "has_dosage": {
        "subject": {"medication", "drug name"},
        "object":  {"drug dosage", "drug dose unit"},
    },
    # A medication is administered via a route.
    "has_route": {
        "subject": {"medication", "drug name"},
        "object":  {"route of administration"},
    },
    # A medication is taken at a specified frequency.
    "has_frequency": {
        "subject": {"medication", "drug name"},
        "object":  {"medication frequency"},
    },
    # A medication is taken for a specified duration.
    "has_duration": {
        "subject": {"medication", "drug name"},
        "object":  {"medication duration"},
    },
    # A condition is managed with a therapeutic intervention.
    "treated_with": {
        "subject": {
            "primary diagnosis", "secondary diagnosis", "comorbidity",
            "disease", "disorder", "injury",
        },
        "object": {
            "medication", "drug name",
            "medical procedure", "surgical procedure", "therapeutic procedure",
            "medical device",
        },
    },
    # A condition is confirmed / evaluated by a diagnostic test.
    "diagnosed_by": {
        "subject": {
            "primary diagnosis", "secondary diagnosis",
            "disease", "disorder", "injury",
        },
        "object": {"diagnostic procedure", "laboratory test", "imaging finding"},
    },
    # An adverse event / allergy is a reaction to a specific agent.
    "adverse_reaction_to": {
        "subject": {"adverse drug reaction", "drug allergy", "allergy"},
        "object":  {"drug name", "medication", "drug class"},
    },
    # A laboratory result belongs to a specific test.
    "test_result_of": {
        "subject": {"laboratory result"},
        "object":  {"laboratory test"},
    },
}

# --------------------------------------------------------------------------- #
# Language-specific hypothesis templates                                        #
# --------------------------------------------------------------------------- #
# Placeholders: {subject} = surface form of the subject entity,
#               {object}  = surface form of the object entity.

RELATION_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "has_finding":        "{subject} presents with {object}",
        "has_location":       "{subject} is located in {object}",
        "medication_for":     "{subject} is prescribed for {object}",
        "has_dosage":         "{subject} is administered at a dose of {object}",
        "has_route":          "{subject} is administered via {object}",
        "has_frequency":      "{subject} is taken {object}",
        "has_duration":       "{subject} is taken for {object}",
        "treated_with":       "{subject} is treated with {object}",
        "diagnosed_by":       "{subject} is diagnosed by {object}",
        "adverse_reaction_to":"{subject} is an adverse reaction to {object}",
        "test_result_of":     "{subject} is the result of {object}",
    },
    "bg": {
        "has_finding":        "{subject} проявява {object}",
        "has_location":       "{subject} е локализиран в {object}",
        "medication_for":     "{subject} е предписан за {object}",
        "has_dosage":         "{subject} се прилага в доза {object}",
        "has_route":          "{subject} се прилага чрез {object}",
        "has_frequency":      "{subject} се приема {object}",
        "has_duration":       "{subject} се приема в продължение на {object}",
        "treated_with":       "{subject} се лекува с {object}",
        "diagnosed_by":       "{subject} се диагностицира чрез {object}",
        "adverse_reaction_to":"{subject} е нежелана реакция към {object}",
        "test_result_of":     "{subject} е резултат от {object}",
    },
}

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_NAME = "MoritzLaurer/xlm-v-base-mnli-xnli"
DEFAULT_NER_TSV    = "gliner_moe_entities.tsv"
DEFAULT_INPUT_DIR  = "texts_top_30"
DEFAULT_OUTPUT_TSV = "fhir_relations.tsv"
DEFAULT_THRESHOLD  = 0.5
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_DISTANCE = 500   # characters between entity spans
MAX_CONTEXT_CHARS  = 512     # characters fed to the NLI model as premise

# --------------------------------------------------------------------------- #
# Model loading                                                                 #
# --------------------------------------------------------------------------- #

def load_nli_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> tuple[object, object, int, str]:
    """Return (tokenizer, model, entailment_index) for the given NLI checkpoint."""
    print(f"Loading NLI model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Locate the entailment label index from the model config.
    id2label: dict = model.config.id2label  # {0: "contradiction", 1: "neutral", 2: "entailment"}
    entailment_idx = next(
        (idx for idx, lbl in id2label.items() if "entail" in lbl.lower()),
        2,  # MNLI convention fallback
    )
    print(f"  Device: {device}  |  Entailment class index: {entailment_idx}")
    return tokenizer, model, int(entailment_idx), device


# --------------------------------------------------------------------------- #
# NLI scoring                                                                   #
# --------------------------------------------------------------------------- #

def batch_entailment_scores(
    premises: list[str],
    hypotheses: list[str],
    tokenizer,
    model,
    entailment_idx: int,
    device: str,
    batch_size: int = 16,
) -> list[float]:
    """Return per-pair entailment probabilities (parallel lists)."""
    scores: list[float] = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i : i + batch_size]
        batch_h = hypotheses[i : i + batch_size]
        enc = tokenizer(
            batch_p,
            batch_h,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, entailment_idx].cpu().tolist())
    return scores


# --------------------------------------------------------------------------- #
# NER TSV loading                                                               #
# --------------------------------------------------------------------------- #

def load_ner_tsv(path: str) -> dict[str, list[dict]]:
    """
    Read the TSV produced by gliner_moe_ner.py.
    Returns {relative_path: [entity_dict, ...]}.
    Each entity_dict has keys: entity_label, entity_kind, start_span, end_span, text.
    """
    records: dict = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records[row["filename"]].append({
                "entity_label": row["entity_label"],
                "entity_kind":  row.get("entity_kind", "concept"),
                "start":        int(row["start_span"]),
                "end":          int(row["end_span"]),
                "text":         row["text"],
            })
    return dict(records)


# --------------------------------------------------------------------------- #
# Language detection                                                            #
# --------------------------------------------------------------------------- #

def detect_language(text: str) -> str:
    """Return 'en' or 'bg' (defaults to 'en' on failure)."""
    try:
        from langdetect import detect
        code = detect(text[:2000])   # sample for speed
        return "bg" if code == "bg" else "en"
    except Exception:
        return "en"


# --------------------------------------------------------------------------- #
# Context extraction                                                            #
# --------------------------------------------------------------------------- #

_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+|\n+')


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans for each sentence-like segment."""
    spans: list[tuple[int, int]] = []
    prev = 0
    for m in _SENT_BOUNDARY.finditer(text):
        spans.append((prev, m.end()))
        prev = m.end()
    if prev < len(text):
        spans.append((prev, len(text)))
    return spans


def extract_context(
    text: str,
    start1: int,
    end1: int,
    start2: int,
    end2: int,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Return the smallest contiguous text window that covers both entity spans,
    expanded to sentence boundaries and capped at *max_chars*.
    """
    lo = min(start1, start2)
    hi = max(end1, end2)

    sent_spans = _sentence_spans(text)

    # Find the sentence(s) that contain each entity
    ctx_start, ctx_end = lo, hi
    for s_start, s_end in sent_spans:
        if s_start <= lo < s_end:
            ctx_start = s_start
        if s_start < hi <= s_end:
            ctx_end = s_end

    # Expand symmetrically when the window is still short
    while (ctx_end - ctx_start) < max_chars:
        extended = False
        for s_start, s_end in sent_spans:
            if s_end == ctx_start and (ctx_end - s_start) <= max_chars:
                ctx_start = s_start
                extended = True
            elif s_start == ctx_end and (s_end - ctx_start) <= max_chars:
                ctx_end = s_end
                extended = True
        if not extended:
            break

    return text[ctx_start:ctx_end].strip()


# --------------------------------------------------------------------------- #
# Candidate pair generation                                                     #
# --------------------------------------------------------------------------- #

def get_candidate_pairs(
    entities: list[dict],
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> list[tuple[dict, dict, list[str]]]:
    """
    Return (subject_entity, object_entity, [applicable_relations]) for every
    ordered pair within *max_distance* characters whose types are compatible
    with at least one FHIR relation.

    Both (e_i → e_j) and (e_j → e_i) orderings are tested so directional
    relations are not missed regardless of which entity appears first in text.
    """
    candidates: list[tuple[dict, dict, list[str]]] = []
    n = len(entities)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ei, ej = entities[i], entities[j]

            # Character distance between the two spans
            gap = max(ei["start"], ej["start"]) - min(ei["end"], ej["end"])
            if gap > max_distance:
                continue

            applicable = [
                rel for rel in RELATIONS
                if (
                    ei["entity_label"] in RELATION_ENTITY_TYPES[rel]["subject"]
                    and ej["entity_label"] in RELATION_ENTITY_TYPES[rel]["object"]
                )
            ]
            if applicable:
                candidates.append((ei, ej, applicable))

    return candidates


# --------------------------------------------------------------------------- #
# Per-file processing                                                           #
# --------------------------------------------------------------------------- #

def process_file(
    filename: str,
    entities: list[dict],
    text: str,
    lang: str,
    tokenizer,
    model,
    entailment_idx: int,
    device: str,
    threshold: float,
    batch_size: int,
    max_distance: int,
) -> list[dict]:
    """
    Run relation extraction for one document.
    Returns a list of relation records (dicts) whose entailment score ≥ threshold.
    """
    templates = RELATION_TEMPLATES.get(lang, RELATION_TEMPLATES["en"])
    candidates = get_candidate_pairs(entities, max_distance)
    if not candidates:
        return []

    # Build flat lists of (premise, hypothesis) for batch NLI scoring
    premises:    list[str]  = []
    hypotheses:  list[str]  = []
    meta:        list[dict] = []  # (subject_entity, object_entity, relation)

    for subj_ent, obj_ent, applicable_relations in candidates:
        context = extract_context(
            text,
            subj_ent["start"], subj_ent["end"],
            obj_ent["start"],  obj_ent["end"],
        )
        for rel in applicable_relations:
            hypothesis = templates[rel].format(
                subject=subj_ent["text"],
                object=obj_ent["text"],
            )
            premises.append(context)
            hypotheses.append(hypothesis)
            meta.append({
                "subject": subj_ent,
                "object":  obj_ent,
                "relation": rel,
                "context": context,
            })

    if not premises:
        return []

    scores = batch_entailment_scores(
        premises, hypotheses, tokenizer, model, entailment_idx, device, batch_size
    )

    # For each (subject, object) pair keep only the highest-scoring relation
    # that clears the threshold (one relation per ordered pair).
    pair_best: dict[tuple, tuple[str, float, str]] = {}  # key → (relation, score, context)

    for record, score in zip(meta, scores):
        subj = record["subject"]
        obj  = record["object"]
        pair_key = (subj["start"], subj["end"], obj["start"], obj["end"])
        if score >= threshold:
            if pair_key not in pair_best or score > pair_best[pair_key][1]:
                pair_best[pair_key] = (record["relation"], score, record["context"])

    results: list[dict] = []
    for (ss, se, os_, oe), (rel, score, context) in pair_best.items():
        # Recover the original entity dicts
        subj_ent = next(e for e in entities if e["start"] == ss and e["end"] == se)
        obj_ent  = next(e for e in entities if e["start"] == os_ and e["end"] == oe)
        results.append({
            "filename":      filename,
            "subject_text":  subj_ent["text"],
            "subject_label": subj_ent["entity_label"],
            "subject_start": subj_ent["start"],
            "subject_end":   subj_ent["end"],
            "relation":      rel,
            "object_text":   obj_ent["text"],
            "object_label":  obj_ent["entity_label"],
            "object_start":  obj_ent["start"],
            "object_end":    obj_ent["end"],
            "score":         round(score, 4),
            "lang":          lang,
        })

    return results


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def run(
    ner_tsv: str,
    input_dir: str,
    output_tsv: str,
    threshold: float,
    lang_override: Optional[str],
    batch_size: int,
    device: Optional[str],
    max_distance: int,
    model_name: str,
) -> None:
    tokenizer, model, entailment_idx, resolved_device = load_nli_model(model_name, device)

    print(f"Loading NER annotations from: {ner_tsv}")
    ner_data = load_ner_tsv(ner_tsv)
    print(f"  {len(ner_data)} document(s) with entity annotations.")

    fieldnames = [
        "filename",
        "subject_text", "subject_label", "subject_start", "subject_end",
        "relation",
        "object_text",  "object_label",  "object_start",  "object_end",
        "score", "lang",
    ]

    total_relations = 0
    with open(output_tsv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for rel_path, entities in sorted(ner_data.items()):
            txt_path = os.path.join(input_dir, rel_path)
            if not os.path.isfile(txt_path):
                print(f"  [WARN] Source text not found: {txt_path} — skipping.")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            if lang_override and lang_override != "auto":
                lang = lang_override
            else:
                lang = detect_language(text)

            print(f"  {rel_path}  ({len(entities)} entities, lang={lang})")

            relations = process_file(
                filename=rel_path,
                entities=entities,
                text=text,
                lang=lang,
                tokenizer=tokenizer,
                model=model,
                entailment_idx=entailment_idx,
                device=resolved_device,
                threshold=threshold,
                batch_size=batch_size,
                max_distance=max_distance,
            )
            for row in relations:
                writer.writerow(row)
            total_relations += len(relations)

    print(f"\nDone. {total_relations} relation(s) written to: {output_tsv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract FHIR relations between NER entities using "
            "MoritzLaurer/xlm-v-base-mnli-xnli zero-shot NLI."
        )
    )
    parser.add_argument(
        "--ner-tsv",
        default=DEFAULT_NER_TSV,
        help=f"NER TSV from gliner_moe_ner.py. Default: {DEFAULT_NER_TSV}",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with original .txt files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-tsv",
        default=DEFAULT_OUTPUT_TSV,
        help=f"Output TSV path. Default: {DEFAULT_OUTPUT_TSV}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum entailment score to accept a relation (0–1). Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "en", "bg"],
        help="Force document language (auto = detect per file). Default: auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"NLI inference batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (cpu / cuda / cuda:N). Default: auto-detect.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        help=(
            f"Maximum character gap between two entity spans for them to be "
            f"considered a candidate pair. Default: {DEFAULT_MAX_DISTANCE}"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace NLI model name or local path. Default: {DEFAULT_MODEL_NAME}",
    )
    args = parser.parse_args()

    run(
        ner_tsv=args.ner_tsv,
        input_dir=args.input_dir,
        output_tsv=args.output_tsv,
        threshold=args.threshold,
        lang_override=args.lang,
        batch_size=args.batch_size,
        device=args.device,
        max_distance=args.max_distance,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
