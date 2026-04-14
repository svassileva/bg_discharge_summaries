"""
Named Entity Recognition on discharge summary txt files using GLiNER-MoE-MultiLingual.
Extracts FHIR-aligned entities (clinical, diagnosis, medication modules) and writes
results to a TSV file with columns: filename, entity_label, start_span, end_span, text.

Install dependencies:
    git clone https://github.com/mayank-rakesh-mck/GLiNER.git
    cd GLiNER && pip install -r requirements.txt

Download model files from HuggingFace and point --model-dir to the local folder:
    huggingface-cli download Mayank6255/GLiNER-MoE-MultiLingual --local-dir ./gliner_moe_model
"""

import os
import glob
import csv
import argparse
import json

import re

import torch
from GLiNER.gliner import GLiNERConfig, GLiNER

# Local directory containing gliner_config.json and pytorch_model.bin
# downloaded from Mayank6255/GLiNER-MoE-MultiLingual
DEFAULT_MODEL_DIR = "./gliner_moe_model"

# Characters per chunk fed to the model.
# GLiNER prepends entity-type tokens to every input, so we leave headroom
# well below the 512-token limit (≈ 4 chars/token → 2 048; minus ~512 chars
# of headroom for entity-type prefix → 1 500 chars of body text per chunk).
MAX_CHUNK_CHARS = 1500

# ---------------------------------------------------------------------------
# FHIR-aligned entity labels for a discharge summary
# Covers Clinical, Diagnosis and Medication modules
# ---------------------------------------------------------------------------
# Labels are split into per-module batches of ~10-12 to stay within GLiNER's
# effective token budget (entity type tokens are prepended to the input).
FHIR_LABEL_BATCHES: list[list[str]] = [
    # --- Clinical: observations & vitals ---
    [
        "symptom",
        "clinical sign",
        "vital sign",
        "body temperature",
        "blood pressure",
        "heart rate",
        "oxygen saturation",
        "laboratory test",
        "laboratory result",
        "imaging finding",
    ],
    # --- Clinical: procedures, allergies & anatomy ---
    [
        "medical procedure",
        "surgical procedure",
        "diagnostic procedure",
        "therapeutic procedure",
        "allergy",
        "drug allergy",
        "adverse drug reaction",
        "body site",
        "anatomical structure",
        "medical device",
        "family history",
    ],
    # --- Diagnosis module ---
    [
        "primary diagnosis",
        "secondary diagnosis",
        "comorbidity",
        "differential diagnosis",
        "disease",
        "disorder",
        "injury",
    ],
    # --- Medication module ---
    [
        "medication",
        "drug name",
        "drug class",
        "drug dosage",
        "drug dose unit",
        "route of administration",
        "medication frequency",
        "medication duration",
        "medication indication",
    ],
    # --- Encounter / administrative ---
    [
        "patient age",
        "patient gender",
        "admission date",
        "discharge date",
        "length of hospital stay",
        "attending physician",
        "medical specialty",
        "hospital department",
    ],
]

# ---------------------------------------------------------------------------
# Entity kind classification
# "concept"  — can be linked to a knowledge base (SNOMED, ICD, RxNorm, …)
# "literal"  — a specific measurement, value, or identifier that cannot be
#              meaningfully mapped to a KB entry
# ---------------------------------------------------------------------------
ENTITY_KIND: dict[str, str] = {
    # observations – concepts
    "symptom": "concept",
    "clinical sign": "concept",
    "vital sign": "concept",
    "laboratory test": "concept",
    "imaging finding": "concept",
    # observations – literals (actual measured values)
    "body temperature": "literal",
    "blood pressure": "literal",
    "heart rate": "literal",
    "oxygen saturation": "literal",
    "laboratory result": "literal",
    # procedures & anatomy – concepts
    "medical procedure": "concept",
    "surgical procedure": "concept",
    "diagnostic procedure": "concept",
    "therapeutic procedure": "concept",
    "allergy": "concept",
    "drug allergy": "concept",
    "adverse drug reaction": "concept",
    "body site": "concept",
    "anatomical structure": "concept",
    "medical device": "concept",
    "family history": "concept",
    # diagnosis – concepts
    "primary diagnosis": "concept",
    "secondary diagnosis": "concept",
    "comorbidity": "concept",
    "differential diagnosis": "concept",
    "disease": "concept",
    "disorder": "concept",
    "injury": "concept",
    # medication – concepts
    "medication": "concept",
    "drug name": "concept",
    "drug class": "concept",
    "route of administration": "concept",
    "medication indication": "concept",
    # medication – literals (specific values)
    "drug dosage": "literal",
    "drug dose unit": "literal",
    "medication frequency": "literal",
    "medication duration": "literal",
    # encounter / administrative – literals
    "patient age": "literal",
    "patient gender": "literal",
    "admission date": "literal",
    "discharge date": "literal",
    "length of hospital stay": "literal",
    "attending physician": "literal",
    # encounter / administrative – concepts
    "medical specialty": "concept",
    "hospital department": "concept",
}


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[int, str]]:
    """Split *text* at sentence boundaries into chunks of at most *max_chars*.

    Returns a list of ``(start_offset, chunk_text)`` pairs where
    ``start_offset`` is the character index of the chunk in the original text.
    Long sentences that already exceed *max_chars* are kept as single chunks
    rather than cut mid-word, which would corrupt entity span offsets.
    """
    # Match the whitespace that follows sentence-final punctuation, or one or
    # more consecutive newlines.  The split points are *after* these separators
    # so each sentence span includes its own trailing whitespace/newline.
    _boundary = re.compile(r'(?<=[.!?])[ \t]+|\n+')

    # Build contiguous sentence spans (start, end) in the original text.
    spans: list[tuple[int, int]] = []
    prev = 0
    for m in _boundary.finditer(text):
        spans.append((prev, m.end()))
        prev = m.end()
    if prev < len(text):
        spans.append((prev, len(text)))

    if not spans:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    chunk_start, chunk_end = spans[0][0], spans[0][0]

    for span_start, span_end in spans:
        span_len = span_end - span_start
        if chunk_end > chunk_start and (chunk_end - chunk_start) + span_len > max_chars:
            chunks.append((chunk_start, text[chunk_start:chunk_end]))
            chunk_start = span_start
        chunk_end = span_end

    if chunk_start < len(text):
        chunks.append((chunk_start, text[chunk_start:]))

    return chunks


def load_model(model_dir: str = DEFAULT_MODEL_DIR) -> GLiNER:
    config_path = os.path.join(model_dir, "gliner_config.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    print(f"Loading model config from: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    model_config = GLiNERConfig(**config)
    model = GLiNER(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading weights from: {weights_path} (device: {device})")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    return model


def extract_entities(model: GLiNER, text: str, label_batches: list[list[str]], threshold: float = 0.5) -> list[dict]:
    """Split *text* into sentence-aligned chunks, run one forward pass per
    (chunk, label-batch) pair, rebase spans to original text offsets, and
    deduplicate by (start, end, label)."""
    seen: set[tuple] = set()
    results: list[dict] = []
    for chunk_offset, chunk_text in _chunk_text(text):
        for batch in label_batches:
            for entity in model.predict_entities(chunk_text, batch, threshold=threshold):
                start = entity["start"] + chunk_offset
                end = entity["end"] + chunk_offset
                key = (start, end, entity["label"])
                if key not in seen:
                    seen.add(key)
                    results.append({**entity, "start": start, "end": end})
    return results


def process_directory(
    input_dir: str,
    output_tsv: str,
    threshold: float = 0.5,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> None:
    model = load_model(model_dir)

    txt_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
    if not txt_files:
        print(f"No .txt files found in: {input_dir}")
        return

    print(f"Found {len(txt_files)} file(s). Writing results to: {output_tsv}")

    with open(output_tsv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["filename", "entity_label", "entity_kind", "start_span", "end_span", "text"])

        for txt_file in txt_files:
            rel_path = os.path.relpath(txt_file, input_dir)
            print(f"  Processing: {rel_path}")

            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

            entities = extract_entities(model, text, FHIR_LABEL_BATCHES, threshold)

            for entity in entities:
                kind = ENTITY_KIND.get(entity["label"], "concept")
                writer.writerow([
                    rel_path,
                    entity["label"],
                    kind,
                    entity["start"],
                    entity["end"],
                    entity["text"],
                ])

    print(f"Done. {len(txt_files)} file(s) processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract FHIR-aligned named entities from discharge summary txt files using GLiNER-MoE-MultiLingual."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="texts_top_30",
        help="Directory containing .txt files (searched recursively). Default: texts_top_30",
    )
    parser.add_argument(
        "output_tsv",
        nargs="?",
        default="gliner_moe_entities.tsv",
        help="Output TSV file path. Default: gliner_moe_entities.tsv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Entity confidence threshold (0–1). Default: 0.5",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Local directory with gliner_config.json and pytorch_model.bin. Default: {DEFAULT_MODEL_DIR}",
    )
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_tsv=args.output_tsv,
        threshold=args.threshold,
        model_dir=args.model_dir,
    )
