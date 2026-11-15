"""
Parse LLM NER output files produced in the format:
*   **Concept N:** <name>
    *   **Type:** <type>
    *   **FHIR Mapping:** <fhir>
    *   **ICD-10 CM Code:** <icd10>
    *   **ATC Code:** <atc>
    *   **LOINC Code:** <loinc>
    *   **SNOMED CT Code:** <snomed>

Outputs a TSV with one row per entity.
"""

import re
import csv
import sys
from pathlib import Path

import tqdm


FIELD_PATTERNS = {
    "type":         re.compile(r"\*\*Type:\*\*\s*(.+)"),
    "fhir_mapping": re.compile(r"\*\*FHIR Mapping:\*\*\s*(.+)"),
    "icd10_code":   re.compile(r"\*\*ICD-10 CM Code:\*\*\s*(.+)"),
    "atc_code":     re.compile(r"\*\*ATC Code:\*\*\s*(.+)"),
    "loinc_code":   re.compile(r"\*\*LOINC Code:\*\*\s*(.+)"),
    "snomed_code":  re.compile(r"\*\*SNOMED CT Code:\*\*\s*(.+)"),
}

CONCEPT_PATTERN = re.compile(r"\*\*Concept \d+:\*\*\s*(.+)")
FILENAME_PATTERN = re.compile(r"^(.+?)_(\d+)_(.+?)_NER\.txt$")


def parse_file(filepath: Path) -> list[dict]:
    """Parse a single NER output file and return a list of entity dicts."""
    text = filepath.read_text(encoding="utf-8")

    # Extract metadata from filename
    fname = filepath.name
    meta = {"source_file": fname, "specialty": "", "condition": "", "note_index": "", "model": ""}
    m = FILENAME_PATTERN.match(fname)
    if m:
        specialty_condition, index, model = m.group(1), m.group(2), m.group(3)
        # specialty is the first underscore-separated token; condition is the rest
        parts = specialty_condition.split("_", 1)
        meta["specialty"] = parts[0]
        meta["condition"] = parts[1] if len(parts) > 1 else ""
        meta["note_index"] = index
        meta["model"] = model

    entities = []
    current = None

    for line in text.splitlines():
        line = line.strip()

        concept_match = CONCEPT_PATTERN.search(line)
        if concept_match:
            if current is not None:
                entities.append(current)
            current = {**meta, "concept": concept_match.group(1).strip(),
                       "type": "", "fhir_mapping": "",
                       "icd10_code": "", "atc_code": "", "loinc_code": "", "snomed_code": ""}
            continue

        if current is not None:
            for field, pattern in FIELD_PATTERNS.items():
                fm = pattern.search(line)
                if fm:
                    value = fm.group(1).strip()
                    current[field] = "" if value == "N/A" else value
                    break

    if current is not None:
        entities.append(current)

    return entities


def locate_source_doc(texts_root: Path, specialty: str, condition: str, note_index: str, suffix: str | None = None) -> Path | None:
    """Return the path to the source document for an NER entry, or None if not found."""
    specialty_condition = f"{specialty}_{condition}"
    if suffix is not None:
        filename = f"{specialty_condition}_{note_index}_{suffix}.txt"
    else:
        filename = f"{specialty_condition}_{note_index}.txt"
    path = texts_root / specialty_condition / filename
    return path if path.exists() else None


def find_mentions(concept: str, doc_text: str) -> list[tuple[int, int, str]]:
    """Find all case-insensitive whole-word occurrences of concept in doc_text.
    Returns a list of (start, end, matched_text) tuples."""
    pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
    return [(m.start(), m.end(), m.group()) for m in pattern.finditer(doc_text)]


def expand_to_spans(entities: list[dict], texts_root: Path, suffix: str | None = None) -> list[dict]:
    """For each entity, find all mentions in its source document.
    Returns a flat list of span-level rows with filename, entity_label, entity_kind,
    start_span, end_span, text, and the LLM-provided metadata columns."""
    rows = []
    _doc_cache: dict[Path, str] = {}
    for entity in entities:
        doc_path = locate_source_doc(
            texts_root, entity["specialty"], entity["condition"], entity["note_index"], suffix
        )
        if doc_path is None:
            continue
        if doc_path not in _doc_cache:
            _doc_cache[doc_path] = doc_path.read_text(encoding="utf-8")
        doc_text = _doc_cache[doc_path]
        mentions = find_mentions(entity["concept"], doc_text)
        for start, end, matched_text in mentions:
            rows.append({
                "filename": doc_path.name,
                "entity_label": entity["type"].lower(),
                "entity_kind": "concept",
                "start_span": start,
                "end_span": end,
                "text": matched_text,
                "fhir_mapping": entity["fhir_mapping"],
                "icd10_code": entity["icd10_code"],
                "atc_code": entity["atc_code"],
                "loinc_code": entity["loinc_code"],
                "snomed_code": entity["snomed_code"],
            })
    return rows


def parse_folder(folder: str | Path, output_tsv: str | Path | None = None) -> list[dict]:
    """Parse all *_NER.txt files in folder and optionally write a TSV."""
    folder = Path(folder)
    all_entities = []

    for txt_file in tqdm.tqdm(sorted(folder.glob("*.txt"))):
        all_entities.extend(parse_file(txt_file))

    if output_tsv and all_entities:
        output_tsv = Path(output_tsv)
        fieldnames = ["source_file", "specialty", "condition", "note_index", "model",
                      "concept", "type", "fhir_mapping",
                      "icd10_code", "atc_code", "loinc_code", "snomed_code"]
        with open(output_tsv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(all_entities)
        print(f"Wrote {len(all_entities)} entities to {output_tsv}")

    return all_entities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse LLM NER output files.")
    parser.add_argument("folder", help="Folder containing *_NER.txt files")
    parser.add_argument("--output", "-o", default=None, help="Output TSV path")
    parser.add_argument(
        "--texts", "-t", default=None,
        help="Root folder of source documents; enables span-level output with "
             "filename, entity_label, entity_kind, start_span, end_span, text columns"
    )
    parser.add_argument(
        "--suffix", "-s", default=None,
        help="Optional suffix to append to source document filenames when locating spans"
    )
    args = parser.parse_args()

    entities = parse_folder(args.folder)

    # --- span-level output (when --texts is provided) ---
    if args.texts:
        spans = expand_to_spans(entities, Path(args.texts), suffix=args.suffix)
        span_fieldnames = [
            "filename", "entity_label", "entity_kind", "start_span", "end_span", "text",
            "fhir_mapping", "icd10_code", "atc_code", "loinc_code", "snomed_code",
        ]
        if args.output:
            out_path = Path(args.output)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=span_fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(spans)
            print(f"Wrote {len(spans)} spans to {out_path}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=span_fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(spans)

    # --- entity-level output (original behaviour, no --texts) ---
    else:
        entity_fieldnames = [
            "source_file", "specialty", "condition", "note_index", "model",
            "concept", "type", "fhir_mapping",
            "icd10_code", "atc_code", "loinc_code", "snomed_code",
        ]
        if args.output:
            out_path = Path(args.output)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=entity_fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(entities)
            print(f"Wrote {len(entities)} entities to {out_path}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=entity_fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(entities)

    # --- type counts (always written when --output is provided) ---
    if args.output:
        type_counts: dict[str, int] = {}
        for e in entities:
            t = e["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        type_counts_file = Path(args.output).with_suffix(".types.tsv")
        with open(type_counts_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["type", "count"])
            for t, count in sorted(type_counts.items()):
                writer.writerow([t, count])
        print(f"Wrote type counts to {type_counts_file}")
