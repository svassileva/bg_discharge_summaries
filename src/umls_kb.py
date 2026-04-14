"""
Build per-entity-type UMLS knowledge-base files for entity linking.

Reads MRCONSO.RRF (and optionally MRSTY.RRF from the same directory),
extracts all English synonyms, and writes one TSV per NER entity type
(concept-kind labels only, as defined in gliner_moe_ner.py).

Output format  (matches the --candidates-tsv format expected by
sapbert_entity_linking.py):

    CUI<TAB>synonym_string

Each (CUI, synonym) pair is one row; the file is sorted and deduplicated.

Usage
-----
    python umls_kb.py --mrconso /path/to/UMLS/MRCONSO.RRF [options]

    # Explicit MRSTY path (default: same folder as MRCONSO.RRF)
    python umls_kb.py --mrconso /data/umls/MRCONSO.RRF \\
                      --mrsty   /data/umls/MRSTY.RRF   \\
                      --output-dir ./umls_kb

Arguments
---------
--mrconso   Path to MRCONSO.RRF  (required)
--mrsty     Path to MRSTY.RRF    (default: <mrconso_dir>/MRSTY.RRF)
--output-dir Directory for output TSV files (default: ./umls_kb)
--suppress  Comma-separated SUPPRESS values to exclude  (default: O,E)
            'O' = obsolete, 'E' = editor-suppressed

MRCONSO.RRF column layout (pipe-separated)
------------------------------------------
0:CUI 1:LAT 2:TS 3:LUI 4:STT 5:SUI 6:ISPREF 7:AUI 8:SAUI 9:SCUI
10:SDUI 11:SAB 12:TTY 13:CODE 14:STR 15:SRL 16:SUPPRESS 17:CVF

MRSTY.RRF column layout (pipe-separated)
-----------------------------------------
0:CUI 1:TUI 2:STN 3:STY 4:ATUI 5:CVF
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# NER entity-type  →  set of UMLS semantic type identifiers (TUIs)
# Only "concept" kind labels from gliner_moe_ner.py::ENTITY_KIND are included.
# Reference: https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html
# ---------------------------------------------------------------------------

# Semantic-type groups (reused for several closely related labels)
_TUIS_FINDING       = {"T033", "T034", "T184"}       # Finding / Sign or Symptom / Lab Result
_TUIS_PROCEDURE     = {"T058", "T059", "T060", "T061"}  # Health Care Activity / Lab / Dx / Tx procedure
_TUIS_DISEASE       = {"T020", "T037", "T047", "T048", "T191"}  # Abnormality / Injury / Disease / BehavDisorder / Neoplasm
_TUIS_DRUG          = {"T109", "T121", "T195"}       # Organic Chem / Pharmacologic Substance / Antibiotic
_TUIS_ANATOMY       = {"T017", "T022", "T023", "T024", "T025", "T029", "T030"}  # Anatomical structures

# ---------------------------------------------------------------------------
# Source-vocabulary filter: only retain concepts that appear in at least one
# of these ontologies.  SAB codes from MRCONSO.RRF column 11.
# ---------------------------------------------------------------------------
ALLOWED_SABS: frozenset[str] = frozenset({
    # SNOMED CT
    "SNOMEDCT_US", "SNOMEDCT",
    # ICD-10 (international, US Clinical Modification, Procedure Coding System, Australian Modification)
    "ICD10", "ICD10CM", "ICD10PCS", "ICD10AM",
    # ATC – Anatomical Therapeutic Chemical Classification
    "ATC",
    # LOINC
    "LNC",
})

LABEL_SEMTYPES: dict[str, set[str]] = {
    # --- findings / symptoms ---
    "symptom":               {"T033", "T184"},
    "clinical sign":         _TUIS_FINDING,
    "vital sign":            {"T033", "T034", "T184"},
    "laboratory test":       {"T059", "T060"},
    "imaging finding":       {"T033", "T034"},
    # --- procedures ---
    "medical procedure":     _TUIS_PROCEDURE,
    "surgical procedure":    {"T058", "T061"},
    "diagnostic procedure":  {"T058", "T059", "T060"},
    "therapeutic procedure": {"T058", "T061"},
    # --- allergy / adverse drug reaction ---
    "allergy":               {"T033", "T184", "T047"},
    "drug allergy":          {"T033", "T184"},
    "adverse drug reaction": {"T033", "T037"},
    # --- anatomy ---
    "body site":             {"T017", "T023", "T029", "T030"},
    "anatomical structure":  _TUIS_ANATOMY,
    # --- medical device ---
    "medical device":        {"T074", "T075", "T122"},
    # --- family history (recorded as a finding) ---
    "family history":        {"T033"},
    # --- diagnoses / diseases / disorders ---
    "primary diagnosis":     _TUIS_DISEASE,
    "secondary diagnosis":   _TUIS_DISEASE,
    "comorbidity":           {"T020", "T047", "T048"},
    "differential diagnosis":_TUIS_DISEASE,
    "disease":               {"T020", "T047", "T048", "T191"},
    "disorder":              {"T020", "T047", "T048", "T191"},
    "injury":                {"T037"},
    # --- medications / drugs ---
    "medication":            _TUIS_DRUG,
    "drug name":             _TUIS_DRUG,
    "drug class":            _TUIS_DRUG,
    "route of administration": {"T058", "T061", "T169"},  # T169 Functional Concept
    "medication indication": {"T020", "T047", "T048"},
    # --- administrative ---
    "medical specialty":     {"T091"},   # Biomedical Occupation or Discipline
    "hospital department":   {"T093"},   # Health Care Related Organization
}


def _label_to_filename(label: str) -> str:
    """Convert an entity label to a safe filename stem, e.g. 'drug name' → 'drug_name'."""
    return label.strip().replace(" ", "_").replace("/", "_")


def load_cui_tuis(mrsty_path: str) -> dict[str, set[str]]:
    """Return {CUI: {TUI, ...}} from MRSTY.RRF."""
    print(f"[1/3] Reading semantic types from: {mrsty_path}", flush=True)
    cui_tuis: dict[str, set[str]] = defaultdict(set)
    with open(mrsty_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("|")
            if len(cols) < 2:
                continue
            cui, tui = cols[0], cols[1]
            cui_tuis[cui].add(tui)
    print(f"    Loaded TUI sets for {len(cui_tuis):,} CUIs.", flush=True)
    return dict(cui_tuis)


def build_label_cui_sets(
    cui_tuis: dict[str, set[str]],
    label_semtypes: dict[str, set[str]],
) -> dict[str, set[str]]:
    """
    For each NER label, compute the set of CUIs whose TUI set intersects the
    label's required semantic types.

    Returns {label: {CUI, ...}}.
    """
    print("[2/3] Matching CUIs to entity labels ...", flush=True)

    # Pre-compute per-label results in one pass over cui_tuis
    label_cuis: dict[str, set[str]] = {label: set() for label in label_semtypes}

    for cui, tuis in cui_tuis.items():
        for label, required_tuis in label_semtypes.items():
            if tuis & required_tuis:  # non-empty intersection
                label_cuis[label].add(cui)

    for label, cuis in label_cuis.items():
        print(f"    {label:<30s}: {len(cuis):>8,} CUIs", flush=True)

    return label_cuis


def write_kb_files(
    mrconso_path: str,
    label_cuis: dict[str, set[str]],
    output_dir: str,
    suppress_values: set[str],
) -> None:
    """
    Stream MRCONSO.RRF once, collect English synonyms for each label's CUI set,
    then write sorted, deduplicated TSV files.

    Output: <output_dir>/kb_<label>.tsv with columns CUI<TAB>STR.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Invert: CUI → list of labels that want this CUI
    cui_to_labels: dict[str, list[str]] = defaultdict(list)
    for label, cuis in label_cuis.items():
        for cui in cuis:
            cui_to_labels[cui].append(label)

    # Accumulate rows per label: label → set of (CUI, STR) to deduplicate
    label_rows: dict[str, set[tuple[str, str]]] = {label: set() for label in label_cuis}

    # Track which relevant CUIs appear in at least one allowed source ontology
    cui_in_allowed_sab: set[str] = set()

    print(f"[3/3] Streaming MRCONSO.RRF: {mrconso_path}", flush=True)
    print(f"    Allowed SABs: {sorted(ALLOWED_SABS)}", flush=True)

    n_english = 0
    n_written = 0
    with open(mrconso_path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                continue
            cols = raw_line.split("|")
            if len(cols) < 17:
                continue

            cui = cols[0]
            lat = cols[1]          # language
            sab = cols[11]         # source vocabulary
            suppress = cols[16]    # SUPPRESS flag
            string = cols[14]      # STR

            # Track concepts that belong to at least one allowed ontology
            if sab in ALLOWED_SABS and cui in cui_to_labels:
                cui_in_allowed_sab.add(cui)

            if lat != "ENG":
                continue
            n_english += 1
            if suppress in suppress_values:
                continue
            if not string:
                continue

            if cui in cui_to_labels:
                for label in cui_to_labels[cui]:
                    label_rows[label].add((cui, string))
                    n_written += 1

    print(f"    English rows seen: {n_english:,}", flush=True)
    print(f"    (CUI, synonym) pairs collected before SAB filter: {n_written:,}", flush=True)
    print(f"    CUIs matched to allowed ontologies: {len(cui_in_allowed_sab):,}", flush=True)

    # Filter: keep only concepts present in at least one allowed ontology
    label_rows = {
        label: {(cui, s) for cui, s in rows if cui in cui_in_allowed_sab}
        for label, rows in label_rows.items()
    }
    n_filtered = sum(len(rows) for rows in label_rows.values())
    print(f"    (CUI, synonym) pairs after SAB filter: {n_filtered:,}", flush=True)

    # Write output files
    all_rows: set[tuple[str, str]] = set()
    for label, rows in sorted(label_rows.items()):
        filename = f"kb_{_label_to_filename(label)}.tsv"
        out_path = os.path.join(output_dir, filename)
        sorted_rows = sorted(rows)  # sort by CUI then STR
        with open(out_path, "w", encoding="utf-8", newline="") as out_fh:
            writer = csv.writer(out_fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerows(sorted_rows)
        print(f"    Wrote {len(sorted_rows):>8,} rows → {out_path}", flush=True)
        all_rows.update(rows)

    # Write merged file covering all entity types (deduplicated)
    all_out_path = os.path.join(output_dir, "kb_all.tsv")
    sorted_all = sorted(all_rows)
    with open(all_out_path, "w", encoding="utf-8", newline="") as out_fh:
        writer = csv.writer(out_fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(sorted_all)
    print(f"    Wrote {len(sorted_all):>8,} rows → {all_out_path}  [merged all types]", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-entity-type UMLS KB files from MRCONSO.RRF."
    )
    parser.add_argument(
        "--mrconso",
        required=True,
        metavar="PATH",
        help="Path to MRCONSO.RRF (required).",
    )
    parser.add_argument(
        "--mrsty",
        default=None,
        metavar="PATH",
        help="Path to MRSTY.RRF. Defaults to MRSTY.RRF in the same directory as --mrconso.",
    )
    parser.add_argument(
        "--output-dir",
        default="./umls_kb",
        metavar="DIR",
        help="Directory for output TSV files (default: ./umls_kb).",
    )
    parser.add_argument(
        "--suppress",
        default="O,E",
        metavar="FLAGS",
        help=(
            "Comma-separated SUPPRESS values to exclude from MRCONSO.RRF "
            "(default: 'O,E' — obsolete and editor-suppressed entries)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    mrconso_path: str = os.path.abspath(args.mrconso)
    if not os.path.isfile(mrconso_path):
        sys.exit(f"ERROR: MRCONSO.RRF not found: {mrconso_path}")

    mrsty_path: str
    if args.mrsty:
        mrsty_path = os.path.abspath(args.mrsty)
    else:
        mrsty_path = os.path.join(os.path.dirname(mrconso_path), "MRSTY.RRF")
    if not os.path.isfile(mrsty_path):
        sys.exit(f"ERROR: MRSTY.RRF not found: {mrsty_path}")

    suppress_values: set[str] = {v.strip() for v in args.suppress.split(",") if v.strip()}
    output_dir: str = os.path.abspath(args.output_dir)

    print(f"MRCONSO : {mrconso_path}")
    print(f"MRSTY   : {mrsty_path}")
    print(f"Output  : {output_dir}")
    print(f"Suppress: {sorted(suppress_values)}")
    print()

    cui_tuis = load_cui_tuis(mrsty_path)
    label_cuis = build_label_cui_sets(cui_tuis, LABEL_SEMTYPES)
    write_kb_files(mrconso_path, label_cuis, output_dir, suppress_values)

    print("\nDone.")


if __name__ == "__main__":
    main()
