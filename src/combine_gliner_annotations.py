"""
Combine automatic NER annotations from all gliner_* model folders into a single TSV.
Output columns: document_id, language, model, entity_type, text, start_index, end_index, umls_cui
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def normalize_doc_id(filename: str) -> str:
    """Strip .txt extension from filename."""
    return filename[:-4] if filename.endswith(".txt") else filename


def detect_language_from_path(filepath: str) -> str:
    """Detect language (en/bg) from file path."""
    name = os.path.basename(filepath).lower()
    if "_bg_" in name or "_bg." in name:
        return "bg"
    if "_en_" in name or "_en." in name:
        return "en"
    return "unknown"


def is_umls_file(filepath: str) -> bool:
    name = os.path.basename(filepath).lower()
    return "umls" in name


def _extract_model(df: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    if "model" not in df.columns:
        df["model"] = os.path.basename(folder_path)
    return df


def _normalise_spans(df: pd.DataFrame) -> pd.DataFrame:
    df["start_index"] = df["start_span"].astype(float).astype(int)
    df["end_index"] = df["end_span"].astype(float).astype(int)
    return df


def load_ner_file(filepath: str, folder_path: str) -> pd.DataFrame:
    """Load a NER-only TSV (no CUI columns)."""
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["filename", "start_span", "end_span"])
    df["document_id"] = df["filename"].apply(normalize_doc_id)
    df = _normalise_spans(df)
    df["entity_type"] = df["entity_label"].str.upper()
    df = _extract_model(df, folder_path)
    df["umls_cui"] = pd.NA
    keep = ["document_id", "entity_type", "start_index", "end_index", "text", "model", "umls_cui"]
    return df[[c for c in keep if c in df.columns]]


def load_umls_file_as_entities(filepath: str, folder_path: str) -> pd.DataFrame:
    """Load a UMLS TSV directly as the entity source (includes CUI when matched)."""
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["filename", "start_span", "end_span"])
    df["document_id"] = df["filename"].apply(normalize_doc_id)
    df = _normalise_spans(df)
    df["entity_type"] = df["entity_label"].str.upper()
    df = _extract_model(df, folder_path)

    # Normalise CUI column across file variants
    if "UMLS cui" in df.columns:
        df["umls_cui"] = df["UMLS cui"]
    elif "umls_cui_1" in df.columns:
        df["umls_cui"] = df["umls_cui_1"]
    else:
        df["umls_cui"] = pd.NA

    df["umls_cui"] = df["umls_cui"].where(
        df["umls_cui"].notna() & (df["umls_cui"].str.strip() != ""), other=pd.NA
    )

    keep = ["document_id", "entity_type", "start_index", "end_index", "text", "model", "umls_cui"]
    return df[[c for c in keep if c in df.columns]]


def process_gliner_folder(folder_path: str) -> pd.DataFrame:
    files = os.listdir(folder_path)
    tsv_files = [f for f in files if f.endswith(".tsv")]

    ner_files = [f for f in tsv_files if not is_umls_file(f)]
    umls_files = [f for f in tsv_files if is_umls_file(f)]

    # Group by language
    umls_langs = {detect_language_from_path(f) for f in umls_files}

    parts = []

    # For each UMLS file, use it directly as the entity source
    for uf in umls_files:
        lang = detect_language_from_path(uf)
        df = load_umls_file_as_entities(os.path.join(folder_path, uf), folder_path)
        df["language"] = lang
        parts.append(df)

    # For NER files whose language has no UMLS counterpart, load the NER file
    for nf in ner_files:
        lang = detect_language_from_path(nf)
        if lang in umls_langs:
            continue  # UMLS file already used for this language
        df = load_ner_file(os.path.join(folder_path, nf), folder_path)
        df["language"] = lang
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Combine GLiNER annotation TSVs into a single TSV."
    )
    parser.add_argument(
        "--annotations-dir", type=Path, required=True,
        help="Root directory containing gliner_* annotation folders.",
    )
    parser.add_argument(
        "--output-tsv", type=Path, required=True,
        help="Path for the combined TSV.",
    )
    args = parser.parse_args(argv)

    if not args.annotations_dir.is_dir():
        parser.error(f"Annotations directory not found: {args.annotations_dir}")

    gliner_folders = [
        path for path in args.annotations_dir.iterdir()
        if path.is_dir() and path.name.startswith("gliner")
    ]
    if not gliner_folders:
        raise SystemExit(f"No gliner_* folders found in: {args.annotations_dir}")

    all_parts = []
    for folder_path in sorted(gliner_folders):
        print(f"Processing {folder_path.name} ...")
        df = process_gliner_folder(str(folder_path))
        if not df.empty:
            all_parts.append(df)

    if not all_parts:
        raise SystemExit(f"No annotation TSV data found in: {args.annotations_dir}")

    combined = pd.concat(all_parts, ignore_index=True)

    # Final column order
    out_cols = ["document_id", "language", "model", "entity_type", "text", "start_index", "end_index", "umls_cui"]
    for col in out_cols:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[out_cols]

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"\nWrote {len(combined):,} rows to {args.output_tsv}")
    print(combined["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
