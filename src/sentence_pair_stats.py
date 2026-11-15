#!/usr/bin/env python3
"""
Compute corpus-level sentence and word statistics from aligned sentence-pair TSV files.

Each TSV has two columns: 'source' (English) and 'hypothesis' (Bulgarian).
One row = one aligned sentence pair.

Outputs:
  - Total sentences per language
  - Average words per sentence per language
  - Average sentences per document
  - Average words per document per language
"""

import argparse
import csv
from pathlib import Path


def word_count(text: str) -> int:
    return len(text.split())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize aligned sentence-pair TSV files."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing sentence-pair TSV files.",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"Input directory not found: {args.input_dir}")

    tsv_files = sorted(args.input_dir.glob("*.tsv"))
    if not tsv_files:
        raise SystemExit(f"No TSV files found in: {args.input_dir}")

    total_sentences = 0
    total_en_words = 0
    total_bg_words = 0

    # Per-document counts for computing averages
    doc_sentence_counts: list[int] = []
    doc_en_word_counts: list[int] = []
    doc_bg_word_counts: list[int] = []

    for tsv_path in tsv_files:
        doc_sentences = 0
        doc_en_words = 0
        doc_bg_words = 0

        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                en_sent = row.get("source", "").strip()
                bg_sent = row.get("hypothesis", "").strip()
                if not en_sent and not bg_sent:
                    continue
                doc_sentences += 1
                doc_en_words += word_count(en_sent)
                doc_bg_words += word_count(bg_sent)

        if doc_sentences == 0:
            continue

        total_sentences += doc_sentences
        total_en_words += doc_en_words
        total_bg_words += doc_bg_words

        doc_sentence_counts.append(doc_sentences)
        doc_en_word_counts.append(doc_en_words)
        doc_bg_word_counts.append(doc_bg_words)

    num_docs = len(doc_sentence_counts)
    if num_docs == 0:
        raise SystemExit(f"No sentence-pair rows found in: {args.input_dir}")

    avg_words_per_sent_en = total_en_words / total_sentences
    avg_words_per_sent_bg = total_bg_words / total_sentences
    avg_sents_per_doc = total_sentences / num_docs
    avg_en_words_per_doc = total_en_words / num_docs
    avg_bg_words_per_doc = total_bg_words / num_docs

    print(f"Documents processed         : {num_docs}")
    print(f"Total sentence pairs        : {total_sentences}")
    print()
    print(f"{'Metric':<40} {'English':>12} {'Bulgarian':>12}")
    print("-" * 66)
    print(f"{'Total sentences':<40} {total_sentences:>12,} {total_sentences:>12,}")
    print(f"{'Total words':<40} {total_en_words:>12,} {total_bg_words:>12,}")
    print(f"{'Avg words per sentence':<40} {avg_words_per_sent_en:>12.2f} {avg_words_per_sent_bg:>12.2f}")
    print(f"{'Avg sentences per document':<40} {avg_sents_per_doc:>12.2f} {avg_sents_per_doc:>12.2f}")
    print(f"{'Avg words per document':<40} {avg_en_words_per_doc:>12.2f} {avg_bg_words_per_doc:>12.2f}")


if __name__ == "__main__":
    main()
