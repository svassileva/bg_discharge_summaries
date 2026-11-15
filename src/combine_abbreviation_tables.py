"""
Combines all markdown table txt files in a folder into a single TSV file.
"""

import argparse
import csv
import re
from pathlib import Path

HEADER = ["English Abbreviation", "English Expansion", "Bulgarian Abbreviation", "Bulgarian Expansion"]


def parse_markdown_table(text: str) -> list[list[str]]:
    """Extract data rows from a markdown table, handling header split across lines."""
    rows = []
    # Normalize: join lines that are continuations (no leading |) to the previous line
    lines = text.splitlines()
    joined_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            joined_lines.append(stripped)
        elif joined_lines:
            # continuation of previous line (e.g. "Expansion |")
            joined_lines[-1] = joined_lines[-1].rstrip() + " " + stripped

    for line in joined_lines:
        # Skip separator rows like |---|---|
        if re.match(r"^\|[-| :]+\|$", line):
            continue
        # Split on | and strip whitespace
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last cells caused by leading/trailing |
        cells = [c for c in cells if c != ""]
        if len(cells) != 4:
            continue
        # Skip header row
        if cells[0].lower() in ("english abbreviation",):
            continue
        rows.append(cells)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Combine markdown abbreviation tables into a TSV."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing the source .txt tables.",
    )
    parser.add_argument(
        "--output-tsv", type=Path, required=True,
        help="Path for the combined TSV.",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"Input directory not found: {args.input_dir}")

    files = sorted(args.input_dir.glob("*.txt"))

    if not files:
        raise SystemExit(f"No .txt files found in: {args.input_dir}")

    all_rows = []
    for filepath in files:
        filename = filepath.name
        text = filepath.read_text(encoding="utf-8")
        rows = parse_markdown_table(text)
        for row in rows:
            all_rows.append([filename] + row)

    # Deduplicate
    unique_rows = list(dict.fromkeys(map(tuple, all_rows)))
    duplicates_removed = len(all_rows) - len(unique_rows)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Source File"] + HEADER)
        writer.writerows(unique_rows)

    print(
        f"Written {len(unique_rows)} rows from {len(files)} files to: "
        f"{args.output_tsv} ({duplicates_removed} duplicates removed)"
    )


if __name__ == "__main__":
    main()
