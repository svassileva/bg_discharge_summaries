"""
Translate the entity text column of a NER TSV result file.

Reads a TSV produced by gliner_moe_ner.py (or any file with a ``text``
column), translates the ``text`` values from Bulgarian to English, and writes
a new TSV with the original columns plus ``expanded_text`` (optional) and
``translated_text`` appended.

Translation back-ends (mutually exclusive):
  --translate          Local Helsinki-NLP/opus-mt-bg-en MarianMT model.
  --azure-translate    Azure AI Foundry GPT deployment
                       (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY env vars).
  --openai-translate   Any OpenAI-compatible API
                       (--openai-url / --openai-api-key, or env vars).

Usage
-----
    # Local MarianMT
    python translate_entities.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --output-tsv entities_translated.tsv \\
        --translate

    # Azure GPT-4.1
    python translate_entities.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --output-tsv entities_translated.tsv \\
        --azure-translate --azure-model gpt-4.1

    # OpenAI-compatible endpoint
    python translate_entities.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --output-tsv entities_translated.tsv \\
        --openai-translate \\
        --openai-url https://api.openai.com/v1 \\
        --openai-model gpt-4o-mini

    # With abbreviation expansion (applied before translation)
    python translate_entities.py \\
        --ner-tsv gliner_moe_entities.tsv \\
        --output-tsv entities_translated.tsv \\
        --translate --abbrev-dict abbreviations_dictionary.tsv
"""

import argparse
import csv
import json
import os
import re


# ---------------------------------------------------------------------------
# Translation model identifiers
# ---------------------------------------------------------------------------
MARIAN_MODEL = "Helsinki-NLP/opus-mt-bg-en"

_TRANSLATE_PROMPT = (
    "You are a medical translator. "
    "You will receive a JSON object with a key \"entities\" containing a list of Bulgarian medical entity strings. "
    "Translate each string to English. "
    "Return a JSON object with a single key \"translations\" whose value is a list of English translations "
    "in the same order and with the same count as the input list. "
    "Output only valid JSON, without any additional commentary."
)


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

def load_abbrev_dict(path: str) -> dict[str, str]:
    """Load an abbreviation dictionary from a two-column TSV (abbrev<TAB>expansion).

    Lines starting with '#' are treated as comments and ignored.
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
    """Replace abbreviations with their expansions (whole-token, longest-first)."""
    if not abbrev:
        return texts
    sorted_keys = sorted(abbrev, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted_keys) + r")(?!\w)"
    )
    expanded = [pattern.sub(lambda m: abbrev[m.group(0)], t) if t is not None else t for t in texts]
    changed = sum(1 for o, e in zip(texts, expanded) if o != e)
    print(f"Abbreviation expansion: {changed}/{len(texts)} entity strings modified.")
    return expanded


# ---------------------------------------------------------------------------
# Local MarianMT translation
# ---------------------------------------------------------------------------

def _build_marian_translator(device):
    from transformers import MarianMTModel, MarianTokenizer
    print(f"Loading translation model: {MARIAN_MODEL}")
    tok = MarianTokenizer.from_pretrained(MARIAN_MODEL)
    mdl = MarianMTModel.from_pretrained(MARIAN_MODEL).to(device)
    return tok, mdl


def translate_marian(texts: list[str], batch_size: int = 64) -> list[str]:
    """Translate via local Helsinki-NLP/opus-mt-bg-en."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, mdl = _build_marian_translator(device)
    results: list[str] = []
    mdl.eval()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(texts), batch_size), 1):
            batch = texts[start : start + batch_size]
            print(f"  [marian] batch {batch_idx}/{total_batches} ({start + 1}–{min(start + batch_size, len(texts))}/{len(texts)}) …", flush=True)
            try:
                inputs = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(device)
                ids = mdl.generate(**inputs)
                results.extend(tok.batch_decode(ids, skip_special_tokens=True))
            except Exception as exc:
                print(f"  [marian] ERROR on batch {batch_idx}: {exc}")
                results.extend([""] * len(batch))
    return results


# ---------------------------------------------------------------------------
# Azure AI Foundry translation
# ---------------------------------------------------------------------------

def translate_azure(texts: list[str], deployment: str, batch_size: int = 32) -> list[str]:
    """Translate via Azure AI Foundry GPT deployment."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise SystemExit("openai package not installed. Run: pip install openai")

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint:
        raise SystemExit("AZURE_OPENAI_ENDPOINT environment variable is not set.")
    if not api_key:
        raise SystemExit("AZURE_OPENAI_API_KEY environment variable is not set.")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    def _batch(batch: list[str]) -> list[str]:
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": _TRANSLATE_PROMPT},
                    {"role": "user", "content": json.dumps({"entities": batch}, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            print(f"  [azure] API error: {exc}; falling back to per-entity calls.")
            return [_single(t) for t in batch]
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  [azure] JSON parse error: {exc}\n  Raw response: {raw[:200]!r}; falling back to per-entity calls.")
            return [_single(t) for t in batch]
        translations = parsed.get("translations", [])
        if len(translations) != len(batch):
            print(
                f"  [azure] count mismatch (expected {len(batch)}, got {len(translations)}); "
                "falling back to per-entity calls."
            )
            return [_single(t) for t in batch]
        return [str(t) for t in translations]

    def _single(text: str) -> str:
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Translate the following Bulgarian medical entity to English. Output only the English translation."},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [azure] API error for entity {text!r}: {exc}")
            return ""

    results: list[str] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(texts), batch_size), 1):
        print(f"  [azure] batch {batch_idx}/{total_batches} ({start + 1}–{min(start + batch_size, len(texts))}/{len(texts)}) …", flush=True)
        results.extend(_batch(texts[start : start + batch_size]))
    return results


# ---------------------------------------------------------------------------
# OpenAI-compatible translation
# ---------------------------------------------------------------------------

def translate_openai(
    texts: list[str],
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int = 32,
) -> list[str]:
    """Translate via any OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed. Run: pip install openai")

    client = OpenAI(base_url=base_url, api_key=api_key)

    def _batch(batch: list[str]) -> list[str]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TRANSLATE_PROMPT},
                    {"role": "user", "content": json.dumps({"entities": batch}, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            print(f"  [openai] API error: {exc}; falling back to per-entity calls.")
            return [_single(t) for t in batch]
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  [openai] JSON parse error: {exc}\n  Raw response: {raw[:200]!r}; falling back to per-entity calls.")
            return [_single(t) for t in batch]
        translations = parsed.get("translations", [])
        if len(translations) != len(batch):
            print(
                f"  [openai] count mismatch (expected {len(batch)}, got {len(translations)}); "
                "falling back to per-entity calls."
            )
            return [_single(t) for t in batch]
        return [str(t) for t in translations]

    def _single(text: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Translate the following Bulgarian medical entity to English. Output only the English translation."},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [openai] API error for entity {text!r}: {exc}")
            return ""

    results: list[str] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(texts), batch_size), 1):
        print(f"  [openai] batch {batch_idx}/{total_batches} ({start + 1}–{min(start + batch_size, len(texts))}/{len(texts)}) …", flush=True)
        results.extend(_batch(texts[start : start + batch_size]))
    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_tsv(path: str) -> tuple[list[str], list[dict]]:
    """Return (fieldnames, rows) from a TSV file."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_tsv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output written to: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(
    ner_tsv: str,
    output_tsv: str,
    text_column: str = "text",
    translate: bool = False,
    azure_translate: bool = False,
    azure_model: str = "gpt-4.1",
    openai_translate: bool = False,
    openai_model: str = "gpt-4o-mini",
    openai_url: str | None = None,
    openai_api_key: str | None = None,
    batch_size: int = 64,
    abbrev_dict_path: str | None = None,
) -> None:
    fieldnames, rows = read_tsv(ner_tsv)
    if not rows:
        print("No rows found in input TSV. Exiting.")
        return
    print(f"Loaded {len(rows)} rows from: {ner_tsv}")

    if text_column not in fieldnames:
        raise SystemExit(
            f"Column '{text_column}' not found in TSV. Available columns: {fieldnames}"
        )

    texts = [r.get(text_column, "") for r in rows]

    # Abbreviation expansion
    abbrev_dict: dict[str, str] = {}
    if abbrev_dict_path:
        abbrev_dict = load_abbrev_dict(abbrev_dict_path)
        expanded = expand_abbreviations(texts, abbrev_dict)
        if "expanded_text" not in fieldnames:
            fieldnames = fieldnames + ["expanded_text"]
        for row, exp in zip(rows, expanded):
            row["expanded_text"] = exp
        query_texts = expanded
    else:
        query_texts = texts

    # Translation
    if "translated_text" not in fieldnames:
        fieldnames = fieldnames + ["translated_text"]

    if openai_translate:
        resolved_url = openai_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_url:
            raise SystemExit("--openai-url (or OPENAI_BASE_URL env var) is required with --openai-translate.")
        if not resolved_key:
            raise SystemExit("--openai-api-key (or OPENAI_API_KEY env var) is required with --openai-translate.")
        print(f"Translating {len(query_texts)} strings via OpenAI-compatible API ('{openai_model}', {resolved_url}) …")
        translated = translate_openai(
            query_texts,
            model=openai_model,
            base_url=resolved_url,
            api_key=resolved_key,
            batch_size=batch_size,
        )
    elif azure_translate:
        print(f"Translating {len(query_texts)} strings via Azure ('{azure_model}') …")
        translated = translate_azure(query_texts, deployment=azure_model, batch_size=batch_size)
    elif translate:
        print(f"Translating {len(query_texts)} strings via local MarianMT …")
        translated = translate_marian(query_texts, batch_size=batch_size)
    else:
        raise SystemExit(
            "No translation back-end selected. "
            "Pass --translate, --azure-translate, or --openai-translate."
        )

    for row, tr in zip(rows, translated):
        row["translated_text"] = tr

    write_tsv(output_tsv, fieldnames, rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a translated_text column to a NER TSV result file."
    )
    parser.add_argument(
        "--ner-tsv",
        required=True,
        help="Input TSV file (e.g. produced by gliner_moe_ner.py).",
    )
    parser.add_argument(
        "--output-tsv",
        required=True,
        help="Output TSV path (original columns + translated_text).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the column containing entity text to translate. Default: text",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Translation batch size. Default: 64",
    )
    parser.add_argument(
        "--abbrev-dict",
        default=None,
        metavar="TSV",
        help=(
            "Path to a TSV with abbreviation expansions "
            "(abbreviation<TAB>expansion, one per line, '#' lines ignored). "
            "Applied before translation; adds an expanded_text column."
        ),
    )

    translate_group = parser.add_mutually_exclusive_group(required=True)
    translate_group.add_argument(
        "--translate",
        action="store_true",
        help="Translate using the local Helsinki-NLP/opus-mt-bg-en MarianMT model.",
    )
    translate_group.add_argument(
        "--azure-translate",
        action="store_true",
        help=(
            "Translate using an Azure AI Foundry GPT deployment "
            "(AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY env vars required)."
        ),
    )
    translate_group.add_argument(
        "--openai-translate",
        action="store_true",
        help=(
            "Translate using any OpenAI-compatible API "
            "(--openai-url and --openai-api-key, or OPENAI_BASE_URL / OPENAI_API_KEY env vars)."
        ),
    )

    parser.add_argument(
        "--azure-model",
        default="gpt-4.1",
        metavar="DEPLOYMENT",
        help="Azure AI Foundry deployment name used with --azure-translate. Default: gpt-4.1",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        metavar="MODEL",
        help="Model name used with --openai-translate. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--openai-url",
        default=None,
        metavar="URL",
        help="Base URL for the OpenAI-compatible API (or OPENAI_BASE_URL env var).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        metavar="KEY",
        help="API key for the OpenAI-compatible endpoint (or OPENAI_API_KEY env var).",
    )

    args = parser.parse_args()

    process(
        ner_tsv=args.ner_tsv,
        output_tsv=args.output_tsv,
        text_column=args.text_column,
        translate=args.translate,
        azure_translate=args.azure_translate,
        azure_model=args.azure_model,
        openai_translate=args.openai_translate,
        openai_model=args.openai_model,
        openai_url=args.openai_url,
        openai_api_key=args.openai_api_key,
        batch_size=args.batch_size,
        abbrev_dict_path=args.abbrev_dict,
    )


if __name__ == "__main__":
    main()
