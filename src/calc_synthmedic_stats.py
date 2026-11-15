#!/usr/bin/env python3
import argparse
import collections
import json
import re
from pathlib import Path
from statistics import mean, median


def sent_split(text: str):
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [p.strip() for p in parts if p.strip()]


def word_tokens(text: str):
    return re.findall(r"\b\w+(?:[-']\w+)*\b", text)


def whitespace_tokens(text: str):
    return text.split()


def quantiles(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(vals):
    return {
        'count': len(vals),
        'mean': round(mean(vals), 2) if vals else None,
        'median': round(median(vals), 2) if vals else None,
        'min': min(vals) if vals else None,
        'max': max(vals) if vals else None,
        'std_approx': round((sum((x - mean(vals)) ** 2 for x in vals) / len(vals)) ** 0.5, 2) if vals else None,
        'p25': round(quantiles(vals, 0.25), 2) if vals else None,
        'p75': round(quantiles(vals, 0.75), 2) if vals else None,
    }


def extract_disease(stem: str) -> str:
    """Strip trailing _<digits>[_suffix] to get the disease label."""
    m = re.match(r'^(.+?)_\d+.*$', stem)
    return m.group(1) if m else stem


def iter_texts(input_dir: Path):
    exts = {'.txt', '.text', '.md'}
    for path in sorted(input_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in exts:
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                text = path.read_text(encoding='latin-1')
            yield path, text


def main():
    ap = argparse.ArgumentParser(description='Calculate corpus document-size statistics.')
    ap.add_argument('input_dir', help='Directory containing one text document per file (.txt, .md, .text)')
    ap.add_argument('--outdir', default='output', help='Output directory for results')
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_words: set[str] = set()
    for path, text in iter_texts(input_dir):
        clean = text.strip()
        sentences = sent_split(clean)
        words = word_tokens(clean)
        ws_toks = whitespace_tokens(clean)
        all_words.update(w.lower() for w in words)
        disease = extract_disease(path.stem)
        rows.append({
            'file': str(path.relative_to(input_dir)),
            'disease': disease,
            'characters': len(clean),
            'characters_no_spaces': len(re.sub(r'\s+', '', clean)),
            'lines': len(clean.splitlines()) if clean else 0,
            'sentences': len(sentences),
            'words_regex': len(words),
            'tokens_whitespace': len(ws_toks),
            'avg_words_per_sentence': round(len(words) / len(sentences), 2) if sentences else None,
        })

    if not rows:
        raise SystemExit('No text files found. Put .txt, .text, or .md files in the input directory.')

    import csv
    with open(outdir / 'document_level_stats.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Per-disease sentence counts
    disease_sentence_counts: dict[str, int] = collections.defaultdict(int)
    disease_doc_counts: dict[str, int] = collections.defaultdict(int)
    for r in rows:
        disease_sentence_counts[r['disease']] += r['sentences']
        disease_doc_counts[r['disease']] += 1

    total_sentences = sum(r['sentences'] for r in rows)

    summary = {
        'documents': len(rows),
        'total_sentences': total_sentences,
        'unique_words': len(all_words),
        'sentences_by_disease': {
            disease: {
                'documents': disease_doc_counts[disease],
                'total_sentences': disease_sentence_counts[disease],
            }
            for disease in sorted(disease_sentence_counts)
        },
        'characters': summarize([r['characters'] for r in rows]),
        'characters_no_spaces': summarize([r['characters_no_spaces'] for r in rows]),
        'lines': summarize([r['lines'] for r in rows]),
        'sentences': summarize([r['sentences'] for r in rows]),
        'words_regex': summarize([r['words_regex'] for r in rows]),
        'tokens_whitespace': summarize([r['tokens_whitespace'] for r in rows]),
        'avg_words_per_sentence': summarize([r['avg_words_per_sentence'] for r in rows if r['avg_words_per_sentence'] is not None]),
    }

    with open(outdir / 'corpus_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    md = [
        '# Corpus size statistics',
        '',
        f"Documents: {summary['documents']}",
        f"Total sentences: {summary['total_sentences']}",
        f"Unique words: {summary['unique_words']}",
        '',
        '## Sentences by disease',
        '',
        '| Disease | Documents | Total sentences |',
        '|---|---:|---:|',
    ]
    for disease, counts in summary['sentences_by_disease'].items():
        md.append(f"| {disease} | {counts['documents']} | {counts['total_sentences']} |")

    md.extend([
        '',
        '## Aggregate metrics',
        '',
        '| Metric | Mean | Median | Min | Max | P25 | P75 | Std |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for key in ['characters', 'characters_no_spaces', 'lines', 'sentences', 'words_regex', 'tokens_whitespace', 'avg_words_per_sentence']:
        s = summary[key]
        md.append(f"| {key} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} | {s['p25']} | {s['p75']} | {s['std_approx']} |")

    md.extend([
        '',
        '## Notes',
        '',
        '- `words_regex` counts word-like units using a regex and is often better than plain whitespace splitting.',
        '- `tokens_whitespace` is a simple whitespace token count, useful as a rough approximation.',
        '- Sentence splitting here is rule-based. For clinical text, consider replacing it with spaCy or Stanza if you need higher accuracy.',
        '- If you want model-specific subword token counts, add a Hugging Face tokenizer (for example `bert-base-uncased` or your downstream model tokenizer).',
    ])

    (outdir / 'README_stats.md').write_text('\n'.join(md), encoding='utf-8')


if __name__ == '__main__':
    main()
