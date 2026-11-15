"""
Evaluate predicted NER entities (GLiNER) against manually annotated ones.

Annotation format (JSON):
    {
        "classes": [...],
        "annotations": [
            [text, {"entities": [[start, end, label], ...]}],
            ...  # multiple sets if multiple annotators
        ]
    }

Prediction format (TSV):
    filename  entity_label  entity_kind  start_span  end_span  text

Metrics (per entity type and overall):
    - Strict:  exact (start, end, label) match
    - Span:    exact (start, end) match, ignoring label
    - Lenient: same normalised surface text and same label (handles off-by-one offsets)

Usage:
    python eval_ner.py [--annotations-dir DIR] [--predictions-tsv TSV]
                       [--annotator {first,all,union}]
                       [--output-dir DIR]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_label(label: str) -> str:
    return label.strip().upper()


def derive_pred_filename(annotation_path: Path) -> str:
    """Derive prediction TSV filename from annotation JSON filename.

    Examples:
        cardiovascular_arterial hypertension_26_BGgpt_annotations (4).json
            → cardiovascular_arterial hypertension_26.txt
        endocrinology_hypothyroidism_34_shortest_annotations (1).json
            → endocrinology_hypothyroidism_34.txt
        infectious diseases_influenza_37_shortest_annotations.json
            → infectious diseases_influenza_37.txt
    """
    stem = annotation_path.stem  # without .json
    # Remove trailing annotation marker: _annotations ... or _annotations(N)
    stem = re.sub(r'_annotations.*$', '', stem, flags=re.IGNORECASE)
    # Remove known translation/variant suffixes
    for suffix in ['_BGgpt', '_shortest', '_BGGpt']:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem + '.txt'


def load_annotations(annotation_path: Path, annotator: str = 'first'):
    """Return a dict mapping pred_filename → list of (start, end, label) tuples.

    annotator: 'first' | 'union' | 'all'
        first – use only the first annotation set
        union – merge all annotation sets (any annotator marked it)
        all   – return separate sets (list of lists); used for IAA
    """
    with open(annotation_path, encoding='utf-8') as f:
        data = json.load(f)

    annotation_sets = []
    for ann in data['annotations']:
        text = ann[0]
        entities_raw = ann[1]['entities']
        entity_set = set()
        for ent in entities_raw:
            start, end, label = int(ent[0]), int(ent[1]), normalise_label(ent[2])
            span_text = text[start:end]
            entity_set.add((start, end, label, span_text))
        annotation_sets.append(entity_set)

    if annotator == 'first':
        return annotation_sets[:1], data.get('classes', [])
    elif annotator == 'union':
        merged = set()
        for s in annotation_sets:
            merged |= s
        return [merged], data.get('classes', [])
    else:  # 'all'
        return annotation_sets, data.get('classes', [])


def load_predictions(tsv_path: Path) -> dict:
    """Return dict mapping filename → set of (start, end, label, text) tuples."""
    df = pd.read_csv(tsv_path, sep='\t')
    preds = defaultdict(set)
    for _, row in df.iterrows():
        if len(row.keys()) < 5:
            print(f'!! WARNING: Skipping malformed row in predictions TSV: {row}')
            continue
        fname = str(row['filename']).strip()  # ensure .txt extension
        fname = fname if fname.endswith('.txt') else fname + '.txt'
        label = normalise_label(str(row['entity_label']))
        if pd.isna(row['start_span']) or pd.isna(row['end_span']):
            print(f'!! WARNING: Missing start/end span in row: {row}')
            continue
        start = int(row['start_span'])
        end = int(row['end_span'])
        text = str(row['text']).strip()
        preds[fname].add((start, end, label, text))
    return dict(preds)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gold: set, pred: set):
    """Compute P, R, F1 for three matching schemes.

    gold / pred: sets of (start, end, label, text)

    Returns dict with keys: strict, span, lenient
    Each value is a dict: {tp, fp, fn, precision, recall, f1}
    """

    def prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return dict(tp=tp, fp=fp, fn=fn,
                    precision=precision, recall=recall, f1=f1)

    # --- Strict: (start, end, label) must all match ---
    gold_strict = {(s, e, l) for s, e, l, _ in gold}
    pred_strict = {(s, e, l) for s, e, l, _ in pred}
    tp_strict = gold_strict & pred_strict
    strict = prf(len(tp_strict), len(pred_strict - tp_strict),
                 len(gold_strict - tp_strict))

    # --- Span: only (start, end) must match ---
    gold_span = {(s, e) for s, e, l, _ in gold}
    pred_span = {(s, e) for s, e, l, _ in pred}
    tp_span = gold_span & pred_span
    span = prf(len(tp_span), len(pred_span - tp_span),
               len(gold_span - tp_span))

    # --- Lenient: same normalised surface text + same label ---
    gold_lenient = {(t.lower(), l) for _, _, l, t in gold}
    pred_lenient = {(t.lower(), l) for _, _, l, t in pred}
    tp_lenient = gold_lenient & pred_lenient
    lenient = prf(len(tp_lenient), len(pred_lenient - tp_lenient),
                  len(gold_lenient - tp_lenient))

    return dict(strict=strict, span=span, lenient=lenient)


def compute_per_type_metrics(gold: set, pred: set, labels: list):
    """Return per-label metrics for strict matching."""
    results = {}
    for label in labels:
        g = {(s, e) for s, e, l, _ in gold if l == label}
        p = {(s, e) for s, e, l, _ in pred if l == label}
        tp = g & p
        precision = len(tp) / len(p) if p else 0.0
        recall = len(tp) / len(g) if g else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results[label] = dict(
            tp=len(tp), fp=len(p - tp), fn=len(g - tp),
            precision=precision, recall=recall, f1=f1,
            n_gold=len(g), n_pred=len(p)
        )
    return results


def print_metrics(metrics: dict, title: str = ''):
    if title:
        print(f'\n{"=" * 60}')
        print(f'  {title}')
        print(f'{"=" * 60}')
    for scheme, m in metrics.items():
        print(f'  [{scheme.upper()}]  '
              f'P={m["precision"]:.3f}  R={m["recall"]:.3f}  F1={m["f1"]:.3f}  '
              f'(TP={m["tp"]}, FP={m["fp"]}, FN={m["fn"]})')


def print_per_type(per_type: dict, title: str = ''):
    if title:
        print(f'\n  -- {title} --')
    header = f'  {"Label":<15} {"Gold":>6} {"Pred":>6} {"TP":>5} {"FP":>5} {"FN":>5} {"P":>7} {"R":>7} {"F1":>7}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for label, m in sorted(per_type.items()):
        print(f'  {label:<15} {m["n_gold"]:>6} {m["n_pred"]:>6} '
              f'{m["tp"]:>5} {m["fp"]:>5} {m["fn"]:>5} '
              f'{m["precision"]:>7.3f} {m["recall"]:>7.3f} {m["f1"]:>7.3f}')


def iaa_f1(set_a: set, set_b: set) -> float:
    """Inter-annotator agreement: treat set_a as gold, set_b as pred (strict)."""
    gold = {(s, e, l) for s, e, l, _ in set_a}
    pred = {(s, e, l) for s, e, l, _ in set_b}
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predicted NER vs manual annotations.')
    parser.add_argument(
        '--annotations-dir',
        default='results/Manual Annotations/Manual Annotations',
        help='Directory containing annotation JSON files.')
    parser.add_argument(
        '--predictions-tsv',
        default='outputs/gliner_moe_entities_en_corpus_full.tsv',
        help='TSV file with predicted entities.')
    parser.add_argument(
        '--annotator', choices=['first', 'union', 'all'], default='first',
        help='Which annotator set(s) to use when a file has multiple annotations.')
    parser.add_argument(
        '--output-dir', default=None,
        help='Optional directory to save per-document result TSVs.')
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    predictions_tsv = Path(args.predictions_tsv)

    if not annotations_dir.exists():
        sys.exit(f'Annotations directory not found: {annotations_dir}')
    if not predictions_tsv.exists():
        sys.exit(f'Predictions TSV not found: {predictions_tsv}')

    # Load predictions
    print(f'Loading predictions from: {predictions_tsv}')
    all_preds = load_predictions(predictions_tsv)
    print(f'  → {len(all_preds)} documents in predictions.')

    # Find annotation files
    json_files = sorted(annotations_dir.glob('*.json'))
    if not json_files:
        sys.exit(f'No JSON annotation files found in {annotations_dir}')
    print(f'Found {len(json_files)} annotation file(s).\n')

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_labels = set()
    doc_results = []  # list of (doc_name, gold_set, pred_set)

    for jf in json_files:
        pred_fname = derive_pred_filename(jf)
        annotation_sets, classes = load_annotations(jf, annotator=args.annotator)
        labels = [normalise_label(c) for c in classes]
        all_labels.update(labels)

        gold_set = annotation_sets[0]  # first (or union) annotation set

        print(f'Document: {jf.name}')
        print(f'  → Mapped to prediction filename: {pred_fname}')
        print(f'  → Gold entities: {len(gold_set)}')

        # Inter-annotator agreement if multiple sets
        if len(annotation_sets) > 1:
            iaa = iaa_f1(annotation_sets[0], annotation_sets[1])
            print(f'  → Inter-annotator agreement (F1, strict): {iaa:.3f}')

        if pred_fname not in all_preds:
            print(f'  !! WARNING: "{pred_fname}" not found in predictions TSV.')
            print(f'     Available similar keys: '
                  f'{[k for k in all_preds if pred_fname.split("_")[0] in k][:5]}')
            pred_set = set()
        else:
            pred_set = all_preds[pred_fname]
            print(f'  → Predicted entities: {len(pred_set)}')

        metrics = compute_metrics(gold_set, pred_set)
        print_metrics(metrics, title=f'Overall – {jf.stem}')

        per_type = compute_per_type_metrics(gold_set, pred_set, labels)
        print_per_type(per_type, title='Per entity type (strict span)')

        # Error analysis
        gold_strict = {(s, e, l) for s, e, l, _ in gold_set}
        pred_strict = {(s, e, l) for s, e, l, _ in pred_set}
        false_negatives = [(s, e, l, gold_set) for s, e, l in (gold_strict - pred_strict)
                           for gs, ge, gl, gt in gold_set if gs == s and ge == e and gl == l]
        false_positives = [(s, e, l, pred_set) for s, e, l in (pred_strict - gold_strict)
                           for ps, pe, pl, pt in pred_set if ps == s and pe == e and pl == l]

        print(f'\n  False Negatives (gold not found in pred): {len(gold_strict - pred_strict)}')
        for s, e, l in sorted(gold_strict - pred_strict)[:10]:
            text = next((t for gs, ge, gl, t in gold_set if gs == s and ge == e and gl == l), '')
            print(f'    [{l}] ({s},{e}) "{text}"')
        if len(gold_strict - pred_strict) > 10:
            print(f'    ... and {len(gold_strict - pred_strict) - 10} more')

        print(f'\n  False Positives (pred not in gold): {len(pred_strict - gold_strict)}')
        for s, e, l in sorted(pred_strict - gold_strict)[:10]:
            text = next((t for ps, pe, pl, t in pred_set if ps == s and pe == e and pl == l), '')
            print(f'    [{l}] ({s},{e}) "{text}"')
        if len(pred_strict - gold_strict) > 10:
            print(f'    ... and {len(pred_strict - gold_strict) - 10} more')

        doc_results.append((pred_fname, gold_set, pred_set, labels))

        # Optional per-doc TSV output
        if output_dir:
            rows = []
            for s, e, l, t in sorted(gold_set):
                in_pred = (s, e, l) in pred_strict
                rows.append(dict(source='gold', start=s, end=e, label=l, text=t, match=in_pred))
            for s, e, l, t in sorted(pred_set):
                in_gold = (s, e, l) in gold_strict
                rows.append(dict(source='pred', start=s, end=e, label=l, text=t, match=in_gold))
            pd.DataFrame(rows).sort_values(['start', 'source']).to_csv(
                output_dir / (pred_fname.replace('.txt', '_eval.tsv')), sep='\t', index=False)

        print()

    # ---------------------------------------------------------------------------
    # Aggregate metrics across all documents
    # ---------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('  AGGREGATE RESULTS (all annotated documents)')
    print('=' * 60)

    agg_gold = set()
    agg_pred = set()
    # Use document-offset to avoid span collisions across docs
    offset = 0
    DOC_OFFSET = 10_000_000

    agg_gold_flat = []
    agg_pred_flat = []
    for i, (fname, gold_set, pred_set, labels) in enumerate(doc_results):
        off = i * DOC_OFFSET
        agg_gold_flat.extend((s + off, e + off, l, t) for s, e, l, t in gold_set)
        agg_pred_flat.extend((s + off, e + off, l, t) for s, e, l, t in pred_set)

    agg_gold_set = set(agg_gold_flat)
    agg_pred_set = set(agg_pred_flat)

    # --- Entity counts per class ---
    print('\n  -- Entity counts per class (aggregate) --')
    agg_labels = sorted(all_labels)
    gold_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    for _, _, l, _ in agg_gold_flat:
        gold_counts[l] += 1
    for _, _, l, _ in agg_pred_flat:
        pred_counts[l] += 1
    all_count_labels = sorted(set(gold_counts) | set(pred_counts))
    count_header = f'  {"Label":<15} {"Gold":>8} {"Pred":>8} {"Diff":>8}'
    print(count_header)
    print('  ' + '-' * (len(count_header) - 2))
    for label in all_count_labels:
        g = gold_counts.get(label, 0)
        p = pred_counts.get(label, 0)
        print(f'  {label:<15} {g:>8} {p:>8} {p - g:>+8}')
    total_g = sum(gold_counts.values())
    total_p = sum(pred_counts.values())
    print('  ' + '-' * (len(count_header) - 2))
    print(f'  {"TOTAL":<15} {total_g:>8} {total_p:>8} {total_p - total_g:>+8}')

    agg_metrics = compute_metrics(agg_gold_set, agg_pred_set)
    print_metrics(agg_metrics)

    agg_per_type = compute_per_type_metrics(agg_gold_set, agg_pred_set, agg_labels)
    print_per_type(agg_per_type, title='Per entity type (strict span, all docs)')


if __name__ == '__main__':
    main()
