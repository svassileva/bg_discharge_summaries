# BGSYNTHMEDIC: SYNTHETIC PARALLEL ENGLISH BULGARIAN DISCHARGE SUMMARY CORPUS WITH AUTOMATIC ANNOTATIONS

```
@inproceedings{boytcheva2026bgsynthmedic,
  author    = {Svetla Boytcheva and Sylvia Vassileva and Pavel Boytchev},
  title     = {BGSynthMedic: Synthetic Parallel English-Bulgarian Discharge Summary Corpus with Automatic Annotations},
  booktitle = {Proceedings of the Seventh International Conference Computational Linguistics in Bulgaria (CLIB 2026)},
  year      = {2026},
  address   = {Sofia, Bulgaria},
  publisher = {Institute for Bulgarian Language, Bulgarian Academy of Sciences},
  pages     = {42--54},
  issn      = {2367-5675},
  url       = {https://dcl.bas.bg/clib/wp-content/uploads/2026/09/CLIB2026_06_Boytcheva-et-al_042-054.pdf}
}
```

This repository contains research scripts for processing English/Bulgarian
discharge-summary text, NER annotations, terminology tables, and translation
evaluation. It does not include clinical source data, model weights, or UMLS
files; obtain and use those separately under their applicable terms.

Run commands from the repository root. Supply paths to your own data rather
than relying on files from the original research workspace.

## Scripts

| Script | Purpose |
| --- | --- |
| `src/calc_synthmedic_stats.py` | Write document-level and corpus summary statistics. |
| `src/combine_abbreviation_tables.py` | Combine markdown abbreviation tables into a TSV. |
| `src/combine_terms_tables.py` | Combine markdown term tables into a TSV. |
| `src/combine_gliner_annotations.py` | Combine GLiNER annotation TSVs from `gliner_*` folders. |
| `src/cross_lingual_entity_matching.py` | Match English and Bulgarian entities within aligned sentence pairs. |
| `src/eval_ner.py` | Evaluate NER predictions against manual annotation JSON files. |
| `src/llm_ner_parse.py` | Parse LLM NER output files, optionally linking mentions to source texts. |
| `src/sentence_pair_stats.py` | Summarize aligned sentence-pair TSV files. |
| `src/translate_entities.py` | Translate entity strings using a local or configured API backend. |

Use each script's `--help` output for all options. Example commands:
```bash
python src/calc_synthmedic_stats.py path/to/texts --outdir results/stats
python src/combine_abbreviation_tables.py \
  --input-dir path/to/abbreviation-tables \
  --output-tsv results/abbreviations.tsv
python src/combine_terms_tables.py \
  --input-dir path/to/term-tables \
  --output-tsv results/terms.tsv
python src/combine_gliner_annotations.py \
  --annotations-dir path/to/annotations \
  --output-tsv results/gliner_annotations.tsv
python src/cross_lingual_entity_matching.py \
  --sentence-pairs-dir results/alignment/sentence_pairs \
  --src-ner-tsv path/to/entities_en.tsv \
  --hyp-ner-tsv path/to/entities_bg.tsv \
  --output-tsv results/entity_matches.tsv
python src/eval_ner.py \
  --annotations-dir path/to/manual-annotations \
  --predictions-tsv path/to/predictions.tsv
python src/llm_ner_parse.py path/to/llm-output --output results/parsed.tsv
python src/sentence_pair_stats.py --input-dir results/alignment/sentence_pairs
python src/translate_entities.py \
  --ner-tsv path/to/entities.tsv \
  --output-tsv results/entities_translated.tsv \
  --translate
```

## Dependencies and setup

The table combiners, sentence-pair statistics, and corpus statistics scripts
use the Python standard library. Other scripts have additional dependencies;
see [src/setup.md](src/setup.md) for setup notes and optional model/API
requirements. Install only the dependencies needed for the scripts you run.

## Data handling and licensing

Do not commit clinical source text, annotation exports, model weights, caches,
or generated outputs unless their release has been separately approved.
Confirm the data and model licenses before redistribution. UMLS data requires
separate access under the UMLS license.

`translate_entities.py` sends entity text to an external provider when
`--azure-translate` or `--openai-translate` is selected. Use those modes only
when the data is authorized for that provider. API credentials should be
provided through environment variables, not committed to this repository.
