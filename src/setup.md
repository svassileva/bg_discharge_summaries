# Setup and dependencies

Run commands from the `bg_discharge_summaries` repository root. Use Python
3.10 or newer. The scripts have different dependencies; install only the
packages needed for the workflow you plan to run.

## Lightweight utilities

The following scripts use only the Python standard library:

- `src/combine_abbreviation_tables.py`
- `src/combine_terms_tables.py`
- `src/calc_synthmedic_stats.py`
- `src/sentence_pair_stats.py`

`src/combine_gliner_annotations.py` and `src/eval_ner.py` require pandas.
`src/llm_ner_parse.py` requires tqdm.

## GLiNER entity extraction

Install the GLiNER fork and its requirements, then download the model files:

```bash
git clone https://github.com/mayank-rakesh-mck/GLiNER.git
python -m pip install -r GLiNER/requirements.txt
huggingface-cli download Mayank6255/GLiNER-MoE-MultiLingual \
  --local-dir ./gliner_moe_model
```

Run extraction with explicit input and output paths:

```bash
python src/gliner_moe_ner.py path/to/texts results/entities.tsv \
  --model-dir ./gliner_moe_model
```

## Entity matching and translation

Entity matching and local translation use PyTorch and Hugging Face
Transformers; SapBERT matching also uses NumPy. Install the versions
appropriate for your hardware and model requirements. `faiss-cpu` is optional
for SapBERT retrieval.

`src/translate_entities.py --translate` uses the local
`Helsinki-NLP/opus-mt-bg-en` model. API modes require the `openai` package and
provider credentials in environment variables:

- Azure: `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
- OpenAI-compatible API: `OPENAI_BASE_URL` and `OPENAI_API_KEY` (or the
  corresponding command-line options)

API translation sends the entity strings to the configured provider. Do not
use it with data that is not authorized for external processing.

## Knowledge-base and evaluation inputs

SapBERT entity linking can consume UMLS-derived candidate files. The UMLS
release files are not included here; users must obtain them under the UMLS
license. Translation evaluation additionally uses BERTScore and
SentenceTransformers; see the requirements documented in
`src/translation_evaluation.py`.

No clinical source data or model weights are bundled. Provide your own input
paths and verify applicable data, model, and third-party licenses before use
or redistribution.