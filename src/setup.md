# 1. Install the forked GLiNER
git clone https://github.com/mayank-rakesh-mck/GLiNER.git
cd GLiNER && pip install -r requirements.txt && cd ..

# 2. Download model files
huggingface-cli download Mayank6255/GLiNER-MoE-MultiLingual --local-dir ./gliner_moe_model

# 3. Run
python gliner_moe_ner.py texts_top_30 gliner_moe_entities.tsv