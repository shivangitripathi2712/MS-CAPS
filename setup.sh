#!/bin/bash
# MS-CAPS environment setup. Run once inside a fresh Python 3.10 environment.
set -e

echo "[1/2] Installing Python packages ..."
pip install -r requirements.txt

echo "[2/2] Downloading spaCy English model ..."
python -m spacy download en_core_web_sm

echo ""
echo "Setup complete."
echo "  Quick test (2 images):  python run_pipeline.py --max-samples 2"
echo "  Full run (50 images):   python run_pipeline.py --max-samples 50"
echo ""
echo "FLUX.1-dev is a gated model: accept its license on Hugging Face and run"
echo "'huggingface-cli login' before the first run."
