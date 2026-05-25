#!/bin/bash
# KG-RD Compress: Environment Setup
# ===================================
# Run this once to set up the environment.

set -e

echo "=========================================="
echo "  KG-RD Compress: Setup"
echo "=========================================="

# 1. Install Python dependencies
echo "[1/4] Installing Python packages ..."
pip install -r requirements.txt

# 2. Download spaCy model
echo "[2/4] Downloading spaCy English model ..."
python -m spacy download en_core_web_sm

# 3. Download Benepar model
echo "[3/4] Downloading Benepar constituency parser ..."
python -c "
import benepar
benepar.download('benepar_en3')
print('Benepar downloaded successfully')
"

# 4. Download NLTK data (needed for WordNet in SPICE)
echo "[4/4] Downloading NLTK data ..."
python -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('NLTK data downloaded')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  "
echo "  Quick test (10 images):"
echo "    python run_pipeline.py --max_samples 10"
echo "  "
echo "  Full run:"
echo "    python run_pipeline.py"
echo "=========================================="
