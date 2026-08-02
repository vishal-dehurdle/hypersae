#!/bin/bash
# setup_vm.sh — Run this on a fresh GCP Deep Learning VM to install all dependencies.
# Usage: bash setup_vm.sh
set -e

echo "=== HyperSAE Cloud VM Setup ==="

# 1. System packages
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv

# 2. Create virtual environment
echo "[2/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA support
echo "[3/6] Installing PyTorch (CUDA 12.x)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install HyperSAE dependencies
echo "[4/6] Installing HyperSAE dependencies..."
pip install geoopt networkx plotly matplotlib transformers datasets accelerate
pip install google-cloud-storage python-dotenv

# 5. Install lm-eval harness (for GPQA & MMLU-Pro)
echo "[5/6] Installing EleutherAI lm-eval harness..."
pip install lm-eval[api]

# 6. Authenticate Hugging Face
echo "[6/6] Authenticating Hugging Face..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -n "$HF_TOKEN" ]; then
    pip install huggingface_hub
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "Hugging Face authentication successful."
else
    echo "WARNING: HF_TOKEN not found in .env. Gated models (Gemma-2) will fail to download."
fi

# Authenticate GCS
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS"
    echo "GCS credentials set to: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "WARNING: GOOGLE_APPLICATION_CREDENTIALS not set. GCS uploads will fail."
fi

echo ""
echo "=== Setup Complete ==="
echo "To start training, run:"
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=src python cloud_run.py"
