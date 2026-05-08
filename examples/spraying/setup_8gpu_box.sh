#!/usr/bin/env bash
# One-shot bootstrap for an 8xA100 (or any multi-GPU) preprocessing box.
# Idempotent: re-runnable; skips work that's already done.
#
# Assumptions:
#   - uv is installed at $HOME/.local/bin/uv (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - aws CLI is installed and credentials configured for s3://vla-data-collection
#   - Python 3.11 and CUDA-matched driver are present
#   - This script lives in the cloned molmoact repo; we use it as the working dir
#
# Usage on the 8-GPU box:
#   curl -sL https://raw.githubusercontent.com/faizan050903/molmoact/spraying-finetune/examples/spraying/setup_8gpu_box.sh | bash
# OR after cloning manually:
#   bash examples/spraying/setup_8gpu_box.sh

set -euo pipefail

cd "$HOME"

echo "[1/7] Cloning repos..."
[ -d molmoact ] || git clone https://github.com/faizan050903/molmoact.git
[ -d Depth-Anything-V2 ] || git clone https://github.com/DepthAnything/Depth-Anything-V2.git
[ -d Aurora-perception ] || git clone --depth=1 https://github.com/mahtabbigverdi/Aurora-perception.git
ln -sfn "$HOME/Aurora-perception/AiT" "$HOME/AiT"

cd "$HOME/molmoact"
git fetch origin spraying-finetune
git checkout spraying-finetune
git pull --ff-only origin spraying-finetune

echo "[2/7] Creating uv venv..."
[ -d .venv ] || uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/7] Installing molmoact + extras..."
uv pip install -e ".[all]"

echo "[4/7] Installing Depth-Anything-V2 deps + lerobot + tensorflow-cpu..."
( cd "$HOME/Depth-Anything-V2" && uv pip install -r requirements.txt )
uv pip uninstall opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
uv pip install --no-cache-dir opencv-python-headless
uv pip install lerobot==0.3.3
uv pip install tensorflow-cpu

echo "[5/7] Downloading depth + VQVAE checkpoints..."
mkdir -p "$HOME/Depth-Anything-V2/checkpoints"
[ -f "$HOME/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth" ] || \
    wget -q --show-progress -O "$HOME/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth" \
        "https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/depth_anything_v2_vitb.pth"
[ -f "$HOME/molmoact/vae-final.pt" ] || \
    wget -q --show-progress -O "$HOME/molmoact/vae-final.pt" \
        "https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/vae-final.pt"

echo "[6/7] Pre-warming Molmo VLM (gripper detection, ~14 GB)..."
huggingface-cli download allenai/Molmo-7B-D-0924

echo "[7/7] Pulling cleaned LeRobot dataset from S3 (~30 GB)..."
mkdir -p "$HOME/.cache/huggingface/lerobot/rishi-10x"
aws s3 sync \
    s3://vla-data-collection/arranged_dummy_data/spraying-v1-cleaned/ \
    "$HOME/.cache/huggingface/lerobot/rishi-10x/spraying-v1-cleaned/"

echo
echo "Setup complete. Activate venv with: source $HOME/molmoact/.venv/bin/activate"
echo "Then run preprocessing: bash $HOME/molmoact/examples/spraying/preprocess_parallel.sh"
