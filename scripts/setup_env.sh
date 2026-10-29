#!/usr/bin/env bash
# Builds the Python 3.12 conda env this project needs: SAM 3.1 requires
# Python 3.12+, PyTorch 2.7+, CUDA 12.6+. 3.12 is also the sweet spot where
# projectaria_tools ships an aarch64 (Grace/GB10) wheel.
#
# Prerequisite: conda ToS must already be accepted for the default channels:
#   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
#   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
set -euo pipefail

ENV_NAME="hackathon"
SAM3_REPO="/home/iairo-dgx1/vad_data/hackathon/repos/sam3"

conda create -n "$ENV_NAME" python=3.12 -y

PIP="$(conda run -n $ENV_NAME which pip)"

# Official SAM3 install instructions pin cu128; GB10's driver is forward-compatible.
"$PIP" install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
"$PIP" install projectaria-tools huggingface_hub[cli]

( cd "$SAM3_REPO" && "$PIP" install -e . )

echo "Env '$ENV_NAME' ready. Next: hf auth login (once SAM 3.1 gated access is approved),"
echo "then run scripts/smoke_test_sam31.py on the DGX Spark."
