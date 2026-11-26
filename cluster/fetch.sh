#!/bin/bash
#SBATCH --account=sunjunresearch
#SBATCH --partition=sunjunresearch
#SBATCH --qos=sunjunresearch-priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/common/home/users/d/doducanh.2026/drawverify/logs/fetch-%j.log
# Qwen3-VL-2B (home) and -4B (scratch, via the symlink in hf/hub) into HF_HOME; job.sh runs offline from it.
set -euo pipefail
export HF_HOME=~/drawverify/hf PYTHONPATH=~/drawverify/pydeps
~/envs/lerobot/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"):
    print(m, snapshot_download(m), flush=True)
PY
du -shL ~/drawverify/hf/hub/*
