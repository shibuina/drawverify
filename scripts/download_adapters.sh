#!/usr/bin/env bash
# Download the trained DrawVerify 2B LoRA adapters and the per-prompt predictions from Hugging Face Hub,
# then verify the adapters' SHA-256 checksums.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPO="${DRAWVERIFY_ADAPTER_REPO:-shibuina/drawverify-adapters}"
PY="${PYTHON:-python3}"

"$PY" - "$REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(repo_id=repo, repo_type="model", local_dir="output/adapters",
                         allow_patterns=["SHA256SUMS", "2b-*/*"])
print(f"Downloaded adapters of {repo} -> {path}")
# per-prompt predictions behind every report table + label-only human test set (no images) -> output/eval, output/
path = snapshot_download(repo_id=repo, repo_type="model", local_dir="output",
                         allow_patterns=["eval/**", "human_test_labels.parquet"])
print(f"Downloaded predictions of {repo} -> {path}/eval")
PY

echo "Verifying SHA-256 checksums..."
sha256sum -c output/adapters/SHA256SUMS
