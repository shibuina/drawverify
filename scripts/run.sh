#!/usr/bin/env bash
# One-click runner for DrawVerify (CS707 artifact package).
#
# Usage:
#   bash scripts/run.sh                       # verify checksums, run tests, reproduce report tables
#   bash scripts/run.sh reproduce             # same as default
#   bash scripts/run.sh verify-checksums      # check SHA-256 of trained LoRA adapters
#   bash scripts/run.sh train stage1-ans [args...]   # train 2B answer-only stage-1 validator
#   bash scripts/run.sh train stage1-rat [args...]   # train 2B rationale stage-1 validator
#   bash scripts/run.sh train stage2 [args...]       # fine-tune stage-2 adapter on human sketches
#   bash scripts/run.sh demo [args...]        # launch interactive local web demo (pass --selftest to smoke-test)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pick_python() {
  if [ -n "${PYTHON:-}" ]; then
    echo "$PYTHON"
    return
  fi
  for cand in python3 "$HOME/.cache/drawverify-demo/venv/bin/python"; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c "import pyarrow.dataset, pyarrow.parquet, numpy" >/dev/null 2>&1; then
        echo "$cand"
        return
      fi
    fi
  done
  echo "python3"
}

PY="$(pick_python)"
MODE="${1:-reproduce}"
shift || true

case "$MODE" in
  -h|--help|help)
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  verify-checksums)
    echo "== Verifying adapter SHA-256 checksums =="
    sha256sum -c output/adapters/SHA256SUMS
    ;;

  reproduce)
    echo "== 1/4 Adapter checksums =="
    if [ -f output/adapters/SHA256SUMS ]; then
      sha256sum -c output/adapters/SHA256SUMS
    else
      echo "No output/adapters/SHA256SUMS found; skipping."
    fi

    echo
    echo "== 2/4 Unit tests & metric self-check =="
    if command -v pytest >/dev/null 2>&1; then
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
    else
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest tests/ -q
    fi
    "$PY" eval/metrics.py

    HUMAN_META="output/human_ui/data/human/test-00000.parquet"   # full human set (local build)
    [ -f "$HUMAN_META" ] || HUMAN_META="output/human_test_labels.parquet"   # label-only copy from scripts/download_adapters.sh
    if [ -f "$HUMAN_META" ] && [ -d output/eval/cluster ] && [ -d output/eval/gpu01 ]; then
      echo
      echo "== 3/4 Table 1 (three-seed means) & Appendix F (data-size sensitivity) =="
      "$PY" eval/report_table.py --human-meta "$HUMAN_META" \
        --row "2B rationale=output/eval/gpu01/v4/full4,output/eval/cluster/full4-s1,output/eval/cluster/full4-s2" \
              "2B answer-only=output/eval/gpu01/s8/full4ans,output/eval/cluster/full4ans-s1,output/eval/cluster/full4ans-s2" \
              "4B rationale=output/eval/gpu01/s8/4b-full4-s0,output/eval/gpu01/s8/4b-full4-s1,output/eval/gpu01/s8/4b-full4-s2" \
              "4B + stage 2=output/eval/gpu01/s8/4b-full4-s0-hum,output/eval/gpu01/s8/4b-full4-s1-hum,output/eval/gpu01/s8/4b-full4-s2-hum"
      if [ -f output/eval/cluster/full4ans-f0p1_human.jsonl ]; then
        echo
        echo "-- Training-data-size sensitivity (2B answer-only, seed 0) --"
        "$PY" eval/report_table.py --human-meta "$HUMAN_META" \
          --row "10% (3,417)=output/eval/cluster/full4ans-f0p1" \
                "25% (8,541)=output/eval/cluster/full4ans-f0p25" \
                "50% (17,083)=output/eval/cluster/full4ans-f0p5" \
                "100% (34,165)=output/eval/gpu01/s8/full4ans"
      fi

      echo
      echo "== 4/4 Seed-level statistics, paired bootstrap intervals, and sketch-type counts =="
      "$PY" eval/report_seeds.py "$HUMAN_META" output/eval
    else
      echo "Saved predictions or human labels not present; run scripts/download_adapters.sh first. Skipping table reproduction."
    fi
    ;;

  train)
    STAGE="${1:-stage1-ans}"
    shift || true
    case "$STAGE" in
      stage1-ans) CFG="configs/train_stage1_ans.yaml" ;;
      stage1-rat) CFG="configs/train_stage1_rat.yaml" ;;
      stage2)     CFG="configs/train_stage2.yaml" ;;
      *) echo "Unknown train target: $STAGE (expected stage1-ans | stage1-rat | stage2)" >&2; exit 2 ;;
    esac
    exec "$PY" training/train.py --config "$CFG" "$@"
    ;;

  demo)
    exec bash demo/run.sh "$@"
    ;;

  *)
    echo "Unknown mode: $MODE (run 'bash scripts/run.sh --help')" >&2
    exit 2
    ;;
esac
