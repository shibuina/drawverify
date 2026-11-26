#!/usr/bin/env python
"""Score a validator on one split: a LoRA adapter, or the base model zero-shot without one.

Writes one JSON line per row (id, target, p_wrong, prediction, reasoning) for eval/score.py.
P(wrong) is read from the answer alone unless --max-new-tokens asks for a rationale first.
The decision is the forced-choice verdict (p_wrong >= 0.5); thresholded gates are set in
score.py.

  python eval/eval_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --adapter output/train/gen-hum \\
      --data <export dir or HF id> --split test --out output/eval/4b-gen-hum.jsonl
  python eval/eval_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --data ... --out output/eval/4b-zero-shot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from validator import load_model, load_processor, load_split, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="LoRA adapter dir or HF model id; omit for zero-shot")
    ap.add_argument("--data", required=True, help="stage-10 export dir or HF dataset id")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    # The default reads P(wrong) from the answer alone: after a generated rationale the
    # answer is settled and P(wrong) saturates at 0/1, which leaves no threshold to tune.
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="rationale budget before the answer; 0 (default) scores the answer alone")
    ap.add_argument("--limit", type=int, help="first N rows only (smoke tests)")
    a = ap.parse_args()

    rows = load_split(a.data, a.split)
    if a.limit:
        rows = rows.select(range(min(a.limit, len(rows))))
    processor = load_processor(a.model)
    model = load_model(a.model, a.adapter).eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for start in range(0, len(rows), a.batch_size):
            batch = [rows[i] for i in range(start, min(start + a.batch_size, len(rows)))]
            for r, s in zip(batch, score(model, processor, batch, a.max_new_tokens)):
                f.write(json.dumps({"id": r.get("id", ""), "target": r["verdict"], **s,
                                    "prediction": "wrong" if s["p_wrong"] >= 0.5 else "right"})
                        + "\n")
            print(f"{start + len(batch)}/{len(rows)}", flush=True)
    print(f"predictions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
