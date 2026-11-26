#!/usr/bin/env python3
"""Rebuild the eval set with YOUR labels as ground truth.

The rendered set ships with the geometric verdict (`sketch_ok`: does the drawing
match the demonstration). This replaces it with the labels from
labeling/label_server.py, which answer the question the validator is actually
asked — is the pair coherent and unambiguous.

Only labelled rows are kept, and `unsure` is dropped. The geometric verdict is
preserved as `geometric_right` so the two can still be compared, and the images
are shared with the original set rather than copied.

    python labeling/apply_labels.py                       # -> out/human-eval-labelled
    python labeling/apply_labels.py --include-unsure wrong
"""

from __future__ import annotations

import argparse
import json
import shutil
import os
from pathlib import Path

def _human_eval_root() -> Path:
    """The human-evaluation working directory: holds out/ and the label files.

    $HUMAN_EVAL_ROOT wins. Otherwise the `human_eval/` folder next to this repo (the lab
    layout is <x>/DrawVLA/{data_gen,human_eval}), then one inside the repo.
    """
    env = os.environ.get("HUMAN_EVAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]          # this file lives in <repo>/labeling/
    for cand in (repo.parent / "human_eval", repo / "human_eval"):
        if cand.is_dir():
            return cand
    return repo.parent / "human_eval"


ROOT = _human_eval_root()
SRC = ROOT / "out" / "human-eval"
LABELS = ROOT / "out" / "labels.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--set", type=Path, default=SRC, dest="src",
                    help="rendered set the labels belong to")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "human-eval-labelled")
    ap.add_argument("--include-unsure", choices=("drop", "right", "wrong"), default="drop",
                    help="what to do with sketches you marked unsure")
    ap.add_argument("--marks", choices=("any", "both"), default="any",
                    help="'both' keeps only drawings that have a circle AND an arrow")
    args = ap.parse_args()

    if not args.labels.is_file():
        raise SystemExit(f"no labels at {args.labels} — run labeling/label_server.py first")
    mine = {}
    for line in args.labels.open(encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            mine[d["response_id"]] = d["label"]          # later lines win, so undo/relabel works

    src = args.src
    records = [json.loads(l) for l in (src / "human_sketches.jsonl").open(encoding="utf-8")]
    kept, dropped = [], {"unlabelled": 0, "unsure": 0, "one_channel": 0}
    for r in records:
        label = mine.get(r["response_id"])
        if label is None:
            dropped["unlabelled"] += 1
            continue
        if label == "unsure":
            if args.include_unsure == "drop":
                dropped["unsure"] += 1
                continue
            label = args.include_unsure
        if args.marks == "both" and not r["arrow"]:
            dropped["one_channel"] += 1
            continue
        row = dict(r)
        row["right"] = label == "right"
        row["verdict"] = label
        row["geometric_right"] = bool(r["right"])
        row["label_source"] = "manual"
        kept.append(row)

    if not kept:
        raise SystemExit("nothing labelled yet")

    out = args.out
    (out / "data" / "human").mkdir(parents=True, exist_ok=True)
    if (out / "images").is_symlink() or (out / "images").exists():
        if (out / "images").is_symlink():
            (out / "images").unlink()
        else:
            shutil.rmtree(out / "images")
    (out / "images").symlink_to(src.resolve() / "images")
    with (out / "human_sketches.jsonl").open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "id": [r["id"] for r in kept],
        "image": [{"bytes": (src / r["image"]).read_bytes(), "path": r["image"]} for r in kept],
        "caption": [r["caption"] for r in kept],
        "right": [r["right"] for r in kept],
        "verdict": [r["verdict"] for r in kept],
        "name_explicit": [False] * len(kept),
        "prompt_level": [r["prompt_level"] for r in kept],
        "response_id": [r["response_id"] for r in kept],
        "has_arrow": [bool(r["arrow"]) for r in kept],
        "geometric_right": [r["geometric_right"] for r in kept],
    })
    pq.write_table(table, out / "data" / "human" / "test-00000.parquet")

    agree = sum(r["right"] == r["geometric_right"] for r in kept)
    print(json.dumps({
        "labelled_rows_kept": len(kept),
        "your_right": sum(r["right"] for r in kept),
        "your_wrong": sum(not r["right"] for r in kept),
        "dropped": dropped,
        "agreement_with_geometric": round(agree / len(kept), 3),
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
