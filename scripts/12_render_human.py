#!/usr/bin/env python
"""Re-render a labelled human-sketch set in the generator's fixed stroke look.

The study stores each mark as a raw pointer path; the old renders gave each drawing a
random colour, width and head style from the generator's former augmentation. This draws
every path in `strokes.UI` (what participants saw while drawing) and keeps the labelled
set's rows and captions. Labels change only by rule (a pick verb with an arrow is wrong) and
by `--overrides`.

  python scripts/12_render_human.py --labelled data/human/data/human/test-00000.parquet \
      --manifest .../human_eval/out/control-clean_arrow/human_sketches.jsonl \
      --scenes .../user_study_draw/public/scenes --out output/human_ui

  # stage-2 training set: the other labelled sketches, minus every evaluated drawer
  python labeling/apply_labels.py --out /tmp/hel      # labels.jsonl -> labelled parquet
  python scripts/12_render_human.py --labelled /tmp/hel/data/human/test-00000.parquet \
      --manifest .../human_eval/out/human-eval/human_sketches.jsonl --scenes ... \
      --exclude .../human_eval/out/control-clean_arrow/human_sketches.jsonl \
      --cal-share 0.3 --overrides /dev/null --out output/human_train
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import _common  # noqa: F401  (puts the repo on sys.path)
from libero_sketch import render as R
from libero_sketch.false_prompts import is_pick


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", type=Path, required=True, help="labelled set parquet")
    ap.add_argument("--manifest", type=Path, required=True, help="JSONL with the raw strokes")
    ap.add_argument("--scenes", type=Path, required=True, help="dir of <image_sha256>.png")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overrides", type=Path,
                    default=Path(__file__).resolve().parents[1] / "labeling" / "human_label_overrides.jsonl",
                    help="JSONL of {response_id, verdict}: label decisions made after labelling")
    ap.add_argument("--exclude", nargs="*", type=Path, default=[],
                    help="eval manifests: drop their sketches and every row by their drawers")
    ap.add_argument("--cal-share", type=float,
                    help="write train/validation splits, holding out this share of drawers")
    a = ap.parse_args()

    strokes = {}
    for line in a.manifest.open():
        r = json.loads(line)
        strokes[r["response_id"]] = r
    table = pq.read_table(a.labelled)
    rows = table.to_pylist()
    held = [json.loads(l) for f in a.exclude for l in f.open()]
    drop = {r["response_id"] for r in held}
    drawers = {r["participant_id"] for r in held}
    rows = [r for r in rows if r["response_id"] not in drop
            and strokes[r["response_id"]]["participant_id"] not in drawers]
    fix = {json.loads(l)["response_id"]: json.loads(l)["verdict"]
           for l in a.overrides.open()} if a.overrides.exists() else {}
    for row in rows:
        src = strokes.get(row["response_id"])
        if src is None:
            raise SystemExit(f"no strokes for {row['id']}")
        scene = np.asarray(Image.open(a.scenes / f"{src['scene_sha256']}.png").convert("RGB"))
        img = R.render_human(scene, src.get("circle_stroke"), src.get("arrow_stroke"))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        row["image"] = {"bytes": buf.getvalue(), "path": f"{row['response_id']}.png"}
        if is_pick(row["caption"]) and row["has_arrow"]:     # labelling rule, labeling/README.md
            row["verdict"], row["right"] = "wrong", False
        if row["response_id"] in fix:
            row["verdict"] = fix.pop(row["response_id"])
            row["right"] = row["verdict"] == "right"
    if fix:
        raise SystemExit(f"overrides for rows not in the set: {sorted(fix)}")
    splits = {"test": rows}
    if a.cal_share:                        # split by drawer, so no hand is on both sides
        cal = lambda r: int(hashlib.md5(strokes[r["response_id"]]["participant_id"].encode())
                            .hexdigest(), 16) % 1000 < a.cal_share * 1000
        splits = {"train": [r for r in rows if not cal(r)],
                  "validation": [r for r in rows if cal(r)]}
    out = a.out / "data" / "human"
    out.mkdir(parents=True, exist_ok=True)
    for split, part in splits.items():
        pq.write_table(pa.Table.from_pylist(part, schema=table.schema), out / f"{split}-00000.parquet")
        print(f"{len(part)} rows -> {out / f'{split}-00000.parquet'}"
              f" ({sum(r['verdict'] == 'wrong' for r in part)} invalid)")


if __name__ == "__main__":
    main()
