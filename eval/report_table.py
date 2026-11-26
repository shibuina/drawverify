#!/usr/bin/env python3
"""Report Table 2 columns for prediction files: own test, LIBERO-90, new scenes, human, non-pick AUROC and the
human set at 0.5 (precision and recall on invalid, R_val, MCC). Several files per row are averaged (seeds).

  python eval/report_table.py --human-meta output/human_ui/data/human/test-00000.parquet \\
      --row "answer-only=output/eval/gpu01/s8/full4ans,output/eval/cluster/full4ans-s1,output/eval/cluster/full4ans-s2"

Each run prefix P needs P_human.jsonl and, if present, P_l90val.jsonl and P_gentest.jsonl.
Non-pick leaves out the human sketches with a pick caption ("pick ...", "grab ...") and an arrow (84 of 339).
Human labels come from --human-meta (the final labels), not from the prediction files, which may predate a relabel.
New scenes = LIBERO-90 rows from the 11 scenes with no LIBERO-10 task (the only training suite with scene names).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import auroc, metrics  # noqa: E402

TRAIN_SCENES = {"KITCHEN_SCENE3", "KITCHEN_SCENE4", "KITCHEN_SCENE6", "KITCHEN_SCENE8", "LIVING_ROOM_SCENE1",
                "LIVING_ROOM_SCENE2", "LIVING_ROOM_SCENE5", "LIVING_ROOM_SCENE6", "STUDY_SCENE1"}   # LIBERO-10


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def au(rows: list[dict]) -> float:
    return auroc([int(r["target"] == "wrong") for r in rows], [r["p_wrong"] for r in rows]) if rows else float("nan")


def run(prefix: str, pick_arrow: set[str], label: dict[str, str]) -> dict:
    hum, l90, gen = (load(Path(f"{prefix}_{s}.jsonl")) for s in ("human", "l90val", "gentest"))
    hum = [{**r, "target": label[r["id"]]} for r in hum]
    scene = lambda r: re.search(r"((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", r["id"]).group(1)  # noqa: E731
    y = [int(r["target"] == "wrong") for r in hum]
    m = metrics(y, [int(r["p_wrong"] >= 0.5) for r in hum], [r["p_wrong"] for r in hum])
    return {"own": au(gen), "l90": au(l90), "new": au([r for r in l90 if scene(r) not in TRAIN_SCENES]),
            "human": m["auroc"], "nonpick": au([r for r in hum if r["id"] not in pick_arrow]),
            "prec": m["prec"], "rec": m["rec"], "r_val": m["r_val"], "mcc": m["mcc"],
            "n": (len(gen), len(l90), len(hum))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-meta", required=True, help="human test parquet (id, caption, has_arrow)")
    ap.add_argument("--row", nargs="+", required=True, metavar="NAME=PREFIX[,PREFIX...]")
    a = ap.parse_args()
    meta = pq.read_table(a.human_meta, columns=["id", "caption", "has_arrow", "verdict"]).to_pylist()
    pick_arrow = {r["id"] for r in meta if r["has_arrow"] and r["caption"].lower().split()[0] in ("pick", "grab")}
    label = {r["id"]: r["verdict"] for r in meta}
    print(f"pick captions with an arrow: {len(pick_arrow)}")
    cols = ["own", "l90", "new", "human", "nonpick", "prec", "rec", "r_val", "mcc"]
    print(f"{'':28s}" + "".join(f"{c:>15s}" for c in cols))
    for spec in a.row:
        name, prefixes = spec.split("=", 1)
        runs = [run(p, pick_arrow, label) for p in prefixes.split(",")]
        for p, r in zip(prefixes.split(","), runs):
            print(f"  {Path(p).name:26s}" + "".join(f"{r[c]:15.3f}" for c in cols) + f"   n={r['n']}")
        if len(runs) > 1:
            def cell(c: str) -> str:
                vals = [r[c] for r in runs if not math.isnan(r[c])]
                if len(vals) > 1:
                    return f"{st.mean(vals):.3f}±{st.stdev(vals):.3f}"
                return f"{vals[0]:.3f}" if vals else "--"
            print(f"{name + ' mean':28s}" + "".join(f"{cell(c):>15s}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
