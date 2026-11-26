#!/usr/bin/env python3
"""Pixel-geometry baseline (paper Sec. VIII-C, Table V): LightGBM on stroke geometry only.

The green circle and the red arrow are segmented out of each rendered image by colour, so the
input must use the fixed stroke colours (`strokes.UI`): every generated render and the human
sketches re-rendered by scripts/12_render_human.py. Features come from pixel positions only (circle centroid and radius,
arrow tail and tip, both relative to the circle); the caption is never read.

  python eval/geometry_baseline.py --fit gen=<gen export>:train --fit hum=<human export>:train \\
      --fit gen+hum=<gen export>:train,<human export>:train \\
      --test test=<human export>:test --test study=<study export>:test --out-dir output/eval/geometry

Writes <out-dir>/<fit>__<test>.jsonl in the eval/score.py layout and prints AUROC per pair.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import auroc  # noqa: E402

from libero_sketch.strokes import UI  # noqa: E402

CIRCLE_RGB, ARROW_RGB = UI["circle_color"], UI["arrow_color"]   # the study interface's pens
COLOR_TOL = 70.0                                        # RGB distance; survives JPEG-95
NAMES = ["c_px", "c_x", "c_y", "c_r", "c_spread",
         "a_px", "tail_x", "tail_y", "tip_x", "tip_y", "a_len", "a_cos", "a_sin",
         "tail_in_circle", "tip_in_circle"]


def read(spec: str) -> tuple[list[str], list[int], list[bytes]]:
    """`dir:split[,dir:split...]` -> ids, invalid flags, encoded images."""
    ids, ys, imgs = [], [], []
    for part in spec.split(","):
        root, split = part.rsplit(":", 1)
        for f in sorted(Path(root).expanduser().glob(f"**/{split}-*.parquet")):
            t = pq.read_table(f, columns=["id", "verdict", "image"]).to_pylist()
            ids += [r["id"] for r in t]
            ys += [int(r["verdict"] == "wrong") for r in t]
            imgs += [r["image"]["bytes"] for r in t]
    if not ids:
        raise SystemExit(f"no rows for {spec}")
    return ids, ys, imgs


def features(png: bytes) -> list[float]:
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    pts = {}
    for key, rgb in (("c", CIRCLE_RGB), ("a", ARROW_RGB)):
        ys, xs = np.nonzero(np.linalg.norm(a - rgb, axis=-1) < COLOR_TOL)
        pts[key] = np.stack([xs / w, ys / h], 1)
    nan = math.nan
    c, ar = pts["c"], pts["a"]
    if len(c) >= 10:
        cen = c.mean(0)
        d = np.linalg.norm(c - cen, axis=1)
        r = float(np.median(d))
        circ = [len(c) / (h * w), *cen, r, float(d.std() / r) if r else nan]
    else:
        cen, r, circ = None, nan, [len(c) / (h * w), nan, nan, nan, nan]
    if len(ar) >= 10:
        mu = ar.mean(0)
        axis = np.linalg.svd(ar - mu, full_matrices=False)[2][0]
        proj = (ar - mu) @ axis
        lo, hi = np.percentile(proj, [2, 98])
        ends = [mu + lo * axis, mu + hi * axis]
        near = [(np.abs(proj - v) < 0.15 * (hi - lo)).sum() for v in (lo, hi)]
        tail, tip = (ends[0], ends[1]) if near[1] >= near[0] else (ends[1], ends[0])  # head = more ink
        v = tip - tail
        length = float(np.linalg.norm(v))
        rel = [float(np.linalg.norm(p - cen) / r) if cen is not None and r else nan for p in (tail, tip)]
        arrow = [len(ar) / (h * w), *tail, *tip, length,
                 *(v / length if length else (nan, nan)), *rel]
    else:
        arrow = [len(ar) / (h * w)] + [nan] * 9
    return circ + arrow


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="append", required=True, metavar="NAME=DIR:SPLIT[,DIR:SPLIT]")
    ap.add_argument("--test", action="append", required=True, metavar="NAME=DIR:SPLIT")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from lightgbm import LGBMClassifier

    def load(spec):
        ids, ys, imgs = read(spec)
        return ids, np.array(ys), np.array([features(b) for b in imgs], dtype=np.float32)

    tests = {name: load(spec) for name, spec in (s.split("=", 1) for s in a.test)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for fit_name, spec in (s.split("=", 1) for s in a.fit):
        _, y, x = load(spec)
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                             random_state=a.seed, verbose=-1).fit(x, y)
        for test_name, (ids, ty, tx) in tests.items():
            p = clf.predict_proba(tx)[:, 1]
            with (a.out_dir / f"{fit_name}__{test_name}.jsonl").open("w") as f:
                for i, t, s in zip(ids, ty, p):
                    f.write(json.dumps({"id": i, "target": "wrong" if t else "right",
                                        "p_wrong": float(s)}) + "\n")
            print(f"{fit_name:>12} -> {test_name:<12} n={len(ids):>6}  AUROC {auroc(ty.tolist(), p.tolist()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
