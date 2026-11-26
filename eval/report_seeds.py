#!/usr/bin/env python3
"""Seed-level numbers of the final report (Sec. 6, Tables 2 and 4) from the per-prompt predictions.

  python eval/report_seeds.py <human test parquet> <eval dir>

<eval dir> holds gpu01/ (arena-gpu01: s8/, v4/, rep/) and cluster/ (origami: seeds 1-2, 4B zero-shot, geometry/).
Prints per seed and mean: shared/new-scene AUROC and valid rejections on LIBERO-90, R_val on the generated test split,
paired AUROC differences (seed i against seed i, 2,000 bootstrap resamples), rejections by sketch type, and the baselines.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

E = Path(sys.argv[2])
META = {r["id"]: r for r in pq.read_table(sys.argv[1], columns=["id", "caption", "has_arrow", "verdict"]).to_pylist()}
PICK = lambda i: META[i]["caption"].lower().split()[0] in ("pick", "grab")  # noqa: E731
NONPICK = {i for i in META if not (META[i]["has_arrow"] and PICK(i))}
TRAIN_SCENES = {"KITCHEN_SCENE3", "KITCHEN_SCENE4", "KITCHEN_SCENE6", "KITCHEN_SCENE8", "LIVING_ROOM_SCENE1",
                "LIVING_ROOM_SCENE2", "LIVING_ROOM_SCENE5", "LIVING_ROOM_SCENE6", "STUDY_SCENE1"}
RUNS = {"rat": [f"{E}/gpu01/v4/full4", f"{E}/cluster/full4-s1", f"{E}/cluster/full4-s2"],
        "ans": [f"{E}/gpu01/s8/full4ans", f"{E}/cluster/full4ans-s1", f"{E}/cluster/full4ans-s2"],
        "4b": [f"{E}/gpu01/s8/4b-full4-s{s}" for s in range(3)],
        "4b+s2": [f"{E}/gpu01/s8/4b-full4-s{s}-hum" for s in range(3)]}


def load(prefix, split):
    p = Path(f"{prefix}_{split}.jsonl")
    if not p.exists():
        return None
    rows = {r["id"]: r for r in map(json.loads, p.open())}
    if split == "human":
        for i, r in rows.items():
            r["target"] = META[i]["verdict"]
    return rows


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s[np.diff(np.sort(s), prepend=np.nan) == 0]):   # average ranks of ties
        ranks[s == v] = ranks[s == v].mean()
    npos = y.sum()
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def ys(rows, ids):
    return np.array([rows[i]["target"] == "wrong" for i in ids], int), np.array([rows[i]["p_wrong"] for i in ids])


def delta(a, b, ids, n=2000, seed=0):
    """AUROC(a) - AUROC(b) on the same rows and its paired 95% bootstrap interval."""
    y, sa = ys(a, ids)
    _, sb = ys(b, ids)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        k = rng.integers(0, len(ids), len(ids))
        if 0 < y[k].sum() < len(k):
            d.append(auroc(y[k], sa[k]) - auroc(y[k], sb[k]))
    return auroc(y, sa) - auroc(y, sb), np.percentile(d, 2.5), np.percentile(d, 97.5)


def scene(i):
    return re.search(r"((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", i).group(1)


def fmt(d):
    lo, hi = d[1], d[2]
    return f"{d[0]:+.3f}{'*' if lo > 0 or hi < 0 else ' '} [{lo:+.3f}, {hi:+.3f}]"


def main():
    hum_ids = sorted(META)
    nonpick = sorted(NONPICK)
    print("== per seed, generated only")
    for name in ("rat", "ans"):
        for p in RUNS[name]:
            l90, gen, hum = load(p, "l90val"), load(p, "gentest"), load(p, "human")
            sh = [i for i in l90 if scene(i) in TRAIN_SCENES]
            valid = [i for i in l90 if l90[i]["target"] == "right"]
            gv = [i for i in gen if gen[i]["target"] == "right"]
            hv = [i for i in hum_ids if hum[i]["target"] == "right"]
            print(f"{name} {Path(p).name:14s} l90 shared {auroc(*ys(l90, sh)):.3f}"
                  f"  l90 valid rejected {np.mean([l90[i]['p_wrong'] >= .5 for i in valid]):.3f}"
                  f"  gentest R_val {1 - np.mean([gen[i]['p_wrong'] >= .5 for i in gv]):.3f}"
                  f"  human valid accepted {1 - np.mean([hum[i]['p_wrong'] >= .5 for i in hv]):.3f}")
    print("== paired differences, seed i vs seed i")
    for a, b, sets in (("ans", "rat", ("human", "nonpick", "l90")), ("4b", "rat", ("human", "nonpick", "l90")),
                       ("ans", "4b", ("human", "nonpick")), ("ans", "4b+s2", ("human", "nonpick"))):
        for split in sets:
            ds = []
            for pa, pb in zip(RUNS[a], RUNS[b]):
                ra, rb = load(pa, "l90val" if split == "l90" else "human"), load(pb, "l90val" if split == "l90" else "human")
                ids = sorted(ra) if split == "l90" else (nonpick if split == "nonpick" else hum_ids)
                ds.append(delta(ra, rb, ids))
            print(f"{a} - {b} {split:8s} mean {np.mean([d[0] for d in ds]):+.3f} | " + " | ".join(map(fmt, ds)))
    print("== rejections by sketch type (valid arrow/non-pick, valid circle/pick, invalid arrow/non-pick, arrow/pick, circle)")
    types = {"V arrow nonpick": lambda r: r["verdict"] == "right" and r["has_arrow"] and not PICK(r["id"]),
             "V circle pick": lambda r: r["verdict"] == "right" and not r["has_arrow"] and PICK(r["id"]),
             "I arrow nonpick": lambda r: r["verdict"] == "wrong" and r["has_arrow"] and not PICK(r["id"]),
             "I arrow pick": lambda r: r["verdict"] == "wrong" and r["has_arrow"] and PICK(r["id"]),
             "I circle": lambda r: r["verdict"] == "wrong" and not r["has_arrow"]}
    for t, f in types.items():
        ids = [i for i, r in META.items() if f({**r, "id": i})]
        cells = []
        for name in ("rat", "ans"):
            cells.append(" / ".join(str(sum(load(p, "human")[i]["p_wrong"] >= .5 for i in ids)) for p in RUNS[name]))
        print(f"{t:16s} n={len(ids):3d}  rat {cells[0]:14s} ans {cells[1]}")
    print("== baselines")
    for name, p in (("zero-shot 4B", f"{E}/cluster/zeroshot-4B"), ("zero-shot 2B", f"{E}/gpu01/rep/zeroshot")):
        hum, l90 = load(p, "human"), load(p, "l90val")
        new = [i for i in l90 if scene(i) not in TRAIN_SCENES]
        print(f"{name}: l90 {auroc(*ys(l90, sorted(l90))):.3f} new {auroc(*ys(l90, new)):.3f}"
              f" human {auroc(*ys(hum, hum_ids)):.3f} nonpick {auroc(*ys(hum, nonpick)):.3f}")
    g = E / "cluster/geometry"
    for fit in ("gen", "hum", "gen+hum"):
        hum = {r["id"]: {**r, "target": META[r["id"]]["verdict"]} for r in map(json.loads, (g / f"{fit}__human.jsonl").open())}
        l90 = {r["id"]: r for r in map(json.loads, (g / f"{fit}__l90.jsonl").open())}
        new = [i for i in l90 if scene(i) not in TRAIN_SCENES]
        y, s = ys(hum, hum_ids)
        p = (s >= .5).astype(int)
        tp, fp, fn, tn = (p & y).sum(), (p & (1 - y)).sum(), ((1 - p) & y).sum(), ((1 - p) & (1 - y)).sum()
        mcc = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1)
        print(f"geometry {fit:8s}: l90 {auroc(*ys(l90, sorted(l90))):.3f} new {auroc(*ys(l90, new)):.3f}"
              f" human {auroc(y, s):.3f} nonpick {auroc(*ys(hum, nonpick)):.3f}"
              f" prec {tp / max(tp + fp, 1):.3f} rec {tp / (tp + fn):.3f} r_val {tn / (tn + fp):.3f} mcc {mcc:+.3f}")


if __name__ == "__main__":
    main()
