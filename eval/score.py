#!/usr/bin/env python3
"""One table from prediction files, in the layout of paper Table IV.

Each file has one JSON line per row: id, target (right/wrong), p_wrong (or p_fail in [0, 100]
for API models) and optionally prediction (right/wrong; default p_wrong >= 0.5). Only rows
present in every file are scored, so paired differences are on identical rows.

  python eval/score.py --pred "Gemini-3.8-Flash=api/gemini.jsonl" "4B gen->hum=output/eval/4b.jsonl" \\
      --ref Gemini-3.8-Flash --families \\
      --cut "4B gen->hum=output/eval/4b_human_train.jsonl@0.85"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import bootstrap_delta_auroc, metrics, threshold_for_recall  # noqa: E402

FAMILY = {"A": "Ref.", "B": "Dir.", "C": "X-mod./Gr.", "D": "Lang."}
GROUNDING = {"no_sketch", "dest_open", "referent_twin", "dest_between", "clean_unresolved"}


def load(path: str) -> dict[str, dict]:
    rows = {}
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        s = r["p_wrong"] if "p_wrong" in r else r["p_fail"] / 100
        pred = r.get("prediction")
        rows[r["id"]] = {"mode": r.get("mode"), "y": int(r["target"] == "wrong"), "s": float(s),
                         "pred": int(pred == "wrong") if pred else int(s >= 0.5)}
    return rows


def family(row_id: str, mode: str | None) -> str:
    """Paper Table IV(e) families; Withh. = invalid rows without a released mode (in the
    generated data: `deixis_resolved` attempts that still left more than one reading)."""
    m = mode or (re.search(r"/([^/#]+)#\d+$", row_id) or [None, None])[1]
    if m in GROUNDING:
        return "X-mod./Gr."
    return FAMILY.get(m[0], "Withh.") if m and re.fullmatch(r"[A-D]\d", m) else "Withh."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--ref", help="model the paired AUROC difference is taken against")
    ap.add_argument("--cut", nargs="*", default=[], metavar="NAME=CAL.jsonl@RECALL",
                    help="extra row: NAME re-decided at the threshold giving RECALL on CAL")
    ap.add_argument("--families", action="store_true", help="recall on invalid rows per family")
    a = ap.parse_args()

    preds = dict(spec.split("=", 1) for spec in a.pred)
    preds = {name: load(path) for name, path in preds.items()}
    ids = sorted(set.intersection(*(set(p) for p in preds.values())))
    for spec in a.cut:
        name, rest = spec.split("=", 1)
        cal_path, recall = rest.rsplit("@", 1)
        cal = list(load(cal_path).values())
        t = threshold_for_recall([r["y"] for r in cal], [r["s"] for r in cal], float(recall))
        preds[f"{name} @{recall} (t={t:.3f})"] = {
            i: {**r, "pred": int(r["s"] >= t)} for i, r in preds[name].items()}
    y = [preds[next(iter(preds))][i]["y"] for i in ids]
    print(f"{len(ids)} rows, {sum(y)} invalid ({sum(y) / len(ids):.1%})\n")
    cols = ("acc", "prec", "rec", "mcc", "r_val", "auroc")
    print(f"{'model':<34}" + "".join(f"{c:>8}" for c in cols) + (f"{'delta':>24}" if a.ref else ""))
    for name, p in preds.items():
        m = metrics(y, [p[i]["pred"] for i in ids], [p[i]["s"] for i in ids])
        line = f"{name:<34}" + "".join(f"{m[c]:>8.3f}" for c in cols)
        if a.ref and name != a.ref:
            d, lo, hi = bootstrap_delta_auroc(y, [p[i]["s"] for i in ids],
                                              [preds[a.ref][i]["s"] for i in ids])
            line += f"{d:>+8.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"
        print(line)
    if a.families:
        first = preds[next(iter(preds))]
        fams = {i: family(i, first[i]["mode"]) for i in ids if first[i]["y"]}
        names = sorted(set(fams.values()))
        print(f"\nrecall on invalid by family ({', '.join(f'{f} {list(fams.values()).count(f)}' for f in names)})")
        for name, p in preds.items():
            cells = [sum(p[i]["pred"] for i in fams if fams[i] == f) / list(fams.values()).count(f)
                     for f in names]
            print(f"{name:<34}" + "".join(f"{f:>12} {c:.2f}" for f, c in zip(names, cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
