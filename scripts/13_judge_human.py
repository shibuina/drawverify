#!/usr/bin/env python
"""Run the generator's own geometry on human sketches: what would stage 09 call each one?

Every human stroke is registered as a template on its scene's first frame (stage 08's
instance masks) and measured with the same `Ctx` / `resolve()` the generator labels with.
The output feeds the label-function ceiling of the paper (Table 3) and the thresholds of the
loose-circle and wandering-arrow modes.

  python scripts/13_judge_human.py --labelled output/human_ui/data/human/test-00000.parquet \
      --manifest .../human_eval/out/control-clean_arrow/human_sketches.jsonl --out gj_test.jsonl

Per row: selected objects, target and rival coverage (rival relative to the demo's target and,
for task-blind labels, relative to the selected object), circle looseness (circle area over
the selected object's visible mask), arrow wander (largest distance of the path from its
chord, over the chord), and where the head lands.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from libero_sketch import false_prompts as FP, strokes as S  # noqa: E402


def wander(path: np.ndarray) -> tuple[float, float]:
    """(largest distance of the path from its tail->head chord, path length), both over the chord."""
    q0, q1 = path[0], path[-1]
    ch = max(float(np.linalg.norm(q1 - q0)), 1e-6)
    u = (q1 - q0) / ch
    off = (path - q0) @ np.array([-u[1], u[0]])
    return float(np.max(np.abs(off)) / ch), float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) / ch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", type=Path, required=True, help="human set parquet (id, response_id, verdict, caption)")
    ap.add_argument("--manifest", type=Path, required=True, help="JSONL with the raw strokes")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = pq.read_table(a.labelled, columns=["id", "response_id", "verdict", "caption"]).to_pylist()
    man = {json.loads(l)["response_id"]: json.loads(l) for l in a.manifest.open()}
    dirs = {}
    for root in ("output/extracted", "output_l90/extracted"):
        for d in glob.glob(str(REPO / root / "*/*/demo_*")):
            p = Path(d)
            dirs[(p.parts[-3], p.parts[-2].split("_", 1)[1], p.name)] = p
    tpl = S.templates()
    with a.out.open("w") as out:
        for r in rows:
            m = man[r["response_id"]]
            d = dirs.get((m["suite"], m["task"], m["demo"]))
            rec = {"id": r["id"], "verdict": r["verdict"], "caption": r["caption"]}
            if d is None:
                rec["skip"] = "no scene"
                out.write(json.dumps(rec) + "\n")
                continue
            meta = json.loads((d / "meta.json").read_text())
            scene = json.loads((d / "scene.json").read_text())
            inst = cv2.imread(str(d / "frames/t0_objects.png"), cv2.IMREAD_GRAYSCALE)
            ctx = FP.Ctx(meta=meta, scene=scene, occupancy=None, caption_pool=[], ladder=None, instances=inst)
            c = arr = None
            cs, as_ = np.asarray(m.get("circle_stroke") or []), np.asarray(m.get("arrow_stroke") or [])
            if len(cs) >= 3 and m.get("circle"):
                cc = m["circle"]
                tpl.append({"circle": ((cs - [cc["cx"], cc["cy"]]) / cc["r"]).tolist()})
                c = {"kind": "circle", "tpl": len(tpl) - 1, "cx": cc["cx"], "cy": cc["cy"], "r": cc["r"],
                     "ecc": 1.0, "rot": 0.0}
            if len(as_) >= 3:
                q0, q1 = as_[0], as_[-1]
                ch = max(float(np.linalg.norm(q1 - q0)), 1e-6)
                u = (q1 - q0) / ch
                rot = np.array([[u[0], u[1]], [-u[1], u[0]]])
                tpl.append({"arrow": ((rot @ (as_ - q0).T).T / ch).tolist()})
                arr = {"kind": "arrow", "tpl": len(tpl) - 1, "x0": q0[0], "y0": q0[1], "x1": q1[0], "y1": q1[1],
                       "width": S.UI["width"]}
            t = ctx.obj(ctx.target_object or "")
            sel = ctx.selected(c) if c else []
            rec.update(task=m["task"], target=ctx.target_object, dest=ctx.dest_object,
                       selected=[o["name"] for o in sel])
            if c is not None:
                rec.update(target_cov=round(ctx.target_coverage(c), 3),
                           rival_cov=round(FP.rival_coverage(ctx, c), 3),
                           fixture_share=round(max([ctx.disc_share(c, o) for o in ctx.visible()
                                                    if o is not t and o.get("fixture") and t is not None
                                                    and not FP._supports(o, t)] or [0]), 3))
                if len(sel) == 1:                   # task-blind: rivals of the object actually circled
                    s0 = sel[0]
                    rec["rival_sel"] = round(max([ctx.coverage(c, o) for o in ctx.visible()
                                                  if o is not s0 and ctx.can_act_on(o) and not FP._supports(o, s0)]
                                                 or [0.0]), 3)
                    H, W = inst.shape
                    ys, xs = np.mgrid[0:H, 0:W]
                    area = float(np.sum(S.circle_contains(c, ((xs + 0.5) / W, (ys + 0.5) / H))))
                    rec["loose"] = round(area / max(len(ctx._px.get(s0["name"], ())), 1), 2)
            if c is not None or arr is not None:
                n_ref, n_dst, _ = FP.resolve(ctx, c, arr, "L3", caption=r["caption"])
                rec.update(n_ref=n_ref, n_dst=n_dst)
            if arr is not None:
                refs = {o["name"] for o in sel} or {ctx.target_object}
                hit = [o["name"] for o in ctx.indicated((arr["x1"], arr["y1"]), exclude=refs)]
                dest = ctx.obj(ctx.dest_object or "")
                # an object with neither a visible mask nor a mask circle has no distance
                md = lambda o, pt: round(ctx.mask_dist(o, pt), 3) if o and (o["name"] in ctx._px or o.get("circle")) else None
                w, pr = wander(as_)
                rec.update(head_hits=hit,
                           head_dist=md(dest, (arr["x1"], arr["y1"])),
                           tail_in_circle=bool(c and S.circle_contains(c, (arr["x0"], arr["y0"]))),
                           tail_dist=md(t, (arr["x0"], arr["y0"])),
                           head_edge=round(min(arr["x1"], arr["y1"], 1 - arr["x1"], 1 - arr["y1"]), 3),
                           wander=round(w, 3), path_ratio=round(pr, 3))
                obj = sel[0]["name"] if len(sel) == 1 else ctx.target_object
                places = {o["name"] for o in FP.sensible_dests(ctx, obj or "")} | \
                    ({ctx.dest_object} if obj == ctx.target_object else set())
                rec["head_on_sensible"] = bool(hit) and set(hit) <= places and len(hit) == 1
            out.write(json.dumps(rec) + "\n")
    print(f"{len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
