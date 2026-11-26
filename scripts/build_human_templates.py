#!/usr/bin/env python
"""Build the human stroke templates the generator draws with (`imprecision.templates`).

Each template is one study drawing, reduced to geometry relative to its scene's clean
annotation, so it can be replayed on any other scene:

  circle      the pointer path, relative to the drawer's own fitted circle (unit radius)
  circle_off  where that circle sat relative to the clean one: centre offset / clean r, r ratio
  arrow       the pointer path, mapped so its first point is (0, 0) and its last (1, 0)
  tail_off    first arrow point minus the clean tail, in the clean arrow's frame (along, across)
  head_off    last arrow point minus the clean head, same frame

Only drawings whose geometry matched their scene (`agreement.sketch_ok`) are kept, so the
offsets are good-faith human sloppiness, never a mistake. Participants who appear in an
evaluation manifest are excluded, so no evaluated drawer's hand is in the training data.

  python scripts/build_human_templates.py \
      --responses '../../SMU_Research/ICRA_26/DrawVLA/human_eval/out/archive/*.jsonl' \
      --exclude ../../SMU_Research/ICRA_26/DrawVLA/human_eval/out/control-clean_arrow/human_sketches.jsonl
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "libero_sketch" / "assets" / "human_strokes.json"


def frame(a: dict):
    p0, p1 = np.array([a["x0"], a["y0"]]), np.array([a["x1"], a["y1"]])
    u = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)
    return p0, p1, u, np.array([-u[1], u[0]])


def template(r: dict) -> dict | None:
    cs, as_ = np.asarray(r.get("circle_stroke") or []), np.asarray(r.get("arrow_stroke") or [])
    c, cc, ac = r.get("circle"), r.get("circle_clean"), r.get("arrow_clean")
    if len(cs) < 8 or len(as_) < 8 or not (c and cc and ac and r.get("arrow")):
        return None
    q0, q1 = as_[0], as_[-1]
    chord = float(np.linalg.norm(q1 - q0))
    if chord < 0.04:                                   # a dab, not an arrow
        return None
    u = (q1 - q0) / chord
    rot = np.array([[u[0], u[1]], [-u[1], u[0]]])     # chord -> +x
    p0, p1, cu, cv = frame(ac)
    return {
        "circle": np.round((cs - [c["cx"], c["cy"]]) / c["r"], 4).tolist(),
        "circle_off": [round((c["cx"] - cc["cx"]) / cc["r"], 4),
                       round((c["cy"] - cc["cy"]) / cc["r"], 4), round(c["r"] / cc["r"], 4)],
        "arrow": np.round((rot @ (as_ - q0).T).T / chord, 4).tolist(),
        "tail_off": [round(float((q0 - p0) @ cu), 4), round(float((q0 - p0) @ cv), 4)],
        "head_off": [round(float((q1 - p1) @ cu), 4), round(float((q1 - p1) @ cv), 4)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True, help="glob of study response JSONL files")
    ap.add_argument("--exclude", nargs="*", default=[], help="eval manifests: drop their drawers")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    held = {json.loads(l)["participant_id"] for f in a.exclude for l in open(f)}
    seen, out, why = set(), [], {"eval_participant": 0, "timed_out": 0, "not_ok": 0,
                                 "unusable": 0, "duplicate": 0}
    for f in sorted(glob.glob(a.responses)):
        for line in open(f):
            r = json.loads(line)
            key = hashlib.md5(json.dumps([r.get("circle_stroke"), r.get("arrow_stroke")])
                              .encode()).hexdigest()
            if key in seen:
                why["duplicate"] += 1
                continue
            seen.add(key)
            if r.get("participant_id") in held:
                why["eval_participant"] += 1
            elif r.get("timed_out"):
                why["timed_out"] += 1
            elif not (r.get("agreement") or {}).get("sketch_ok"):
                why["not_ok"] += 1
            elif (t := template(r)) is None:
                why["unusable"] += 1
            else:
                out.append(t)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n": len(out), "dropped": why, "templates": out}))
    print(f"{len(out)} templates -> {a.out}; dropped {why}")


if __name__ == "__main__":
    main()
