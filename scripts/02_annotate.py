#!/usr/bin/env python
"""Stage 02 (sim-free): clean circle+arrow -> K human-imprecision variants per demo.

Cheap and re-runnable: sweep imprecision configs without touching the simulator.
"""

import json
import zlib

from _common import base_parser, setup, iter_demo_dirs
from libero_sketch.strokes import sample_variant_pair


def main():
    p = base_parser("Sample augmented sketch variants from extracted metadata")
    p.add_argument("--variants", type=int, default=None, help="K variants per demo")
    args = p.parse_args()
    cfg, root, suites = setup(args)
    K = args.variants or cfg.annotate.variants_per_demo
    imp = json.loads(json.dumps(cfg["imprecision"]))  # plain dicts
    flavor = cfg.annotate.arrow_flavor

    for suite in suites:
        n = 0
        for task_dir, demo_dir in iter_demo_dirs(root, suite):
            meta = json.loads((demo_dir / "meta.json").read_text())
            circle = meta["circle_clean"]
            tgt = meta["points_px"]["target_t0"]
            H, W = meta["camera"]["H"], meta["camera"]["W"]
            target_xy = (tgt[0] / W, tgt[1] / H)

            variants = []
            base = zlib.crc32(f"{suite}/{task_dir.name}/{demo_dir.name}".encode()) + cfg.seed
            for j in range(K):
                arrow = _pick_arrow(meta, flavor, j, cfg.annotate.min_arrow_len)
                variants.append(sample_variant_pair(circle, arrow, target_xy,
                                                    seed=base + 31 * j, cfg=imp))
            (demo_dir / "annotations.json").write_text(json.dumps(
                {"clean": {"circle": circle,
                           "arrow_dest": meta["arrow_dest_clean"],
                           "arrow_motion": meta["arrow_motion_clean"]},
                 "variants": variants}, indent=1))
            n += 1
        print(f"[{suite}] annotated {n} demos ({K} variants each)")


def _pick_arrow(meta, flavor, j, min_len):
    dest, motion = meta["arrow_dest_clean"], meta["arrow_motion_clean"]
    if flavor == "motion":
        arrow = motion
    elif flavor == "both":
        arrow = (dest if j % 2 == 0 and dest else motion)
    else:
        arrow = dest or motion  # 'dest': falls back to motion for manipulate tasks
    # near-zero arrows (e.g. knob turns) are noise — circle-only is the honest prompt
    if arrow is not None:
        length = ((arrow["x1"] - arrow["x0"]) ** 2 + (arrow["y1"] - arrow["y0"]) ** 2) ** 0.5
        if length < min_len:
            return None
    return arrow


if __name__ == "__main__":
    main()
