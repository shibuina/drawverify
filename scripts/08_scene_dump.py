#!/usr/bin/env python
"""Stage 08: dump per-demo scene geometry (needs sim, like stage 01).

False prompts must reference objects other than the target, which stage 01 never persisted.
This adds scene.json + frames/t0_occupancy.png next to each extracted demo.

  MUJOCO_GL=egl python 08_scene_dump.py --suite spatial --tasks 0 --max-demos 3
"""

import json

from _common import base_parser, data_root, setup
from libero_sketch.scene import dump_suite_scenes


def main():
    p = base_parser("Dump per-demo scene geometry for false-prompt generation")
    p.add_argument("--data-root", default=None, help="LIBERO datasets root")
    p.add_argument("--tasks", default=None, help="comma-separated task indices")
    p.add_argument("--max-demos", type=int, default=None, help="cap demos per task")
    args = p.parse_args()
    cfg, root, suites = setup(args)
    droot = data_root(cfg, args.data_root)
    task_filter = [int(x) for x in args.tasks.split(",")] if args.tasks else None

    summaries = []
    for suite in suites:
        print(f"==== scene dump '{suite}' ====")
        summaries.append(dump_suite_scenes(suite, droot, root, cfg, task_filter,
                                           args.max_demos))
    (root / "scene_dump_summary.json").write_text(json.dumps(summaries, indent=1))
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
