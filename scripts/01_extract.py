#!/usr/bin/env python
"""Stage 01: replay demos, extract ground truth (needs sim; see setup_env.sh).

Examples:
  python 01_extract.py --suite spatial --tasks 0 --max-demos 2     # smoke run
  python 01_extract.py --suite all --data-root ~/libero_datasets
  MUJOCO_GL=egl python 01_extract.py --suite long --tasks 0,1,2    # shardable
"""

import json

from _common import base_parser, setup, data_root
from libero_sketch.extract import extract_suite


def main():
    p = base_parser("Extract ground truth from LIBERO demos")
    p.add_argument("--data-root", default=None, help="LIBERO datasets root")
    p.add_argument("--tasks", default=None, help="comma-separated task indices (shard filter)")
    p.add_argument("--max-demos", type=int, default=None, help="cap demos per task (smoke runs)")
    args = p.parse_args()
    cfg, root, suites = setup(args)
    droot = data_root(cfg, args.data_root)
    task_filter = [int(x) for x in args.tasks.split(",")] if args.tasks else None

    summaries = []
    for suite in suites:
        print(f"==== extracting suite '{suite}' ====")
        summaries.append(extract_suite(suite, droot, root, cfg, task_filter, args.max_demos))

    out = root / "extraction_summary.json"
    prev = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(prev + summaries, indent=1))
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
