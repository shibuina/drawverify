"""Shared CLI plumbing for the numbered pipeline scripts."""

import argparse
import sys
from pathlib import Path

DATA_GEN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_GEN_ROOT))

from libero_sketch.config import load_config, data_root  # noqa: E402,F401  (data_root is re-exported)
from libero_sketch.suites import resolve_suites, add_suite_arg  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    add_suite_arg(p)
    p.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    p.add_argument("--out-root", default=None, help="pipeline output root")
    p.add_argument("--seed", type=int, default=None)
    return p


def setup(args):
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    root = Path(args.out_root) if args.out_root else (DATA_GEN_ROOT / cfg.paths.out_root)
    root.mkdir(parents=True, exist_ok=True)
    return cfg, root, resolve_suites(args.suite)


def iter_demo_dirs(out_root_p: Path, suite: str):
    ext = out_root_p / "extracted" / suite
    for task_dir in sorted(ext.glob("[0-9]*_*")):
        for demo_dir in sorted(task_dir.glob("demo_*")):
            if (demo_dir / "meta.json").exists():
                yield task_dir, demo_dir
