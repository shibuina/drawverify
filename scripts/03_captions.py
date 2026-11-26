#!/usr/bin/env python
"""Stage 03 (sim-free): degraded-caption ladder per task, templates + optional LLM.

Set GEMINI_API_KEY or OPENAI_API_KEY (env or data_gen/.env) for --provider gemini|openai;
--provider none produces the template-only ladder.
"""

import json

from _common import base_parser, setup, iter_demo_dirs
from libero_sketch.captions import build_captions
from libero_sketch.llm import LLMClient


def main():
    p = base_parser("Generate degraded captions (L0-L3)")
    p.add_argument("--provider", default=None, choices=["gemini", "openai", "none"])
    p.add_argument("--model", default=None)
    p.add_argument("--n-paraphrases", type=int, default=None)
    args = p.parse_args()
    cfg, root, suites = setup(args)

    client = LLMClient(args.provider or cfg.captions.provider,
                       args.model or cfg.captions.model,
                       cache_dir=root / "llm_cache")
    n_para = args.n_paraphrases if args.n_paraphrases is not None else cfg.captions.n_paraphrases

    for suite in suites:
        cap_dir = root / "captions" / suite
        cap_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for task_dir, demo_dir in iter_demo_dirs(root, suite):
            if task_dir.name in seen:
                continue
            seen.add(task_dir.name)
            meta = json.loads((demo_dir / "meta.json").read_text())
            caps = build_captions(
                meta["instruction"], meta["goal"]["kind"], meta["goal"]["target_object"],
                meta["goal"]["destination"], meta.get("inventory", []),
                client, n_para)
            (cap_dir / f"{task_dir.name}.json").write_text(json.dumps(caps, indent=1))
        print(f"[{suite}] captions for {len(seen)} tasks "
              f"(llm calls={client.stats['calls']}, cache hits={client.stats['cache_hits']})")


if __name__ == "__main__":
    main()
