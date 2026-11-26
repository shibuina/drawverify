#!/usr/bin/env python
"""Stage 00: download LIBERO demonstration datasets from the HuggingFace mirror.

Calls snapshot_download directly instead of LIBERO's wrapper: LIBERO's downloader passes
force_download=True (no resume — reruns start from zero) and its overwrite prompt deletes
partial data. This path resumes, skips completed files, parallelizes, and uses the
hf_transfer Rust backend when installed. The HF repo has a separate libero_10 folder, so
`--suite long` avoids downloading the complete libero_100 bundle. libero_90 is excluded.

Fallback: --original-links shells out to LIBERO's own script (interactive, not resumable).
"""

import os
import subprocess
import sys

from _common import base_parser, setup, data_root, DATA_GEN_ROOT
from libero_sketch.suites import dataset_dir, download_group

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def main():
    p = base_parser("Download LIBERO datasets")
    p.add_argument("--data-root", default=None, help="where datasets are stored")
    p.add_argument("--max-workers", type=int, default=16,
                   help="parallel file downloads (default 16)")
    p.add_argument("--original-links", action="store_true",
                   help="use LIBERO's original download links instead of HuggingFace")
    args = p.parse_args()
    cfg, _, suites = setup(args)
    target = data_root(cfg, args.data_root)
    target.mkdir(parents=True, exist_ok=True)

    if args.original_links:
        return download_original(suites, target)

    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        print("hf_transfer enabled")
    except ImportError:
        print("hf_transfer not installed — using default downloader "
              "(pip install hf_transfer for a faster one)")

    from huggingface_hub import snapshot_download
    patterns = [f"{dataset_dir(m)}/*" for m in suites]
    print(f"== downloading {patterns} -> {target} (resumable)", flush=True)
    snapshot_download(
        repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(target),
        allow_patterns=patterns, max_workers=args.max_workers)
    for m in suites:
        n = len(list((target / dataset_dir(m)).glob("*.hdf5")))
        print(f"[{m}] {n} task files in {target / dataset_dir(m)}")
    print("done.")


def download_original(suites, target):
    dl_script = DATA_GEN_ROOT / "third_party" / "LIBERO" / "benchmark_scripts" / \
        "download_libero_datasets.py"
    for g in sorted({download_group(m) for m in suites}):
        cmd = [sys.executable, str(dl_script), "--datasets", g,
               "--download-dir", str(target)]
        r = subprocess.run(cmd, input="y\n", text=True)
        if r.returncode != 0:
            sys.exit(f"downloader failed for {g}; run manually:\n  {' '.join(cmd)}")


if __name__ == "__main__":
    main()
