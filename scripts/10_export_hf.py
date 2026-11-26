#!/usr/bin/env python
"""Stage 10 (sim-free): split the prompt-validation samples into datasets and export to a
HuggingFace-loadable layout (parquet + dataset card), optionally pushing to the Hub.

"Different datasets" are HF **configs** (subsets) inside one repo, so they load with
`datasets.load_dataset(repo, "<config>")`. Split granularity is your choice:

  --split-by family   A/B/C/D + grounding + positive (6 configs, default)
  --split-by mode     A2, A3, ... D4, POS         (fine-grained)
  --split-by verdict  right / wrong                (the binary label)

Images are embedded as an HF Image column by default (`--images embed`), or dropped for a
tiny metadata-only dataset (`--images none`) since every sketch re-renders deterministically
from the row's prompt fields.

  python scripts/10_export_hf.py --split-by family
  python scripts/10_export_hf.py --split-by family --push --repo shibuina/drawvla-... --private

Output: output/hf/<repo-name>/  (data/<config>/<split>-*.parquet + README.md).
"""

import argparse
import io
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from _common import DATA_GEN_ROOT
from libero_sketch.suites import resolve_suites

FAMILY_CONFIG = {"A": "referential", "B": "directional", "C": "crossmodal",
                 "D": "language", "G": "grounding",
                 "P": "positive"}


def split_of(demo_id: str) -> str:
    """Deterministic 80/10/10 split keyed on the full suite/task/demo identity.

    The bare `demo` field ("demo_0") repeats across tasks, so hashing it alone groups every
    task's demo_0 into one split. The full path gives stable approximate splits where no
    demo straddles train, validation, or test, regardless of which suites are enabled.
    """
    bucket = zlib.crc32(demo_id.encode()) % 10
    return "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def label_reason(rec: dict) -> str:
    """Explain the binary label using only generic, auditable failure criteria."""
    labels = rec["labels"]
    if rec["verdict"] == "right":
        return (
            "right because the circle, arrow, and caption are mutually consistent and "
            "the rendered prompt leaves exactly one actionable reading"
        )

    failures = []
    if not labels.get("circle_ok", True):
        failures.append("the circle does not identify the intended referent")
    if not labels.get("arrow_ok", True):
        failures.append("the arrow does not identify the intended destination or motion")
    if not labels.get("caption_ok", True):
        failures.append("the caption does not express the intended action")
    if not labels.get("consistency_ok", True):
        failures.append("the visual and language channels are inconsistent")
    if not labels.get("sufficient_ok", True):
        n_readings = int(rec["resolution"]["n_readings"])
        failures.append(
            f"the complete prompt remains under-specified with {n_readings} possible "
            f"reading{'s' if n_readings != 1 else ''}"
        )
    if not failures:
        raise ValueError(f"wrong row has no auditable failure reason: {rec.get('id')}")
    return "wrong because " + "; ".join(failures)


def config_key(rec: dict, by: str) -> str:
    if by == "family":
        return FAMILY_CONFIG.get(rec["category"], rec["category"])
    if by == "mode":
        return rec.get("slot") or rec["mode"]
    if by == "verdict":
        return rec["verdict"]
    raise ValueError(by)


def _embedded_image(
    path: Path, image_format: str, jpeg_quality: int
) -> tuple[bytes, str]:
    """Read an image, optionally converting bulky lossless PNGs for hosted training."""
    if image_format == "source":
        return path.read_bytes(), path.name
    output = io.BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
        )
    return output.getvalue(), f"{path.stem}.jpg"


def _row(
    rec: dict,
    root: Path,
    embed_images: bool,
    image_format: str = "jpeg",
    jpeg_quality: int = 95,
    split: str = "train",
) -> dict:
    resolution = rec["resolution"]
    rendered = rec.get("rendered_resolution", {})
    has_reason = split in {"train", "validation"}
    row = {
        "id": rec["id"], "suite": rec["suite"], "task": rec["task"], "demo": rec["demo"],
        "mode": rec["mode"] if has_reason else None,
        "slot": rec.get("slot", rec["mode"]) if has_reason else None,
        "category": rec["category"] if has_reason else None,
        "category_name": rec["category_name"] if has_reason else None,
        "tier": rec["tier"] if has_reason else None,
        "solution": rec["solution"] if has_reason else None,
        "data_amount": rec["data_amount"] if has_reason else None,
        "verdict": rec["verdict"], "right": bool(rec["labels"]["right"]),
        "reason_available": has_reason,
        "label_reason": label_reason(rec) if has_reason else None,
        "circle_ok": bool(rec["labels"]["circle_ok"]) if has_reason else None,
        "arrow_ok": bool(rec["labels"]["arrow_ok"]) if has_reason else None,
        "caption_ok": bool(rec["labels"]["caption_ok"]) if has_reason else None,
        "consistency_ok": bool(rec["labels"]["consistency_ok"]) if has_reason else None,
        "sufficient_ok": (
            bool(rec["labels"].get("sufficient_ok", True)) if has_reason else None
        ),
        "severity": float(rec["severity"]) if has_reason else None,
        "caption": rec["prompt"]["caption"], "caption_level": resolution["caption_level"],
        "n_referents": resolution["n_referents"] if has_reason else None,
        "n_destinations": resolution["n_destinations"] if has_reason else None,
        "n_readings": resolution["n_readings"] if has_reason else None,
        "resolved_by": resolution.get("resolved_by", "") if has_reason else None,
        "grounding_kind": resolution.get("kind", "") if has_reason else None,
        "rendered_n_referents": (
            rendered.get("n_referents", resolution["n_referents"]) if has_reason else None
        ),
        "rendered_n_destinations": (
            rendered.get("n_destinations", resolution["n_destinations"])
            if has_reason else None
        ),
        "rendered_n_readings": (
            rendered.get("n_readings", resolution["n_readings"]) if has_reason else None
        ),
        "render_verified": bool(rendered.get("verified", False)) if has_reason else None,
        "circle_json": json.dumps(rec["prompt"]["circle"]),
        "arrow_json": json.dumps(rec["prompt"]["arrow"]),
        "note": rec["note"] if has_reason else None,
        "source_frame": rec.get("source_frame", ""),
        "name_explicit": bool(rec["name_explicit"]),
    }
    if embed_images:
        img_path = root / rec["render"] if rec.get("render") else None
        data, filename = (
            _embedded_image(img_path, image_format, jpeg_quality)
            if img_path and img_path.exists()
            else (None, None)
        )
        # HF `Image` feature = struct{bytes, path}; the viewer renders it inline
        row["image"] = {"bytes": data, "path": filename}
    return row


def _schema(embed_images: bool) -> pa.Schema:
    f = [
        ("id", pa.string()), ("suite", pa.string()), ("task", pa.string()),
        ("demo", pa.string()), ("mode", pa.string()), ("slot", pa.string()),
        ("category", pa.string()), ("category_name", pa.string()), ("tier", pa.string()),
        ("solution", pa.string()), ("data_amount", pa.string()), ("verdict", pa.string()),
        ("right", pa.bool_()), ("reason_available", pa.bool_()),
        ("label_reason", pa.string()), ("circle_ok", pa.bool_()), ("arrow_ok", pa.bool_()),
        ("caption_ok", pa.bool_()), ("consistency_ok", pa.bool_()),
        ("sufficient_ok", pa.bool_()), ("severity", pa.float32()),
        ("caption", pa.string()), ("caption_level", pa.string()),
        ("n_referents", pa.int32()), ("n_destinations", pa.int32()),
        ("n_readings", pa.int32()), ("resolved_by", pa.string()),
        ("grounding_kind", pa.string()), ("circle_json", pa.string()),
        ("rendered_n_referents", pa.int32()), ("rendered_n_destinations", pa.int32()),
        ("rendered_n_readings", pa.int32()), ("render_verified", pa.bool_()),
        ("arrow_json", pa.string()), ("note", pa.string()), ("source_frame", pa.string()),
        ("name_explicit", pa.bool_()),
    ]
    if embed_images:
        f.append(("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])))
    return pa.schema(f)


def dataset_card(
    repo: str, by: str, configs: dict, counts: dict, embed: bool, image_format: str,
    note: str = "",
) -> str:
    """README.md with the HF `configs:` block that makes each subset a loadable config."""
    image_note = (
        f"Embedded images use {image_format.upper()} encoding."
        if embed
        else "This metadata-only export does not embed images."
    )
    yaml_cfgs = []
    for cfg in sorted(configs):
        yaml_cfgs.append(f"- config_name: {cfg}\n  data_files:\n"
                         f"  - split: train\n    path: data/{cfg}/train-*.parquet\n"
                         f"  - split: validation\n    path: data/{cfg}/validation-*.parquet\n"
                         f"  - split: test\n    path: data/{cfg}/test-*.parquet")
    feats = ("image, " if embed else "") + ("id, mode, category, tier, solution, verdict, "
             "right, reason_available, label_reason, "
             "circle_ok/arrow_ok/caption_ok/consistency_ok/sufficient_ok, severity, "
             "caption, caption_level, n_referents/n_destinations/n_readings, resolved_by, "
             "grounding_kind, rendered_n_readings/render_verified, circle_json, arrow_json, "
             "name_explicit, note")
    rows = "\n".join(f"| `{c}` | {counts[c]['train']:,} | "
                     f"{counts[c]['validation']:,} | {counts[c]['test']:,} | "
                     f"{counts[c]['n']:,} | {counts[c]['right']:,} / {counts[c]['wrong']:,} |"
                     for c in sorted(configs))
    total = sum(c["n"] for c in counts.values())
    return f"""---
license: mit
task_categories:
- image-classification
- robotics
tags:
- vision-language-action
- robot-manipulation
- libero
- prompt-validation
- sketch
pretty_name: DrawVerify Prompt-Validation
configs:
{chr(10).join(yaml_cfgs)}
---

# DrawVerify — Sketch-Prompt Validation

Circle (**which**) + arrow (**where**) + caption (**what**) visual instructions overlaid on
[LIBERO](https://libero-project.github.io/) observations, each labelled with a **binary
verdict** for training a prompt validator:

- **`right`** — every channel is correct and exactly one reading survives; execute.
- **`wrong`** — a channel is incorrect or the deictic prompt remains under-determined;
  reject. Formerly ambiguous prompts are retained in this class.

All captions are name-free L2/L3 deixis (for example, "move this there"). Object identity
must come from the circle/arrow sketch, not an explicit noun. `n_readings = n_referents x
n_destinations` is retained as diagnostic metadata. {image_note}

Train and validation rows include `label_reason`, a human-readable explanation derived
from the channel checks and declared grounding count. The demo-disjoint test split keeps
its ground-truth `verdict` for scoring but sets `reason_available=false` and nulls all
reason/rationale fields (`label_reason`, `note`, mode/category, failure-channel flags,
severity, and resolution diagnostics), preventing rationale leakage during final testing.

Generated from the failure taxonomy (`failure_modes.md`). Split into **{len(configs)}
configs** by `{by}`; each loads independently:

```python
from datasets import load_dataset
ds = load_dataset("{repo}", "{sorted(configs)[0]}", split="train")
```

## Configs ({total:,} samples)

| config | train | validation | test | total | right / wrong |
|---|---:|---:|---:|---:|---:|
{rows}

## Fields

{feats}. The sketch geometry (`circle_json`, `arrow_json`, image-plane, normalized) lets
you re-render at any resolution. This release samples only L2 (relational deixis) and L3
(pure deixis); L0/L1 are excluded because they explicitly name the task or object category.
Splits are approximately 80/10/10 by demo; no demo straddles train, validation, or test.

Built from LIBERO demonstrations with the DrawVerify generator (stages 01, 08, 09, 10).
{note}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-by", default="family", choices=["family", "mode", "verdict"])
    p.add_argument("--images", default="embed", choices=["embed", "none"])
    p.add_argument(
        "--image-format",
        default="jpeg",
        choices=["jpeg", "source"],
        help="embedded encoding; JPEG greatly reduces Hub/download size",
    )
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--suite", default="all", help="which suites to include")
    p.add_argument("--split-as", choices=["validation", "test"], default=None,
                   help="put every row in this split (required for suite 90, held out of training)")
    p.add_argument("--out-root", default=str(DATA_GEN_ROOT / "output"))
    p.add_argument("--repo", default="shibuina/drawvla-prompt-validation")
    p.add_argument("--push", action="store_true", help="upload to the HF Hub")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--rows-per-shard", type=int, default=4000)
    p.add_argument("--card-note", default="",
                   help="extra markdown appended to the dataset card (e.g. rendering style)")
    args = p.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        p.error("--jpeg-quality must be between 1 and 100")

    root = Path(args.out_root)
    # Use the shared suite registry rather than accepting arbitrary output directories.
    # In particular, this prevents stale output/false_prompts/90 artifacts from being
    # exported after libero_90 was removed from dataset generation.
    suites = resolve_suites(args.suite)
    if "90" in suites and args.split_as is None:
        raise SystemExit("libero_90 never enters training: export it with --split-as validation")
    embed = args.images == "embed"
    repo_name = args.repo.split("/")[-1]
    export = root / "hf" / repo_name
    if export.exists():
        import shutil
        shutil.rmtree(export)

    # bucket rows by (config, split)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    n_total = 0
    for suite in suites:
        sp = root / "false_prompts" / suite / "samples.jsonl"
        if not sp.exists():
            print(f"[skip] {sp} missing")
            continue
        with sp.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("verdict") not in {"right", "wrong"}:
                    raise ValueError(
                        f"{sp} contains legacy verdict {rec.get('verdict')!r}; rerun stage "
                        "09 to create the binary deictic-only dataset before exporting"
                    )
                if rec.get("resolution", {}).get("caption_level") not in {"L2", "L3"}:
                    raise ValueError(
                        f"{rec.get('id', '<unknown>')} has a name-explicit caption level; "
                        "only L2/L3 rows may be exported"
                    )
                if rec.get("name_explicit") is not False:
                    raise ValueError(
                        f"{rec.get('id', '<unknown>')} lacks the verified name-free marker; "
                        "rerun stage 09 before exporting"
                    )
                if rec.get("category") == "P" and rec["verdict"] != "right":
                    raise ValueError(f"non-right row in positive family: {rec['id']}")
                cfg = config_key(rec, args.split_by)
                split = args.split_as or split_of(f"{rec['suite']}/{rec['task']}/{rec['demo']}")
                buckets[(cfg, split)].append(
                    _row(
                        rec,
                        root,
                        embed,
                        args.image_format,
                        args.jpeg_quality,
                        split,
                    )
                )
                counts[cfg]["n"] += 1
                counts[cfg][rec["verdict"]] += 1
                counts[cfg][split] += 1
                n_total += 1

    schema = _schema(embed)
    configs = sorted({c for c, _ in buckets})
    for cfg in configs:
        cfg_dir = export / "data" / cfg
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            rows = buckets.get((cfg, split), [])
            # Keep every generated config loadable for every split, even an empty one.
            chunks = [rows[i:i + args.rows_per_shard]
                      for i in range(0, len(rows), args.rows_per_shard)] or [[]]
            for shard, chunk in enumerate(chunks):
                tbl = pa.Table.from_pylist(chunk, schema=schema)
                pq.write_table(tbl, cfg_dir / f"{split}-{shard:05d}.parquet")
    (export / "README.md").write_text(
        dataset_card(
            args.repo,
            args.split_by,
            dict.fromkeys(configs),
            counts,
            embed,
            args.image_format,
            note=args.card_note,
        ))

    gb = sum(f.stat().st_size for f in export.rglob("*.parquet")) / 1e9
    print(f"exported {n_total:,} rows into {len(configs)} configs ({args.split_by}), "
          f"{gb:.2f} GB, images={'embedded' if embed else 'none'} -> {export}")
    for c in configs:
        print(f"  {c:14s} {counts[c]['n']:>7,}  "
              f"(train {counts[c]['train']}, validation {counts[c]['validation']}, "
              f"test {counts[c]['test']}; right {counts[c]['right']}, "
              f"wrong {counts[c]['wrong']})")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=args.private,
                        exist_ok=True)
        print(f"uploading to https://huggingface.co/datasets/{args.repo} ...")
        # The shard count changes as the corpus grows. Remove prior data shards in the
        # same atomic commit so old glob-matched parquet files cannot leak into a new
        # dataset revision. README.md is replaced by the uploaded folder normally.
        api.upload_folder(
            folder_path=str(export),
            repo_id=args.repo,
            repo_type="dataset",
            delete_patterns="data/**",
            commit_message="Publish regenerated binary prompt-validation dataset",
        )
        print("pushed.")
    else:
        print("dry run (no --push). Inspect output/hf/, then re-run with --push.")


if __name__ == "__main__":
    main()
