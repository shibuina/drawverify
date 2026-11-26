#!/usr/bin/env python
"""Stage 09 (sim-free): generate binary right/wrong deictic prompt data.

Implements the failure taxonomy in ../failure_modes.md (paper Table I). Needs stage 01 (meta.json),
stage 08 (scene.json + occupancy) and, for the language modes, stage 03 captions.

  python 09_false_prompts.py --suite spatial --tasks 2 --max-demos 3 --per-mode 1
  python 09_false_prompts.py --suite spatial --modes A3,A5,B1 --per-mode 2

Writes output/false_prompts/<suite>/:
  samples.jsonl            one labelled record per generated prompt
  images/<task>/<demo>/<MODE>_<i>.png    rendered prompt (hand-drawn strokes by default)
  summary.json             per-mode yield / applicability
"""

import json
import shutil
import zlib
from collections import Counter, defaultdict

import numpy as np

from _common import base_parser, iter_demo_dirs, setup
from libero_sketch import false_prompts as FP
from libero_sketch import render as R


def main():
    p = base_parser("Generate binary name-free prompt-validation data")
    p.add_argument("--tasks", default=None, help="comma-separated task indices")
    p.add_argument("--max-demos", type=int, default=None, help="cap demos per task")
    p.add_argument("--modes", default="all",
                   help=f"comma list of {list(FP.DEFAULT_MODES)}, or 'all'")
    p.add_argument("--per-mode", type=int, default=None, help="samples per mode per demo")
    p.add_argument("--positives", type=int, default=None,
                   help="clean/augmented sketches with a name-free deictic caption")
    p.add_argument("--per-grounding", type=int, default=None,
                   help="samples per deictic grounding config per demo")
    p.add_argument("--style", default="hand", choices=["hand", "clean", "shape"],
                   help="stroke rendering: hand = full stroke model; clean = exact geometry, "
                        "fixed colour/width/head; shape = hand-drawn circle shape and arrow "
                        "path over exact placement, fixed colour/width/head")
    p.add_argument("--no-render", action="store_true", help="labels only, skip images")
    p.add_argument("--positives-clean", action="store_true",
                   help="build every positive from the exact stage-01 annotation instead of a "
                        "stage-02 jittered variant (no circle/arrow-tip placement augmentation)")
    p.add_argument("--dedupe", action="store_true",
                   help="drop records whose declared circle/arrow/caption repeats within a "
                        "demo (they are pixel-identical once hand-drawn jitter is removed)")
    args = p.parse_args()
    cfg, root, suites = setup(args)
    fp_cfg = json.loads(json.dumps(cfg.get("false_prompts", {})))
    level_mix = fp_cfg.get("caption_level_mix", FP.DEFAULT_LEVEL_MIX)
    per_mode = args.per_mode if args.per_mode is not None else int(
        fp_cfg.get("samples_per_mode", 2)
    )
    positives = args.positives if args.positives is not None else int(
        fp_cfg.get("positives_per_demo", 4)
    )
    per_grounding = args.per_grounding if args.per_grounding is not None else int(
        fp_cfg.get("grounding_per_kind", 2)
    )

    import cv2

    want = (list(FP.DEFAULT_MODES) if args.modes == "all"
            else [m.strip().upper() for m in args.modes.split(",") if m.strip()])
    unknown = [m for m in want if m not in FP.REGISTRY]
    if unknown:
        raise SystemExit(f"unknown mode(s) {unknown}; have {sorted(FP.REGISTRY)}")
    explicit = [m for m in want if m in FP.NAME_EXPLICIT_MODES]
    if explicit:
        raise SystemExit(
            f"name-explicit modes are disabled in the deictic-only dataset: {explicit}"
        )
    task_filter = [int(x) for x in args.tasks.split(",")] if args.tasks else None
    seed = int(cfg.seed)
    imp = json.loads(json.dumps(cfg["imprecision"]))      # plain dicts for strokes.py
    alt_per_demo = int(fp_cfg.get("alt_positives_per_demo", 0))
    pick_per_demo = int(fp_cfg.get("pick_positives_per_demo", 0))
    # Human templates already carry the sloppiness; stage 02's parametric variants on top
    # would jitter the positives twice.
    stage02_variants = imp["circle"].get("source") != "human"

    # Cross-task caption pools were used only by the now-disabled name-explicit C2/C3 modes.
    pool: list[dict] = []

    for suite in suites:
        out_dir = root / "false_prompts" / suite
        # A regeneration must not leave disabled name-explicit images from an older run.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        records, yield_ct, na_ct = [], Counter(), Counter()
        inconsistent: list[str] = []
        n_demos = 0
        n_dup = 0
        n_no_masks = n_off_target = n_unresolved = n_render_drop = 0

        for task_dir, demo_dir in iter_demo_dirs(root, suite):
            ti = int(task_dir.name.split("_")[0])
            if task_filter and ti not in task_filter:
                continue
            if args.max_demos:
                di = int(demo_dir.name.split("_")[-1])
                if di >= args.max_demos:
                    continue
            scene_p = demo_dir / "scene.json"
            if not scene_p.exists():
                continue
            meta = json.loads((demo_dir / "meta.json").read_text())
            scene = json.loads(scene_p.read_text())
            occ_p = demo_dir / "frames" / "t0_occupancy.png"
            occ = (cv2.imread(str(occ_p), cv2.IMREAD_GRAYSCALE) > 127
                   if occ_p.exists() else None)
            inst_p = demo_dir / "frames" / "t0_objects.png"
            inst = (cv2.imread(str(inst_p), cv2.IMREAD_GRAYSCALE)
                    if inst_p.exists() else None)
            if inst is None:
                n_no_masks += 1
            cap_p = root / "captions" / suite / f"{task_dir.name}.json"
            ladder = json.loads(cap_p.read_text()) if cap_p.exists() else None

            ctx = FP.Ctx(meta=meta, scene=scene, occupancy=occ, caption_pool=pool,
                         ladder=ladder, radius_scale=float(cfg.circle.radius_scale),
                         instances=inst)
            # A demo whose own clean circle misses its projected target is already broken
            # upstream: its positive is mislabelled and every mode built on it inherits
            # that. Skip and count rather than emit label noise.
            if not ctx.annotation_consistent():
                inconsistent.append(f"{suite}/{task_dir.name}/{demo_dir.name}")
                continue
            demo_key = f"{suite}/{task_dir.name}/{demo_dir.name}"
            recs = FP.generate(ctx, want, per_mode, seed, key=demo_key,
                               level_mix=level_mix)

            recs += FP.generate_grounding(ctx, list(FP.GROUNDING_KINDS),
                                          per_grounding, seed, key=demo_key)

            # Positives use the same name-free L2/L3 language as the negatives; only the
            # visual grounding determines whether the complete prompt is right or wrong.
            ann_p = demo_dir / "annotations.json"
            variants = (json.loads(ann_p.read_text()).get("variants", [])
                        if ann_p.exists() else [])
            rng = np.random.default_rng(zlib.crc32(f"{seed}/{demo_key}/POS".encode()))

            def add_positive(c, k, v=None, slot="POS", goal=None, make=FP.positive):
                nonlocal n_off_target, n_unresolved
                sample = make(c, rng, v)
                pick = make is FP.pick_positive
                level = "L3" if pick else FP._sample_level(rng, level_mix)
                if not FP.circle_selects_target(c, sample.circle) or \
                        not FP.arrow_visible(sample.circle, sample.arrow) or \
                        not FP.head_on_frame(sample.arrow) or \
                        not FP.head_lands_on_destination(c, sample.arrow):
                    n_off_target += 1                  # a jittered circle that lost the target
                    return                             # or swallowed its arrow is no positive
                sample = FP._finalize(c, sample, level, rng, keep_caption=pick)
                if sample.n_referents * sample.n_destinations != 1 or \
                        FP.rival_coverage(c, sample.circle) > FP.VALID_MARGIN:
                    # The jitter, not the scene, made it ambiguous: its circle clips a
                    # neighbour or its head grazes one. That sits between the positives
                    # and the deliberate ambiguity negatives (A5, referent_twin,
                    # dest_between), so it is dropped rather than relabelled wrong.
                    n_unresolved += 1
                    return
                FP.preserve_positive_resolution_guard(c, sample)
                pr = sample.as_record("POS")
                pr["index"], pr["slot"] = k, slot
                if goal:
                    pr["goal_override"] = goal
                recs.append(pr)

            alts = FP.alternatives(ctx)
            for k in range(positives):
                v = None if k == 0 or not variants or args.positives_clean \
                    or not stage02_variants else variants[int(rng.integers(len(variants)))]
                add_positive(ctx, k, v)
            # "grab this" with only a circle: picking up is the whole action (place scenes)
            for k in range(pick_per_demo if ctx.kind == "place" else 0):
                if alts and rng.random() < 0.5:            # any clearly visible object
                    tgt, reg = alts[int(rng.integers(len(alts)))]
                    add_positive(FP.alt_context(ctx, tgt, reg), k, slot="POS_pick",
                                 goal={"target_object": tgt, "destination": reg},
                                 make=FP.pick_positive)
                else:
                    add_positive(ctx, k, slot="POS_pick", make=FP.pick_positive)
            # Task-blind: a clean drawing of another sensible action is right too.
            for k in range(alt_per_demo if alts else 0):
                tgt, reg = alts[int(rng.integers(len(alts)))]
                actx = FP.alt_context(ctx, tgt, reg)
                if actx.annotation_consistent():
                    add_positive(actx, k, slot="POS_alt",
                                 goal={"target_object": tgt, "destination": reg})

            got = {r["mode"] for r in recs}
            for m in want:
                (yield_ct if m in got else na_ct)[m] += 1

            if args.dedupe:
                # Rows that share declared geometry and caption differ only in the sampled
                # hand-drawn stroke; with --style clean they would be exact duplicates.
                seen_keys: set = set()
                unique = []
                for r in recs:
                    k = (json.dumps(r["prompt"]["circle"], sort_keys=True),
                         json.dumps(r["prompt"]["arrow"], sort_keys=True),
                         r["prompt"]["caption"])
                    if k in seen_keys:
                        n_dup += 1
                        continue
                    seen_keys.add(k)
                    unique.append(r)
                recs = unique

            rgb = cv2.imread(str(demo_dir / "frames" / "t0_rgb.png"))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb is not None else None
            img_dir = out_dir / "images" / task_dir.name / demo_dir.name

            for r in recs:
                # POS is the diagnostic generation mode for both clean and grounding
                # attempts; `category` separates verified positives (P) from failures (G).
                slot = r.get("slot") or r["resolution"].get("kind") or r["mode"]
                r["slot"] = slot
                rctx = FP.alt_context(ctx, **r["goal_override"]) if r.get("goal_override") \
                    else ctx
                r["id"] = f"{suite}/{task_dir.name}/{demo_dir.name}/{slot}#{r['index']}"
                r["suite"], r["task"], r["demo"] = suite, task_dir.name, demo_dir.name
                r["source_frame"] = str((demo_dir / "frames" / "t0_rgb.png").relative_to(root))
                r["name_explicit"] = not FP.is_name_free_deictic_caption(
                    ctx, r["prompt"]["caption"])
                if r["name_explicit"]:
                    raise RuntimeError(f"name-explicit caption escaped filtering: {r['id']}")
                if r["category"] == "P" and r["verdict"] != "right":
                    raise RuntimeError(f"non-right row entered positive family: {r['id']}")
                if not args.no_render and rgb is not None:
                    img_dir.mkdir(parents=True, exist_ok=True)
                    pc, pa = r["prompt"]["circle"], r["prompt"]["arrow"]
                    seed0 = zlib.crc32(r["id"].encode())
                    var = R.clean_variant(pc, pa)
                    # Verify the actual sampled stroke, not only its declared geometry.
                    # 40 drawings as people place them, then 20 of the same human shapes at the
                    # declared position, before the clean strokes as a last resort.
                    for attempt in range(1 if args.style == "clean" else 60):
                        if args.style == "hand":
                            var = FP.hand_variant(
                                pc, pa, r.get("guard"),
                                imp if attempt < 40 else FP.shape_imprecision(imp),
                                seed0 + 15485863 * attempt
                            )
                            r["stroke_source"] = "human" if attempt < 40 else "human_shape"
                        elif args.style == "shape":
                            var = FP.shape_variant(
                                pc, pa, r.get("guard"), imp, seed0 + 15485863 * attempt
                            )
                        n_ref, n_dst, rendered_by = FP.resolve_rendered(
                            rctx, var, r["resolution"]["caption_level"], r["prompt"]["caption"]
                        )
                        rendered_readings = n_ref * n_dst
                        expected_ok = (
                            rendered_readings == 1 if r["category"] == "P" else
                            rendered_readings > 1 if r["category"] == "G" else True
                        ) and FP.rendered_ok(rctx, r, var)
                        if expected_ok:
                            break
                    if not expected_ok:
                        var = R.clean_variant(pc, pa)
                        r["stroke_source"] = "clean"
                        n_ref, n_dst, rendered_by = FP.resolve_rendered(
                            rctx, var, r["resolution"]["caption_level"], r["prompt"]["caption"]
                        )
                        rendered_readings = n_ref * n_dst
                        expected_ok = (
                            rendered_readings == 1 if r["category"] == "P" else
                            rendered_readings > 1 if r["category"] == "G" else True
                        ) and FP.rendered_ok(rctx, r, var)
                    if not expected_ok:
                        n_render_drop += 1             # not even the exact strokes show it
                        continue
                    r["rendered_resolution"] = {
                        "n_referents": n_ref,
                        "n_destinations": n_dst,
                        "n_readings": rendered_readings,
                        "resolved_by": rendered_by,
                        "verified": expected_ok if r["category"] in {"P", "G"} else True,
                    }
                    r["guard_fallback"] = any(
                        (var.get(ch) or {}).get("guard_fallback", False)
                        for ch in ("circle", "arrow"))
                    keep = ("tpl", "cx", "cy", "r", "x0", "y0", "x1", "y1")
                    r["drawn"] = {ch: {k: v for k, v in (var.get(ch) or {}).items() if k in keep}
                                  for ch in ("circle", "arrow") if var.get(ch)}
                    if (var.get("circle") or {}).get("also"):
                        r["drawn"]["circle"]["also"] = {k: v for k, v in var["circle"]["also"].items()
                                                        if k in keep}
                    img = R.render_variant(var, rgb.shape[0], rgb.shape[1], rgb)
                    fp = img_dir / f"{slot}_{r['index']}.png"
                    cv2.imwrite(str(fp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    r["render"] = str(fp.relative_to(root))
                records.append(r)
            n_demos += 1

        with (out_dir / "samples.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        by_mode = defaultdict(int)
        for r in records:
            by_mode[r["mode"]] += 1
        verdicts = Counter(r["verdict"] for r in records)
        summary = {
            "suite": suite, "demos": n_demos, "samples": len(records),
            "verdicts": dict(verdicts),          # right | wrong
            "generation": {
                "samples_per_mode": per_mode,
                "positive_attempts_per_demo": positives,
                "grounding_attempts_per_kind": per_grounding,
                "caption_level_mix": level_mix,
                "style": args.style,
                "dedupe": bool(args.dedupe),
                "positives_clean": bool(args.positives_clean),
            },
            "deduplicated": n_dup,
            "demos_without_instance_map": n_no_masks,
            "positive_attempts_off_target": n_off_target,
            "positive_attempts_unresolved": n_unresolved,
            "rows_dropped_at_render": n_render_drop,
            "skipped_inconsistent_demos": len(inconsistent),
            "skipped_examples": inconsistent[:10],
            "per_mode": dict(sorted(by_mode.items())),
            "per_tier": dict(Counter(r["tier"] for r in records if r["mode"] != "POS")),
            "per_solution": dict(Counter(r["solution"] for r in records
                                         if r["mode"] != "POS")),
            "applicability": {m: {"demos_with_sample": yield_ct[m],
                                  "demos_not_applicable": na_ct[m]} for m in want},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[{suite}] {n_demos} demos -> {len(records)} samples "
              f"(right {verdicts['right']} / wrong {verdicts['wrong']}"
              f"{f', {n_dup} duplicates dropped' if args.dedupe else ''}) -> {out_dir}")
        if inconsistent:
            print(f"  [!] skipped {len(inconsistent)} demos whose stage-01 circle_clean "
                  f"does not contain the projected target (e.g. {inconsistent[0]})")
        for m in want:
            if na_ct[m]:
                print(f"  [{m}] n/a on {na_ct[m]}/{yield_ct[m] + na_ct[m]} demos")


if __name__ == "__main__":
    main()
