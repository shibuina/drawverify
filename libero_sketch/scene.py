"""Per-demo scene geometry dump (sim-dependent).

Stage 01 persists only the *target's* mask plus the target/destination pixel anchors —
enough for positive prompts, but false prompts must reference the *other* objects: circle a
wrong object (A2), sample genuinely empty table (A4), grow a circle until it encloses two
objects (A5), retarget an arrow at a different container (B1), compose a mismatched
circle/arrow pair (C1). This re-opens the sim at each demo's t0 state and records every
scene object's mask circle, projected center, and visibility.

Calibration (seg id offset + pixel variant) is read back from meta.json rather than
recomputed, so dumped geometry lands in exactly the same pixel frame as `circle_clean`.

`frames/t0_objects.png` is the instance map the generator resolves prompts against: pixel
value i+1 is `scene["objects"][i]`, ROBOT_ID marks the robot, 0 is everything else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry as G
from .extract import (body_mask, camera_matrices, make_env, obs_at_state, obs_get,
                      sim_of)

ROBOT_ID = 255
_ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")


def robot_mask(seg2d: np.ndarray, sim, offset: int) -> np.ndarray:
    """Pixels of the arm, gripper and mount: never an object, never empty table."""
    geoms = [g for g in range(sim.model.ngeom)
             if (sim.model.body_id2name(int(sim.model.geom_bodyid[g])) or "")
             .startswith(_ROBOT_PREFIXES)]
    return np.isin(seg2d, np.array(geoms) + offset)
from .suites import benchmark_name


def dump_demo_scene(env, state0: np.ndarray, meta: dict) -> dict:
    """Scene geometry for one demo at its t0 state, in meta.json's pixel frame."""
    import cv2

    H, W = meta["camera"]["H"], meta["camera"]["W"]
    variant = meta["camera"]["pixel_variant"]
    seg_off = meta["camera"].get("seg_offset")

    obs0 = obs_at_state(env, state0)
    seg0 = obs_get(obs0, "segmentation_element", "segmentation_instance",
                   "segmentation_class")
    sim = sim_of(env)
    K, T_wc = camera_matrices(env, H, W)

    base = getattr(env, "env", env)
    body_ids = dict(getattr(base, "obj_body_id", {}) or {})
    fixtures = set(getattr(base, "fixtures_dict", {}) or {})
    objects, seg2d, instances = [], None, None
    if seg0 is not None:
        seg2d = seg0[..., 0] if np.asarray(seg0).ndim == 3 else np.asarray(seg0)
        instances = np.zeros((H, W), dtype=np.uint8)

    for name, bid in sorted(body_ids.items()):
        anchor_w = np.array(sim.data.body_xpos[int(bid)])
        uv = G.project(anchor_w, K, T_wc, H, W, variant)[0]
        rec = {
            "name": name,
            "body_id": int(bid),
            "fixture": name in fixtures,          # cabinets, stoves, racks: never picked up
            "anchor_3d": [float(x) for x in anchor_w],
            "center_px": [float(uv[0]), float(uv[1])],
            "circle": None,
            "bbox_px": None,
            "n_px": 0,
            "visible": False,
        }
        mask = (body_mask(seg2d, sim, int(bid), seg_off)
                if (seg2d is not None and seg_off is not None) else None)
        if mask is not None:
            ys, xs = np.nonzero(mask)
            (mx, my), mr = cv2.minEnclosingCircle(
                np.stack([xs, ys], axis=1).astype(np.float32))
            rec["circle"] = {"cx": float(mx / W), "cy": float(my / H),
                             "r": float(mr / min(H, W))}
            rec["bbox_px"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            rec["n_px"] = int(mask.sum())
            rec["visible"] = True
            instances[mask] = len(objects) + 1
        objects.append(rec)

    # union of all object masks -> lets the generator sample provably-empty table (A4/B5)
    occupancy = None
    if seg2d is not None and seg_off is not None:
        occ = np.zeros((H, W), dtype=bool)
        for name, bid in body_ids.items():
            m = body_mask(seg2d, sim, int(bid), seg_off)
            if m is not None:
                occ |= m
        robot = robot_mask(seg2d, sim, seg_off)
        occupancy = occ | robot
        instances[robot & (instances == 0)] = ROBOT_ID

    return {"objects": objects,
            "n_objects": len(objects),
            "n_visible": sum(o["visible"] for o in objects)}, occupancy, instances


def dump_suite_scenes(suite: str, data_root: Path, out_root: Path, cfg,
                      task_filter: list[int] | None = None,
                      max_demos: int | None = None) -> dict:
    """Write scene.json (+ t0_occupancy.png) next to each demo's meta.json."""
    import os

    import cv2
    import h5py
    from libero.libero import benchmark, get_libero_path

    bname = benchmark_name(suite)
    suite_obj = benchmark.get_benchmark_dict()[bname]()
    bddl_root = get_libero_path("bddl_files")
    H, W = cfg.render.height, cfg.render.width
    summary = {"suite": suite, "demos_ok": 0, "demos_failed": 0, "skipped": 0}

    for ti in range(suite_obj.n_tasks):
        if task_filter and ti not in task_filter:
            continue
        task = suite_obj.get_task(ti)
        demo_path = data_root / suite_obj.get_task_demonstration(ti)
        if not demo_path.exists():
            continue
        task_dir = out_root / "extracted" / suite / f"{ti:02d}_{task.name}"
        if not task_dir.exists():
            continue
        bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        env, _ = make_env(bddl_path, H, W)
        try:
            env.reset()
            with h5py.File(demo_path, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                if max_demos:
                    demo_keys = demo_keys[:max_demos]
                for dk in demo_keys:
                    out_dir = task_dir / dk
                    meta_p = out_dir / "meta.json"
                    if not meta_p.exists():
                        summary["skipped"] += 1
                        continue
                    try:
                        meta = json.loads(meta_p.read_text())
                        states = np.asarray(f["data"][dk]["states"])
                        scene, occ, inst = dump_demo_scene(env, states[0], meta)
                        (out_dir / "scene.json").write_text(json.dumps(scene, indent=1))
                        if occ is not None:
                            cv2.imwrite(str(out_dir / "frames" / "t0_occupancy.png"),
                                        (occ.astype(np.uint8) * 255))
                            cv2.imwrite(str(out_dir / "frames" / "t0_objects.png"), inst)
                        summary["demos_ok"] += 1
                    except Exception as e:
                        summary["demos_failed"] += 1
                        print(f"[{suite}] {task.name}/{dk} scene dump FAILED: {e}")
            print(f"[{suite}] task {ti:02d} {task.name}: scenes dumped")
        finally:
            env.close()
    return summary
