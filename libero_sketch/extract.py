"""Stage 01: replay LIBERO demos, dump per-demo ground truth for annotation.

Everything robosuite/LIBERO-version-sensitive lives here behind defensive helpers:
obs-key lookup, segmentation id offset, body-name resolution, and the pixel-convention
auto-calibration (projected target centroid must land inside its segmentation mask).
Any fallback taken is appended to meta["quality"].

Requires the `libero_sketch` conda env (see setup_env.sh). All downstream stages are
sim-free and consume only the meta.json/frames written here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import geometry as G
from .bddl import Goal, goal_from_predicates, parse_bddl_file
from .suites import benchmark_name

CAM = "agentview"


# ------------------------------------------------------------ env helpers --

def make_env(bddl_path: str, H: int, W: int):
    """OffScreenRenderEnv with seg+depth; retries without them on old robosuite."""
    from libero.libero.envs import OffScreenRenderEnv
    base = dict(bddl_file_name=bddl_path, camera_heights=H, camera_widths=W)
    for extra in (dict(camera_depths=True, camera_segmentations="element"),
                  dict(camera_depths=True), dict()):
        try:
            env = OffScreenRenderEnv(**base, **extra)
            return env, list(extra.keys())
        except (TypeError, ValueError):
            continue
    raise RuntimeError(f"could not construct OffScreenRenderEnv for {bddl_path}")


def obs_get(obs: dict, *suffixes) -> np.ndarray | None:
    for suf in suffixes:
        for key in (f"{CAM}_{suf}", f"{CAM}{suf}"):
            if key in obs:
                return obs[key]
    return None


def sim_of(env):
    for path in ("env.sim", "sim"):
        cur = env
        try:
            for attr in path.split("."):
                cur = getattr(cur, attr)
            return cur
        except AttributeError:
            continue
    raise AttributeError("cannot find mujoco sim on env")


def set_state_fast(env, state: np.ndarray) -> None:
    """State-only update, no observation/render (for track extraction)."""
    sim = sim_of(env)
    sim.set_state_from_flattened(np.asarray(state))
    sim.forward()


def obs_at_state(env, state: np.ndarray) -> dict:
    out = env.set_init_state(np.asarray(state))
    if isinstance(out, dict):
        return out
    return env.env._get_observations(force_update=True)


def body_id_for(env, name: str, quality: list) -> int | None:
    sim = sim_of(env)
    mapping = getattr(getattr(env, "env", env), "obj_body_id", None)
    if isinstance(mapping, dict) and name in mapping:
        return int(mapping[name])
    for cand in (name, f"{name}_main", f"{name}_1", f"{name}_1_main"):
        try:
            return int(sim.model.body_name2id(cand))
        except Exception:
            continue
    hits = [i for i in range(sim.model.nbody)
            if name in (sim.model.body_id2name(i) or "")]
    if hits:
        quality.append(f"fuzzy body match for '{name}' -> {sim.model.body_id2name(hits[0])}")
        return hits[0]
    quality.append(f"body not found: {name}")
    return None


def descendant_geoms(sim, body_id: int) -> list[int]:
    """All geom ids attached to body_id or its descendants."""
    nbody = sim.model.nbody
    desc = {body_id}
    changed = True
    while changed:
        changed = False
        for b in range(nbody):
            if b not in desc and int(sim.model.body_parentid[b]) in desc:
                desc.add(b)
                changed = True
    return [g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) in desc]


def body_mask(seg2d: np.ndarray, sim, body_id: int, offset: int) -> np.ndarray | None:
    ids = np.array(descendant_geoms(sim, body_id)) + offset
    mask = np.isin(seg2d, ids)
    return mask if mask.sum() >= 8 else None


def calibrate_projection(seg: np.ndarray | None, sim, body_ids: list[int],
                         K: np.ndarray, T_wc: np.ndarray, H: int, W: int,
                         quality: list) -> tuple[int | None, dict]:
    """Jointly calibrate the segmentation id offset and the pixel-convention variant.

    Both are ambiguous in isolation: the seg offset heuristic can be fooled (consecutive
    table geom ids), and a single-object variant check silently locks onto a wrong mask.
    Scoring (offset, variant) pairs by how many scene objects' projected centers land
    inside their own masks disambiguates both at once — the correct pair scores near
    len(body_ids), wrong pairs score ~0-2. Returns (offset, variant); offset is None
    when no segmentation is available (variant falls back to LIBERO's usual flip_v)."""
    default_variant = {"flip_u": False, "flip_v": True, "transpose": False}
    if seg is None:
        quality.append("no segmentation: pixel variant defaulted, no offset")
        return None, default_variant
    seg2d = seg[..., 0] if seg.ndim == 3 else seg
    best = (-1, None, None, 0)  # score, offset, variant, n_masks
    for off in (0, 1):
        masks = {b: m for b in body_ids
                 if (m := body_mask(seg2d, sim, b, off)) is not None}
        if not masks:
            continue
        centers = {b: np.array(sim.data.body_xpos[b]) for b in masks}
        for var in G.PIXEL_VARIANTS:
            score = 0
            for b, m in masks.items():
                uv = G.project(centers[b], K, T_wc, H, W, var)[0]
                u, v = int(round(uv[0])), int(round(uv[1]))
                if 0 <= u < W and 0 <= v < H and m[v, u]:
                    score += 1
            if score > best[0]:
                best = (score, off, var, len(masks))
    score, off, var, n = best
    if var is None or score < max(2, n // 2):
        quality.append(f"projection calibration weak (score {score}/{n}); "
                       "using defaults")
        return (off if var is not None else None), (var or default_variant)
    return off, var


def target_mask(seg: np.ndarray | None, sim, body_id: int, offset: int | None,
                quality: list) -> np.ndarray | None:
    if seg is None or offset is None:
        return None
    seg2d = seg[..., 0] if seg.ndim == 3 else seg
    mask = body_mask(seg2d, sim, body_id, offset)
    if mask is None:
        quality.append("target mask empty/tiny")
    return mask


def ee_pos(env, quality: list) -> np.ndarray | None:
    sim = sim_of(env)
    for site in ("gripper0_grip_site", "gripper0_ee", "grip_site"):
        try:
            return np.array(sim.data.site_xpos[sim.model.site_name2id(site)])
        except Exception:
            continue
    try:
        return np.array(sim.data.body_xpos[sim.model.body_name2id("gripper0_eef")])
    except Exception:
        quality.append("EE site not found")
        return None


def resolve_target(env, goal: Goal, quality: list):
    """Map the goal's target to (body_id, anchor_world|None, extent_m|None).

    Place predicates name objects directly. Manipulate predicates (open/close/turn*)
    usually name a REGION ('wooden_cabinet_1_top_region'): anchor the circle to the
    region center with a radius from the region extent, and use the parent body
    (the cabinet) for masks/tracks."""
    bid = body_id_for(env, goal.target_object, [])
    if bid is not None:
        return bid, None, None
    parsed = getattr(getattr(env, "env", env), "parsed_problem", None) or {}
    entry = parsed.get("regions", {}).get(goal.target_object)
    if entry:
        parent = entry.get("target", "")
        pbid = body_id_for(env, parent, quality)
        if pbid is not None:
            anchor = region_world_pos(env, goal.target_object, quality)
            r = (entry.get("ranges") or [[0, 0, 0, 0]])[0]
            ext = float(np.hypot(r[2] - r[0], r[3] - r[1]))
            quality.append(f"target is region; anchored to parent '{parent}'")
            return pbid, anchor, (ext if ext > 1e-6 else None)
    quality.append(f"target unresolvable: {goal.target_object}")
    return None, None, None


def region_world_pos(env, region_name: str, quality: list) -> np.ndarray | None:
    """BDDL region centroid in world coords (metadata only; hindsight endpoint is the
    label of record)."""
    sim = sim_of(env)
    parsed = getattr(getattr(env, "env", env), "parsed_problem", None)
    regions = (parsed or {}).get("regions", {})
    if region_name in regions:
        entry = regions[region_name]
        parent = entry.get("target", "")
        ranges = entry.get("ranges") or [[0, 0, 0, 0]]
        r = np.asarray(ranges[0], dtype=np.float64)
        local = np.array([(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 0.0])
        bid = body_id_for(env, parent, quality)
        if bid is not None:
            R = np.array(sim.data.body_xmat[bid]).reshape(3, 3)
            return np.array(sim.data.body_xpos[bid]) + R @ local
    bid = body_id_for(env, region_name, [])
    if bid is not None:
        return np.array(sim.data.body_xpos[bid])
    try:
        return np.array(sim.data.site_xpos[sim.model.site_name2id(region_name)])
    except Exception:
        quality.append(f"region not resolvable: {region_name}")
        return None


# ------------------------------------------------------------- keyframes --

def find_keyframes(actions: np.ndarray, ee_track: np.ndarray | None,
                   obj_track: np.ndarray | None, kind: str) -> dict:
    """Gripper-action heuristic; falls back to EE-object proximity for manipulate tasks."""
    T = len(actions)
    grip = actions[:, -1]

    def first_sustained(x: np.ndarray, start: int, hold: int = 5) -> int | None:
        run = 0
        for t in range(start, len(x)):
            run = run + 1 if x[t] else 0
            if run >= hold:
                return t - hold + 1
        return None

    grasp = first_sustained(grip > 0.5, 0)
    release = first_sustained(grip < -0.5, (grasp or 0) + 1) if grasp is not None else None
    if grasp is None and kind == "manipulate" and ee_track is not None and obj_track is not None:
        d = np.linalg.norm(ee_track - obj_track, axis=1)
        grasp = int(np.argmin(d))
    return {"grasp": int(grasp) if grasp is not None else 0,
            "release": int(release) if release is not None else T - 1, "T": T}


# --------------------------------------------------------------- extract --

def parse_goal(env, bddl_path: str, quality: list) -> Goal:
    parsed = getattr(getattr(env, "env", env), "parsed_problem", None)
    if parsed and parsed.get("goal_state"):
        try:
            return goal_from_predicates([[str(x) for x in p] for p in parsed["goal_state"]])
        except ValueError:
            pass
    quality.append("goal parsed from bddl file, not env")
    return parse_bddl_file(bddl_path)


def camera_matrices(env, H: int, W: int):
    from robosuite.utils import camera_utils as CU
    sim = sim_of(env)
    K = CU.get_camera_intrinsic_matrix(sim, CAM, H, W)
    T = CU.get_camera_extrinsic_matrix(sim, CAM)  # camera pose in world
    return np.asarray(K, dtype=np.float64), np.asarray(T, dtype=np.float64)


def extract_demo(env, states: np.ndarray, actions: np.ndarray, goal: Goal,
                 instruction: str, cfg, out_dir: Path, demo_key: str,
                 source_hdf5: str, suite: str, task_id: int, task_name: str) -> dict:
    import cv2

    quality: list[str] = []
    H, W = cfg.render.height, cfg.render.width
    stride = cfg.extract.track_stride
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # --- t0 observation (RGB / seg / depth) ---
    obs0 = obs_at_state(env, states[0])
    rgb0 = obs_get(obs0, "image", "rgb")
    seg0 = obs_get(obs0, "segmentation_element", "segmentation_instance", "segmentation_class")
    depth0 = obs_get(obs0, "depth")
    if rgb0 is None:
        raise RuntimeError(f"no {CAM} RGB in obs; keys={sorted(obs0.keys())[:20]}")
    rgb0 = np.asarray(rgb0)[..., :3]

    sim = sim_of(env)
    K, T_wc = camera_matrices(env, H, W)
    tgt_bid, tgt_anchor_w, tgt_extent = resolve_target(env, goal, quality)
    if tgt_bid is None:
        raise RuntimeError(f"target '{goal.target_object}' has no body and is no region")

    # --- joint segmentation-offset + pixel-convention calibration ---
    scene_bids = list(getattr(getattr(env, "env", env), "obj_body_id", {}).values()) \
        or [tgt_bid]
    seg_off, variant = calibrate_projection(seg0, sim, scene_bids, K, T_wc, H, W, quality)
    mask0 = target_mask(seg0, sim, tgt_bid, seg_off, quality)
    tgt_w0 = (tgt_anchor_w if tgt_anchor_w is not None
              else np.array(sim.data.body_xpos[tgt_bid]))

    # RGB orientation must match projection space: LIBERO-stored agentview frames are
    # 180-degree rotated vs the raw render; we save frames exactly as rendered and keep all
    # pixel coords in that same frame, so overlays and projections always agree here.
    dest_region_w = region_world_pos(env, goal.destination, quality) if goal.destination else None

    # --- tracks (state-only replay, no rendering) ---
    idx = list(range(0, len(states), stride))
    if idx[-1] != len(states) - 1:
        idx.append(len(states) - 1)
    obj_track_w, ee_track_w = [], []
    for t in idx:
        set_state_fast(env, states[t])
        obj_track_w.append(np.array(sim.data.body_xpos[tgt_bid]))
        e = ee_pos(env, quality if t == 0 else [])
        ee_track_w.append(e if e is not None else np.full(3, np.nan))
    obj_track_w = np.stack(obj_track_w)
    ee_track_w = np.stack(ee_track_w)

    kf = find_keyframes(actions,
                        ee_track_w if not np.isnan(ee_track_w).any() else None,
                        obj_track_w, goal.kind)

    # --- projections (all into the t0 camera; camera is static in LIBERO) ---
    def proj(pw):
        return G.project(pw, K, T_wc, H, W, variant)

    tgt_px0 = proj(tgt_w0)[0]
    tgt_pxF = proj(obj_track_w[-1])[0]
    obj_track_px = proj(obj_track_w)
    ee_track_px = proj(ee_track_w) if not np.isnan(ee_track_w).any() else None
    dest_region_px = proj(dest_region_w)[0] if dest_region_w is not None else None

    # --- clean circle ---
    if tgt_anchor_w is not None:
        # region target (open/close/turn*): circle the region, not the whole parent body
        ccx, ccy = tgt_px0
        r_world = max(tgt_extent or 0.12, 0.08) / 2
        cr = G.sphere_radius_px(tgt_w0, r_world, K, T_wc) * cfg.circle.radius_scale
        circle_src = "region"
    elif mask0 is not None:
        ys, xs = np.nonzero(mask0)
        (ccx, ccy), cr = cv2.minEnclosingCircle(
            np.stack([xs, ys], axis=1).astype(np.float32))
        cr *= cfg.circle.radius_scale
        circle_src = "mask"
    else:
        rbound = max((float(sim.model.geom_rbound[g]) for g in descendant_geoms(sim, tgt_bid)),
                     default=0.05)
        ccx, ccy = tgt_px0
        cr = G.sphere_radius_px(tgt_w0, rbound, K, T_wc) * cfg.circle.radius_scale
        circle_src = "sphere"
        quality.append("circle from bounding sphere (no mask)")
    circle_clean = {"cx": float(ccx / W), "cy": float(ccy / H),
                    "r": float(cr / min(H, W)), "source": circle_src}

    # --- clean arrows ---
    def arrow_dict(p_tail, p_head):
        return {"x0": float(p_tail[0] / W), "y0": float(p_tail[1] / H),
                "x1": float(p_head[0] / W), "y1": float(p_head[1] / H)}

    # destination arrow: hindsight — where the object actually ends up
    arrow_dest = arrow_dict(tgt_px0, tgt_pxF) if goal.kind == "place" else None
    # motion arrow: coarse EE/object displacement between grasp and release
    g_i = min(range(len(idx)), key=lambda i: abs(idx[i] - kf["grasp"]))
    r_i = min(range(len(idx)), key=lambda i: abs(idx[i] - kf["release"]))
    track_px = obj_track_px if goal.kind == "place" else (ee_track_px if ee_track_px is not None else obj_track_px)
    seg_pts = G.rdp(track_px[g_i:r_i + 1], eps=0.02 * min(H, W)) if r_i > g_i else track_px[[g_i, r_i]]
    arrow_motion = arrow_dict(seg_pts[0], seg_pts[-1])

    # --- depth (for the depth-lifted arrow ablation) ---
    depth_at = {
        "target": float(-G.world_to_cam(tgt_w0, T_wc)[0, 2]),
        "final": float(-G.world_to_cam(obj_track_w[-1], T_wc)[0, 2]),
    }

    # --- write artifacts ---
    cv2.imwrite(str(frames_dir / "t0_rgb.png"), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    if mask0 is not None:
        cv2.imwrite(str(frames_dir / "t0_seg_target.png"), mask0.astype(np.uint8) * 255)
        
    # --- generate and write arrow mask ---
    # Prioritize destination arrow if available (e.g. place task), fallback to motion arrow
    a = arrow_dest or arrow_motion
    arrow_mask = np.zeros((H, W), dtype=np.uint8)
    if a is not None:
        # Convert the normalized clean arrow coords back to absolute pixel coordinates
        x0, y0 = int(a["x0"] * W), int(a["y0"] * H)
        x1, y1 = int(a["x1"] * W), int(a["y1"] * H)
        # Thickness scaled proportionally to screen size (e.g., 5px thickness for 256x256 maps)
        thickness = max(2, int(min(H, W) * 0.02)) 
        cv2.arrowedLine(arrow_mask, (x0, y0), (x1, y1), 255, thickness=thickness, tipLength=0.2)
    cv2.imwrite(str(frames_dir / "t0_seg_arrow.png"), arrow_mask)

    if depth0 is not None:
        try:
            from robosuite.utils import camera_utils as CU
            real = CU.get_real_depth_map(sim, np.asarray(depth0))
        except Exception:
            real = np.asarray(depth0)
        np.savez_compressed(frames_dir / "t0_depth.npz",
                            depth=np.squeeze(real).astype(np.float16))

    inventory = sorted(getattr(getattr(env, "env", env), "obj_body_id", {}).keys())

    meta = {
        "suite": suite, "task_id": task_id, "task_name": task_name,
        "source_hdf5": source_hdf5, "demo_key": demo_key, "instruction": instruction,
        "goal": goal.to_dict(), "inventory": inventory,
        "camera": {"name": CAM, "H": H, "W": W, "K": K.tolist(),
                   "T_world_cam": T_wc.tolist(), "pixel_variant": variant,
                   "seg_offset": seg_off},
        "keyframes": kf,
        "points_px": {
            "target_t0": tgt_px0.tolist(), "target_final": tgt_pxF.tolist(),
            "dest_region": dest_region_px.tolist() if dest_region_px is not None else None,
            "obj_track": obj_track_px.tolist(),
            "ee_track": ee_track_px.tolist() if ee_track_px is not None else None,
            "track_indices": idx,
        },
        "points_3d": {
            "target_t0": tgt_w0.tolist(), "target_final": obj_track_w[-1].tolist(),
            "dest_region": dest_region_w.tolist() if dest_region_w is not None else None,
            "depth": depth_at,
        },
        "circle_clean": circle_clean,
        "arrow_dest_clean": arrow_dest,
        "arrow_motion_clean": arrow_motion,
        "quality": quality,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def extract_suite(suite: str, data_root: Path, out_root: Path, cfg,
                  task_filter: list[int] | None = None, max_demos: int | None = None) -> dict:
    """Drive extraction for one suite. Returns summary stats."""
    import h5py
    from libero.libero import benchmark, get_libero_path

    bname = benchmark_name(suite)
    suite_obj = benchmark.get_benchmark_dict()[bname]()
    bddl_root = get_libero_path("bddl_files")
    H, W = cfg.render.height, cfg.render.width
    summary = {"suite": suite, "tasks": 0, "demos_ok": 0, "demos_failed": 0, "warnings": 0}

    for ti in range(suite_obj.n_tasks):
        if task_filter and ti not in task_filter:
            continue
        task = suite_obj.get_task(ti)
        demo_rel = suite_obj.get_task_demonstration(ti)
        demo_path = data_root / demo_rel
        if not demo_path.exists():
            print(f"[{suite}] task {ti}: missing {demo_path}, skipping")
            continue
        bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        env, _ = make_env(bddl_path, H, W)
        try:
            env.reset()
            goal_q: list[str] = []
            goal = parse_goal(env, bddl_path, goal_q)
            with h5py.File(demo_path, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                if max_demos:
                    demo_keys = demo_keys[:max_demos]
                for dk in demo_keys:
                    grp = f["data"][dk]
                    states = np.asarray(grp["states"])
                    actions = np.asarray(grp["actions"])
                    out_dir = out_root / "extracted" / suite / f"{ti:02d}_{task.name}" / dk
                    try:
                        meta = extract_demo(
                            env, states, actions, goal, task.language, cfg, out_dir,
                            dk, str(demo_path), suite, ti, task.name)
                        meta["quality"].extend(goal_q)
                        summary["demos_ok"] += 1
                        summary["warnings"] += len(meta["quality"])
                    except Exception as e:
                        summary["demos_failed"] += 1
                        print(f"[{suite}] {task.name}/{dk} FAILED: {e}")
            summary["tasks"] += 1
            print(f"[{suite}] task {ti:02d} {task.name}: done")
        finally:
            env.close()
    return summary
