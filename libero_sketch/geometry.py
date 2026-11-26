"""Projection utilities and polyline helpers. Sim-free (pure numpy).

Pixel convention here: (u, v) = (column, row), origin top-left, matching OpenCV.
Because robosuite/LIBERO offscreen renders differ in orientation across versions,
projection goes through a `pixel_variant` — one of 8 (flip_u, flip_v, transpose)
combinations — chosen once per extraction run by `calibrate_pixel_variant` (the
variant that lands the projected object centroid inside its segmentation mask).
"""

from __future__ import annotations

import itertools

import numpy as np

PIXEL_VARIANTS = [
    {"flip_u": fu, "flip_v": fv, "transpose": tr}
    for fu, fv, tr in itertools.product([False, True], repeat=3)
]


def world_to_cam(points_w: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """points_w: (N,3) world coords; T_world_cam: 4x4 camera pose in world (camera->world).
    Returns (N,3) in camera frame."""
    pts = np.atleast_2d(points_w).astype(np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (T_cam_world @ pts_h.T).T[:, :3]


def project(points_w: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
            H: int, W: int, variant: dict | None = None) -> np.ndarray:
    """Project world points to (u, v) pixels. MuJoCo cameras look down -z."""
    pc = world_to_cam(points_w, T_world_cam)
    z = -pc[:, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * (pc[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (-pc[:, 1] / z) + K[1, 2]
    uv = np.stack([u, v], axis=1)
    if variant:
        uv = apply_variant(uv, H, W, variant)
    return uv


def apply_variant(uv: np.ndarray, H: int, W: int, variant: dict) -> np.ndarray:
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64)).copy()
    if variant.get("transpose"):
        uv = uv[:, ::-1]
    if variant.get("flip_u"):
        uv[:, 0] = (W - 1) - uv[:, 0]
    if variant.get("flip_v"):
        uv[:, 1] = (H - 1) - uv[:, 1]
    return uv


def calibrate_pixel_variant(point_w: np.ndarray, mask: np.ndarray, K: np.ndarray,
                            T_world_cam: np.ndarray) -> tuple[dict | None, float]:
    """Pick the pixel variant whose projection of `point_w` lands inside `mask`
    (HxW bool). Ties broken by distance to mask centroid. Returns (variant, score);
    variant is None if no variant lands in-bounds."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, np.inf
    centroid = np.array([xs.mean(), ys.mean()])
    best, best_d = None, np.inf
    for var in PIXEL_VARIANTS:
        uv = project(point_w, K, T_world_cam, H, W, var)[0]
        u, v = int(round(uv[0])), int(round(uv[1]))
        if not (0 <= u < W and 0 <= v < H):
            continue
        inside = bool(mask[v, u])
        d = float(np.linalg.norm(uv - centroid)) + (0.0 if inside else 1e4)
        if d < best_d:
            best, best_d = var, d
    return best, best_d


def sphere_radius_px(point_w: np.ndarray, r_world: float, K: np.ndarray,
                     T_world_cam: np.ndarray) -> float:
    """Projected radius of a world-space bounding sphere (fallback circle size)."""
    z = float(-world_to_cam(point_w, T_world_cam)[0, 2])
    f = float((K[0, 0] + K[1, 1]) / 2.0)
    return abs(f * r_world / max(abs(z), 1e-9))


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification. points: (N,2)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-12:
        d = np.linalg.norm(pts - start, axis=1)
    else:
        rel = pts - start
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len   # 2-D cross (np.cross 2-D removed in numpy 2.5)
    i = int(np.argmax(d))
    if d[i] > eps:
        left = rdp(pts[: i + 1], eps)
        right = rdp(pts[i:], eps)
        return np.concatenate([left[:-1], right])
    return np.stack([start, end])


def normalize_uv(uv, H: int, W: int) -> np.ndarray:
    """Pixels -> [0,1]^2 (x by width, y by height)."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    return uv / np.array([[W, H]], dtype=np.float64)


def denormalize_uv(uv_n, H: int, W: int) -> np.ndarray:
    uv_n = np.atleast_2d(np.asarray(uv_n, dtype=np.float64))
    return uv_n * np.array([[W, H]], dtype=np.float64)
