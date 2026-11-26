"""Hand-drawn stroke model: clean circle/arrow -> sloppy human-like variants.

All coordinates are normalized [0,1] (x by width, y by height). A variant is a dict of
sampled parameters + the seed that produced it; `circle_polyline` / `arrow_polylines`
regenerate the exact geometry, so storage is parameters-only and rendering is
deterministic at any resolution.

Two sources of shape (`imprecision.<circle|arrow>.source`):
  human       replay a recorded study drawing (assets/human_strokes.json, built by
              scripts/build_human_templates.py): its pointer path and where it sat relative
              to the clean annotation. A variant stores the template index (`tpl`).
  parametric  ellipse arc / quadratic bezier with sampled jitter and wobble.
Either way every stroke is drawn in the study UI's look (`UI`): one colour per mark, one
width, the same V head. Nothing about colour, width or head shape varies.

Sanity guards: a sampled variant is rejected if the circle no longer contains the target
point, or the arrow head drifts too far from the intended endpoint — sloppiness must not
flip the label the stroke encodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# What participants saw while drawing (user_study_draw/public/app.js): green circle, red
# arrow, line width 0.8% of the canvas, V head with wings 3.2 line widths long at +-0.45 rad
# off the direction from the 6th-last pointer sample to the last.
UI = {"circle_color": [23, 160, 94], "arrow_color": [217, 73, 73], "width": 0.008 * 256,
      "head_wing": 3.2, "head_angle": 0.45, "head_anchor": 6}
TEMPLATES = Path(__file__).resolve().parent / "assets" / "human_strokes.json"
_TPL: list | None = None
_MASKS: dict = {}


def templates() -> list[dict]:
    global _TPL
    if _TPL is None:
        _TPL = json.loads(TEMPLATES.read_text())["templates"]
    return _TPL


def ui_head(shaft: np.ndarray, width: float = UI["width"]) -> np.ndarray:
    """The study UI's V head on the end of a pointer path, as one polyline."""
    tip = shaft[-1]
    d = tip - shaft[max(0, len(shaft) - UI["head_anchor"])]
    ang = float(np.arctan2(d[1], d[0]))
    size = UI["head_wing"] * width / 256
    wing = [tip - size * np.array([np.cos(ang + s * UI["head_angle"]),
                                   np.sin(ang + s * UI["head_angle"])]) for s in (-1, 1)]
    return np.stack([wing[0], tip, wing[1]])


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- circle ----

def sample_circle_variant(clean: dict, seed: int, cfg: dict) -> dict:
    """clean: {cx, cy, r} normalized. cfg: imprecision.circle section (plain dict);
    cfg["place"] False keeps the clean centre and radius (shape only)."""
    g = _rng(seed)
    r = float(clean["r"])
    if cfg.get("source") == "human":
        i = int(g.integers(len(templates())))
        dx, dy, rr = templates()[i]["circle_off"] if cfg.get("place", True) else (0, 0, 1)
        return {"kind": "circle", "seed": int(seed), "tpl": i,
                "cx": float(clean["cx"] + dx * r), "cy": float(clean["cy"] + dy * r),
                "r": float(r * rr), "ecc": 1.0, "rot": 0.0,
                "width": UI["width"], "color": UI["circle_color"]}
    var = {
        "kind": "circle",
        "seed": int(seed),
        "cx": float(clean["cx"] + g.normal(0, cfg["center_jitter"] * r)),
        "cy": float(clean["cy"] + g.normal(0, cfg["center_jitter"] * r)),
        "r": float(r * np.exp(g.normal(0, cfg["radius_lognorm_sigma"]))),
        "ecc": float(1.0 + abs(g.normal(0, cfg["ecc_sigma"]))),
        "rot": float(g.uniform(0, np.pi)),
        "theta0": float(g.uniform(0, 2 * np.pi)),
        "span": float(g.uniform(cfg["span_min"], cfg["span_max"]) * 2 * np.pi),
        "wobble_amps": [float(g.normal(0, cfg["wobble"] / k)) for k in range(1, 5)],
        "wobble_phases": [float(g.uniform(0, 2 * np.pi)) for _ in range(4)],
        "width": UI["width"], "color": UI["circle_color"],
    }
    return var


def circle_polyline(var: dict, n: int = 120) -> np.ndarray:
    """(N,2) normalized polyline of the hand-drawn circle (template path or ellipse arc)."""
    if "tpl" in var:
        return np.array([var["cx"], var["cy"]]) + var["r"] * np.asarray(
            templates()[var["tpl"]]["circle"])
    th = var["theta0"] + np.linspace(0, var["span"], n)
    wob = np.ones_like(th)
    for k, (a, p) in enumerate(zip(var["wobble_amps"], var["wobble_phases"]), start=1):
        wob += a * np.sin(k * th + p)
    a_ax = var["r"] * var["ecc"] * wob
    b_ax = var["r"] / var["ecc"] * wob
    x = a_ax * np.cos(th)
    y = b_ax * np.sin(th)
    c, s = np.cos(var["rot"]), np.sin(var["rot"])
    xr = c * x - s * y + var["cx"]
    yr = s * x + c * y + var["cy"]
    return np.stack([xr, yr], axis=1)


def _tpl_mask(var: dict, n: int = 256) -> np.ndarray:
    """Inside of a template circle on a (3n)^2 grid covering [-1, 2]^2: everything the
    closed path separates from the border (an open arc is closed by its chord, a loop
    drawn twice is filled once)."""
    import cv2
    key = (var["tpl"], round(var["cx"], 4), round(var["cy"], 4), round(var["r"], 4))
    m = _MASKS.get(key)
    if m is None:
        canvas = np.zeros((3 * n, 3 * n), np.uint8)
        pts = np.round((circle_polyline(var) + 1) * n).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, 1, 1)
        cv2.floodFill(canvas, np.zeros((3 * n + 2, 3 * n + 2), np.uint8), (0, 0), 2)
        m = canvas != 2
        if len(_MASKS) > 2048:
            _MASKS.clear()
        _MASKS[key] = m
    return m


def circle_contains(var: dict, pt_xy, margin: float = 0.0) -> bool:
    """Is `pt_xy` (normalized; scalars or arrays) inside the drawn circle? A template
    circle uses its own path; an ellipse ignores wobble."""
    if "tpl" in var:
        m = _tpl_mask(var)
        n = m.shape[0] // 3
        xi = np.floor((np.asarray(pt_xy[0], dtype=float) + 1) * n).astype(int)
        yi = np.floor((np.asarray(pt_xy[1], dtype=float) + 1) * n).astype(int)
        ok = (xi >= 0) & (xi < 3 * n) & (yi >= 0) & (yi < 3 * n)
        out = np.zeros(np.broadcast(xi, yi).shape, bool)
        out[ok] = m[yi[ok], xi[ok]]
        return bool(out) if out.ndim == 0 else out
    dx, dy = pt_xy[0] - var["cx"], pt_xy[1] - var["cy"]
    c, s = np.cos(-var["rot"]), np.sin(-var["rot"])
    u = c * dx - s * dy
    v = s * dx + c * dy
    a_ax = var["r"] * var["ecc"] * (1 - margin)
    b_ax = var["r"] / var["ecc"] * (1 - margin)
    return (u / max(a_ax, 1e-9)) ** 2 + (v / max(b_ax, 1e-9)) ** 2 <= 1.0


# ----------------------------------------------------------------- arrow ----

def sample_arrow_variant(clean: dict, seed: int, cfg: dict) -> dict:
    """clean: {x0, y0, x1, y1} normalized (tail -> head). cfg: imprecision.arrow section;
    cfg["place"] False keeps the clean endpoints (shape only)."""
    g = _rng(seed)
    p0 = np.array([clean["x0"], clean["y0"]], dtype=np.float64)
    p1 = np.array([clean["x1"], clean["y1"]], dtype=np.float64)
    if cfg.get("source") == "human":
        i = int(g.integers(len(templates())))
        t = templates()[i]
        d = p1 - p0
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        v = np.array([-u[1], u[0]])
        if cfg.get("place", True):                   # offsets in the clean arrow's frame
            p0 = p0 + u * t["tail_off"][0] + v * t["tail_off"][1]
            p1 = p1 + u * t["head_off"][0] + v * t["head_off"][1]
        return {"kind": "arrow", "seed": int(seed), "tpl": i,
                "x0": float(p0[0]), "y0": float(p0[1]), "x1": float(p1[0]), "y1": float(p1[1]),
                "head_style": "v", "width": UI["width"], "color": UI["arrow_color"]}
    jit = cfg["endpoint_jitter"]
    p0j = p0 + g.normal(0, jit, 2)
    p1j = p1 + g.normal(0, jit, 2)
    p1j = p0j + (p1j - p0j) * g.normal(1.0, cfg["length_sigma"])  # tail-anchored length noise
    L = float(np.linalg.norm(p1j - p0j))
    var = {
        "kind": "arrow",
        "seed": int(seed),
        "x0": float(p0j[0]), "y0": float(p0j[1]),
        "x1": float(p1j[0]), "y1": float(p1j[1]),
        "curv": float(g.normal(0, cfg["curvature_sigma"]) * L),
        "wobble": float(abs(g.normal(0, cfg["shaft_wobble"]))),
        "wobble_phase": float(g.uniform(0, 2 * np.pi)),
        "head_style": "v", "width": UI["width"], "color": UI["arrow_color"],
    }
    return var


def arrow_polylines(var: dict, n: int = 60) -> list[np.ndarray]:
    """Shaft (template path, or bezier + wobble) and the UI head, (N,2) normalized each."""
    p0 = np.array([var["x0"], var["y0"]])
    p1 = np.array([var["x1"], var["y1"]])
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return [np.stack([p0, p1])]
    perp = np.array([-d[1], d[0]]) / L
    if "tpl" in var:                                 # chord (0,0)->(1,0) onto tail->head
        q = np.asarray(templates()[var["tpl"]]["arrow"])
        shaft = p0 + q[:, :1] * d + q[:, 1:] * L * perp
        return [shaft, ui_head(shaft, var["width"])]
    t = np.linspace(0, 1, n)[:, None]
    ctrl = (p0 + p1) / 2 + perp * var["curv"]
    shaft = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    shaft += perp * (var["wobble"] * np.sin(3 * np.pi * t + var["wobble_phase"]))
    return [shaft, ui_head(shaft, var["width"])]


def arrow_head(var: dict) -> np.ndarray:
    return np.array([var["x1"], var["y1"]])


# ------------------------------------------------------------- sampling ----

def sample_variant_pair(circle_clean: dict, arrow_clean: dict | None, target_xy,
                        seed: int, cfg: dict, max_tries: int = 25) -> dict:
    """Sample one (circle, arrow) variant pair with sanity guards.

    target_xy: normalized target-object centroid the circle must keep containing.
    Returns {"seed", "circle", "arrow"} (arrow None if arrow_clean is None).
    """
    ccfg, acfg = cfg["circle"], cfg["arrow"]
    tau = cfg["arrow"]["head_tolerance"]
    for k in range(max_tries):
        s = seed + 7919 * k
        cvar = sample_circle_variant(circle_clean, s, ccfg)
        if not circle_contains(cvar, target_xy):
            continue
        avar = None
        if arrow_clean is not None:
            avar = sample_arrow_variant(arrow_clean, s + 1, acfg)
            drift = np.linalg.norm(arrow_head(avar) - np.array([arrow_clean["x1"], arrow_clean["y1"]]))
            if drift > tau:
                continue
        return {"seed": int(s), "circle": cvar, "arrow": avar}
    # guards exhausted: fall back to a barely-perturbed variant so output count is stable
    tight = {**ccfg, "center_jitter": 0.02, "radius_lognorm_sigma": 0.03}
    cvar = sample_circle_variant(circle_clean, seed + 104729, tight)
    avar = None
    if arrow_clean is not None:
        tight_a = {**acfg, "endpoint_jitter": 0.004, "length_sigma": 0.02}
        avar = sample_arrow_variant(arrow_clean, seed + 104730, tight_a)
    return {"seed": int(seed + 104729), "circle": cvar, "arrow": avar, "fallback": True}
