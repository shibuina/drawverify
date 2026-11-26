"""Rasterize stroke variants onto images (OpenCV). Deterministic given the variant dict."""

from __future__ import annotations

import numpy as np

from . import strokes as S


def _to_px(poly_n: np.ndarray, H: int, W: int) -> np.ndarray:
    pts = np.asarray(poly_n, dtype=np.float64) * np.array([[W, H]])
    return np.round(pts).astype(np.int32)


def _draw_polys(canvas: np.ndarray, polys: list[np.ndarray], color, width_px: int,
                closed_fill: np.ndarray | None = None) -> None:
    import cv2
    H, W = canvas.shape[:2]
    for poly in polys:
        pts = _to_px(poly, H, W).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, width_px, lineType=cv2.LINE_AA)
    if closed_fill is not None:
        cv2.fillPoly(canvas, [_to_px(closed_fill, H, W).reshape(-1, 1, 2)], color,
                     lineType=cv2.LINE_AA)


def render_variant(variant: dict, H: int, W: int, rgb: np.ndarray | None = None,
                   alpha: float = 1.0) -> np.ndarray:
    """Render one {"circle", "arrow"} variant.

    rgb None  -> returns an RGBA overlay (H,W,4) with strokes only.
    rgb given -> returns RGB with strokes alpha-composited on top.
    Stroke widths in the variant are calibrated for 256px and scale with resolution.
    """
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    scale = max(1.0, min(H, W) / 256.0)

    cvar = variant.get("circle")
    if cvar:
        w_px = max(1, int(round(cvar["width"] * scale)))
        polys = [S.circle_polyline(cvar)]
        if cvar.get("also"):                            # a second, separate loop
            polys.append(S.circle_polyline(cvar["also"]))
        if cvar.get("double"):
            shrunk = {**cvar, "r": cvar["r"] * 0.93, "theta0": cvar["theta0"] + 0.4}
            polys.append(S.circle_polyline(shrunk))
        _draw_alpha(overlay, polys, cvar["color"], w_px)

    avar = variant.get("arrow")
    if avar:
        w_px = max(1, int(round(avar["width"] * scale)))
        polys = S.arrow_polylines(avar)
        fill = None
        if avar["head_style"] == "tri" and len(polys) >= 4:
            fill = np.stack([polys[0][-1], polys[1][1], polys[2][1]])
        _draw_alpha(overlay, polys, avar["color"], w_px, fill)

    if rgb is None:
        return overlay
    return composite(rgb, overlay, alpha)


def render_human(rgb: np.ndarray, circle_stroke, arrow_stroke) -> np.ndarray:
    """A study drawing (raw pointer paths, normalized) in the same fixed look the
    generator draws with (strokes.UI), so human and generated sketches differ only in
    the drawing itself."""
    H, W = rgb.shape[:2]
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    w_px = max(1, int(round(S.UI["width"] * max(1.0, min(H, W) / 256.0))))
    if circle_stroke is not None and len(circle_stroke) >= 2:
        _draw_alpha(overlay, [np.asarray(circle_stroke, dtype=np.float64)],
                    S.UI["circle_color"], w_px)
    if arrow_stroke is not None and len(arrow_stroke) >= 2:
        shaft = np.asarray(arrow_stroke, dtype=np.float64)
        _draw_alpha(overlay, [shaft, S.ui_head(shaft)], S.UI["arrow_color"], w_px)
    return composite(rgb, overlay)


def _draw_alpha(overlay_rgba: np.ndarray, polys, color, width_px: int, fill=None) -> None:
    stroke = np.zeros(overlay_rgba.shape[:2], dtype=np.uint8)
    _draw_polys(stroke, polys, 255, width_px, fill)
    m = stroke > 0
    overlay_rgba[m, :3] = np.array(color, dtype=np.uint8)
    overlay_rgba[m, 3] = np.maximum(overlay_rgba[m, 3], stroke[m])


def composite(rgb: np.ndarray, overlay_rgba: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    a = (overlay_rgba[..., 3:4].astype(np.float32) / 255.0) * alpha
    out = rgb.astype(np.float32) * (1 - a) + overlay_rgba[..., :3].astype(np.float32) * a
    return out.astype(np.uint8)


def clean_variant(circle: dict | None, arrow: dict | None) -> dict:
    """Variant dict for the un-augmented annotation (perfect circle + straight arrow)."""
    variant = {"circle": None, "arrow": None}
    if circle:
        variant["circle"] = {
            "cx": circle["cx"], "cy": circle["cy"], "r": circle["r"], "ecc": 1.0,
            "rot": 0.0, "theta0": 0.0, "span": 2 * np.pi,
            "wobble_amps": [0, 0, 0, 0], "wobble_phases": [0, 0, 0, 0],
            "double": False, "width": S.UI["width"], "color": S.UI["circle_color"]}
    if arrow:
        variant["arrow"] = {
            "x0": arrow["x0"], "y0": arrow["y0"], "x1": arrow["x1"], "y1": arrow["y1"],
            "curv": 0.0, "wobble": 0.0, "wobble_phase": 0.0, "head_style": "v",
            "width": S.UI["width"], "color": S.UI["arrow_color"]}
    return variant


def render_clean(circle: dict | None, arrow: dict | None, H: int, W: int,
                 rgb: np.ndarray | None = None) -> np.ndarray:
    return render_variant(clean_variant(circle, arrow), H, W, rgb)
