import numpy as np

from libero_sketch import strokes as S
from libero_sketch.render import clean_variant, composite, render_clean, render_variant

CIRCLE = {"cx": 0.5, "cy": 0.5, "r": 0.1}
ARROW = {"x0": 0.5, "y0": 0.5, "x1": 0.8, "y1": 0.7}


def test_overlay_has_strokes(imp):
    v = S.sample_variant_pair(CIRCLE, ARROW, (0.5, 0.5), seed=5, cfg=imp)
    ov = render_variant(v, 256, 256)
    assert ov.shape == (256, 256, 4)
    assert (ov[..., 3] > 0).sum() > 100


def test_composite_changes_pixels(imp):
    v = S.sample_variant_pair(CIRCLE, ARROW, (0.5, 0.5), seed=5, cfg=imp)
    rgb = np.full((256, 256, 3), 120, dtype=np.uint8)
    out = render_variant(v, 256, 256, rgb=rgb)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert (out != rgb).any()


def test_render_deterministic(imp):
    v = S.sample_variant_pair(CIRCLE, ARROW, (0.5, 0.5), seed=9, cfg=imp)
    assert np.array_equal(render_variant(v, 256, 256), render_variant(v, 256, 256))


def test_clean_render_and_missing_arrow():
    out = render_clean(CIRCLE, None, 128, 128)
    assert (out[..., 3] > 0).any()
    v = clean_variant(CIRCLE, ARROW)
    assert v["arrow"]["head_style"] == "v"


def test_resolution_independence(imp):
    v = S.sample_variant_pair(CIRCLE, ARROW, (0.5, 0.5), seed=4, cfg=imp)
    small = render_variant(v, 128, 128)
    big = render_variant(v, 512, 512)
    # stroke mass should land in the same normalized region at both resolutions
    ys, xs = np.nonzero(small[..., 3])
    yb, xb = np.nonzero(big[..., 3])
    assert abs(xs.mean() / 128 - xb.mean() / 512) < 0.03
    assert abs(ys.mean() / 128 - yb.mean() / 512) < 0.03


def test_alpha_blend_bounds():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    ov = np.zeros((64, 64, 4), dtype=np.uint8)
    ov[10:20, 10:20] = [255, 0, 0, 255]
    half = composite(rgb, ov, alpha=0.5)
    assert half[15, 15, 0] in range(120, 136)
