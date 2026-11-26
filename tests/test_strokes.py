import numpy as np

from libero_sketch import strokes as S

CIRCLE = {"cx": 0.5, "cy": 0.45, "r": 0.08}
ARROW = {"x0": 0.5, "y0": 0.45, "x1": 0.75, "y1": 0.6}
TARGET = (0.5, 0.45)


def test_deterministic_given_seed(imp):
    a = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=123, cfg=imp)
    b = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=123, cfg=imp)
    assert a == b
    pa = S.circle_polyline(a["circle"])
    pb = S.circle_polyline(b["circle"])
    assert np.allclose(pa, pb)


def test_different_seeds_differ(imp):
    a = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=1, cfg=imp)
    b = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=2, cfg=imp)
    assert a["circle"]["cx"] != b["circle"]["cx"]


def test_guard_target_stays_inside(imp):
    for seed in range(50):
        v = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=seed, cfg=imp)
        assert S.circle_contains(v["circle"], TARGET)


def test_guard_arrow_head_bounded(imp):
    tau = imp["arrow"]["head_tolerance"]
    clean_head = np.array([ARROW["x1"], ARROW["y1"]])
    for seed in range(50):
        v = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=seed, cfg=imp)
        if v.get("fallback"):
            continue
        drift = np.linalg.norm(S.arrow_head(v["arrow"]) - clean_head)
        assert drift <= tau + 1e-9


def test_fallback_when_guards_impossible(imp):
    # target far outside the circle: containment can never hold -> tight fallback
    v = S.sample_variant_pair(CIRCLE, ARROW, (0.95, 0.95), seed=7, cfg=imp, max_tries=5)
    assert v.get("fallback") is True


def test_arrow_without_circle_arrow_none(imp):
    v = S.sample_variant_pair(CIRCLE, None, TARGET, seed=11, cfg=imp)
    assert v["arrow"] is None


def test_polyline_shapes(imp):
    v = S.sample_variant_pair(CIRCLE, ARROW, TARGET, seed=3, cfg=imp)
    circ = S.circle_polyline(v["circle"], n=100)
    assert circ.ndim == 2 and circ.shape[1] == 2 and len(circ) >= 8
    polys = S.arrow_polylines(v["arrow"])
    assert len(polys) >= 1 and all(p.shape[1] == 2 for p in polys)


def test_human_strokes_replay_a_template_in_the_ui_look(imp):
    looks = set()
    for seed in range(30):
        c = S.sample_circle_variant(CIRCLE, seed, imp["circle"])
        a = S.sample_arrow_variant(ARROW, seed, imp["arrow"])
        looks.add((tuple(c["color"]), c["width"], tuple(a["color"]), a["width"], a["head_style"]))
        shaft, head = S.arrow_polylines(a)
        assert np.allclose(shaft[0], [a["x0"], a["y0"]]) and np.allclose(shaft[-1], [a["x1"], a["y1"]])
        assert np.allclose(head[1], shaft[-1]) and len(head) == 3      # V: wing, tip, wing
        # the drawn path decides containment: its own centre in, a point 3 radii out not
        assert not S.circle_contains(c, (c["cx"] + 3 * c["r"], c["cy"]))
    assert looks == {(tuple(S.UI["circle_color"]), S.UI["width"], tuple(S.UI["arrow_color"]),
                      S.UI["width"], "v")}
    inside = [S.circle_contains(S.sample_circle_variant(CIRCLE, s, imp["circle"]), TARGET)
              for s in range(60)]
    assert np.mean(inside) > 0.8           # good-faith human circles mostly hold the target


def test_template_containment_is_vectorised(imp):
    c = S.sample_circle_variant(CIRCLE, 5, imp["circle"])
    xs = np.array([c["cx"], c["cx"] + 3 * c["r"], -0.5])
    out = S.circle_contains(c, (xs, np.full(3, c["cy"])))
    assert out.shape == (3,) and not out[1] and not out[2]
