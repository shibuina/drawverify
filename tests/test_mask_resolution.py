import numpy as np

from libero_sketch import false_prompts as fp
from libero_sketch.captions import template_levels

S = 256


def disc(inst, idx, cx, cy, r):
    ys, xs = np.mgrid[0:S, 0:S]
    inst[(xs + 0.5 - cx * S) ** 2 + (ys + 0.5 - cy * S) ** 2 <= (r * S) ** 2] = idx


def obj(name, cx, cy, r, z=0.9, fixture=False):
    return {"name": name, "fixture": fixture, "visible": True, "anchor_3d": [cx, cy, z],
            "center_px": [cx * S, cy * S], "circle": {"cx": cx, "cy": cy, "r": r}}


def ctx_for(objects, *, kind="place", dest="plate_1"):
    inst = np.zeros((S, S), np.uint8)
    for i, o in enumerate(objects):                   # later objects occlude earlier ones
        disc(inst, i + 1, o["circle"]["cx"], o["circle"]["cy"], o["circle"]["r"])
    bowl = objects[-1]
    meta = {"camera": {"H": S, "W": S}, "instruction": "put this there",
            "goal": {"kind": kind, "target_object": bowl["name"], "destination": dest},
            "points_px": {"target_t0": bowl["center_px"]},
            "circle_clean": {**bowl["circle"], "r": bowl["circle"]["r"] * 1.12},
            "arrow_dest_clean": {"x0": bowl["circle"]["cx"], "y0": bowl["circle"]["cy"],
                                 "x1": 0.8, "y1": 0.2}}
    return fp.Ctx(meta=meta, scene={"objects": objects}, occupancy=None, caption_pool=[],
                  ladder=None, instances=inst)


def test_support_and_fixture_under_the_target_do_not_compete():
    plate = obj("plate_1", 0.8, 0.2, 0.08)
    for under in (obj("flat_stove_1", 0.5, 0.5, 0.06, z=0.85, fixture=True),
                  obj("ramekin_1", 0.5, 0.5, 0.06, z=0.85)):
        ctx = ctx_for([plate, under, obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
        assert [o["name"] for o in ctx.selected(ctx.circle_clean)] == ["akita_black_bowl_1"]
        assert fp.resolve(ctx, ctx.circle_clean, ctx.arrow_clean, "L3")[:2] == (1, 1)


def test_twins_compete_and_a_grazed_neighbour_does_not():
    twin = obj("akita_black_bowl_2", 0.62, 0.5, 0.05)
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), twin, obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    assert ctx.coverage(ctx.circle_clean, twin) < fp.COVER       # touching, mostly outside
    assert fp.resolve(ctx, ctx.circle_clean, ctx.arrow_clean, "L3")[:2] == (1, 1)
    pair = {"cx": 0.56, "cy": 0.5, "r": 0.12}
    assert fp.resolve(ctx, pair, ctx.arrow_clean, "L3")[0] == 2


def test_arrow_head_between_two_masks_is_ambiguous_and_on_one_is_not():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.06), obj("cookies_1", 0.8, 0.4, 0.06),
                   obj("akita_black_bowl_1", 0.3, 0.7, 0.05)])
    on_plate = {"x0": 0.3, "y0": 0.7, "x1": 0.8, "y1": 0.2}
    between = {**on_plate, "y1": 0.3}
    bare = {**on_plate, "x1": 0.3, "y1": 0.2}
    assert fp.resolve(ctx, ctx.circle_clean, on_plate, "L3")[1] == 1
    assert fp.resolve(ctx, ctx.circle_clean, between, "L3")[1] == 2
    assert fp.resolve(ctx, ctx.circle_clean, bare, "L3")[1] == 1   # bare table is one spot


def test_compound_instruction_takes_the_place_verb():
    l2 = template_levels("open the top drawer and put the bowl inside", "place")["L2"]
    assert l2 == ["put this there"]
    assert template_levels("open the middle drawer of the cabinet", "manipulate")["L2"] == ["open this"]


def test_head_on_destination_only_when_it_lands_on_the_destination():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.06), obj("cookies_1", 0.8, 0.5, 0.06),
                   obj("akita_black_bowl_1", 0.3, 0.7, 0.05)])
    sample = lambda x1, y1: fp.Sample(circle=ctx.circle_clean, caption="", labels={},  # noqa: E731
                                      arrow={"x0": 0.3, "y0": 0.7, "x1": x1, "y1": y1})
    assert fp.head_on_destination(ctx, sample(0.8, 0.2))
    assert not fp.head_on_destination(ctx, sample(0.8, 0.5))
    assert not fp.head_on_destination(ctx, sample(0.3, 0.2))


def test_rival_coverage_ignores_supports_and_counts_neighbours():
    under = obj("ramekin_1", 0.5, 0.5, 0.06, z=0.85)
    side = obj("akita_black_bowl_2", 0.6, 0.5, 0.05)
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), under, side, obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    assert fp.rival_coverage(ctx, ctx.circle_clean) == ctx.coverage(ctx.circle_clean, side)
    assert fp.rival_coverage(ctx, {"cx": 0.2, "cy": 0.8, "r": 0.05}) == 0.0


def test_rendered_ok_follows_the_drawn_arrow():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.06), obj("akita_black_bowl_1", 0.3, 0.7, 0.05)])
    circle = {**ctx.circle_clean, "ecc": 1.0, "rot": 0.0}
    drawn = lambda x1, y1: {"circle": circle, "arrow": {"x0": 0.3, "y0": 0.7, "x1": x1, "y1": y1}}  # noqa: E731
    right = {"verdict": "right", "mode": "POS", "labels": {}}
    b3 = {"verdict": "wrong", "mode": "B3", "labels": {"arrow_ok": False}}
    assert fp.rendered_ok(ctx, right, drawn(0.8, 0.2))
    assert not fp.rendered_ok(ctx, right, drawn(0.95, 0.05))      # overshoots the plate
    assert not fp.rendered_ok(ctx, b3, drawn(0.8, 0.2))           # corruption landed back on it
    assert fp.rendered_ok(ctx, b3, drawn(0.95, 0.05))


def test_valid_arrow_must_land_on_the_destination_not_beside_it():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.06), obj("akita_black_bowl_1", 0.3, 0.7, 0.05)])
    arrow = lambda x1: {"x0": 0.3, "y0": 0.7, "x1": x1, "y1": 0.2}  # noqa: E731
    assert fp.head_lands_on_destination(ctx, arrow(0.8))
    assert not fp.head_lands_on_destination(ctx, arrow(0.9))           # 10 px past the rim
    assert fp.head_on_destination(ctx, fp.Sample(circle=None, arrow=arrow(0.9), caption="", labels={}))


def test_demo_with_a_fully_hidden_target_is_skipped():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    assert ctx.annotation_consistent()
    ctx._px.pop("akita_black_bowl_1")                 # no target pixel in the instance map
    assert not ctx.annotation_consistent()


def test_task_blind_alternatives_and_corruptions():
    stove = obj("flat_stove_1", 0.2, 0.8, 0.08, z=0.85, fixture=True)
    twin = obj("akita_black_bowl_2", 0.3, 0.3, 0.05)
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), stove, twin,
                   obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    alts = set(fp.alternatives(ctx))
    assert ("akita_black_bowl_2", "plate_1") in alts                 # the other bowl -> plate
    assert ("akita_black_bowl_1", "flat_stove_1_cook_region") in alts
    assert ("akita_black_bowl_1", "plate_1") not in alts             # the demo itself
    assert ("akita_black_bowl_1", "akita_black_bowl_2") in alts       # stacking is valid
    actx = fp.alt_context(ctx, "akita_black_bowl_2", "plate_1")
    assert actx.target_object == "akita_black_bowl_2" and actx.dest_object == "plate_1"
    assert fp.circle_selects_target(actx, actx.circle_clean)
    assert fp.head_lands_on_destination(actx, actx.arrow_clean)
    to_stove = {"x0": 0.5, "y0": 0.5, "x1": 0.2, "y1": 0.8}
    assert fp.head_on_alternative(ctx, to_stove)                     # valid under task-blind
    # stacking is never generated as valid, but no corruption may point there either
    assert fp.head_on_alternative(ctx, {**to_stove, "x1": 0.3, "y1": 0.3})
    b3 = {"verdict": "wrong", "mode": "B3", "resolution": {}}
    assert not fp.rendered_ok(ctx, b3, {"circle": ctx.circle_clean, "arrow": to_stove})
    rng = np.random.default_rng(0)
    for _ in range(20):                                              # B1 never picks the stove
        s = fp.b1_wrong_destination(ctx, rng)
        assert s is None or (s.arrow["x1"], s.arrow["y1"]) != (0.2, 0.8)


def test_detached_tail_must_be_clearly_off_the_circle():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("ramekin_1", 0.3, 0.5, 0.05),
                   obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    circ = ctx.scene_circle(ctx.obj("ramekin_1"))       # C1: circle the ramekin
    rec = {"verdict": "wrong", "mode": "C1", "resolution": {}}
    far = {"x0": 0.5, "y0": 0.5, "x1": 0.8, "y1": 0.2}  # tail on the bowl, well clear
    near = {**far, "x0": circ["cx"] + circ["r"] + 0.01, "y0": circ["cy"]}  # tail at the rim
    assert fp.rendered_ok(ctx, rec, {"circle": circ, "arrow": far})
    assert not fp.rendered_ok(ctx, rec, {"circle": circ, "arrow": near})


def test_circle_only_is_valid_with_a_pick_verb_only():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("ramekin_1", 0.2, 0.8, 0.05),
                   obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    c = ctx.circle_clean
    assert fp.resolve(ctx, c, None, "L3", caption="grab this")[:2] == (1, 1)
    assert fp.resolve(ctx, c, None, "L3", caption="put this there")[1] > 1    # where?
    rng = np.random.default_rng(0)
    s = fp._finalize(ctx, fp.pick_positive(ctx, rng), "L3", rng, keep_caption=True)
    assert fp.is_pick(s.caption) and s.labels["sufficient_ok"]


def test_b8_head_stops_just_short_of_the_destination():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("akita_black_bowl_1", 0.3, 0.7, 0.05)])
    rng = np.random.default_rng(1)
    s = fp.b8_stops_short(ctx, rng)
    gap = ctx.mask_dist(ctx.obj("plate_1"), (s.arrow["x1"], s.arrow["y1"]))
    assert 0.045 <= gap <= 0.095 and not fp.head_lands_on_destination(ctx, s.arrow)


def test_a7_bad_circle_with_a_pick_verb_is_wrong():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("akita_black_bowl_2", 0.62, 0.5, 0.05),
                   obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    seen = 0
    for k in range(20):
        rng = np.random.default_rng(k)
        s = fp.a7_bad_circle_pick(ctx, rng)
        if s is None:
            continue
        rec = fp._finalize(ctx, s, "L3", rng, keep_caption=True).as_record("A7")
        assert s.arrow is None and fp.is_pick(s.caption) and rec["verdict"] == "wrong"
        seen += 1
    assert seen


def test_two_separate_loops_are_two_referents():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("ramekin_1", 0.25, 0.75, 0.05),
                   obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    two = {**ctx.circle_clean, "also": ctx.scene_circle(ctx.obj("ramekin_1"))}
    assert fp.resolve(ctx, two, ctx.arrow_clean, "L3")[0] == 2
    assert fp.resolve(ctx, ctx.circle_clean, ctx.arrow_clean, "L3")[0] == 1


def test_a_pick_verb_with_an_arrow_is_wrong_and_never_on_a_sketch_labelled_row():
    ctx = ctx_for([obj("plate_1", 0.8, 0.2, 0.08), obj("akita_black_bowl_1", 0.5, 0.5, 0.05)])
    ctx.ladder = {"levels": {"L3": {"templates": ["grab this", "pick up this", "put this there"]}}}
    for k in range(30):
        rng = np.random.default_rng(k)
        assert not fp.is_pick(fp._finalize(ctx, fp.positive(ctx, rng), "L3", rng).caption)
    picks = 0
    for k in range(40):
        rng = np.random.default_rng(k)
        rec = fp._finalize(ctx, fp.d3_wrong_verb(ctx, rng), "L3", rng).as_record("D3")
        assert rec["verdict"] == "wrong"
        picks += fp.is_pick(rec["prompt"]["caption"])
    assert picks
