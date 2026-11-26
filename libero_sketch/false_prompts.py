"""Binary prompt generation: controlled corruption of name-free deictic sketch prompts.

Implements the failure taxonomy in ../failure_modes.md (paper Table I). Sim-free — consumes stage-01
`meta.json`, stage-08 `scene.json` (+ occupancy) and the stage-03 caption pool.

Core principle: `strokes.py` already samples semantic-preserving prompts behind two sanity
guards (`circle_contains(target)`, arrow head `drift < head_tolerance`). A wrong prompt is
the same generator with those guards inverted or the corruption pushed past the
semantic-preserving threshold, so negatives share the positive rendering path and the
corruption magnitude is a scalar (`severity`) that sweeps the decision boundary.

Each mode is `fn(ctx, rng) -> Sample | None`; None means "not applicable to this demo"
(e.g. B1 needs a second container). Registry METADATA carries the category/tier/solution
axes of the taxonomy.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import strokes as _S

# ---------------------------------------------------------------- registry ----

# mode -> (category, tier, solution, data_amount)
METADATA = {
    "A2": ("A", "easy",   "vlm",       "large"),
    "A3": ("A", "hard",   "vlm",       "curated"),
    "A4": ("A", "easy",   "vlm",       "large"),
    "A5": ("A", "hard",   "vlm",       "curated"),
    "A6": ("A", "easy",   "algorithm", "large"),
    "A7": ("A", "medium", "vlm",       "curated"),
    "B1": ("B", "hard",   "vlm",       "curated"),
    "B3": ("B", "hard",   "algorithm", "curated"),
    "B4": ("B", "medium", "algorithm", "curated"),
    "B5": ("B", "easy",   "algorithm", "large"),
    "B6": ("B", "hard",   "algorithm", "curated"),
    "B7": ("B", "hard",   "algorithm", "curated"),
    "B8": ("B", "hard",   "algorithm", "curated"),
    "C1": ("C", "hard",   "vlm",       "curated"),
    "C2": ("C", "hard",   "vlm",       "curated"),
    "C3": ("C", "medium", "vlm",       "curated"),
    "D1": ("D", "medium", "vlm",       "curated"),
    "D2": ("D", "easy",   "vlm",       "large"),
    "D3": ("D", "medium", "vlm",       "curated"),
    "D4": ("D", "hard",   "llm",       "curated"),
}

CATEGORY_NAMES = {
    "A": "Referential failures (circle / which)",
    "B": "Directional failures (arrow / where)",
    "C": "Cross-modal consistency",
    "D": "Language-only failures (caption / what)",
    "G": "Deictic grounding failures",
    "P": "Verified positives",
}

REGISTRY: dict[str, Callable] = {}


def mode(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


# ------------------------------------------------------------------ sample ----

@dataclass
class Sample:
    """One generated prompt. ``labels.right`` mirrors the binary complete verdict.

    `guard` states what the *rendered* stroke must still encode, so hand-drawn jitter can
    never silently flip this sample's label (see `hand_variant`):
      contains -- normalized points the drawn circle must enclose
      excludes -- normalized points it must NOT enclose
    """
    circle: dict | None
    arrow: dict | None
    caption: str
    labels: dict
    severity: float = 1.0
    note: str = ""
    guard: dict = field(default_factory=lambda: {"contains": [], "excludes": []})
    # ambiguity axis — how many readings survive the prompt (computed by resolve())
    n_referents: int = 1
    n_destinations: int = 1
    caption_level: str = "L0"
    resolved_by: str = ""
    grounding_kind: str = ""      # grounding configuration for positive-base samples

    def as_record(self, mode_name: str) -> dict:
        cat, tier, solution, amount = METADATA.get(mode_name, ("P", "n/a", "n/a", "n/a"))
        labels = {**self.labels}
        labels["right"] = verdict_of(labels) == "right"
        return {
            "mode": mode_name,
            "category": cat,
            "category_name": CATEGORY_NAMES.get(cat, "Positive"),
            "tier": tier,
            "solution": solution,
            "data_amount": amount,
            "severity": round(float(self.severity), 4),
            "labels": labels,
            "verdict": verdict_of(labels),
            "resolution": {"n_referents": int(self.n_referents),
                           "n_destinations": int(self.n_destinations),
                           "n_readings": int(self.n_referents * self.n_destinations),
                           "caption_level": self.caption_level,
                           "resolved_by": self.resolved_by,
                           "kind": self.grounding_kind},
            "prompt": {"circle": self.circle, "arrow": self.arrow,
                       "caption": self.caption},
            "guard": self.guard,
            "note": self.note,
        }


def _ok(**over) -> dict:
    base = {"circle_ok": True, "arrow_ok": True, "caption_ok": True,
            "consistency_ok": True, "sufficient_ok": True}
    base.update(over)
    return base


# channels whose failure makes a prompt *wrong* (as opposed to merely under-determined)
_WRONG_CHANNELS = ("circle_ok", "arrow_ok", "caption_ok", "consistency_ok")


def verdict_of(labels: dict) -> str:
    """Binary verdict for the complete multimodal prompt.

    All captions are deliberately deictic: the sketch must ground "this" and "there".
    A prompt is right only when every channel is correct *and* exactly one executable
    reading survives. Formerly ambiguous/under-determined prompts are retained as wrong.
    """
    channels_ok = all(labels.get(k, True) for k in _WRONG_CHANNELS)
    return "right" if channels_ok and labels.get("sufficient_ok", True) else "wrong"


def _guard(contains=(), excludes=()) -> dict:
    return {"contains": [list(map(float, p)) for p in contains],
            "excludes": [list(map(float, p)) for p in excludes]}


def set_record_category(record: dict, category: str) -> dict:
    """Reclassify a record without changing its diagnostic mode or verdict."""
    record["category"] = category
    record["category_name"] = CATEGORY_NAMES[category]
    if category == "G":
        record.update(tier="hard", solution="vlm", data_amount="large")
    return record


# Fixed look shared by the `clean` and `shape` render styles (see render.clean_variant).
SHAPE_STYLE = {
    "circle": {"width": _S.UI["width"], "color": _S.UI["circle_color"]},
    "arrow": {"width": _S.UI["width"], "color": _S.UI["arrow_color"], "head_style": "v"},
}


def shape_imprecision(imp: dict) -> dict:
    """The stroke model with every placement term removed.

    The circle keeps its exact centre and radius and the arrow its exact endpoints; only how
    the stroke is drawn varies — circle eccentricity, angular gaps/overshoot, radial wobble
    and double strokes; arrow shaft curvature and wobble. Width, colour, head and start blob
    are pinned by SHAPE_STYLE.
    """
    c = {**imp["circle"], "center_jitter": 0.0, "radius_lognorm_sigma": 0.0, "place": False}
    a = {**imp["arrow"], "endpoint_jitter": 0.0, "length_sigma": 0.0, "place": False}
    return {"circle": c, "arrow": a}


def shape_variant(circle: dict | None, arrow: dict | None, guard: dict | None,
                  imp: dict, seed: int, max_tries: int = 40) -> dict:
    """Hand-drawn *shape* over exact placement: `hand_variant` with the placement terms
    zeroed, then colour/width/head pinned and the declared geometry re-asserted (the guard
    fallback inside hand_variant may nudge the centre; undo that)."""
    var = hand_variant(circle, arrow, guard, shape_imprecision(imp), seed, max_tries)
    if var.get("circle") and circle:
        var["circle"].update(cx=circle["cx"], cy=circle["cy"], r=circle["r"],
                             **SHAPE_STYLE["circle"])
    if var.get("arrow") and arrow:
        var["arrow"].update(x0=arrow["x0"], y0=arrow["y0"], x1=arrow["x1"], y1=arrow["y1"],
                            **SHAPE_STYLE["arrow"])
    return var


def hand_variant(circle: dict | None, arrow: dict | None, guard: dict | None,
                 imp: dict, seed: int, max_tries: int = 40) -> dict:
    """Hand-drawn strokes over declared params, resampled until the drawn stroke still
    encodes the intended label. Returns a render.py-compatible variant dict.

    Stage 02's guards keep sloppiness from flipping a *correct* label. Here the label is
    deliberately flipped, so the guards must be re-aimed rather than skipped: jitter must
    not drag a correct channel off its referent, and — just as important — must not nudge a
    near-miss (A3 at 1.05r) back onto the target it is supposed to miss. Both directions
    are label noise. Falls back to reduced jitter if the guards can't be satisfied.
    """
    from . import strokes as S

    g = guard or {"contains": [], "excludes": []}
    var = {"circle": None, "arrow": None}

    def _holds(cv) -> bool:
        return (all(S.circle_contains(cv, p) for p in g.get("contains", []))
                and not any(S.circle_contains(cv, p) for p in g.get("excludes", [])))

    if circle:
        ok = None
        for k in range(max_tries):
            cv = S.sample_circle_variant(circle, seed + 7919 * k, imp["circle"])
            if _holds(cv):
                ok = cv
                break
        if ok is None:                      # guards exhausted -> near-clean fallback
            tight = {**imp["circle"], "center_jitter": 0.02,
                     "radius_lognorm_sigma": 0.03, "wobble": 0.02, "place": False}
            ok = S.sample_circle_variant(circle, seed + 104729, tight)
            ok["guard_fallback"] = True
            if not _holds(ok):
                # Even reduced jitter can miss when the guard is tight (A2 circling an
                # object adjacent to the target must contain one and exclude the other).
                # The declared geometry satisfies the guard by construction, so drop to
                # zero jitter: a label that survives is worth more than a wobblier stroke.
                ok = {**S.sample_circle_variant(circle, seed + 104729,
                                                {**tight, "source": "parametric"}),
                      "cx": circle["cx"], "cy": circle["cy"], "r": circle["r"],
                      "ecc": 1.0, "rot": 0.0, "wobble_amps": [0, 0, 0, 0],
                      "guard_fallback": True, "guard_exact": True}
        if circle.get("also"):                     # the second loop of a two-loop A5
            o = circle["also"]
            for k in range(max_tries):
                cv2 = S.sample_circle_variant(o, seed + 7919 * k + 3, imp["circle"])
                if S.circle_contains(cv2, (o["cx"], o["cy"])):
                    break
            else:
                cv2 = S.sample_circle_variant(o, seed + 3, {**imp["circle"], "place": False})
            ok["also"] = cv2
        var["circle"] = ok

    if arrow:
        # the head must land where the corruption put it, or the sample loses its meaning
        tau = imp["arrow"]["head_tolerance"]
        intended = np.array([arrow["x1"], arrow["y1"]])
        ok = None
        for k in range(max_tries):
            av = S.sample_arrow_variant(arrow, seed + 104729 * k + 1, imp["arrow"])
            if np.linalg.norm(S.arrow_head(av) - intended) <= tau:
                ok = av
                break
        if ok is None:
            tight = {**imp["arrow"], "endpoint_jitter": 0.004, "length_sigma": 0.02,
                     "place": False}
            ok = S.sample_arrow_variant(arrow, seed + 15485863, tight)
            ok["guard_fallback"] = True
        var["arrow"] = ok
    return var


# ----------------------------------------------------------------- context ----

# vendor/material qualifiers that never surface in LIBERO instructions
_STOPWORDS = {"akita", "glazed", "rim", "porcelain", "flat", "new", "salad",
              "chefmate", "target"}
_VERBS = ["pick up", "put", "place", "open", "close", "turn on", "turn off", "push"]


# Mask-based grounding (stage 08's frames/t0_objects.png). A circle selects an object when it
# encloses at least COVER of the object's visible mask; an arrow head indicates the objects
# whose masks lie within max(FLOOR, REL x nearest) of it, or bare table (one unique "there")
# when no mask is within NEAR.
COVER, FLOOR, REL, NEAR = 0.5, 0.02, 1.5, 0.06
# A valid prompt keeps a margin from that cut: no competing object more than 30% inside its
# circle. Between 30% and the deliberate ambiguity negatives (100%) nothing is generated.
VALID_MARGIN = 0.30
MIN_TARGET_PX = 60     # shortcut: absolute visible-pixel floor at 256 px; a ratio needs full masks
TAIL_CLEAR = 0.08      # A2/C1/B6: drawn tail at least ~20 px from the drawn circle
NEAR_TAIL_CLEAR = 0.05 # A3/D4: the tail on the target, ~13 px off the near-miss circle
EDGE_CLEAR = 0.06      # a valid head at least ~15 px inside the image
BETWEEN_CLEAR = 0.04   # dest_between: drawn head ~10 px off both objects, so it touches neither
MIN_DEST_PX = 150
MIN_ARROW_VIS = 0.06
NEAR_MISS_COVER = 0.15 # A3/D4: a near miss takes in at most 15% of the target
LAND = 0.012           # a valid head tip within ~3 px of the destination mask (FLOOR is 5)   # on-image arrow length (~15 px)      # an alternative destination must be visibly there (not a sliver)      # A2/C1/B6: drawn tail at least ~13 px from the drawn circle

# Task-blind validity: a prompt is right when it shows one sensible action, whichever object
# it circles, as the human labellers judge it (labeling/README.md). "Sensible" generalises
# LIBERO's own goals (every On/In goal of libero_90/10/goal/spatial/object) by object class.
# Destinations the arrow alone cannot resolve (cabinet drawer vs top, shelf layer, table
# region) are left out, and so is stacking a bowl on a bowl.
# shortcut: hand-curated from the BDDL goals; extend PLACE_INTO when a new category appears.
_GROCERY = {"alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cream_cheese",
            "ketchup", "milk", "orange_juice", "salad_dressing", "new_salad_dressing",
            "tomato_sauce"}
_BOWL = {"akita_black_bowl", "white_bowl"}
_MUG = {"porcelain_mug", "red_coffee_mug", "white_yellow_mug"}
PLACE_INTO = {          # destination category -> (predicate, goal region suffix, what it takes)
    "basket": ("In", "contain_region", _GROCERY),
    "wooden_tray": ("In", "contain_region", _GROCERY | _BOWL),
    "plate": ("On", "", _BOWL | _MUG),
    "flat_stove": ("On", "cook_region", _BOWL | {"moka_pot", "chefmate_8_frypan"}),
    "microwave": ("In", "heating_region", _MUG),
    "wine_rack": ("On", "top_region", {"wine_bottle"}),
    "desk_caddy": ("In", "front_contain_region", {"black_book", "yellow_book"}),
    "akita_black_bowl": ("On", "", {"cream_cheese"} | _BOWL),   # stacking (libero_90)
    "white_bowl": ("On", "", _BOWL),
}
# Places people accept but that we do not generate as valid (user, 2026-10-24): no
# corruption may point at them.
SENSIBLE_EXTRA = {"cookies": _BOWL, "flat_stove": {"plate"}}
# Holders whose region the arrow alone cannot pick (drawer vs top, shelf layer): no valid
# alternative is generated onto them, but an arrow there is still a sensible place, so no
# corruption may point at one either.
OPEN_HOLDERS = {"wooden_cabinet", "white_cabinet", "wooden_two_layer_shelf"}
# Fixtures an open/close/turn task's circle could equally mean (A2 must avoid them).
MANIPULABLE = {"Open": {"wooden_cabinet", "white_cabinet", "microwave"},
               "Close": {"wooden_cabinet", "white_cabinet", "microwave"},
               "TurnOn": {"flat_stove"}, "TurnOff": {"flat_stove"}}


@dataclass
class Ctx:
    meta: dict
    scene: dict
    occupancy: np.ndarray | None
    caption_pool: list[dict]          # [{"task","instruction","kind"}] across the suite
    ladder: dict | None               # this task's L0-L3 caption ladder
    radius_scale: float = 1.12
    instances: np.ndarray | None = None   # t0_objects.png: i+1 = scene["objects"][i]
    _by_name: dict = field(default_factory=dict)
    _px: dict = field(default_factory=dict)
    _arrow: dict | None = None

    def __post_init__(self):
        self._by_name = {o["name"]: o for o in self.scene["objects"]}
        if self.instances is not None:
            H, W = self.instances.shape
            for i, o in enumerate(self.scene["objects"]):
                ys, xs = np.nonzero(self.instances == i + 1)
                if len(xs):
                    self._px[o["name"]] = np.stack([(xs + 0.5) / W, (ys + 0.5) / H], 1)
        self._arrow = self._snap_head(self.meta.get("arrow_dest_clean"))

    def _owner(self, name: str | None) -> str | None:
        """The scene object a goal argument names, or owns ("wooden_cabinet_1_top_region"
        -> "wooden_cabinet_1"); None for a table region."""
        names = sorted(self._by_name, key=len, reverse=True)
        return next((n for n in names if name == n or (name or "").startswith(n + "_")), None)

    @property
    def target_object(self) -> str | None:
        return self._owner(self.target_name)

    def can_act_on(self, o: dict) -> bool:
        """Affordance filter for referents: a place task moves an object, never a fixture;
        open/turn-on acts on a fixture. Scene dumps without the flag keep every object."""
        if "fixture" not in o:
            return True
        return o["fixture"] == (self.kind != "place")

    # -- geometry -------------------------------------------------------------
    @property
    def H(self) -> int: return self.meta["camera"]["H"]

    @property
    def W(self) -> int: return self.meta["camera"]["W"]

    @property
    def circle_clean(self) -> dict:
        c = self.meta["circle_clean"]
        return {"cx": c["cx"], "cy": c["cy"], "r": c["r"]}

    @property
    def arrow_clean(self) -> dict | None:
        return self._arrow

    def _snap_head(self, a: dict | None) -> dict | None:
        """Stage 01's arrow, its head moved inside the destination's mask when it misses it.

        The head is the object's final position, which for a drawer task is the pulled-out
        drawer: off the closed cabinet the t0 frame shows. It moves to the nearest point at
        least 8 px deep in the mask, so it does not sit on the edge next to a neighbour.
        Table regions have no mask and keep their head.
        """
        dest = self.dest_object
        if a is None or self.instances is None or dest not in self._px:
            return a
        px = self._px[dest]
        if np.hypot(px[:, 0] - a["x1"], px[:, 1] - a["y1"]).min() <= FLOOR:
            return a
        import cv2
        H, W = self.instances.shape
        mask = (self.instances == self.scene["objects"].index(self._by_name[dest]) + 1)
        depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        ys, xs = np.nonzero(depth >= 8) if (depth >= 8).any() else np.nonzero(mask)
        d = np.hypot((xs + 0.5) / W - a["x1"], (ys + 0.5) / H - a["y1"])
        i = int(d.argmin())
        return {**a, "x1": float((xs[i] + 0.5) / W), "y1": float((ys[i] + 0.5) / H)}

    @property
    def dest_object(self) -> str | None:
        return self._owner(self.dest_name)

    @property
    def target_name(self) -> str:
        return self.meta["goal"]["target_object"]

    @property
    def dest_name(self) -> str | None:
        return self.meta["goal"].get("destination")

    @property
    def kind(self) -> str:
        return self.meta["goal"].get("kind", "place")

    @property
    def suite(self) -> str:
        return self.meta.get("suite", "")

    @property
    def instruction(self) -> str:
        return self.meta["instruction"]

    @property
    def target_xy(self) -> np.ndarray:
        px = self.meta["points_px"]["target_t0"]
        return np.array([px[0] / self.W, px[1] / self.H])

    def obj(self, name: str) -> dict | None:
        return self._by_name.get(name)

    def visible(self) -> list[dict]:
        return [o for o in self.scene["objects"] if o["visible"] and o["circle"]]

    def others(self, exclude_same_category: bool = False) -> list[dict]:
        """Visible objects that are not the target (optionally excluding same-category)."""
        tb = _basename(self.target_name)
        out = []
        for o in self.visible():
            if o["name"] == self.target_name:
                continue
            if exclude_same_category and _basename(o["name"]) == tb:
                continue
            out.append(o)
        return out

    def scene_circle(self, o: dict) -> dict:
        """An object's mask circle, scaled like stage-01's clean circle."""
        c = o["circle"]
        return {"cx": c["cx"], "cy": c["cy"], "r": c["r"] * self.radius_scale}

    def annotation_consistent(self) -> bool:
        """Does stage 01's clean circle actually contain the projected target?

        It should by construction, but a partial/occluded mask can yield a tiny enclosing
        circle far from the body centre. Such a demo's *positive* prompt is already wrong,
        so every mode built on it would be mislabelled — skip the demo instead.
        """
        if self.instances is not None and len(self._px.get(self.target_object, ())) < MIN_TARGET_PX:
            return False                    # target (nearly) hidden: long/00 soup behind the milk
        return circle_holds(self.circle_clean, self.target_xy)

    def twins(self) -> list[dict]:
        """Visible objects of the target's own category, excluding the target itself.

        LIBERO spatial is built on these: every task ships two identical black bowls and
        only the L0 spatial phrase tells them apart.
        """
        tb = _basename(self.target_name)
        return [o for o in self.visible()
                if o["name"] != self.target_name and _basename(o["name"]) == tb]

    def in_circle(self, circle: dict) -> list[dict]:
        """Visible objects whose centre falls inside `circle`."""
        cv = {**circle, "ecc": 1.0, "rot": 0.0}
        from . import strokes as S
        return [o for o in self.visible() if S.circle_contains(cv, _center(o))]

    def coverage(self, circle: dict, o: dict) -> float:
        """Share of `o`'s visible mask inside the (possibly elliptical) drawn circle."""
        from . import strokes as S
        px = self._px.get(o["name"])
        if px is None:
            return 0.0
        cv = {"ecc": 1.0, "rot": 0.0, **circle}
        return float(np.mean(S.circle_contains(cv, (px[:, 0], px[:, 1]))))

    def target_coverage(self, circle: dict | None) -> float:
        t = self.obj(self.target_object or "")
        return self.coverage(circle, t) if circle is not None and t is not None else 0.0

    def disc_share(self, circle: dict, o: dict) -> float:
        """Share of the drawn circle's area lying on `o`: a circle on a drawer or a knob
        marks part of a fixture without enclosing it."""
        from . import strokes as S
        H, W = self.instances.shape
        ys, xs = np.mgrid[0:H, 0:W]
        cv = {"ecc": 1.0, "rot": 0.0, **circle}
        inside = S.circle_contains(cv, ((xs + 0.5) / W, (ys + 0.5) / H))
        if not inside.any():
            return 0.0
        idx = self.scene["objects"].index(o) + 1
        return float(np.mean(self.instances[inside] == idx))

    def selected(self, circle: dict) -> list[dict]:
        """Objects the circle selects among those the task can act on: >= COVER of their
        mask enclosed, or, for a fixture, >= COVER of the circle lying on it. Without an
        instance map (old scene dumps) fall back to object centres."""
        if self.instances is None:
            return self.in_circle(circle)
        sel = [o for o in self.visible() if self.can_act_on(o) and (
            self.coverage(circle, o) >= COVER
            or (o.get("fixture") and self.disc_share(circle, o) >= COVER))]
        # A bowl standing on a ramekin: circling the stack means the top object.
        return [o for o in sel if not any(_supports(o, u) for u in sel if u is not o)]

    def gap_midpoint(self, o1: dict, o2: dict) -> np.ndarray:
        """Midpoint of the closest pixel pair of two masks (centres without masks)."""
        p, q = self._px.get(o1["name"]), self._px.get(o2["name"])
        if p is None or q is None:
            return (np.asarray(_center(o1)) + np.asarray(_center(o2))) / 2
        p, q = p[:: max(1, len(p) // 800)], q[:: max(1, len(q) // 800)]
        d = np.hypot(p[:, None, 0] - q[None, :, 0], p[:, None, 1] - q[None, :, 1])
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        return (p[i] + q[j]) / 2

    def mask_dist(self, o: dict, pt) -> float:
        px = self._px.get(o["name"])
        if px is None:
            return float(np.linalg.norm(np.asarray(_center(o)) - np.asarray(pt)))
        return float(np.min(np.hypot(px[:, 0] - pt[0], px[:, 1] - pt[1])))

    def indicated(self, pt, exclude=(), bare_table: bool = True) -> list[dict]:
        """Objects a stroke endpoint indicates, by distance to their masks. [] = bare table
        (a unique spot) when `bare_table` and no mask is within NEAR."""
        if self.instances is None:
            return self.pointed_at(pt)
        ds = [(self.mask_dist(o, pt), o) for o in self.visible() if o["name"] not in exclude]
        if not ds:
            return []
        dmin = min(d for d, _ in ds)
        if bare_table and dmin > NEAR:
            return []
        return [o for d, o in ds if d <= max(dmin * REL, FLOOR)]

    def pointed_at(self, pt, floor: float = 0.05, rel: float = 1.4) -> list[dict]:
        """Which objects a stroke endpoint plausibly indicates.

        Relative, not fixed-radius: an endpoint sitting on an object indicates just that
        one, but an endpoint placed *between* two comparably-distant objects indicates
        both (that is the whole point of an ambiguous arrow — a fixed tolerance around the
        midpoint would land near neither and miss the ambiguity). Counts every object
        within `rel`x the nearest, with a `floor` so a bullseye still resolves to one.
        """
        vis = self.visible()
        if not vis:
            return []
        p = np.asarray(pt, dtype=float)
        ds = [(float(np.linalg.norm(np.asarray(_center(o)) - p)), o) for o in vis]
        dmin = min(d for d, _ in ds)
        thr = max(dmin * rel, floor)
        return [o for d, o in ds if d <= thr]

    # -- language -------------------------------------------------------------
    def surface(self, obj_name: str) -> str | None:
        """Longest suffix n-gram of the object name that appears in the instruction."""
        return _surface_form(obj_name, self.instruction)

    def level(self, lv: str, rng) -> str | None:
        """Sample a caption from a degradation level (templates + LLM paraphrases)."""
        if not self.ladder:
            return None
        node = self.ladder.get("levels", {}).get(lv, {})
        # Deictic training captions use deterministic templates only. This prevents an LLM
        # paraphrase from hallucinating an explicit noun that happens not to be in the
        # current scene inventory and therefore escaping an object-name lookup.
        pool = list(node.get("templates", []))
        if lv in _DEICTIC:
            pool = [caption for caption in pool
                    if is_name_free_deictic_caption(self, str(caption))]
        return str(pool[rng.integers(len(pool))]) if pool else None


_DEICTIC = ("L2", "L3")            # captions that name no object ("this"/"there")
_NAME_FREE_DEICTIC_RE = re.compile(
    r"^(?:(?:pick up|put|place|move|push|open|close|turn on|turn off|grab|grasp|take|"
    r"stack|do|interact with|handle|set)\s+)?this(?:\s+one)?"
    r"(?:\s*,?\s*(?:there|over there|here))?[.!]?$",
    re.IGNORECASE,
)


def caption_names_scene_object(ctx: Ctx, caption: str) -> bool:
    """Whether a caption explicitly names any scene object or destination.

    Object suffix matching catches natural forms such as ``black bowl`` for simulator
    identifiers such as ``akita_black_bowl_1``. Generation uses this as a hard invariant:
    the model must resolve object identity from the sketch, never from a noun.
    """
    names = {o["name"] for o in ctx.scene.get("objects", [])}
    names.add(ctx.target_name)
    if ctx.dest_name:
        names.add(ctx.dest_name)
    return any(_surface_form(name, caption) is not None for name in names)


def is_name_free_deictic_caption(ctx: Ctx, caption: str) -> bool:
    """Accept only a small deictic grammar and reject every explicit scene-object name."""
    return bool(_NAME_FREE_DEICTIC_RE.fullmatch(caption.strip())) and not \
        caption_names_scene_object(ctx, caption)


PICK_CAPTIONS = ("grab this", "pick up this", "take this")
_PICK_RE = re.compile(r"^(?:grab|pick up|take|grasp) this$")


def is_pick(caption: str) -> bool:
    """A pick verb with no destination: picking the object up is the whole action."""
    return bool(_PICK_RE.fullmatch((caption or "").strip().lower()))


def resolve(ctx: Ctx, circle: dict | None, arrow: dict | None,
            caption_level: str, how: str = "", caption: str = "") -> tuple[int, int, str]:
    """How many (referent, destination) readings survive a prompt.

    This applies to every mode: given whatever circle/arrow/caption a sample carries, count
    the readings each channel leaves open. A complete prompt is binary ``wrong`` when this
    product exceeds one, even if no individual channel is geometrically incorrect.

    Referent — what to act on — is pinned by, in order: the circle (the objects it selects,
    `Ctx.selected`), else the arrow's tail (the object it starts on), else the caption (a
    deictic caption leaves every visible object open). Destination — where to put it — is
    pinned by the arrow head (`Ctx.indicated`, never the referent itself; bare table is one
    spot), else the caption; manipulate tasks have none. `circle`/`arrow` may be declared
    geometry or a rendered stroke variant (its ellipse is honoured).
    """
    nvis = max(len(ctx.visible()), 1)

    # -- referent --
    if circle is not None:
        refs = ctx.selected(circle)
        if circle.get("also"):                          # a second loop: both are referents
            refs += [o for o in ctx.selected(circle["also"]) if o not in refs]
        why_r = "circle"
    elif arrow is not None:
        refs = [o for o in ctx.indicated((arrow["x0"], arrow["y0"]), bare_table=False)
                if ctx.can_act_on(o)]
        why_r = "arrow-tail"
    else:
        refs = None
        why_r = "deixis" if caption_level in _DEICTIC else "caption"
    n_ref = (len(refs) or 1) if refs is not None else (nvis if why_r == "deixis" else 1)

    # -- destination --
    if ctx.kind != "place":
        n_dst, why_d = 1, "none"
    elif arrow is None and is_pick(caption):
        n_dst, why_d = 1, "pick"                        # "grab this": no destination needed
    elif arrow is not None:
        skip = {o["name"] for o in refs or []} or {ctx.target_name}
        n_dst = len(ctx.indicated((arrow["x1"], arrow["y1"]), exclude=skip)) or 1
        why_d = "arrow-head"
    elif caption_level in _DEICTIC or not (ctx.dest_name and ctx.surface(ctx.dest_name)):
        n_dst = max(nvis - 1, 1)                        # any object but the referent
        why_d = "deixis"
    else:
        n_dst, why_d = 1, "caption"

    return n_ref, n_dst, f"ref={how}{why_r},dst={how}{why_d}"


def resolve_rendered(ctx: Ctx, variant: dict, caption_level: str,
                     caption: str = "") -> tuple[int, int, str]:
    """Resolve the actual sampled stroke parameters used to render an image."""
    return resolve(ctx, variant.get("circle"), variant.get("arrow"), caption_level,
                   how="rendered-", caption=caption)


def rival_coverage(ctx: Ctx, circle: dict | None) -> float:
    """Largest share of a competing object (actable, not the target, not under it) inside."""
    t = ctx.obj(ctx.target_object or "")
    if circle is None or t is None or ctx.instances is None:
        return 0.0
    return max([ctx.coverage(circle, o) for o in ctx.visible()
                if o["name"] != t["name"] and ctx.can_act_on(o) and not _supports(o, t)] or [0.0])


def circle_selects_target(ctx: Ctx, circle: dict | None) -> bool:
    return circle is not None and any(o["name"] == ctx.target_object
                                      for o in ctx.selected(circle))


def head_on_destination(ctx: Ctx, s: Sample) -> bool:
    """Whether the arrow head indicates exactly the true destination object (an overshoot
    that stays on a big basket still points at the basket). False for table regions."""
    if s.arrow is None or ctx.kind != "place" or ctx.dest_object is None:
        return False
    refs = {o["name"] for o in ctx.selected(s.circle)} if s.circle else set()
    hit = ctx.indicated((s.arrow["x1"], s.arrow["y1"]), exclude=refs or {ctx.target_object})
    return {o["name"] for o in hit} == {ctx.dest_object}


def _crosses(ctx: Ctx, arrow: dict, name: str | None) -> bool:
    """Whether the on-frame part of the shaft runs over object `name`'s mask."""
    if name is None or ctx.instances is None:
        return False
    H, W = ctx.instances.shape
    idx = ctx.scene["objects"].index(ctx._by_name[name]) + 1
    t = np.linspace(0, 1, 60)
    x = arrow["x0"] + t * (arrow["x1"] - arrow["x0"])
    y = arrow["y0"] + t * (arrow["y1"] - arrow["y0"])
    on = (x >= 0) & (x < 1) & (y >= 0) & (y < 1)
    return bool(np.any(ctx.instances[(y[on] * H).astype(int), (x[on] * W).astype(int)] == idx))


def rendered_ok(ctx: Ctx, rec: dict, var: dict) -> bool:
    """Whether the drawn strokes still show what the label claims. Hand-drawn rendering
    moves the arrow head up to `head_tolerance` (31 px), so a valid arrow can end past its
    destination and a corrupted one can land back on it; resample until they agree."""
    from . import strokes as S
    c, a = var.get("circle"), var.get("arrow")
    probe = Sample(circle=c, arrow=a, caption="", labels={})
    pts = None if c is None else S.circle_polyline(
        {"ecc": 1.0, "rot": 0.0, "theta0": 0.0, "span": 2 * np.pi,
         "wobble_amps": [0] * 4, "wobble_phases": [0] * 4, **c})
    shaft = None
    if a is not None:                               # ~15 px of the drawn path on the image,
        t = np.linspace(0, 1, 50)[:, None]          # starting on it
        shaft = S.arrow_polylines(a)[0] if "tpl" in a else \
            (1 - t) * [a["x0"], a["y0"]] + t * [a["x1"], a["y1"]]
        on = np.all((shaft >= 0) & (shaft <= 1), axis=1)
        seg = np.hypot(*np.diff(shaft, axis=0).T)
        if not on[0] or float(seg[on[1:] & on[:-1]].sum()) < MIN_ARROW_VIS:
            return False
    if pts is not None:                             # a mark nobody can see shows nothing
        if np.mean(np.all((pts >= 0) & (pts <= 1), axis=1)) < 0.5:
            return False
    if rec["verdict"] == "right":
        if c is not None and (not circle_selects_target(ctx, c)
                              or rival_coverage(ctx, c) > VALID_MARGIN):
            return False
        t = ctx.obj(ctx.target_object or "")
        if c is not None and t is not None and any(  # a loop that lies mostly on a fixture
                o is not t and o.get("fixture") and not _supports(o, t)
                and ctx.disc_share(c, o) > VALID_MARGIN for o in ctx.visible()):
            return False
        if a is not None:
            if not (arrow_visible(c, a) and head_on_frame(a)):
                return False
            if not head_lands_on_destination(ctx, a):
                return False
            if not (EDGE_CLEAR <= a["x1"] <= 1 - EDGE_CLEAR and EDGE_CLEAR <= a["y1"] <= 1 - EDGE_CLEAR):
                return False                        # a tip at the border reads as running off
            t = ctx.obj(ctx.target_object or "")
            tail = (a["x0"], a["y0"])
            if c is not None and t is not None and not (
                    S.circle_contains(c, tail) or ctx.mask_dist(t, tail) <= FLOOR):
                return False
        return True
    mode = rec["mode"]
    if mode in ("B1", "B3", "B4", "B5", "B8") and a is not None and (
            head_on_destination(ctx, probe) or head_on_alternative(ctx, a) or any(
                ctx.mask_dist(ctx.obj(n), (a["x1"], a["y1"])) <= BETWEEN_CLEAR
                for n in {ctx.dest_object} | {d["name"] for d in sensible_dests(
                    ctx, ctx.target_object or "")} if n and ctx.obj(n))):
        return False                                # touching a place it could go
    if mode == "B5" and a is not None and not head_on_frame(a) and any(
            float(np.min([ctx.mask_dist(ctx.obj(n), q) for q in shaft[::2]])) < 0.05
            for n in {ctx.dest_object} | {d["name"] for d in sensible_dests(
                ctx, ctx.target_object or "")} if n and ctx.obj(n)):
        return False                                # an off-frame arrow past a sensible place
    if mode == "B8" and a is not None and ctx.obj(ctx.dest_object or "") is not None and \
            ctx.mask_dist(ctx.obj(ctx.dest_object), (a["x1"], a["y1"])) > 0.12:
        return False                                # short of it, not somewhere else
    if mode in ("A2", "A3", "A4", "C1", "D4") and ctx.target_coverage(c) > VALID_MARGIN:
        return False                                # still visibly circles the target
    if mode == "A7" and c is not None:
        sel = resolve(ctx, c, None, "L3", caption="grab this")[0] if c.get("also") else \
            len(ctx.selected(c))
        if sel == 1 or (sel == 0 and ctx.target_coverage(c) > NEAR_MISS_COVER):
            return False                            # it now reads as grabbing one object
    if mode in ("A3", "D4") and c is not None and a is not None and pts is not None and (
            circle_holds(c, (a["x0"], a["y0"])) or float(np.min(np.hypot(
                *(pts - [a["x0"], a["y0"]]).T))) < NEAR_TAIL_CLEAR):
        return False                                # an arrow from the circle reads as valid
    if mode in ("A3", "D4") and c is not None and (ctx.selected(c)
                                                   or ctx.target_coverage(c) > NEAR_MISS_COVER):
        return False                                # landed on another object / still on it
    tgt = ctx.obj(ctx.target_object or "")
    if mode in ("A2", "A3", "A4", "C1", "D4") and c is not None and tgt is not None and \
            tgt.get("fixture") and ctx.disc_share(c, tgt) > VALID_MARGIN:
        return False                                # still lies on the drawer / stove
    if mode in ("A2", "C1", "B6") and a is not None and c is not None:
        # the failure is "the arrow starts at another object": a tail on or near the drawn
        # circle reads as starting from the circled one, a valid prompt
        if circle_holds(c, (a["x0"], a["y0"])) or float(np.min(np.hypot(
                *(pts - [a["x0"], a["y0"]]).T))) < TAIL_CLEAR:
            return False
        inside = circle_holds(c, (shaft[:, 0], shaft[:, 1]))
        if np.any(inside[: max(1, len(shaft) * 2 // 3)]):
            return False                            # nor run through the circle on its way
    if rec.get("resolution", {}).get("kind") == "dest_between" and a is not None:
        head = (a["x1"], a["y1"])                   # clearly between: on neither, near both
        near = sorted((ctx.mask_dist(o, head), o["name"]) for o in ctx.visible()
                      if o["name"] != ctx.target_object)
        ds = [d for d, _ in near]
        if len(ds) < 2 or ds[0] <= BETWEEN_CLEAR or ds[1] > 1.25 * ds[0]:
            return False
        places = {ctx.dest_object} | {d["name"] for d in sensible_dests(ctx, ctx.target_object or "")}
        if not {near[0][1], near[1][1]} <= places:
            return False                            # between a basket and a bottle reads as the basket
        gt = ctx.arrow_clean                        # a table-spot goal: not on the spot either
        if ctx.dest_object is None and gt is not None and \
                np.hypot(head[0] - gt["x1"], head[1] - gt["y1"]) <= 2 * BETWEEN_CLEAR:
            return False
    if mode == "A5" or rec.get("resolution", {}).get("kind") == "referent_twin":
        n_ref, _, _ = resolve(ctx, c, a, rec["resolution"]["caption_level"])
        if n_ref < 2 or not circle_selects_target(ctx, c):
            return False                            # the drawn circle lost one of the two
    return True


def head_lands_on_destination(ctx: Ctx, arrow: dict | None) -> bool:
    """A valid arrow's head lies ON the destination: within FLOOR of its mask, or for a
    table region within 10 px of the ground-truth spot and on no object."""
    if arrow is None or ctx.kind != "place":
        return True
    head, gt = (arrow["x1"], arrow["y1"]), ctx.arrow_clean
    dest = ctx.obj(ctx.dest_object or "")
    if dest is not None:
        return ctx.mask_dist(dest, head) <= LAND
    if gt is None:
        return True
    return (np.hypot(head[0] - gt["x1"], head[1] - gt["y1"]) <= 0.04
            and all(ctx.mask_dist(o, head) > FLOOR for o in ctx.visible()))


def head_on_frame(arrow: dict | None) -> bool:
    return arrow is None or (0 <= arrow["x1"] <= 1 and 0 <= arrow["y1"] <= 1)


def arrow_visible(circle: dict | None, arrow: dict | None, margin: float = 0.03) -> bool:
    """Whether the arrow head clears the circle: a head inside it hides the arrow."""
    if circle is None or arrow is None:
        return True
    return np.hypot(arrow["x1"] - circle["cx"], arrow["y1"] - circle["cy"]) > \
        circle["r"] + margin


def preserve_positive_resolution_guard(ctx: Ctx, sample: Sample) -> None:
    """Keep hand jitter on the target and off the centres the declared circle leaves out.
    Selection itself is re-checked on the rendered stroke (`resolve_rendered`)."""
    if sample.circle is None:
        return
    cv = {**sample.circle, "ecc": 1.0, "rot": 0.0}
    from . import strokes as S
    sample.guard = _guard(
        contains=[ctx.target_xy],
        excludes=[_center(o) for o in ctx.visible()
                  if o["name"] != ctx.target_name and not S.circle_contains(cv, _center(o))],
    )


def circle_holds(circle: dict, pt) -> bool:
    """Does this (un-jittered) circle contain `pt`?"""
    from . import strokes as S
    return S.circle_contains({**circle, "ecc": 1.0, "rot": 0.0}, pt)


def _seg_dist(p, arrow: dict) -> float:
    """Distance from point `p` to the arrow's straight shaft."""
    a, b, q = (np.array([arrow["x0"], arrow["y0"]]), np.array([arrow["x1"], arrow["y1"]]),
               np.asarray(p, dtype=float))
    t = np.clip(np.dot(q - a, b - a) / max(float(np.dot(b - a, b - a)), 1e-12), 0, 1)
    return float(np.linalg.norm(q - (a + t * (b - a))))


def _supports(lower: dict, upper: dict) -> bool:
    """`lower` sits directly under `upper` (anchors within 7 cm horizontally, lower in z)."""
    a, b = np.asarray(lower["anchor_3d"]), np.asarray(upper["anchor_3d"])
    return float(np.hypot(*(a[:2] - b[:2]))) < 0.07 and a[2] < b[2] - 0.005


def _center(o: dict) -> tuple[float, float]:
    """An object's mask-circle centre in normalized coords."""
    return (o["circle"]["cx"], o["circle"]["cy"])


def _basename(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _readable(obj_name: str) -> str:
    toks = [t for t in _basename(obj_name).split("_") if t not in _STOPWORDS]
    return " ".join(toks) if toks else _basename(obj_name).replace("_", " ")


def feasible_dests(ctx: Ctx, obj_name: str, free_only: bool = True) -> list[dict]:
    """Visible objects `obj_name` could sensibly be put on or in (PLACE_INTO)."""
    o = ctx.obj(obj_name)
    if o is None or o.get("fixture"):
        return []
    cat = _basename(obj_name)
    return [d for d in ctx.visible() if d is not o and len(ctx._px.get(d["name"], ())) >= MIN_DEST_PX
            and cat in PLACE_INTO.get(_basename(d["name"]), ("", "", ()))[2]
            and not (free_only and any(_holds_on_top(ctx, d, x) for x in ctx.visible()
                                       if not x.get("fixture") and x is not d and x is not o))]


def _holds_on_top(ctx: Ctx, d: dict, x: dict) -> bool:
    """Object `x` sits on `d` in the image: most of its mask lies inside d's outline (its
    convex hull), and it is not lower. A bowl on a stove burner is 17 cm from the
    stove's 3D anchor, so anchors alone miss it."""
    import cv2
    pd, px = ctx._px.get(d["name"]), ctx._px.get(x["name"])
    if pd is None or px is None or x["anchor_3d"][2] < d["anchor_3d"][2]:
        return False
    H, W = ctx.instances.shape
    hull = cv2.convexHull(np.round(pd * [W, H]).astype(np.int32))
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    ix = np.clip((px * [W, H]).astype(int), 0, [W - 1, H - 1])
    return float(m[ix[:, 1], ix[:, 0]].mean()) >= 0.5


def sensible_dests(ctx: Ctx, obj_name: str) -> list[dict]:
    """Everywhere a person would read an arrow from `obj_name` as sensible: the
    generated alternatives plus cabinets and shelves."""
    o = ctx.obj(obj_name)
    if o is None or o.get("fixture"):
        return []
    # Cabinets and shelves are left out: an arrow to one without saying which drawer is
    # wrong in the human labels (the demo's own cabinet stays protected as its destination).
    cat = _basename(obj_name)
    return feasible_dests(ctx, obj_name, free_only=False) + [
        d for d in ctx.visible() if d is not o and cat in SENSIBLE_EXTRA.get(_basename(d["name"]), ())]


def alternatives(ctx: Ctx) -> list[tuple[str, str]]:
    """(target, destination region) of every sensible place action in the scene other than
    the demonstrated one: what a person circling another object might mean."""
    if ctx.kind != "place" or ctx.instances is None:
        return []
    out = []
    for o in ctx.visible():
        if o.get("fixture") or len(ctx._px.get(o["name"], ())) < MIN_DEST_PX:
            continue                                # only clearly visible objects
        for d in feasible_dests(ctx, o["name"]):
            _, suffix, _ = PLACE_INTO[_basename(d["name"])]
            region = f"{d['name']}_{suffix}" if suffix else d["name"]
            if (o["name"], d["name"]) != (ctx.target_object, ctx.dest_object):
                out.append((o["name"], region))
    return out


def alt_context(ctx: Ctx, target_object: str, destination: str) -> Ctx:
    """The scene re-goaled to another place action: same frame, masks and captions, the
    clean circle around `target_object` and the clean arrow from it to `destination`."""
    o, d = ctx.obj(target_object), ctx.obj(ctx._owner(destination))
    pred = PLACE_INTO[_basename(d["name"])][0]
    tx, ty = o["center_px"][0] / ctx.W, o["center_px"][1] / ctx.H
    meta = {**ctx.meta,
            "instruction": f"put the {_readable(target_object)} "
                           f"{'in' if pred == 'In' else 'on'} the {_readable(d['name'])}",
            "goal": {**ctx.meta["goal"], "kind": "place", "predicate": pred,
                     "target_object": target_object, "destination": destination},
            "points_px": {**ctx.meta["points_px"], "target_t0": o["center_px"]},
            "circle_clean": ctx.scene_circle(o),
            "arrow_dest_clean": {"x0": tx, "y0": ty, "x1": d["circle"]["cx"],
                                 "y1": d["circle"]["cy"]}}
    return Ctx(meta=meta, scene=ctx.scene, occupancy=ctx.occupancy,
               caption_pool=ctx.caption_pool, ladder=ctx.ladder,
               radius_scale=ctx.radius_scale, instances=ctx.instances)


def head_on_alternative(ctx: Ctx, arrow: dict | None) -> bool:
    """Whether the head points at one object the target could sensibly go to: under
    task-blind labels that arrow is a valid instruction, not a corruption."""
    if arrow is None or ctx.kind != "place" or ctx.target_object is None:
        return False
    hit = ctx.indicated((arrow["x1"], arrow["y1"]), exclude={ctx.target_object})
    ok = {d["name"] for d in sensible_dests(ctx, ctx.target_object)}
    return len(hit) == 1 and hit[0]["name"] in ok


def _surface_form(obj_name: str, text: str) -> str | None:
    toks = _readable(obj_name).split()
    low = text.lower()
    for i in range(len(toks)):                      # longest -> shortest suffix
        cand = " ".join(toks[i:])
        if cand and cand in low:
            return cand
    return None


# =============================================================== A · circle ===

@mode("A2")
def a2_wrong_category(ctx: Ctx, rng) -> Sample | None:
    """Circle a clearly different object (plate, stove)."""
    # An object large enough to geometrically contain the target cannot be a wrong-object
    # circle: "circle the cabinet, not the bowl" is unsatisfiable when the bowl sits in the
    # cabinet's drawer. Such a circle would enclose both -> that is A5's ambiguity, not A2.
    fits = MANIPULABLE.get(ctx.meta["goal"].get("predicate"), set())
    cands = [o for o in ctx.others(exclude_same_category=True)
             if not circle_holds(ctx.scene_circle(o), ctx.target_xy)
             and not circle_selects_target(ctx, ctx.scene_circle(o))
             and not (ctx.kind != "place" and _basename(o["name"]) in fits)]
    if not cands:
        return None
    o = cands[rng.integers(len(cands))]
    return Sample(circle=ctx.scene_circle(o), arrow=ctx.arrow_clean,
                  caption=ctx.instruction, labels=_ok(circle_ok=False),
                  severity=1.0, note=f"circled {o['name']} instead of {ctx.target_name}",
                  guard=_guard(contains=[_center(o)], excludes=[ctx.target_xy]))


@mode("A3")
def a3_off_target(ctx: Ctx, rng) -> Sample | None:
    """Offset the center so the target falls outside / only grazes the ellipse.

    Severity is continuous: `k` is the offset in units of radius. k>1 puts the target
    centre outside the circle; k in (1, 1.3] is the genuinely hard near-miss band.
    """
    c = ctx.circle_clean
    k = float(rng.uniform(1.05, 2.2))
    th = float(rng.uniform(0, 2 * np.pi))
    off = k * c["r"]
    # Offset from the *target point*, not the circle centre. For elongated objects the
    # mask's enclosing-circle centre sits well off the projected body centre, so offsetting
    # from the centre can move the circle *toward* the target and silently un-miss it.
    tx, ty = ctx.target_xy
    bad = {"cx": tx + off * np.cos(th), "cy": ty + off * np.sin(th), "r": c["r"]}
    if not (0.02 < bad["cx"] < 0.98 and 0.02 < bad["cy"] < 0.98):
        th += np.pi                                  # keep it inside the frame
        bad = {"cx": tx + off * np.cos(th), "cy": ty + off * np.sin(th), "r": c["r"]}
    if circle_selects_target(ctx, bad):          # still covers most of the target
        return None
    sev = float(np.clip((k - 1.0) / 1.2, 0, 1))
    return Sample(circle=bad, arrow=ctx.arrow_clean, caption=ctx.instruction,
                  labels=_ok(circle_ok=False), severity=sev,
                  note=f"center offset {k:.2f}r from target",
                  guard=_guard(excludes=[ctx.target_xy]))


@mode("A4")
def a4_empty_region(ctx: Ctx, rng) -> Sample | None:
    """Circle bare table / background — sampled outside every object mask."""
    if ctx.occupancy is None:
        return None
    H, W = ctx.occupancy.shape
    free = ~ctx.occupancy
    r_px = ctx.circle_clean["r"] * min(H, W)
    for _ in range(200):
        u = int(rng.integers(int(r_px) + 2, W - int(r_px) - 2))
        v = int(rng.integers(int(r_px) + 2, H - int(r_px) - 2))
        if not free[v, u]:
            continue
        y0, y1 = max(0, v - int(r_px)), min(H, v + int(r_px))
        x0, x1 = max(0, u - int(r_px)), min(W, u + int(r_px))
        if ctx.occupancy[y0:y1, x0:x1].any():        # keep the disc fully on empty table
            continue
        return Sample(circle={"cx": u / W, "cy": v / H, "r": ctx.circle_clean["r"]},
                      arrow=ctx.arrow_clean, caption=ctx.instruction,
                      labels=_ok(circle_ok=False), severity=1.0,
                      note="circle on empty table (no object enclosed)",
                      guard=_guard(excludes=[ctx.target_xy]))
    return None


@mode("A5")
def a5_multi_enclosure(ctx: Ctx, rng) -> Sample | None:
    """Grow the circle until it encloses the target AND a neighbour -> unresolvable.

    Every enclosed object is a plausible referent, so the complete prompt is ``wrong``
    through ``sufficient_ok=False`` rather than through ``circle_ok=False``.
    Prefers a same-category neighbour (two identical bowls) — the maximally ambiguous
    pattern — and falls back to the nearest object of any category.
    """
    t = ctx.obj(ctx.target_name)
    if t is None or not t["circle"]:
        return None
    tc = np.array([t["circle"]["cx"], t["circle"]["cy"]])
    tb = _basename(ctx.target_name)
    cands = [o for o in ctx.visible() if o["name"] != ctx.target_name and ctx.can_act_on(o)]
    if not cands:
        return None
    same = [o for o in cands if _basename(o["name"]) == tb]
    pool = same or cands
    o = min(pool, key=lambda q: np.hypot(q["circle"]["cx"] - tc[0],
                                         q["circle"]["cy"] - tc[1]))
    if rng.random() < 0.4:                             # people draw two separate loops
        two = {**ctx.circle_clean, "also": ctx.scene_circle(o)}
        if len({x["name"] for x in ctx.selected(two["also"]) + ctx.selected(ctx.circle_clean)}) >= 2:
            return Sample(circle=two, arrow=ctx.arrow_clean, caption=ctx.instruction,
                          labels=_ok(sufficient_ok=False), severity=0.8, n_referents=2,
                          n_destinations=1,
                          note=f"two loops: {ctx.target_name} and {o['name']}",
                          guard=_guard(contains=[ctx.target_xy]))
    oc = np.array([o["circle"]["cx"], o["circle"]["cy"]])
    mid = (tc + oc) / 2
    d = float(np.linalg.norm(oc - tc))
    r = d / 2 + max(t["circle"]["r"], o["circle"]["r"]) * 1.05
    circ = {"cx": float(mid[0]), "cy": float(mid[1]), "r": float(r)}
    # The circle does contain the target, but it is insufficient. The binary verdict is
    # therefore wrong while the channel-level circle_ok diagnostic remains true.
    inside = ctx.selected(circ)
    return Sample(circle=circ, arrow=ctx.arrow_clean, caption=ctx.instruction,
                  labels=_ok(sufficient_ok=False), severity=1.0 if same else 0.7,
                  n_referents=len(inside), n_destinations=1,
                  note=(f"encloses {len(inside)} objects "
                        f"({', '.join(x['name'] for x in inside)}); intended pair "
                        f"{ctx.target_name} + {o['name']}"
                        f"{' (same category — maximally ambiguous)' if same else ''}"),
                  guard=_guard(contains=[ctx.target_xy, _center(o)]))


@mode("A6")
def a6_degenerate_size(ctx: Ctx, rng) -> Sample | None:
    """Dot-sized or whole-frame circle (centred correctly — only the scale is wrong)."""
    c = ctx.circle_clean
    th = np.linspace(0, 2 * np.pi, 180)
    big = float(rng.uniform(0.35, 0.48))
    on_frame = np.mean((np.abs(c["cx"] + big * np.cos(th) - 0.5) <= 0.5)
                       & (np.abs(c["cy"] + big * np.sin(th) - 0.5) <= 0.5))
    dot = max(0.012, c["r"] * float(rng.uniform(0.1, 0.2)))    # ~3-5 px, not nothing
    a = ctx.arrow_clean
    hidden = a is not None and _seg_dist((c["cx"], c["cy"]), a) < dot + 0.02  # under the tail
    if (rng.random() < 0.5 or on_frame < 0.5) and not hidden:
        r, what = dot, "dot"
    elif on_frame >= 0.5:                       # at least half the outline on-frame
        r, what = big, "frame-sized"
    else:
        return None
    return Sample(circle={"cx": c["cx"], "cy": c["cy"], "r": r},
                  arrow=ctx.arrow_clean, caption=ctx.instruction,
                  labels=_ok(circle_ok=False), severity=1.0,
                  note=f"{what} circle (r={r:.4f} vs clean {c['r']:.4f})",
                  guard=_guard())


@mode("A7")
def a7_bad_circle_pick(ctx: Ctx, rng) -> Sample | None:
    """Circle only with a pick verb ("grab this"), but the circle picks out nothing or two
    things: empty table, the robot, a dot, or two objects."""
    if ctx.kind != "place":
        return None
    kind = int(rng.integers(4))
    if kind == 3 and ctx.instances is not None:        # the robot is not something to grab
        ys, xs = np.nonzero(ctx.instances == 255)
        if not len(xs):
            return None
        H, W = ctx.instances.shape
        k = int(rng.integers(len(xs)))
        circ = {"cx": float((xs[k] + 0.5) / W), "cy": float((ys[k] + 0.5) / H),
                "r": ctx.circle_clean["r"]}
        if ctx.selected(circ):
            return None
        base = Sample(circle=circ, arrow=None, caption="", labels=_ok(circle_ok=False),
                      note="circle on the robot", guard=_guard(excludes=[ctx.target_xy]))
    else:
        base = (a4_empty_region, a5_multi_enclosure, a6_degenerate_size)[kind](ctx, rng)
    if base is None:
        return None
    return Sample(circle=base.circle, arrow=None,
                  caption=PICK_CAPTIONS[int(rng.integers(len(PICK_CAPTIONS)))],
                  labels={**base.labels, "arrow_ok": True}, severity=base.severity,
                  note=f"[pick] {base.note}", guard=base.guard)


# ================================================================ B · arrow ===

@mode("B1")
def b1_wrong_destination(ctx: Ctx, rng) -> Sample | None:
    """Retarget the head at a different container/region; tail stays on the target."""
    a = ctx.arrow_clean
    if a is None:
        return None
    ok = {d["name"] for d in sensible_dests(ctx, ctx.target_object or "")}
    cands = [o for o in ctx.visible()
             if o["name"] not in (ctx.target_name, ctx.dest_name, ctx.dest_object)
             and o["name"] not in ok]
    if not cands:
        return None
    o = cands[rng.integers(len(cands))]
    return Sample(circle=ctx.circle_clean,
                  arrow={"x0": a["x0"], "y0": a["y0"],
                         "x1": o["circle"]["cx"], "y1": o["circle"]["cy"]},
                  caption=ctx.instruction, labels=_ok(arrow_ok=False), severity=1.0,
                  note=f"arrow -> {o['name']} instead of {ctx.dest_name}",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("B3")
def b3_wrong_magnitude(ctx: Ctx, rng) -> Sample | None:
    """Correct bearing, length pushed past tolerance — undershoot or overshoot."""
    a = ctx.arrow_clean
    if a is None:
        return None
    p0 = np.array([a["x0"], a["y0"]])
    p1 = np.array([a["x1"], a["y1"]])
    d = p1 - p0
    lo = max(0.10, (ctx.circle_clean["r"] + 0.04) / max(float(np.linalg.norm(d)), 1e-6))
    if rng.random() < 0.5 and lo <= 0.45:
        f, what = float(rng.uniform(lo, max(lo, 0.35))), "undershoot"
    else:
        f, what = float(rng.uniform(1.8, 3.0)), "overshoot"
    q1 = p0 + d * f
    q1 = np.clip(q1, 0.01, 0.99)
    sev = float(np.clip(abs(np.log(f)) / np.log(3.0), 0, 1))
    return Sample(circle=ctx.circle_clean,
                  arrow={"x0": a["x0"], "y0": a["y0"],
                         "x1": float(q1[0]), "y1": float(q1[1])},
                  caption=ctx.instruction, labels=_ok(arrow_ok=False), severity=sev,
                  note=f"{what}: length x{f:.2f}",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("B4")
def b4_wrong_bearing(ctx: Ctx, rng) -> Sample | None:
    """Correct length, head rotated about the tail by 90 or 180 degrees."""
    a = ctx.arrow_clean
    if a is None:
        return None
    p0 = np.array([a["x0"], a["y0"]])
    d = np.array([a["x1"] - a["x0"], a["y1"] - a["y0"]])
    deg = float(rng.choice([90.0, -90.0, 180.0]))
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    q1 = p0 + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    if not (0.01 < q1[0] < 0.99 and 0.01 < q1[1] < 0.99):
        th = np.pi                                   # 180 always stays on-frame-ish
        q1 = p0 - d
        deg = 180.0
    q1 = np.clip(q1, 0.01, 0.99)
    return Sample(circle=ctx.circle_clean,
                  arrow={"x0": a["x0"], "y0": a["y0"],
                         "x1": float(q1[0]), "y1": float(q1[1])},
                  caption=ctx.instruction, labels=_ok(arrow_ok=False),
                  severity=1.0 if abs(deg) == 180 else 0.75,
                  note=f"bearing rotated {deg:+.0f} deg",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("B5")
def b5_arrow_to_nowhere(ctx: Ctx, rng) -> Sample | None:
    """Head on provably-empty table, or off-frame entirely."""
    a = ctx.arrow_clean
    if a is None:
        return None
    if ctx.occupancy is not None and rng.random() < 0.5:
        H, W = ctx.occupancy.shape
        free = np.argwhere(~ctx.occupancy)
        if len(free):
            v, u = free[rng.integers(len(free))]
            return Sample(circle=ctx.circle_clean,
                          arrow={"x0": a["x0"], "y0": a["y0"],
                                 "x1": float(u / W), "y1": float(v / H)},
                          caption=ctx.instruction, labels=_ok(arrow_ok=False),
                          severity=1.0, note="head on empty table",
                          guard=_guard(contains=[ctx.target_xy]))
    edge = rng.integers(4)
    x1, y1 = {0: (1.18, a["y1"]), 1: (-0.18, a["y1"]),
              2: (a["x1"], 1.18), 3: (a["x1"], -0.18)}[int(edge)]
    return Sample(circle=ctx.circle_clean,
                  arrow={"x0": a["x0"], "y0": a["y0"], "x1": float(x1), "y1": float(y1)},
                  caption=ctx.instruction, labels=_ok(arrow_ok=False), severity=1.0,
                  note="head off-frame", guard=_guard(contains=[ctx.target_xy]))


@mode("B6")
def b6_detached_tail(ctx: Ctx, rng) -> Sample | None:
    """Translate the tail off the circled object, preserving shaft direction/length.

    Circle and arrow are each individually plausible; the arrow just no longer claims the
    object the circle selects.
    """
    a = ctx.arrow_clean
    if a is None:
        return None
    c = ctx.circle_clean
    d = np.array([a["x1"] - a["x0"], a["y1"] - a["y0"]])
    k = float(rng.uniform(2.0, 3.5))                 # tail displaced k*r off the circle
    th = float(rng.uniform(0, 2 * np.pi))
    p0 = np.array([a["x0"] + k * c["r"] * np.cos(th),
                   a["y0"] + k * c["r"] * np.sin(th)])
    p0 = np.clip(p0, 0.02, 0.98)
    p1 = np.clip(p0 + d, 0.01, 0.99)
    return Sample(circle=c,
                  arrow={"x0": float(p0[0]), "y0": float(p0[1]),
                         "x1": float(p1[0]), "y1": float(p1[1])},
                  caption=ctx.instruction,
                  labels=_ok(arrow_ok=False, consistency_ok=False), severity=1.0,
                  note=f"tail displaced {k:.1f}r off the circled object",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("B7")
def b7_spec_mismatch(ctx: Ctx, rng) -> Sample | None:
    """Under-specification (place task, arrow dropped) or over-specification (manipulate
    task given a spurious destination arrow)."""
    if ctx.kind == "place":
        if ctx.arrow_clean is None:
            return None
        return Sample(circle=ctx.circle_clean, arrow=None, caption=ctx.instruction,
                      labels=_ok(arrow_ok=False), severity=1.0,
                      note="place task with no destination arrow (under-specified)",
                      guard=_guard(contains=[ctx.target_xy]))
    cands = ctx.others()
    if not cands:
        return None
    o = cands[rng.integers(len(cands))]
    c = ctx.circle_clean
    return Sample(circle=c,
                  arrow={"x0": c["cx"], "y0": c["cy"],
                         "x1": o["circle"]["cx"], "y1": o["circle"]["cy"]},
                  caption=ctx.instruction, labels=_ok(arrow_ok=False), severity=1.0,
                  note=f"manipulate task ({ctx.meta['goal']['predicate']}) given a "
                       f"spurious arrow -> {o['name']} (over-specified)",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("B8")
def b8_stops_short(ctx: Ctx, rng) -> Sample | None:
    """Right direction, but the head stops just short of the destination (~11-23 px off
    it): the near miss people draw and the human labellers reject."""
    a, dest = ctx.arrow_clean, ctx.obj(ctx.dest_object or "")
    if a is None or dest is None:
        return None
    p0, p1 = np.array([a["x0"], a["y0"]]), np.array([a["x1"], a["y1"]])
    gap = float(rng.uniform(0.045, 0.09))
    for t in np.linspace(1, 0, 201):
        q = p0 + t * (p1 - p0)
        if ctx.mask_dist(dest, q) >= gap:
            break
    else:
        return None
    if float(np.linalg.norm(q - p0)) < MIN_ARROW_VIS + 0.03:
        return None                                 # the object sits right at the destination
    return Sample(circle=ctx.circle_clean,
                  arrow={"x0": a["x0"], "y0": a["y0"], "x1": float(q[0]), "y1": float(q[1])},
                  caption=ctx.instruction, labels=_ok(arrow_ok=False), severity=0.6,
                  note=f"head stops {gap * 256:.0f} px short of {ctx.dest_object}",
                  guard=_guard(contains=[ctx.target_xy]))


# ========================================================= C · cross-modal ===

@mode("C1")
def c1_circle_arrow_mismatch(ctx: Ctx, rng) -> Sample | None:
    """Valid circle on object A + valid arrow drawn for object B."""
    a = ctx.arrow_clean
    if a is None:
        return None
    fixtures = [f for f in ctx.visible() if f.get("fixture") and f["name"] != ctx.dest_object
                and len(ctx._px.get(f["name"], ())) >= MIN_DEST_PX]
    if fixtures and ctx.kind == "place" and rng.random() < 0.3:   # circle on a drawer handle
        f = fixtures[int(rng.integers(len(fixtures)))]
        px = ctx._px[f["name"]]
        for _ in range(30):
            cx, cy = px[int(rng.integers(len(px)))]
            circ = {"cx": float(cx), "cy": float(cy), "r": ctx.circle_clean["r"]}
            if ctx.disc_share(circ, f) >= COVER and ctx.target_coverage(circ) == 0 and \
                    not circle_holds(circ, (a["x0"], a["y0"])):
                return Sample(circle=circ, arrow=a, caption=ctx.instruction,
                              labels=_ok(consistency_ok=False), severity=1.0,
                              note=f"circle on {f['name']} but arrow originates at {ctx.target_name}",
                              guard=_guard(contains=[(float(cx), float(cy))]))
    cands = [o for o in ctx.others() if ctx.can_act_on(o)
             and ctx.target_coverage(ctx.scene_circle(o)) <= VALID_MARGIN
             and not circle_holds(ctx.scene_circle(o), (a["x0"], a["y0"]))]
    if not cands:
        return None
    o = cands[rng.integers(len(cands))]
    return Sample(circle=ctx.scene_circle(o), arrow=a, caption=ctx.instruction,
                  labels=_ok(consistency_ok=False), severity=1.0,
                  note=(f"circle on {o['name']} but arrow originates at "
                        f"{ctx.target_name} — each channel plausible alone"),
                  guard=_guard(contains=[_center(o)]))


@mode("C2")
def c2_caption_sketch_mismatch(ctx: Ctx, rng) -> Sample | None:
    """Correct sketch, caption cross-paired in from a different task.

    Prefers a caption from the *same* suite: it shares the object vocabulary, so the
    mismatch cannot be spotted from wording alone — the sketch has to be read.
    """
    same = [c for c in ctx.caption_pool
            if c["instruction"] != ctx.instruction and c.get("suite") == ctx.suite]
    pool = same or [c for c in ctx.caption_pool if c["instruction"] != ctx.instruction]
    if not pool:
        return None
    c = pool[rng.integers(len(pool))]
    return Sample(circle=ctx.circle_clean, arrow=ctx.arrow_clean,
                  caption=c["instruction"],
                  labels=_ok(caption_ok=False, consistency_ok=False), severity=1.0,
                  note=f"caption borrowed from task '{c['task']}'; sketch is correct",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("C3")
def c3_predicate_mismatch(ctx: Ctx, rng) -> Sample | None:
    """Caption's predicate conflicts with the sketch's form (place vs manipulate).

    Must draw across suites: spatial and object are entirely `place` tasks, so the
    opposite-kind caption comes from goal. With a single-suite pool this mode is
    silently 0%-yield rather than wrong — hence the cross-suite pool.
    """
    want = "manipulate" if ctx.kind == "place" else "place"
    pool = [c for c in ctx.caption_pool
            if c["kind"] == want and c["instruction"] != ctx.instruction]
    if not pool:
        return None
    c = pool[rng.integers(len(pool))]
    return Sample(circle=ctx.circle_clean, arrow=ctx.arrow_clean,
                  caption=c["instruction"],
                  labels=_ok(caption_ok=False, consistency_ok=False), severity=0.8,
                  note=(f"{ctx.kind}-style sketch with a {want} caption "
                        f"('{c['task']}' from suite {c.get('suite')})"),
                  guard=_guard(contains=[ctx.target_xy]))


# ============================================================= D · language ===

@mode("D1")
def d1_wrong_target_noun(ctx: Ctx, rng) -> Sample | None:
    """Swap the target noun for a distractor's (or an out-of-inventory) noun."""
    surf = ctx.surface(ctx.target_name)
    if not surf:
        return None
    cands = [o for o in ctx.others() if _readable(o["name"]) != _readable(ctx.target_name)]
    if not cands:
        return None
    o = cands[rng.integers(len(cands))]
    repl = _readable(o["name"])
    cap = re.sub(re.escape(surf), repl, ctx.instruction, count=1, flags=re.I)
    if cap.lower() == ctx.instruction.lower():
        return None
    return Sample(circle=ctx.circle_clean, arrow=ctx.arrow_clean, caption=cap,
                  labels=_ok(caption_ok=False, consistency_ok=False), severity=1.0,
                  note=f"target noun '{surf}' -> '{repl}'",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("D2")
def d2_wrong_destination_noun(ctx: Ctx, rng) -> Sample | None:
    """Swap the destination noun."""
    if not ctx.dest_name:
        return None
    surf = ctx.surface(ctx.dest_name)
    if not surf:
        return None
    cands = [o for o in ctx.others() if _readable(o["name"]) != _readable(ctx.dest_name)]
    if not cands:
        return None
    o = cands[rng.integers(len(cands))]
    repl = _readable(o["name"])
    cap = re.sub(re.escape(surf), repl, ctx.instruction, count=1, flags=re.I)
    if cap.lower() == ctx.instruction.lower():
        return None
    return Sample(circle=ctx.circle_clean, arrow=ctx.arrow_clean, caption=cap,
                  labels=_ok(caption_ok=False, consistency_ok=False), severity=1.0,
                  note=f"destination noun '{surf}' -> '{repl}'",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("D3")
def d3_wrong_verb(ctx: Ctx, rng) -> Sample | None:
    """Use a name-free action that conflicts with the demonstrated goal kind."""
    if ctx.kind == "place":                     # a pick verb has no destination for the arrow
        pool = ("open this", "close this", "turn on this", "turn off this") + PICK_CAPTIONS
    else:
        pool = ("move this there", "put this there", "place this there")
    cap = pool[int(rng.integers(len(pool)))]
    return Sample(circle=ctx.circle_clean, arrow=ctx.arrow_clean, caption=cap,
                  labels=_ok(caption_ok=False, consistency_ok=False), severity=0.9,
                  note=f"name-free action conflicts with {ctx.kind} goal",
                  guard=_guard(contains=[ctx.target_xy]))


@mode("D4")
def d4_underdetermined_unresolved(ctx: Ctx, rng) -> Sample | None:
    """An L3 caption ('put this there') is correct with a good sketch — and a failure
    when the sketch that must resolve it is itself wrong."""
    if not ctx.ladder:
        return None
    lv = ctx.ladder.get("levels", {}).get("L3", {})
    pool = list(lv.get("templates", [])) + list(lv.get("llm", []))
    if not pool:
        return None
    cap = pool[rng.integers(len(pool))]
    bad = a3_off_target(ctx, rng)                    # the sketch fails to disambiguate
    if bad is None:
        return None
    return Sample(circle=bad.circle, arrow=ctx.arrow_clean, caption=cap,
                  labels=_ok(circle_ok=False, consistency_ok=False),
                  severity=bad.severity, guard=bad.guard,
                  note=("L3 caption carries no referent and the circle that must supply "
                        f"it is off-target ({bad.note})"))


# =============================================================== positives ===

def pick_positive(ctx: Ctx, rng, variant: dict | None = None) -> Sample:
    """Circle only + a pick verb ("grab this"): picking the object up is a complete action."""
    return Sample(circle=ctx.circle_clean, arrow=None,
                  caption=PICK_CAPTIONS[int(rng.integers(len(PICK_CAPTIONS)))],
                  labels=_ok(), severity=0.0, note="circle only with a pick verb",
                  guard=_guard(contains=[ctx.target_xy]))


def positive(ctx: Ctx, rng, variant: dict | None = None) -> Sample:
    """The correct prompt — clean, or an existing guarded stage-02 stroke variant."""
    if variant is None:
        return Sample(circle=ctx.circle_clean, arrow=ctx.arrow_clean,
                      caption=ctx.instruction, labels=_ok(), severity=0.0,
                      note="clean ground-truth annotation",
                      guard=_guard(contains=[ctx.target_xy]))
    c, a = variant.get("circle"), variant.get("arrow")
    circ = {"cx": c["cx"], "cy": c["cy"], "r": c["r"]} if c else None
    if ctx.kind != "place":
        a = None                                # B7: an arrow over-specifies open/turn-on
    arr = {"x0": a["x0"], "y0": a["y0"], "x1": a["x1"], "y1": a["y1"]} if a else None
    raw, snapped = ctx.meta.get("arrow_dest_clean"), ctx.arrow_clean
    if arr and raw and snapped:                 # stage 02 jittered the unsnapped head
        arr["x1"] += snapped["x1"] - raw["x1"]
        arr["y1"] += snapped["y1"] - raw["y1"]
    return Sample(circle=circ, arrow=arr, caption=ctx.instruction, labels=_ok(),
                  severity=0.0, note="guarded hand-drawn augmentation (still valid)",
                  guard=_guard(contains=[ctx.target_xy]))


# ======================================================= deictic grounding ====
#
# `resolve()` measures under-determination for every sample. These configurations retain
# the old ambiguous examples over the positive base, but the public verdict is now binary:
# "put this there" is right with a resolving sketch and wrong without one.

def _amb_deixis_resolved(ctx, rng):
    """L3 deixis + a full, correct sketch — deictic language is fine when the sketch
    resolves it. The right end of the binary grounding task."""
    return ctx.circle_clean, ctx.arrow_clean, "L3", [ctx.target_xy], False, \
        "deictic caption, full sketch resolves it (right)"


def _amb_no_sketch(ctx, rng):
    """L3 deixis, no sketch at all — every object is a candidate referent."""
    return None, None, "L3", [], False, "deictic caption, no sketch to ground it"


def _amb_dest_open(ctx, rng):
    """L3 + circle only — 'this' resolved, 'there' open (place tasks)."""
    if ctx.kind != "place" or ctx.arrow_clean is None:
        return None
    return ctx.circle_clean, None, "L3", [ctx.target_xy], False, \
        "circle resolves the referent; deictic caption leaves the destination open"


def _amb_referent_twin(ctx, rng):
    """L2 + a circle over the identical pair — both bowls are 'this'."""
    tw = ctx.twins()
    t = ctx.obj(ctx.target_name)
    if not tw or t is None or not t["circle"]:
        return None
    o = tw[int(rng.integers(len(tw)))]
    tc, oc = np.array(_center(t)), np.array(_center(o))
    mid = (tc + oc) / 2
    r = float(np.linalg.norm(oc - tc)) / 2 + max(t["circle"]["r"], o["circle"]["r"]) * 1.05
    circ = {"cx": float(mid[0]), "cy": float(mid[1]), "r": r}
    return circ, ctx.arrow_clean, "L2", [_center(t), _center(o)], True, \
        f"circle encloses the identical pair {ctx.target_name}/{o['name']}"


def _amb_dest_between(ctx, rng):
    """L3 + arrow head midway across the gap between the true destination and its nearest
    rival. With masks the midpoint of the closest pixel pair is equidistant from both; the
    midpoint of two centres can land on the larger object (a plate rim) and point at it."""
    a = ctx.arrow_clean
    if a is None:
        return None
    cands = [o for o in ctx.visible() if o["name"] != ctx.target_name]
    if len(cands) < 2:
        return None
    head = (a["x1"], a["y1"])
    cands.sort(key=lambda o: ctx.mask_dist(o, head))
    mid = ctx.gap_midpoint(cands[0], cands[1])
    arr = {"x0": a["x0"], "y0": a["y0"], "x1": float(mid[0]), "y1": float(mid[1])}
    return ctx.circle_clean, arr, "L3", [ctx.target_xy], False, \
        f"arrow aimed between {cands[0]['name']} and {cands[1]['name']}"


def _amb_category(ctx, rng):
    """L1 + no circle in a scene with an identical twin — the degraded caption names a
    category with >1 instance and no sketch disambiguates."""
    tw = ctx.twins()
    if not tw:
        return None
    cap = ctx.level("L1", rng)
    if not cap or cap.strip().lower().rstrip(".") == ctx.instruction.strip().lower().rstrip("."):
        return None                                    # L1 still == L0: not ambiguous
    # arrow dropped too: a clean arrow tail on the target would itself resolve which twin,
    # so category ambiguity requires no grounding sketch at all
    return None, None, "L1", [], True, \
        f"L1 names a category with {1 + len(tw)} instances; no sketch"


# kind -> builder(ctx, rng) -> (circle, arrow, caption_level, guard_pts, needs_twin, note)
_GROUNDING_CONFIGS = {
    "deixis_resolved": _amb_deixis_resolved,   # right when the sketch resolves deixis
    "no_sketch": _amb_no_sketch,               # wrong: nothing grounds the caption
    "dest_open": _amb_dest_open,               # wrong: destination remains open
    "referent_twin": _amb_referent_twin,       # wrong: identical pair remains open
    "dest_between": _amb_dest_between,          # wrong when two destinations survive
    # `category` is deliberately omitted: it needs an explicit L1 object-category noun.
}
GROUNDING_KINDS = tuple(_GROUNDING_CONFIGS)


def grounding_sample(ctx: Ctx, kind: str, rng) -> Sample | None:
    """Positive-channel prompt realizing one deictic grounding configuration."""
    built = _GROUNDING_CONFIGS[kind](ctx, rng)
    if built is None:
        return None
    circle, arrow, level, guard_pts, _needs_twin, note = built
    # Deictic text is assigned centrally in `_finalize`; use the source instruction only as
    # a placeholder so all modes follow the same name-free invariant.
    return Sample(circle=circle, arrow=arrow, caption=ctx.instruction,
                  labels=_ok(), severity=0.0, caption_level=level,
                  guard=_guard(contains=guard_pts), note=f"[{kind}] {note}")


# ============================================================== generation ===

DEFAULT_LEVEL_MIX = {"L2": 0.5, "L3": 0.5}

# These modes make their negative by inserting or replacing explicit object/task names.
# They are incompatible with the deictic-only dataset and cannot be generated, even when
# requested explicitly on the CLI.
NAME_EXPLICIT_MODES = frozenset({"C2", "C3", "D1", "D2"})
DEFAULT_MODES = tuple(sorted(set(REGISTRY).difference(NAME_EXPLICIT_MODES)))


def _sample_level(rng, mix: dict) -> str:
    unsupported = set(mix).difference(_DEICTIC)
    if unsupported:
        raise ValueError(
            f"Name-explicit caption levels are disabled; use only L2/L3, got {sorted(mix)}"
        )
    lv, w = zip(*mix.items())
    return str(rng.choice(lv, p=np.array(w) / sum(w)))


def _finalize(ctx: Ctx, s: Sample, level: str, rng, keep_caption: bool = False) -> Sample:
    """Assign a name-free caption and compute whether exactly one reading survives."""
    if level not in _DEICTIC:
        raise ValueError(f"Only name-free L2/L3 captions are allowed, got {level!r}")
    owns_caption = keep_caption or not s.labels.get("caption_ok", True)  # D4 owns its L3
    if not owns_caption and level != "L0":
        for _ in range(10):
            cap = ctx.level(level, rng)
            # a pick verb settles the label in a place scene: valid with a circle alone,
            # wrong with an arrow (D3). Keep it off every row whose label is the sketch's.
            if not (cap and ctx.kind == "place" and is_pick(cap)):
                break
        else:
            cap = "put this there"
        if cap:
            s.caption = cap
    if not is_name_free_deictic_caption(ctx, s.caption):
        raise ValueError(f"Caption is not strictly name-free deixis: {s.caption!r}")
    n_ref, n_dst, why = resolve(ctx, s.circle, s.arrow, level, caption=s.caption)
    s.caption_level, s.n_referents, s.n_destinations, s.resolved_by = level, n_ref, n_dst, why
    s.labels["sufficient_ok"] = (n_ref * n_dst == 1)
    return s


def generate(ctx: Ctx, modes: list[str], per_mode: int, seed: int, key: str = "",
             level_mix: dict | None = None) -> list[dict]:
    """Generate `per_mode` corruption samples for each requested mode. Skips N/A modes.

    Every sample is passed through `_finalize`, which draws a name-free L2/L3 caption and
    computes sufficiency via `resolve()`. Modes that require explicit names are disabled.

    `key` (the demo id) seeds the RNG via zlib.crc32 (not hash(), which salts strings per
    process and broke reproducibility) and prevents every demo drawing the same stream.
    """
    mix = level_mix or DEFAULT_LEVEL_MIX
    out = []
    for m in modes:
        if m in NAME_EXPLICIT_MODES:
            raise ValueError(f"Mode {m} is disabled because it requires explicit names")
        fn = REGISTRY.get(m)
        if fn is None:
            continue
        for j in range(per_mode):
            for t in range(5):                         # retry a hidden arrow
                tag = f"{seed}/{key}/{m}/{j}" + (f"/{t}" if t else "")
                rng = np.random.default_rng(zlib.crc32(tag.encode()))
                try:
                    s = fn(ctx, rng)
                except Exception as e:                # never let one mode kill a demo
                    s = None
                    print(f"  [warn] mode {m} raised: {e}")
                if s is None or (arrow_visible(s.circle, s.arrow) and not (
                        not s.labels.get("arrow_ok", True) and (
                            head_on_destination(ctx, s) or head_on_alternative(ctx, s.arrow)))):
                    break
                s = None
            if s is None:
                continue
            level = _sample_level(rng, mix)
            rec = _finalize(ctx, s, "L3" if m == "A7" else level, rng,
                            keep_caption=(m == "A7")).as_record(m)
            if rec["verdict"] == "right":              # the corruption corrupted nothing
                continue
            rec["index"] = j
            out.append(rec)
    return out


def generate_grounding(ctx: Ctx, kinds: list[str], per_kind: int, seed: int,
                       key: str = "") -> list[dict]:
    """Deictic grounding samples from the un-corrupted base (mode ``POS``).

    ``deixis_resolved`` is right when its sketch resolves the caption; under-determined
    configurations are the former ambiguous samples and now receive verdict ``wrong``.
    """
    out = []
    for kind in kinds:
        for j in range(per_kind):
            rng = np.random.default_rng(
                zlib.crc32(f"{seed}/{key}/GROUND/{kind}/{j}".encode())
            )
            try:
                s = grounding_sample(ctx, kind, rng)
            except Exception as e:
                s = None
                print(f"  [warn] grounding {kind} raised: {e}")
            if s is None or not arrow_visible(s.circle, s.arrow) or (
                    kind == "deixis_resolved" and (not circle_selects_target(ctx, s.circle)
                                                   or rival_coverage(ctx, s.circle) > VALID_MARGIN)):
                continue
            s.grounding_kind = kind
            preserve_positive_resolution_guard(ctx, s)
            rec = _finalize(ctx, s, s.caption_level, rng).as_record("POS")
            if kind != "deixis_resolved" and rec["verdict"] != "wrong":
                # The attempted failure did not actually leave multiple readings.
                continue
            if rec["verdict"] == "wrong":
                set_record_category(rec, "G")
            rec["index"] = j
            out.append(rec)
    return out
