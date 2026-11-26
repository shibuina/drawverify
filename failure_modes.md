# Failure taxonomy (paper Sec. IV, Table I)

What the validator must reject. Four corruption families define 18 modes (A–D); a separate
Grounding family covers prompts whose channels are each fine but leave more than one
reading. Code: `libero_sketch/false_prompts.py` (modes, `resolve()`, guards) driven by
`scripts/09_false_prompts.py`, with `scripts/08_scene_dump.py` supplying every object's
geometry. Mode IDs are the generator's keys and are non-contiguous (A1, B2 were dropped).

## A prompt and its verdict

A prompt `P = (I, C, A, L)` overlays a circle (`{cx, cy, r}`, *which* object), an arrow
(`{x0, y0, x1, y1}`, tail on the object, head on the destination) and a name-free caption
(L2/L3 of the caption ladder, e.g. "move this there") on the t0 frame of a LIBERO demo.
Every record carries the paper's predicates:

| paper | record field | true when |
|---|---|---|
| `v_C` | `circle_ok` | the circle selects the intended target |
| `v_A` | `arrow_ok` | the arrow has the right bearing and magnitude, anchored on the target, ending on the destination |
| `v_L` | `caption_ok` | the caption's verb matches the intended action |
| `v_X` | `consistency_ok` | circle, arrow and caption describe the same object, destination and predicate |
| `v_S` | `sufficient_ok` | `n_readings = n_referents × n_destinations == 1` (computed by `resolve()`) |

`verdict` is `right` (paper: valid) iff all five hold, else `wrong` (invalid).

`resolve()` counts readings against the object masks stage 08 writes to
`frames/t0_objects.png`:

- **Referent.** The circle selects an object when it encloses at least half of the object's
  visible mask, or, for a fixture, when at least half of the circle lies on it (a drawer, a
  knob). Only objects the task can act on compete: movable objects for place tasks,
  fixtures for open/turn-on. An object directly under the target (a bowl on a ramekin) does
  not compete with it.
- **Destination.** The arrow head indicates the objects whose masks lie within
  max(5 px, 1.5 x the nearest) of it, never the referent itself; a head more than 15 px
  from every mask is on bare table, which is one spot.
- **Ground truth.** The clean arrow's head is the object's final position. For a drawer the
  drawer is pulled out by then, so the head is moved to the nearest point 8 px inside the
  closed cabinet's mask (goal/03; every other task's head already lies on its destination).

It runs on the declared and on the rendered strokes, and a row is kept only when both
agree. Until 2026-10-10 it counted object *centre points* inside the circle and near the
head. That labelled 2,102 correct sketches wrong (a stove or cabinet under the bowl, a
cabinet top near the wine rack); `reports/` holds the audit before and after the fix.

## Guards: augmentation never flips a label

Hand-drawn jitter could drag a valid circle off its object or nudge a near-miss back onto
it. Every sample therefore declares `guard {contains: [...], excludes: [...]}` — the points
the *drawn* circle must and must not enclose — and `hand_variant()` resamples the stroke
until the guard holds. Valid prompts keep the target in (in-point guard) and distractors out
(out-point guard); negatives keep their corruption.

Every failure must also be visible: an arrow whose head lies inside its circle is resampled
(up to five times) or dropped, an undershoot still leaves the circle, a dot circle is
3-5 px rather than invisible, a frame-sized circle has r <= 0.48 so most of it stays
on-frame, "empty table" marks avoid the robot (stage 08 adds it to the occupancy map),
strokes are never black, and a corruption that resolves to a valid prompt is dropped.
Positives of open/turn-on tasks are circle-only: an arrow there is B7's over-specification.

## Corruption modes

| family | ID | mode | generation | in data |
|---|---|---|---|---|
| **Referential** (circle, `v_C`) | A2 | Wrong-category object | circle a different-category object from `scene.json` | ✓ |
| | A3 | Near-miss circle | centre offset by `k·r`, `k ∈ [1.05, 2.2]`, from the target point | ✓ |
| | A4 | Empty-region circle | disc fully on empty table (`t0_occupancy.png`) | ✓ |
| | A5 | Multi-object enclosure | radius grown over a neighbour; `circle_ok` stays true, `sufficient_ok` fails | ✓ |
| | A6 | Degenerate size | dot-sized or frame-sized circle, centre correct | ✓ |
| **Directional** (arrow, `v_A`) | B1 | Wrong destination | head on another container or region | ✓ |
| | B3 | Wrong magnitude | correct bearing, length ×0.10–0.35 or ×1.8–3.0 | ✓ |
| | B4 | Wrong bearing | correct length, head rotated 90° or 180° about the tail | ✓ |
| | B5 | Null destination | head on empty table or off-frame | ✓ |
| | B6 | Detached tail | tail moved 2–3.5·r off the circled object | ✓ |
| | B7 | Spec mismatch | place task without arrow, or manipulate task with one | ✓ |
| **Cross-modal** (`v_X`) | C1 | Circle–arrow mismatch | valid circle on object A, valid arrow drawn for object B | ✓ |
| | C2 | Caption–sketch mismatch | caption borrowed from another task | n/a |
| | C3 | Predicate mismatch | caption from a task of the other kind (place vs manipulate) | n/a — see below |
| **Language** (caption, `v_L`) | D1 | Wrong target noun | swap the target noun | n/a |
| | D2 | Wrong destination noun | swap the destination noun | n/a |
| | D3 | Wrong verb | place task with "open/close/turn on/off this" or a pick verb ("grab/pick up/take this": no destination, so the arrow contradicts it); manipulate task with "move/put/place this there" | ✓ |
| | D4 | Under-determined caption | L3 "put this there" whose circle is off-target (A3) | ✓ |

C2, C3, D1 and D2 are implemented but disabled (`NAME_EXPLICIT_MODES`): their captions name
objects, which the name-free dataset forbids, and the generator refuses them even when asked.
**The paper's Table I marks C3 as instantiated; the generator does not produce it.** With
name-free captions the predicate conflict C3 describes is what D3 generates, so the data
holds 14 instantiated modes, not 15.

## Grounding configurations (mode `POS`, slot = kind)

| kind | caption | sketch | verdict |
|---|---|---|---|
| `deixis_resolved` (resolved) | L3 | full circle + arrow | right when one reading survives |
| `no_sketch` (no sketch) | L3 | none | wrong |
| `dest_open` (open destination) | L3 | circle only, place task | wrong |
| `referent_twin` (twin referents) | L2 | circle over an identical pair | wrong |
| `dest_between` (arrow between destinations) | L3 | head midway between two containers | wrong |
The paper lists these five. Builds before 2026-10-11 also hold `clean_unresolved`: hand-jittered
positives whose circle clipped a neighbour (median 55% of it enclosed) or whose head grazed one,
relabelled wrong although their note still read "still valid". They sat between the positives
(a competing object at most 37% enclosed for 95% of them) and the deliberate ambiguity
negatives (100% enclosed), so the generator now drops such attempts instead.

## Measured composition (stage-09 defaults, regenerated 2026-10-27, generator v4)

`--suite all --max-demos 20`: 795 of 800 demos over 40 tasks -> **42,795 rows**, 16,082 right /
26,713 wrong (37.6% valid); train 34,165 / validation 4,223 / test 4,407 (demo-disjoint,
`crc32` of `suite/task/demo`). Five demos are skipped because the target shows fewer than 60
pixels (long/00 soup behind the milk carton). Labels are task-blind: besides the demo's own
action (`POS`), a valid prompt may show another sensible place action (`POS_alt`,
`PLACE_INTO`) or circle one object with a pick verb and no arrow (`POS_pick`). Strokes replay
human drawings in the study app's fixed look; 65 rows fell back to the human shape at the
declared position and 1,025 rows were dropped because no drawing kept their label.

| slot | rows | | slot | rows |
|---|---:|---|---|---:|
| A2 / A3 / A4 / A5 / A6 / A7 | 1,502 / 1,419 / 1,586 / 1,531 / 1,308 / 1,497 | | POS (right) | 9,159 |
| B1 / B3 / B4 / B5 / B6 | 1,505 / 1,406 / 1,481 / 1,493 / 1,342 | | POS_alt (right) | 3,979 |
| B7 / B8 | 1,590 / 906 | | POS_pick (right) | 1,420 |
| C1 | 1,484 | | deixis_resolved (right) | 1,524 |
| D3 / D4 | 1,590 / 1,427 | | no_sketch / dest_open | 1,590 / 1,510 |
| | | | dest_between / referent_twin | 234 / 312 |

A7 is a circle-only "grab this" whose circle picks out nothing or two things (empty table,
the robot, a dot, a frame-sized loop, two objects); B8 is an arrow that stops 11-23 px short
of the destination. 40% of A5 rows draw two separate loops and 30% of C1 rows put the circle
on a fixture. The rules behind each check came from two manual passes over the pilot and the
user's decisions on the open cases (changelog, 2026-10-23/24).

Every row is also checked on the strokes actually drawn (`rendered_ok`), since the
hand-drawn jitter moves the head up to ~31 px: a valid arrow head must touch the
destination mask (or sit on the table spot, clear of every object), a corruption must
not land back on it, A2/A3/A4/C1/D4 circles must cover at most 30% of the target, C1 tails
must start outside the circle, and A5/`referent_twin` circles must still hold both objects.
A row that fails falls back to its clean strokes, and 160 rows that fail even then are
dropped. Arrows always carry a head (`head_none_p: 0`). These rules came from a manual pass
over all 1,499 contact sheets of the previous build: 780 tiles flagged, 560 of them valid
place prompts whose head stopped short of or overshot the destination. An arrow
corruption (B1/B3/B4/B5) whose head still lands on the destination object is resampled:
B1 used to compare against the region name and re-pick the stove or basket itself, and
overshoots stayed on big destinations.

| slot | rows | | slot | rows |
|---|---:|---|---|---:|
| A2 / A3 / A4 / A5 / A6 | 1,506 / 1,470 / 1,590 / 1,372 / 1,312 | | POS (right) | 8,536 |
| B1 / B3 / B4 / B5 / B6 | 1,513 / 1,426 / 1,505 / 1,357 / 1,365 | | deixis_resolved (right) | 1,532 |
| B7 | 1,594 | | no_sketch | 1,594 |
| C1 | 1,471 | | dest_open | 1,514 |
| D3 / D4 | 1,594 / 1,479 | | dest_between | 1,116 |
| | | | referent_twin | 312 |

The August 2026 build (37,970 rows, 24.0% valid) is what the Hub datasets
`shibuina/drawvla-prompt-validation{,-clean,-v3}` hold; `reports/before` audits it.
The v3 build above is `shibuina/drawvla-prompt-validation-taskblind` (private, pushed 2026-10-27).

The paper reports a different release: 8,522 prompts (6,463 / 900 / 1,159; 31.8% valid).
That subset is not reproducible from this repository's defaults; see README.

`referent_twin` and A5 fire mostly in the spatial suite, the only one with identical
duplicate objects.
