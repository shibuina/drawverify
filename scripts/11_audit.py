#!/usr/bin/env python
"""Audit a stage-09 output against the scene masks and write a local HTML report.

Every row is checked for label and visibility defects. Definitional checks need no oracle
(the simulator's own annotation labelled wrong, a caption verb the task cannot take); the
mask checks read stage 08's instance maps (`frames/t0_objects.png`) through the grounding
rule of `false_prompts.resolve`; the image checks compare each render with its clean frame.
Pass `--masks-root` to audit an older output whose demos lack instance maps.

  python scripts/11_audit.py --out-root output --report reports/after --compare reports/before/audit.json
"""

import argparse
import html
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import DATA_GEN_ROOT
from libero_sketch import false_prompts as FP

CHECKS = {
    "gt_sketch_labelled_wrong": "The simulator's own annotation (clean positive or full deictic sketch) is labelled wrong.",
    "valid_sketch_labelled_wrong": "Wrong only by sufficiency, but the masks show one referent and one destination.",
    "jittered_positive_relabelled_wrong": "A hand-jittered positive turned into a wrong grounding row (clean_unresolved) while its note says still valid.",
    "caption_verb_conflict": "Labelled right, yet the caption's verb cannot apply (\"open this there\" on a place task).",
    "valid_target_not_selected": "Labelled right, yet the circle encloses less than half of the target.",
    "corruption_labelled_right": "A corruption mode emitted a row labelled right.",
    "valid_arrow_on_manipulate": "Labelled right, yet an open/turn-on task carries an arrow (B7 calls that wrong).",
    "arrow_corruption_on_destination": "A B1/B3/B4/B5 negative whose head still lands on the true destination object.",
    "arrow_corruption_on_alternative": "A B1/B3/B4/B5 negative whose head lands on another object the target could sensibly go to (valid under task-blind labels).",
    "valid_head_off_destination": "Labelled right, yet the arrow head does not land on the destination object.",
    "valid_neighbour_over_margin": "Labelled right, yet a competing object is more than 30% inside the circle.",
    "arrow_hidden_in_circle": "The arrow head lies inside the circle, so the arrow cannot be seen.",
    "head_off_frame": "The arrow head is off-frame outside B5 (off-frame is B5's own failure).",
    "empty_mark_on_robot": "An 'empty table' mark (A4 circle, B5 head) lands on the robot.",
    "frame_circle_offscreen": "A frame-sized A6 circle keeps less than 40% of its outline on-frame.",
    "sketch_invisible": "Fewer than 25 pixels of the render differ from the clean frame.",
    "black_stroke": "Most stroke pixels are near-black (unreadable on the dark tabletops).",
}
FAMILY = {"A": "referential", "B": "directional", "C": "crossmodal", "D": "language",
          "G": "grounding", "P": "positive"}
PLACE_VERBS = ("put", "place", "move", "push", "pick up", "grab", "take", "stack", "set")
MANIP_VERBS = ("open", "close", "turn on", "turn off")
S = 256


def load_rows(out_root: Path) -> list[dict]:
    rows = []
    for f in sorted((out_root / "false_prompts").glob("*/samples.jsonl")):
        if f.parent.name in {"spatial", "object", "goal", "long", "90"}:
            rows += [json.loads(line) for line in f.open()]
    return rows


class Demo:
    """Masks, scene and meta of one demo, from the masks root."""

    def __init__(self, root: Path):
        self.meta = json.loads((root / "meta.json").read_text())
        self.scene = json.loads((root / "scene.json").read_text())
        self.inst = cv2.imread(str(root / "frames" / "t0_objects.png"), cv2.IMREAD_GRAYSCALE)
        self.rgb = np.asarray(Image.open(root / "frames" / "t0_rgb.png").convert("RGB"))
        self.ctx = FP.Ctx(meta=self.meta, scene=self.scene, occupancy=None, caption_pool=[],
                          ladder=None, instances=self.inst)


def caption_verb(caption: str) -> str | None:
    c = caption.lower()
    return next((v for v in MANIP_VERBS + PLACE_VERBS if c.startswith(v + " ")), None)


def audit_row(r: dict, d: Demo, out_root: Path) -> list[str]:
    hits = []
    ctx = FP.alt_context(d.ctx, **r["goal_override"]) if r.get("goal_override") else d.ctx
    lab = r["labels"]
    c, a = r["prompt"]["circle"], r["prompt"]["arrow"]
    slot, right = r["slot"], r["verdict"] == "right"
    level = r["resolution"]["caption_level"]
    clean_gt = (slot in {"POS", "clean_unresolved"} and "clean ground-truth" in r["note"]) \
        or slot == "deixis_resolved"
    if clean_gt and not right:
        hits.append("gt_sketch_labelled_wrong")
    if slot == "clean_unresolved" and not clean_gt:
        hits.append("jittered_positive_relabelled_wrong")
    channels_ok = all(lab.get(k, True) for k in FP._WRONG_CHANNELS)
    if not right and channels_ok and not lab.get("sufficient_ok", True) and \
            slot in {"POS", "clean_unresolved", "deixis_resolved", "dest_between"}:
        n_ref, n_dst, _ = FP.resolve(ctx, c, a, level, caption=r["prompt"]["caption"])
        if n_ref * n_dst == 1 and (c is None or FP.circle_selects_target(ctx, c)):
            hits.append("valid_sketch_labelled_wrong")
    verb = caption_verb(r["prompt"]["caption"])
    kind = ctx.kind
    if right and verb and ((kind == "place" and verb in MANIP_VERBS)
                           or (kind != "place" and verb in PLACE_VERBS)):
        hits.append("caption_verb_conflict")
    if right and c is not None and not FP.circle_selects_target(ctx, c):
        hits.append("valid_target_not_selected")
    if right and r["mode"] != "POS":
        hits.append("corruption_labelled_right")
    if right and a and kind != "place":
        hits.append("valid_arrow_on_manipulate")
    probe = FP.Sample(circle=c, arrow=a, caption="", labels={})
    if slot in {"B1", "B3", "B4", "B5"} and FP.head_on_destination(ctx, probe):
        hits.append("arrow_corruption_on_destination")
    if slot in {"B1", "B3", "B4", "B5"} and FP.head_on_alternative(ctx, a):
        hits.append("arrow_corruption_on_alternative")
    if right and a and kind == "place" and ctx.dest_object and \
            not FP.head_on_destination(ctx, probe):
        hits.append("valid_head_off_destination")
    if right and c is not None and FP.rival_coverage(ctx, c) > FP.VALID_MARGIN:
        hits.append("valid_neighbour_over_margin")
    if c and a and not FP.arrow_visible(c, a):
        hits.append("arrow_hidden_in_circle")
    if a and slot != "B5" and not (0 <= a["x1"] <= 1 and 0 <= a["y1"] <= 1):
        hits.append("head_off_frame")
    robot = d.inst == 255
    if slot == "A4" and c:
        yy, xx = np.mgrid[0:S, 0:S]
        disc = (xx + 0.5 - c["cx"] * S) ** 2 + (yy + 0.5 - c["cy"] * S) ** 2 <= (c["r"] * S) ** 2
        if robot[disc].mean() > 0.2:
            hits.append("empty_mark_on_robot")
    if slot == "B5" and a and 0 <= a["x1"] < 1 and 0 <= a["y1"] < 1 and \
            robot[int(a["y1"] * S), int(a["x1"] * S)]:
        hits.append("empty_mark_on_robot")
    if slot == "A6" and c and c["r"] > 0.3:
        th = np.linspace(0, 2 * np.pi, 360)
        x, y = c["cx"] + c["r"] * np.cos(th), c["cy"] + c["r"] * np.sin(th)
        if np.mean((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)) < 0.4:
            hits.append("frame_circle_offscreen")
    if r.get("render") and slot != "no_sketch":        # no_sketch is blank by design
        img = np.asarray(Image.open(out_root / r["render"]).convert("RGB")).astype(int)
        diff = np.abs(img - d.rgb.astype(int)).max(-1) > 40
        if diff.sum() < 25:
            hits.append("sketch_invisible")
        elif (img[diff].max(-1) < 70).mean() > 0.6:
            hits.append("black_stroke")
    return hits


def tile(r: dict, d: Demo, out_root: Path, why: str) -> Image.Image:
    font = ImageFont.load_default()
    img = Image.open(out_root / r["render"]).convert("RGB").resize((S, S))
    gt = Image.fromarray(d.rgb.copy())
    dr = ImageDraw.Draw(gt)
    for o in d.scene["objects"]:
        x, y = o["center_px"]
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 0))
        dr.text((x + 3, y - 4), o["name"][:16], fill=(255, 255, 0), font=font)
    tx, ty = d.meta["points_px"]["target_t0"]
    dr.line([tx - 7, ty, tx + 7, ty], fill=(0, 255, 0), width=2)
    dr.line([tx, ty - 7, tx, ty + 7], fill=(0, 255, 0), width=2)
    a = d.ctx.arrow_clean
    if a:
        hx, hy = a["x1"] * S, a["y1"] * S
        dr.line([hx - 6, hy - 6, hx + 6, hy + 6], fill=(0, 200, 255), width=2)
        dr.line([hx - 6, hy + 6, hx + 6, hy - 6], fill=(0, 200, 255), width=2)
    t = Image.new("RGB", (2 * S + 4, S + 80), "white")
    t.paste(img, (0, 0))
    t.paste(gt, (S + 4, 0))
    td = ImageDraw.Draw(t)
    g = d.meta["goal"]
    lines = [f"{r['id'].split('/')[0]}/{r['task'][:3]}{r['demo']} {r['slot']}#{r['index']} -> "
             f"{r['verdict'].upper()}  [{why}]",
             f"caption: {r['prompt']['caption']!r}   goal: {g['predicate']} "
             f"{g['target_object']} -> {g['destination']}"]
    lines += textwrap.wrap("note: " + r.get("note", ""), 84)[:3]
    for i, line in enumerate(lines):
        td.text((4, S + 4 + 13 * i), line, fill="black", font=font)
    return t


def sheet(tiles: list[Image.Image], path: Path, cols: int = 3) -> None:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + 6), rows * (h + 6)), (210, 210, 210))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    out.save(path)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def write_html(report: Path, title: str, res: dict, prev: dict | None) -> None:
    n = res["rows"]
    th = "<th>check</th><th>what it means</th>" + (
        "<th>before</th>" if prev else "") + "<th>rows</th><th>share</th>"
    body = []
    for k, desc in CHECKS.items():
        v = res["defects"].get(k, 0)
        before = f"<td class=num>{prev['defects'].get(k, 0):,}</td>" if prev else ""
        body.append(f"<tr><td><code>{k}</code></td><td>{html.escape(desc)}</td>{before}"
                    f"<td class=num>{v:,}</td><td class=num>{pct(v, n)}</td></tr>")
    tasks = sorted(res["tasks"].items(), key=lambda kv: kv[1]["right"] / max(kv[1]["n"], 1))
    task_rows = "".join(
        f"<tr{' class=low' if t['right'] / t['n'] < 0.15 else ''}><td>{html.escape(k)}</td>"
        + (f"<td class=num>{pct(prev['tasks'][k]['right'], prev['tasks'][k]['n'])}</td>"
           if prev and k in prev["tasks"] else ("<td></td>" if prev else ""))
        + f"<td class=num>{t['n']:,}</td><td class=num>{t['right']:,}</td>"
        f"<td class=num>{pct(t['right'], t['n'])}</td></tr>" for k, t in tasks)
    slots = "".join(f"<tr><td>{html.escape(k)}</td><td class=num>{v.get('right', 0):,}</td>"
                    f"<td class=num>{v.get('wrong', 0):,}</td></tr>"
                    for k, v in sorted(res["slots"].items()))
    figs = "".join(f"<h3><code>{k}</code></h3><p>{html.escape(CHECKS[k])}</p>"
                   f"<img src='img/{k}.png' alt='{k} examples'>" for k in res["examples"])
    figs = figs or "<p>No row hit any check.</p>"
    samples = "".join(f"<h3>{f}</h3><img src='img/sample_{f}.png' alt='{f} sample'>"
                      for f in sorted(set(FAMILY.values())))
    summary = (f"{n:,} rows: {res['right']:,} right ({pct(res['right'], n)}) / "
               f"{n - res['right']:,} wrong")
    if prev:
        summary += (f"; before: {prev['rows']:,} rows, {prev['right']:,} right "
                    f"({pct(prev['right'], prev['rows'])})")
    (report / "index.html").write_text(f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1400px;padding:0 16px;color:#222;background:#fff}}
table{{border-collapse:collapse;margin:8px 0 24px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f2f2f2}}.num{{text-align:right;font-variant-numeric:tabular-nums}}tr.low td{{background:#fde8e8}}
img{{max-width:100%;border:1px solid #ccc}}code{{font-size:13px}}</style></head><body>
<h1>{html.escape(title)}</h1><p>{html.escape(res['out_root'])}<br>{summary}</p>
<h2>Defects</h2><table><tr>{th}</tr>{''.join(body)}</table>
<h2>Valid share per task</h2><p>Tasks below 15% valid are shaded.</p>
<table><tr><th>task</th>{'<th>before</th>' if prev else ''}<th>rows</th><th>right</th><th>right share</th></tr>{task_rows}</table>
<h2>Rows per slot</h2><table><tr><th>slot</th><th>right</th><th>wrong</th></tr>{slots}</table>
<h2>Examples</h2>{figs}
<h2>Random sample per family</h2><p>Left: the rendered prompt. Right: the clean frame with
object centres (yellow), the target (green +) and the ground-truth arrow head (cyan x).</p>
{samples}</body></html>""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=DATA_GEN_ROOT / "output")
    ap.add_argument("--masks-root", type=Path, help="output root holding the instance maps")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--title", default="Prompt generator audit")
    ap.add_argument("--compare", type=Path, help="audit.json of an earlier state")
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    masks = a.masks_root or a.out_root
    rows = load_rows(a.out_root)
    demos, defects, examples = {}, Counter(), defaultdict(list)
    tasks, slots = defaultdict(Counter), defaultdict(Counter)
    for i, r in enumerate(rows):
        key = str(Path(r["source_frame"]).parent.parent)
        if key not in demos:
            demos[key] = Demo(masks / key)
        hits = audit_row(r, demos[key], a.out_root)
        defects.update(hits)
        for h in hits:
            examples[h].append(r)
        t = f"{r['suite']}/{r['task']}"
        tasks[t]["n"] += 1
        tasks[t]["right"] += r["verdict"] == "right"
        slots[r["slot"]][r["verdict"]] += 1
        if i % 5000 == 0:
            print(f"{i:,}/{len(rows):,}", flush=True)
    a.report.mkdir(parents=True, exist_ok=True)
    (a.report / "img").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    shown = []
    for k in CHECKS:
        pool = examples.get(k)
        if not pool:
            continue
        pick = [pool[j] for j in sorted(rng.choice(len(pool), min(a.examples, len(pool)),
                                                   replace=False))]
        sheet([tile(r, demos[str(Path(r["source_frame"]).parent.parent)], a.out_root, k)
               for r in pick], a.report / "img" / f"{k}.png")
        shown.append(k)
    families = defaultdict(list)                  # a random look at every family
    for r in rows:
        families[FAMILY.get(r["category"], r["category"])].append(r)
    for fam, pool in sorted(families.items()):
        pick = [pool[j] for j in sorted(rng.choice(len(pool), min(a.examples, len(pool)),
                                                   replace=False))]
        sheet([tile(r, demos[str(Path(r["source_frame"]).parent.parent)], a.out_root, fam)
               for r in pick], a.report / "img" / f"sample_{fam}.png")
    res = {"out_root": str(a.out_root), "rows": len(rows),
           "right": sum(r["verdict"] == "right" for r in rows), "defects": dict(defects),
           "tasks": {k: dict(v) for k, v in tasks.items()},
           "slots": {k: dict(v) for k, v in slots.items()}, "examples": shown}
    (a.report / "audit.json").write_text(json.dumps(res, indent=1))
    prev = json.loads(a.compare.read_text()) if a.compare else None
    write_html(a.report, a.title, res, prev)
    print(json.dumps(res["defects"], indent=1))
    print(f"report -> {a.report / 'index.html'}")


if __name__ == "__main__":
    main()
