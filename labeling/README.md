# Labelling the human sketches (paper Sec. VI-B)

Every human sketch is labelled valid or invalid by an annotator who sees only the drawing and
its caption, never a model verdict.

| tool | what it does |
|---|---|
| `label_server.py` | Blind right / wrong / unsure labelling of rendered human sketches in a local web UI. Hides the geometric verdict, the instruction the participant saw and every model prediction. Appends to a JSONL, so a session resumes. Keys: `R` `W` `U`, space = keep and next, `B` back, `N` next unlabelled. `--seed` fixes the shuffle order. |
| `apply_labels.py` | Rebuilds a rendered set with the labels as ground truth (`out/human-eval-labelled`) and keeps the geometric verdict as `geometric_right`. |

Labels are task-blind: a sketch is valid when it shows exactly one sensible action. Two rules
come from the annotator's later decisions and are applied by `scripts/12_render_human.py`:

- A pick verb with no destination ("grab this", "pick up this", "take this") plus a circle
  alone is valid; the same caption with an arrow is invalid, since the arrow contradicts it
  (2026-10-24, 2026-10-27).
- Individual relabels after re-review live in `human_label_overrides.jsonl` and win over both
  the original label and the rule.

Both read and write a human-eval working directory that is not in git: `$HUMAN_EVAL_ROOT`,
else `../human_eval` next to this repository, else `human_eval/` inside it.

```text
<HUMAN_EVAL_ROOT>/out/human-eval/    rendered set: images/ + human_sketches.jsonl
<HUMAN_EVAL_ROOT>/out/labels.jsonl   label_server.py output (last label per response wins)
```

```bash
python labeling/label_server.py --port 8791     # open http://localhost:8791
python labeling/apply_labels.py                  # -> out/human-eval-labelled
```

The sketches come from the study web app in `shibuina/user-study` (branch
`prompt-to-sketch`). To train or evaluate on a labelled set, export it as
`data/<config>/<split>-*.parquet` with `image`, `caption`, `verdict` and `id` columns and pass the
directory as `--data` to `training/train.py` or `eval/eval_vlm.py`.
