# DrawVerify

Code for *DrawVerify: Learning to Validate Circle-and-Arrow Robot Instructions from Simulator
Ground Truth* (the CS707 project's pilot). It generates exactly labelled sketch prompts from
LIBERO demonstrations, fine-tunes Qwen VLM validators on them, and scores validators in the
paper's convention.

## Paper → code

| paper | code |
|---|---|
| Sec. III, predicates `v_C v_A v_L v_X v_S` and `V(P)` | `libero_sketch/false_prompts.py` (`_ok`, `verdict_of`, `resolve`) |
| Sec. IV, Table I taxonomy | [`failure_modes.md`](failure_modes.md), `false_prompts.REGISTRY` |
| Sec. V-A simulator-grounded extraction | `scripts/00_download.py`, `01_extract.py` (`extract.py`), `08_scene_dump.py` (`scene.py`) |
| Sec. V-B guarded human-imprecision augmentation | `scripts/02_annotate.py`, `strokes.py`, `false_prompts.hand_variant` (guards) |
| Sec. V-C controlled counterfactual failures | `scripts/09_false_prompts.py` |
| Sec. V-D name-free caption ladder (L0–L3, sampled L2/L3) | `scripts/03_captions.py` (`captions.py`, `bddl.py`, `llm.py`) |
| Sec. V-E composition, formats, demo-disjoint splits | `scripts/10_export_hf.py` |
| Sec. VI-A validator training (staged LoRA, `<answer>` scoring) | `training/train.py`, `training/validator.py` |
| Sec. VI-B human sketch labels | `labeling/` |
| Tables III–IV metrics, paired bootstrap, recall by family, gate thresholds | `eval/eval_vlm.py`, `eval/score.py`, `eval/metrics.py` |
| Sec. VIII-C, Table V pixel-geometry baselines | `eval/geometry_baseline.py` |

## Quick start (one-click reproduction)

```bash
conda env create -f environment.yml && conda activate drawverify
# Adapters + the per-prompt predictions behind every report table (Hugging Face Hub), checksums verified
bash scripts/download_adapters.sh
# Verify adapter checksums, run unit tests, and reproduce Tables 1, 6, 7 and the seed statistics
bash scripts/run.sh
```

Other modes of `scripts/run.sh`:

```bash
bash scripts/run.sh verify-checksums            # verify SHA-256 of output/adapters/*
bash scripts/run.sh train stage1-ans --seed 0   # train 2B answer-only stage-1 LoRA (configs/train_stage1_ans.yaml)
bash scripts/run.sh train stage1-rat --seed 0   # train 2B rationale stage-1 LoRA (configs/train_stage1_rat.yaml)
bash scripts/run.sh train stage2 --seed 0       # fine-tune stage-2 LoRA on human sketches (configs/train_stage2.yaml)
bash scripts/run.sh demo --selftest             # smoke-test the interactive web demo
```

## Setup

Conda environment (data processing, evaluation, and training):

```bash
conda env create -f environment.yml
conda activate drawverify
```

Or install by component:

Generator (needs the simulator for stages 01 and 08):

```bash
bash setup_env.sh                 # conda env `libero_sketch`: LIBERO, robosuite 1.4, mujoco 2.3.7
conda activate libero_sketch
export MUJOCO_GL=egl
```

Validator (a CUDA machine; Qwen3-VL needs `transformers>=4.57`):

```bash
pip install -r training/requirements.txt
```

Trained LoRA adapters (`output/adapters/{2b-full4-ans,2b-full4,2b-full4-hum-s0}/`, 402 MB total)
are hosted on Hugging Face Hub (`shibuina/drawverify-adapters`) and can be downloaded and
verified with `bash scripts/download_adapters.sh` (or verified locally with
`sha256sum -c output/adapters/SHA256SUMS`). The same script fetches the per-prompt predictions
(`output/eval/`, 48 MB) and the human test labels without images (`output/human_test_labels.parquet`),
which `scripts/run.sh` turns into the report tables. SLURM cluster scripts for multi-seed
runs, data-size sensitivity, and baselines live in `cluster/job.sh` and `cluster/fetch.sh`.

## 1. Generate the prompts

```bash
python scripts/00_download.py --suite all                      # LIBERO demos (~40 GB)
python scripts/01_extract.py  --suite all --max-demos 20       # sim: target, destination, frames
python scripts/02_annotate.py --suite all                      # guarded hand-drawn variants
python scripts/03_captions.py --suite all --provider none      # L0-L3 ladder (gemini|openai adds paraphrases)
python scripts/08_scene_dump.py --suite all --max-demos 20     # sim: object geometry + instance map
python scripts/09_false_prompts.py --suite all --max-demos 20  # labelled prompts + images
python scripts/10_export_hf.py --split-by family               # parquet export, 6 family configs
python scripts/11_audit.py --report reports/now                # label + visibility audit (HTML)
```

`--suite` takes `spatial`, `object`, `goal`, `long` (= libero_10), a comma list, or `all`;
20 demos per task over the four suites gives the paper's 800 demonstrations and 40 tasks.
Stages 02–10 are sim-free. Add `--push --repo <user>/<name>` to stage 10 to publish.

Rendering styles (stage 09 `--style`):

| style | strokes | use |
|---|---|---|
| `hand` (default) | recorded human drawings replayed: their path and where they sat relative to the clean mark | training |
| `shape` (+ `--positives-clean`) | the same human paths at the exact declared position | placement ablation |
| `clean` (+ `--positives-clean --dedupe`) | exact geometry, perfect circle and straight arrow | ablation without augmentation |

Every style draws in the study app's look (`strokes.UI`): green circle, red arrow, 0.8% of the
image wide, V head; colour, width and head never vary. The templates
(`libero_sketch/assets/human_strokes.json`) come from `scripts/build_human_templates.py`: 71
drawings that matched their scene, by 12 participants who are not in the human eval set.
`scripts/12_render_human.py` re-renders a labelled human set in the same look, so the two sets
differ only in the drawing.

Labels are task-blind, like the human labels (`labeling/README.md`): a prompt is right when it
shows one sensible action, whichever object it circles. Stage 09 adds
`alt_positives_per_demo` valid prompts for other place actions in the scene (`POS_alt`, from
`false_prompts.PLACE_INTO`), and arrow corruptions never land on such an action.
A circle with no arrow is right with a pick verb ("grab this", "pick up this", "take this":
`POS_pick`) and wrong with any caption that needs a destination. Corruptions also avoid places
people accept but that are never generated as valid (`SENSIBLE_EXTRA`, cabinets, shelves), and
B8 adds the arrow that stops just short of the destination. Human-set label decisions made
after labelling live in `labeling/human_label_overrides.jsonl`, applied by stage 12.

With the defaults (generator v4, 2026-10-27) the generator yields 42,795 rows (16,082 valid /
26,713 invalid; train / validation / test 34,165 / 4,223 / 4,407); per-mode counts are in `failure_modes.md`. Stage 08
must have written `frames/t0_objects.png`: prompts are resolved against object masks, and the
older centre-point rule labelled 2,102 correct sketches invalid (see `reports/`). The Hub dataset
`shibuina/drawvla-prompt-validation-taskblind` (public) is this v4 build, 42,795 rows. Train and
validation rows carry `label_reason`, the rationale used as the training target; the test
split redacts it and every diagnostic field.

## 2. Train a validator

```bash
# stage 1: generated prompts (paper defaults: r 32, alpha 64, 2 epochs, batch 16, lr 1e-4)
python training/train.py --model Qwen/Qwen3-VL-4B-Instruct \
    --data shibuina/drawvla-prompt-validation-taskblind --output-dir output/train/4b-gen
# stage 2: the same adapter on the labelled human sketches by drawers outside the eval set
# (scripts/12_render_human.py --exclude ... --cal-share 0.3: 144 train / 39 calibration)
python training/train.py --model Qwen/Qwen3-VL-4B-Instruct --data output/human_train \
    --init-adapter output/train/4b-gen --epochs 4 --batch-size 8 --grad-accum 1 --lr 3e-5 \
    --output-dir output/train/4b-gen-hum
```

`--answer-only` drops the rationale from the targets (the paper's answer-only variant). The
other backbones are `Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen2-VL-2B-Instruct`. `--data` takes a
stage-10 export directory or a Hub dataset id; any export laid out as
`data/<config>/<split>-*.parquet` with `image`, `caption`, `verdict` (and optionally
`label_reason`, `id`) columns works, including human sets.

## 3. Evaluate

```bash
python eval/eval_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --adapter output/train/4b-gen-hum \
    --data <export> --split test --out output/eval/4b-gen-hum.jsonl
python eval/eval_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --data <export> --out output/eval/4b-zero-shot.jsonl
# P(wrong) is read from the answer alone (unsaturated, so a threshold works); --max-new-tokens 96
# generates the rationale first (the earlier protocol: human AUROC 0.736 vs 0.771 for full v3 2B)
python eval/score.py --ref "4B zero-shot" --families \
    --pred "4B zero-shot=output/eval/4b-zero-shot.jsonl" "4B gen->hum=output/eval/4b-gen-hum.jsonl" \
    --cut "4B gen->hum=output/eval/4b-gen-hum_human-train.jsonl@0.85"
python eval/geometry_baseline.py --fit gen=<gen export>:train --fit hum=<human export>:train \
    --fit gen+hum=<gen export>:train,<human export>:train \
    --test test=<human export>:test --out-dir output/eval/geometry
```

Invalid is the positive class: Prec and Rec are on invalid prompts, `r_val` is recall on
valid ones, AUROC is computed from P(wrong). `--ref` adds the paired AUROC difference with a
2,000-resample bootstrap interval (`*` when it excludes zero); `--cut` re-decides a model at
the threshold that gives the stated invalid recall on a calibration file (the live gate used
0.85 on the 82 human training sketches); `--families` prints recall per failure family. Any
model's outputs (an API model's `p_fail` in [0, 100] included) can be scored if written as
`{"id", "target", "p_wrong" | "p_fail", "prediction"?}` lines.

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/` (sim-free; the collator test needs the
Qwen3-VL processor) and `python eval/metrics.py`.

## 4. Demo

`demo/run.sh` serves a local page (http://127.0.0.1:8765): pick a LIBERO frame, draw a circle
and an arrow, choose a caption, and see P(wrong) from three validators (answer-only; rationale
targets; rationale targets + stage 2). Strokes are rendered by `render.render_human` and scored
by `validator.score` with no rationale generated. It needs `output/extracted` (stage 01) and the
adapters in `output/adapters/{2b-full4-ans,2b-full4,2b-full4-hum-s0}/` (`adapter_config.json` +
`adapter_model.safetensors` from the training runs). `demo/run.sh --selftest` loads the models
and scores one sketch. With less than 6 GB of free GPU memory it loads an 8-bit (or 4-bit) copy:
on a 4 GB RTX 3050 the 8-bit copy takes 3.1 GB and about 2.3 s per sketch for the three models, and
gives the same verdict as the bf16 runs on 98 / 95 / 100 % of 60 human test sketches (4-bit: 97 /
87 / 95 % of 150).

## What this repository does not reproduce

- **The paper's 8,522-prompt release** (Table II: 6,463 / 900 / 1,159, 31.8% valid). The
  defaults above give 35,158 rows; the subset the paper used is not recorded here, and it
  was drawn from the pre-fix generator.
- **Taxonomy count.** Table I marks C3 as instantiated, but the generator disables C3 (its
  captions name objects); the data holds 14 instantiated modes. See `failure_modes.md`.
- **The paper's adapters.** `training/` implements the Sec. VI-A recipe as written; the exact
  instruction text behind the reported numbers is not in this repository. The Hub adapters
  `shibuina/drawvla-verify-*` are older (rank 16, `verdict: right|wrong` targets, no staging)
  and are not the paper's models. Sec. VI-A gives no stage-2 warm-up; the default 30 steps
  cover most of the ≈41 stage-2 steps, so set `--warmup-steps` explicitly.
- **The API benchmark** (seven models, the 4-shot prompt, cached replies) and the ranking
  analysis (Spearman ρ, Williams' test) of Sec. VII.
- **The human studies**: the web apps live in `shibuina/user-study` (branches
  `prompt-to-sketch` and `sketch-to-prompt`); the in-loop analysis of Sec. IX (Tables VI–VIII,
  Wilson intervals, the one-retry replay) is not here.
