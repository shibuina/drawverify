#!/usr/bin/env python
"""Fine-tune a Qwen VLM validator with LoRA (paper Sec. VI-A).

Defaults are the paper's stage 1: rank 32, alpha 64, dropout 0.05 on q/k/v/o and the
gate/up/down MLP projections; 2 epochs, effective batch 16, lr 1e-4 cosine, 30 warm-up
steps, AdamW without weight decay, gradient clipping 1.0, 512x512 images.

  # stage 1: generated prompts
  python training/train.py --data shibuina/drawvla-prompt-validation --output-dir output/train/gen
  # stage 2: continue the same adapter on the human training sketches (scripts/12_render_human.py)
  python training/train.py --data output/human_train --init-adapter output/train/gen \\
      --epochs 4 --batch-size 8 --grad-accum 1 --lr 3e-5 --output-dir output/train/gen-hum
  # answer-only variant: add --answer-only to both stages
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from validator import Collator, load_model, load_processor, load_split

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML config file (CLI flags override config values)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--data", help="stage-10 export dir or HF dataset id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--init-adapter", type=Path, help="continue this adapter (stage 2)")
    ap.add_argument("--answer-only", action="store_true", help="drop the rationale from targets")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample-frac", type=float, help="shuffled fraction of training rows in (0, 1] for sample-size sensitivity")
    ap.add_argument("--limit", type=int, help="first N rows only (smoke tests)")
    pre, _ = ap.parse_known_args()
    if pre.config:
        import yaml
        cfg = yaml.safe_load(pre.config.read_text()) or {}
        for k in ("output_dir", "init_adapter"):
            if cfg.get(k) is not None:
                cfg[k] = Path(cfg[k])
        ap.set_defaults(**cfg)
    a = ap.parse_args()
    if not a.data or not a.output_dir:
        ap.error("--data and --output-dir are required (via CLI or --config)")

    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(a.seed)
    rows = load_split(a.data, a.split)
    if a.subsample_frac is not None:
        if not (0.0 < a.subsample_frac <= 1.0):
            ap.error("--subsample-frac must be in (0, 1]")
        if a.subsample_frac < 1.0:
            n_keep = max(1, int(round(len(rows) * a.subsample_frac)))
            rows = rows.shuffle(seed=a.seed).select(range(n_keep))
    if a.limit:
        rows = rows.select(range(min(a.limit, len(rows))))
    processor = load_processor(a.model)
    model = load_model(a.model, a.init_adapter, trainable=True)
    if a.init_adapter is None:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(a.output_dir), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=a.warmup_steps,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch", bf16=bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="epoch", save_total_limit=1, report_to="none",
        remove_unused_columns=False, seed=a.seed)
    # a rerun into the same output dir continues from its last epoch checkpoint (GPU claims expire)
    Trainer(model=model, args=args, train_dataset=rows,
            data_collator=Collator(processor, a.answer_only)).train(
        resume_from_checkpoint=any(a.output_dir.glob("checkpoint-*")) or None)
    model.save_pretrained(a.output_dir)
    processor.save_pretrained(a.output_dir)
    (a.output_dir / "run_config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2) + "\n")
    print(f"adapter -> {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
