"""DrawVerify validator: chat format, data loading, model loading and forced-choice scoring.

Paper Sec. VI-A. The user turn is the 512x512 marked image plus the instruction and the
name-free caption. The assistant turn is the rationale (the export's `label_reason`) followed
by `<answer>valid</answer>` or `<answer>wrong</answer>`; answer-only runs, and rows without a
rationale (test split, human sketches), train on the answer alone. Loss covers the assistant
turn only. At inference the rationale is generated greedily, the prefix up to `<answer>` is
forced, and the score is P(wrong) from the likelihoods of the two answers.
"""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image

ANSWER = {"right": "valid", "wrong": "wrong"}   # dataset verdict -> answer word
IMAGE_SIZE = 512
INSTRUCTION = (
    "The image shows a robot workspace with a drawn instruction: the circle marks the object "
    "to act on and the arrow marks where it should go. The caption names no object, so the "
    "drawing must say which object and where. Decide whether the complete instruction (circle, "
    "arrow and caption together) is valid: every part correct, all parts consistent, and "
    "exactly one action possible. Explain briefly, then answer <answer>valid</answer> or "
    "<answer>wrong</answer>."
)


def user_turn(caption: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": f"{INSTRUCTION}\n\nCaption: {caption}"}]}]


def target(verdict: str, reason: str | None = None) -> str:
    answer = f"<answer>{ANSWER[verdict]}</answer>"
    return f"{reason.strip()}\n{answer}" if reason else answer


def prompt_text(processor, caption: str) -> str:
    return processor.apply_chat_template(user_turn(caption), tokenize=False,
                                         add_generation_prompt=True)


def transcript(processor, caption: str, response: str) -> str:
    """Prompt + assistant turn, closed by the template's own end-of-turn tokens."""
    messages = user_turn(caption) + [
        {"role": "assistant", "content": [{"type": "text", "text": response}]}]
    text = processor.apply_chat_template(messages, tokenize=False)
    if not text.startswith(prompt_text(processor, caption)):
        raise ValueError("chat template no longer extends the generation prompt")
    return text


def load_image(value, size: int = IMAGE_SIZE) -> Image.Image:
    """HF Image column, raw {bytes, path} parquet struct, path, or PIL image -> RGB size x size."""
    if isinstance(value, dict):
        value = Image.open(io.BytesIO(value["bytes"])) if value.get("bytes") else value["path"]
    if not isinstance(value, Image.Image):
        value = Image.open(value)
    return value.convert("RGB").resize((size, size), Image.BICUBIC)


def load_split(data: str | Path, split: str):
    """Every config of one split from a stage-10 export dir (data/<config>/<split>-*.parquet)
    or a Hugging Face dataset id, as one `datasets.Dataset`."""
    import pyarrow.parquet as pq
    from datasets import load_dataset

    root = Path(data).expanduser()
    if not root.exists():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(str(data), repo_type="dataset",
                                      allow_patterns=[f"data/*/{split}-*.parquet"]))
    # zero-row shards (an empty family/split pair) break the parquet builder
    files = [str(f) for f in sorted(root.glob(f"**/{split}-*.parquet"))
             if pq.ParquetFile(f).metadata.num_rows]
    if not files:
        raise FileNotFoundError(f"no non-empty {split}-*.parquet under {root}")
    ds = load_dataset("parquet", data_files=files, split="train")
    missing = {"image", "caption", "verdict"}.difference(ds.column_names)
    if missing:
        raise ValueError(f"{root} lacks {sorted(missing)}; export with --images embed")
    return ds


def load_processor(model_name: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_name)


def load_model(model_name: str, adapter: str | Path | None = None, trainable: bool = False):
    from transformers import AutoModelForImageTextToText
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=dtype, device_map="auto" if cuda else None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    return model


class Collator:
    """Training batches whose labels cover the assistant turn only."""

    def __init__(self, processor, answer_only: bool = False):
        self.processor = processor
        self.answer_only = answer_only

    def __call__(self, rows: list[dict]) -> dict:
        self.processor.tokenizer.padding_side = "right"
        images = [load_image(r["image"]) for r in rows]
        prompts = [prompt_text(self.processor, r["caption"]) for r in rows]
        texts = [transcript(self.processor, r["caption"], target(
            r["verdict"], None if self.answer_only else r.get("label_reason"))) for r in rows]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        prompt_len = self.processor(text=prompts, images=images, padding=True,
                                    return_tensors="pt")["attention_mask"].sum(1)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_len.tolist()):
            labels[i, :n] = -100
        batch["labels"] = labels
        return batch


def _device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def score(model, processor, rows: list[dict], max_new_tokens: int = 96) -> list[dict]:
    """Greedy rationale, then P(wrong) from log P(">valid</answer>") vs log P(">wrong</answer>")
    after the forced `...<answer` prefix. Both answers are scored as whole continuations from
    the first token where they differ, because BPE merges `>w` but not `>v`."""
    tok, dev = processor.tokenizer, _device(model)
    images = [load_image(r["image"]) for r in rows]
    prompts = [prompt_text(processor, r["caption"]) for r in rows]

    texts = [""] * len(rows)              # max_new_tokens=0: answer with no rationale
    if max_new_tokens:
        tok.padding_side = "left"
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(dev)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    reasons = [t.split("<answer>")[0].strip() for t in texts]

    tok.padding_side = "right"
    forced = [p + (r + "\n" if r else "") + f"<answer>{w}</answer>"
              for p, r in zip(prompts, reasons) for w in ANSWER.values()]
    batch = processor(text=forced, images=[im for im in images for _ in ANSWER],
                      padding=True, return_tensors="pt").to(dev)
    logits = model(**batch).logits
    lengths = batch["attention_mask"].sum(1).tolist()
    ids = batch["input_ids"]
    results = []
    for i, reason in enumerate(reasons):
        v, w = 2 * i, 2 * i + 1
        a, b = ids[v, :lengths[v]], ids[w, :lengths[w]]
        k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
        ll = [torch.log_softmax(logits[j, k - 1:n - 1].float(), -1)
              .gather(-1, ids[j, k:n, None]).sum().item()
              for j, n in ((v, lengths[v]), (w, lengths[w]))]
        p_wrong = torch.sigmoid(torch.tensor(ll[1] - ll[0])).item()
        results.append({"reasoning": reason, "generation": texts[i], "p_wrong": p_wrong})
    return results
