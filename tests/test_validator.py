import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "training"))
validator = pytest.importorskip("validator")


def test_target_format():
    assert validator.target("right") == "<answer>valid</answer>"
    assert validator.target("wrong", " the arrow misses ") == "the arrow misses\n<answer>wrong</answer>"


def test_loss_covers_the_assistant_turn_only():
    Image = pytest.importorskip("PIL.Image")
    try:
        processor = validator.load_processor("Qwen/Qwen3-VL-2B-Instruct")
    except Exception:
        pytest.skip("Qwen3-VL processor not available offline")
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buf, format="PNG")
    rows = [{"image": {"bytes": buf.getvalue()}, "caption": "move this there", "verdict": v,
             "label_reason": r} for v, r in (("right", "all channels agree"), ("wrong", None))]
    batch = validator.Collator(processor)(rows)
    for row, ids, labels in zip(rows, batch["input_ids"], batch["labels"]):
        supervised = processor.tokenizer.decode(ids[labels != -100])
        assert supervised.startswith(validator.target(row["verdict"], row["label_reason"]))
        assert supervised.rstrip().endswith("<|im_end|>")
