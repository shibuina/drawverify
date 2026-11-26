import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("drawvla_export_hf", SCRIPTS / "10_export_hf.py")
export_hf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_hf)


def record(*, verdict="wrong", labels=None, readings=2):
    labels = labels or {
        "right": verdict == "right",
        "circle_ok": True,
        "arrow_ok": True,
        "caption_ok": True,
        "consistency_ok": True,
        "sufficient_ok": verdict == "right",
    }
    return {
        "id": "spatial/task/demo_0/POS#0",
        "suite": "spatial",
        "task": "task",
        "demo": "demo_0",
        "mode": "POS",
        "slot": "no_sketch",
        "category": "G" if verdict == "wrong" else "P",
        "category_name": "grounding" if verdict == "wrong" else "positive",
        "tier": "hard",
        "solution": "vlm",
        "data_amount": "large",
        "verdict": verdict,
        "labels": labels,
        "severity": 0.0,
        "prompt": {"caption": "do this", "circle": None, "arrow": None},
        "resolution": {
            "caption_level": "L3",
            "n_referents": readings,
            "n_destinations": 1,
            "n_readings": readings,
            "resolved_by": "deixis",
            "kind": "no_sketch",
        },
        "rendered_resolution": {
            "n_referents": readings,
            "n_destinations": 1,
            "n_readings": readings,
            "verified": True,
        },
        "note": "no sketch to ground the caption",
        "source_frame": "frame.png",
        "name_explicit": False,
    }


def test_split_is_deterministic_and_includes_all_three_partitions():
    demos = [f"suite/task/demo_{index}" for index in range(200)]
    first = [export_hf.split_of(demo) for demo in demos]
    second = [export_hf.split_of(demo) for demo in demos]
    assert first == second
    assert set(first) == {"train", "validation", "test"}


def test_train_and_validation_have_human_readable_label_reasons():
    for split in ("train", "validation"):
        row = export_hf._row(record(), Path("."), False, split=split)
        assert row["reason_available"] is True
        assert row["label_reason"].startswith("wrong because ")
        assert "2 possible readings" in row["label_reason"]
        assert row["note"]
        assert row["category"] == "G"


def test_test_split_redacts_all_rationale_fields():
    row = export_hf._row(record(), Path("."), False, split="test")
    assert row["reason_available"] is False
    assert row["verdict"] == "wrong"  # retained only as the scoring target
    redacted = {
        "label_reason",
        "note",
        "mode",
        "slot",
        "category",
        "category_name",
        "tier",
        "solution",
        "data_amount",
        "circle_ok",
        "arrow_ok",
        "caption_ok",
        "consistency_ok",
        "sufficient_ok",
        "severity",
        "n_referents",
        "n_destinations",
        "n_readings",
        "resolved_by",
        "grounding_kind",
        "rendered_n_referents",
        "rendered_n_destinations",
        "rendered_n_readings",
        "render_verified",
    }
    assert all(row[field] is None for field in redacted)


def test_positive_reason_explains_unique_consistent_reading():
    row = export_hf._row(
        record(verdict="right", readings=1), Path("."), False, split="train"
    )
    assert row["label_reason"].startswith("right because ")
    assert "exactly one actionable reading" in row["label_reason"]
