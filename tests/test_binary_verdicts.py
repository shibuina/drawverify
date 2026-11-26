import pytest

from libero_sketch import false_prompts as fp


def test_complete_prompt_verdict_is_binary():
    assert fp.verdict_of(fp._ok()) == "right"
    assert fp.verdict_of(fp._ok(sufficient_ok=False)) == "wrong"
    assert fp.verdict_of(fp._ok(circle_ok=False)) == "wrong"


def test_default_generation_is_deictic_only():
    assert fp.DEFAULT_LEVEL_MIX == {"L2": 0.5, "L3": 0.5}
    assert "category" not in fp.GROUNDING_KINDS
    assert not set(fp.DEFAULT_MODES).intersection(fp.NAME_EXPLICIT_MODES)
    assert "D3" in fp.DEFAULT_MODES


def test_record_schema_has_no_third_class_or_legacy_ambiguity_block():
    record = fp.Sample(
        circle=None,
        arrow=None,
        caption="do this",
        labels=fp._ok(sufficient_ok=False),
        severity=0.0,
        n_referents=2,
    ).as_record("POS")
    assert record["verdict"] == "wrong"
    assert record["labels"]["right"] is False
    assert "valid" not in record["labels"]
    assert "resolution" in record
    assert "ambiguity" not in record


@pytest.mark.parametrize(
    "caption",
    ["do this", "move this there", "put this over there", "open this", "this one"],
)
def test_name_free_deictic_grammar_accepts_supported_prompts(caption):
    assert fp._NAME_FREE_DEICTIC_RE.fullmatch(caption)


@pytest.mark.parametrize(
    "caption",
    ["move the black bowl there", "put this in the basket", "open the drawer"],
)
def test_name_free_deictic_grammar_rejects_explicit_nouns(caption):
    assert fp._NAME_FREE_DEICTIC_RE.fullmatch(caption) is None


@pytest.mark.parametrize("level", ["L0", "L1"])
def test_name_explicit_caption_levels_are_rejected(level):
    with pytest.raises(ValueError, match="Name-explicit caption levels are disabled"):
        fp._sample_level(__import__("numpy").random.default_rng(17), {level: 1.0})
