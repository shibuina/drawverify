import pytest

from libero_sketch.bddl import (goal_from_predicates, parse_bddl_text,
                                strip_spatial_qualifiers)
from libero_sketch.captions import audit_llm_levels, build_captions, template_levels
from libero_sketch.llm import LLMClient, _extract_json

BDDL = """
(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:objects akita_black_bowl_1 akita_black_bowl_2 plate_1)
  (:goal (And (On akita_black_bowl_1 plate_1)))
)
"""


def test_parse_bddl_place():
    g = parse_bddl_text(BDDL)
    assert g.kind == "place"
    assert g.target_object == "akita_black_bowl_1"
    assert g.destination == "plate_1"


def test_goal_manipulate():
    g = goal_from_predicates([["open", "wooden_cabinet_1_top_region"]])
    assert g.kind == "manipulate" and g.destination is None


def test_goal_prefers_place():
    g = goal_from_predicates([["open", "drawer_1"], ["in", "bowl_1", "drawer_1"]])
    assert g.kind == "place" and g.target_object == "bowl_1"


def test_strip_qualifiers():
    out = strip_spatial_qualifiers("pick up the black bowl next to the plate and place it")
    assert "next to" not in out.lower()
    assert "bowl" in out


def test_template_levels_place():
    lv = template_levels("put the red mug on the left plate", "place")
    assert lv["L0"] == ["put the red mug on the left plate"]
    assert all("this" in s for s in lv["L2"])
    assert lv["L3"]


def test_template_levels_manipulate():
    lv = template_levels("open the top drawer of the cabinet", "manipulate")
    assert lv["L2"] == ["open this"]


def test_audit_drops_spatial_leaks():
    clean, dropped = audit_llm_levels(
        {"L1": ["pick up the bowl"], "L2": ["grab the left one", "grab this"], "L3": []})
    assert dropped == 1
    assert clean["L2"] == ["grab this"]


def test_build_captions_provider_none(tmp_path):
    client = LLMClient("none", cache_dir=tmp_path)
    caps = build_captions("pick up the milk on the left", "place", "milk_1", "basket_1",
                          ["milk_1", "milk_2"], client, n_paraphrases=3)
    assert caps["levels"]["L0"]["templates"]
    assert caps["levels"]["L1"]["templates"][0] == "pick up the milk"
    assert caps["levels"]["L2"]["llm"] == []


def test_llm_client_rejects_bad_provider(tmp_path):
    with pytest.raises(ValueError):
        LLMClient("claude", cache_dir=tmp_path)


def test_extract_json_fenced():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('noise {"b": [1,2]} trailing') == {"b": [1, 2]}
