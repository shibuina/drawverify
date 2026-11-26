"""Degraded-caption ladder (proposal section 2.4): L0 clean ... L3 fully deictic.

Templates are deterministic and always produced; an LLM (Gemini/OpenAI) optionally adds
paraphrase diversity. A regex audit drops LLM outputs that re-introduce disambiguating
spatial qualifiers at L2/L3 (that would defeat the manufactured ambiguity).
"""

from __future__ import annotations

import re

from .bddl import strip_spatial_qualifiers
from .llm import LLMClient

L3_POOL = ["move this there", "grab this", "put this over there", "do this",
           "take this there", "this one, over there"]

_VERB_RE = re.compile(
    r"^(pick up|put|place|move|push|open|close|turn on|turn off|grasp|stack|take)\b",
    re.IGNORECASE)

_MANIPULATE_VERBS = ("open", "close", "turn on", "turn off")
_PLACE_CLAUSE_RE = re.compile(r"\band (pick up|put|place|move|push|stack|take)\b", re.IGNORECASE)

_SPATIAL_AUDIT_RE = re.compile(
    r"\b(left|right|front|back|behind|middle|top|bottom|next to|near|between|closest)\b",
    re.IGNORECASE)


def template_levels(instruction: str, goal_kind: str) -> dict[str, list[str]]:
    """Deterministic degradations of one instruction."""
    instr = instruction.strip().rstrip(".")
    m = _VERB_RE.match(instr)
    verb = (m.group(1).lower() if m else "move")
    if goal_kind == "place" and verb in _MANIPULATE_VERBS:
        # "open the top drawer and put the bowl inside": the circled object is the bowl, so
        # its caption takes the place clause's verb, not "open this there".
        m = _PLACE_CLAUSE_RE.search(instr)
        verb = m.group(1).lower() if m else "put"

    l1 = strip_spatial_qualifiers(instr)
    if goal_kind == "manipulate":
        l2 = [f"{verb} this"]
        l3 = ["do this", "this one"]
    else:
        l2 = list(dict.fromkeys(
            [f"{verb} this" if verb in ("pick up", "grasp") else f"{verb} this there",
             "put this there"]))
        l3 = list(L3_POOL[:4])
    return {"L0": [instr], "L1": [l1] if l1 and l1.lower() != instr.lower() else [instr],
            "L2": l2, "L3": l3}


def llm_prompt(instruction: str, target: str, destination: str | None,
               inventory: list[str], n: int) -> str:
    dest_line = f'The destination is "{destination}".' if destination else \
        "There is no destination (the task manipulates the object in place)."
    return f"""You generate degraded robot-task instructions for a research dataset.
Original instruction: "{instruction}"
The manipulated object is "{target}". {dest_line}
Objects present in the scene: {', '.join(inventory) if inventory else 'unknown'}.

Produce natural casual paraphrases at three degradation levels:
- "L1": keep the object category but REMOVE any spatial or relational qualifiers
  (left/right/near/next-to etc.), so the instruction could match several similar objects.
- "L2": replace the object reference with a deictic like "this"/"this one" but keep the verb
  (e.g. "put this in the basket" -> "put this there").
- "L3": fully deictic, verb may be generic (e.g. "move this there", "grab this").

Rules: never mention colors, positions, or relations that would single out one object at
L2/L3. Keep each under 10 words. Return JSON: {{"L1": [{n} strings], "L2": [{n} strings], "L3": [{n} strings]}}"""


def audit_llm_levels(levels: dict) -> tuple[dict[str, list[str]], int]:
    """Drop L2/L3 strings containing disambiguating spatial words. Returns (clean, n_dropped)."""
    out, dropped = {}, 0
    for lv in ("L1", "L2", "L3"):
        items = [s.strip() for s in levels.get(lv, []) if isinstance(s, str) and s.strip()]
        if lv in ("L2", "L3"):
            keep = [s for s in items if not _SPATIAL_AUDIT_RE.search(s)]
            dropped += len(items) - len(keep)
            items = keep
        out[lv] = items
    return out, dropped


def build_captions(instruction: str, goal_kind: str, target: str, destination: str | None,
                   inventory: list[str], client: LLMClient, n_paraphrases: int) -> dict:
    tpl = template_levels(instruction, goal_kind)
    result = {lv: {"templates": v, "llm": []} for lv, v in tpl.items()}
    meta = {"provider": client.provider, "model": client.model, "llm_dropped": 0}
    if client.enabled and n_paraphrases > 0:
        resp = client.complete_json(
            llm_prompt(instruction, target, destination, inventory, n_paraphrases))
        if resp:
            clean, dropped = audit_llm_levels(resp)
            meta["llm_dropped"] = dropped
            for lv in ("L1", "L2", "L3"):
                result[lv]["llm"] = clean.get(lv, [])
    return {"levels": result, "meta": meta}
