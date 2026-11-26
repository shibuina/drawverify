"""BDDL goal parsing: which object is manipulated, where it should end up.

Primary path uses LIBERO's own parser when importable; a minimal s-expression
fallback parser keeps this module usable in sim-free environments (tests, CI).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

PLACE_PREDICATES = {"on", "in", "inside", "ontop", "up"}
MANIP_PREDICATES = {"open", "close", "turnon", "turnoff"}


@dataclass
class Goal:
    predicate: str
    target_object: str
    destination: str | None  # region/container name, None for open/close/turn*
    kind: str                # "place" | "manipulate"
    raw_goal: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _sexp(text: str):
    """Tiny s-expression reader (enough for BDDL)."""
    toks = re.findall(r"\(|\)|[^\s()]+", text)
    def read(i):
        out = []
        while i < len(toks):
            t = toks[i]
            if t == "(":
                node, i = read(i + 1)
                out.append(node)
            elif t == ")":
                return out, i + 1
            else:
                out.append(t.lower())
                i += 1
        return out, i
    tree, _ = read(0)
    return tree


def _find_section(tree, name: str):
    for node in tree if isinstance(tree, list) else []:
        if isinstance(node, list):
            if node and node[0] == name:
                return node
            got = _find_section(node, name)
            if got is not None:
                return got
    return None


def parse_bddl_text(text: str) -> Goal:
    goal_sec = _find_section(_sexp(text), ":goal")
    if goal_sec is None:
        raise ValueError("BDDL has no (:goal ...) section")
    preds = []
    def collect(node):
        if isinstance(node, list) and node and isinstance(node[0], str):
            if node[0] in ("and", "or", ":goal"):
                for ch in node[1:]:
                    collect(ch)
            else:
                preds.append([str(x) for x in node])
    collect(goal_sec)
    return goal_from_predicates(preds)


def goal_from_predicates(preds: list[list[str]]) -> Goal:
    """Choose the goal predicate of record: prefer place-type over manipulate-type."""
    place = [p for p in preds if p[0].lower() in PLACE_PREDICATES and len(p) >= 3]
    manip = [p for p in preds if p[0].lower() in MANIP_PREDICATES and len(p) >= 2]
    if place:
        p = place[0]
        return Goal(p[0].lower(), p[1], p[2], "place", preds)
    if manip:
        p = manip[0]
        return Goal(p[0].lower(), p[1], None, "manipulate", preds)
    if preds:  # unknown predicate family: keep it, treat as manipulate
        p = preds[0]
        return Goal(p[0].lower(), p[1] if len(p) > 1 else "", None, "manipulate", preds)
    raise ValueError("no goal predicates found")


def parse_bddl_file(path: str) -> Goal:
    try:
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        parsed = robosuite_parse_problem(path)
        preds = [[str(x) for x in p] for p in parsed["goal_state"]]
        return goal_from_predicates(preds)
    except Exception:
        with open(path) as f:
            return parse_bddl_text(f.read())


_QUALIFIER_RE = re.compile(
    r"\b(on the (left|right|top|bottom|middle|front|back)( side)?( of[\w\s]+?)?|"
    r"(left|right|front|back|middle|top|bottom)\s+|"
    r"next to the [\w\s]+?|near the [\w\s]+?|between the [\w\s]+? and the [\w\s]+?|"
    r"closest to [\w\s]+?|behind the [\w\s]+?|in front of the [\w\s]+?)\b",
    re.IGNORECASE)


# The destination clause ("... and place it on the plate"). Everything before it is the
# target clause; only there is a trailing prepositional phrase a *disambiguator*. In
# "put the bowl on the stove" the same PP is the destination and must survive, which is
# why the target-side strip only runs when a destination clause is present.
_DEST_CLAUSE_RE = re.compile(
    r"\s+and\s+(?:place|put|drop|move|set|stack)\s+(?:it|them)\b", re.IGNORECASE)

# A trailing PP on the target clause: "... from table center", "... on the stove",
# "... in the top drawer of the wooden cabinet", "... between the plate and the ramekin".
_TARGET_PP_RE = re.compile(
    r"\s+(?:on|in|from|under|at|beside|near|next\s+to|behind|in\s+front\s+of|"
    r"between|closest\s+to)\s+.*$", re.IGNORECASE)


# A dangling preposition pair ("push the plate to the *of the* stove") means the strip cut
# out a word the sentence still needed — reject rather than emit a mangled caption.
_DAMAGED_RE = re.compile(
    r"\b(?:to|on|in|at|of|from|with)\s+the\s+(?:of|to|on|in|at|from)\b|"
    r"\b(?:to|on|in|at|of|from)\s*$", re.IGNORECASE)


def _tidy(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s).strip(" ,.").strip()


def strip_spatial_qualifiers(instruction: str) -> str:
    """Remove the disambiguating spatial phrase from an instruction (caption level L1).

    L1's whole purpose is to manufacture ambiguity by deleting the phrase that separates
    the target from an identical twin, so a miss here silently yields an L1 equal to L0 —
    an undegraded caption wearing a degraded label.

    The clause-based strip runs *first*: it removes the target's entire trailing PP, where
    `_QUALIFIER_RE` alone either missed it ("from table center", "on the stove" — half the
    spatial suite) or bit off only an adjective, leaving "in the drawer of the wooden
    cabinet" still disambiguating. The regex is the fallback for instructions with no
    destination clause, and its output is rejected if it mangled the sentence.
    """
    m = _DEST_CLAUSE_RE.search(instruction)
    if m:
        head, rest = instruction[:m.start()], instruction[m.start():]
        stripped = _TARGET_PP_RE.sub("", head).strip(" ,.")
        # must leave a verb + object behind, else we ate the referent
        if stripped.lower() != head.lower() and len(stripped.split()) >= 3:
            return _tidy(stripped + rest)

    out = _tidy(_QUALIFIER_RE.sub("", instruction))
    return instruction if (not out or _DAMAGED_RE.search(out)) else out
