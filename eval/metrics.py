"""Validator metrics in the paper's convention: INVALID is the positive class.

`y` is 1 for an invalid prompt (verdict `wrong`), `pred` is 1 for a rejection, and `s` is a
score that grows with the probability of invalid (P(wrong), or p_fail for API models). Prec
and Rec are on invalid prompts, R_val is recall on valid ones (paper Tables III-V).
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Mann-Whitney U with average ranks for ties; nan if only one class is present."""
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return math.nan
    ranked = sorted(zip(s, y))
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(label for _, label in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def metrics(y: Sequence[int], pred: Sequence[int], s: Sequence[float] | None = None) -> dict:
    if not y or len(y) != len(pred) or (s is not None and len(s) != len(y)):
        raise ValueError("metric inputs must be nonempty and of equal length")
    tp = sum(t and p for t, p in zip(y, pred))
    fn = sum(t and not p for t, p in zip(y, pred))
    fp = sum(p and not t for t, p in zip(y, pred))
    tn = len(y) - tp - fn - fp
    div = lambda a, b: a / b if b else math.nan  # noqa: E731
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    rec, r_val = div(tp, tp + fn), div(tn, tn + fp)
    return {"n": len(y), "invalid": tp + fn, "acc": (tp + tn) / len(y), "prec": div(tp, tp + fp),
            "rec": rec, "mcc": (tp * tn - fp * fn) / math.sqrt(den) if den else 0.0,
            "r_val": r_val, "bal_acc": (rec + r_val) / 2,
            "auroc": auroc(y, s if s is not None else pred)}


def bootstrap_delta_auroc(y, s1, s2, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC(s1) - AUROC(s2) on the same rows with a paired 95% percentile interval."""
    rng = random.Random(seed)
    idx = range(len(y))
    deltas = []
    for _ in range(n):
        b = [rng.choice(idx) for _ in idx]
        d = auroc([y[i] for i in b], [s1[i] for i in b]) - auroc([y[i] for i in b], [s2[i] for i in b])
        if not math.isnan(d):
            deltas.append(d)
    deltas.sort()
    return (auroc(y, s1) - auroc(y, s2),
            deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1])


def threshold_for_recall(y: Sequence[int], s: Sequence[float], recall: float) -> float:
    """Highest threshold t with recall(s >= t) >= `recall` on invalid rows (the live gate's cut)."""
    inv = sorted((v for v, t in zip(s, y) if t), reverse=True)
    if not inv:
        raise ValueError("calibration rows contain no invalid prompt")
    return inv[max(0, math.ceil(recall * len(inv)) - 1)]


if __name__ == "__main__":
    y, s = [1, 1, 0, 0, 1, 0], [0.9, 0.8, 0.7, 0.2, 0.4, 0.1]
    assert abs(auroc(y, s) - 8 / 9) < 1e-12
    assert auroc([1, 0], [0.5, 0.5]) == 0.5
    m = metrics(y, [v >= 0.5 for v in s], s)
    assert (m["prec"], m["rec"], m["r_val"]) == (2 / 3, 2 / 3, 2 / 3), m
    assert threshold_for_recall(y, s, 0.6) == 0.8 and threshold_for_recall(y, s, 1.0) == 0.4
    d, lo, hi = bootstrap_delta_auroc(y, s, [1 - v for v in s], n=200)
    assert lo <= d <= hi
    print("ok")
