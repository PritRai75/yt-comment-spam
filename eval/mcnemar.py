"""Paired comparisons of two systems on the same comments (H1 and H2 of eval/preregistration.md).

  mcnemar      exact McNemar test on paired spam/ham predictions. A comment is an error
               for a system when its prediction differs from the label. Only the
               discordant comments count: b = A right and B wrong, c = A wrong and B
               right. Under the null both directions are equally likely, so
               p = min(1, 2 x P(X <= min(b, c))) with X ~ Binomial(b + c, 1/2), the
               two-sided exact test statsmodels gives with `mcnemar(..., exact=True)`.
               No discordant comment -> p = 1.
  holm         Holm's step-down adjustment of a family of p-values (decision 49).
  f1_difference  F1(A) - F1(B) with a seeded percentile bootstrap 95% interval:
               comments resampled with replacement, both systems scored on the same
               resample (paired), 2,000 draws, seed 20261125 (as decision 36).

The test answers "do the two systems make different errors on these comments?"; the F1
difference and its interval say how large the difference is. Neither covers variation
across new videos.
"""
import math

import numpy as np
from scipy.stats import binom

from eval.metrics import N_BOOT, SEED, _preds, f1_from


def discordant(y_true, pred_a, pred_b) -> dict:
    y, a = _preds(y_true, pred_a)
    _, b = _preds(y_true, pred_b)
    if len(a) != len(b):
        raise ValueError("the two systems must predict the same comments")
    ra, rb = a == y, b == y
    return {"n": len(y), "both_right": int((ra & rb).sum()), "a_only_right": int((ra & ~rb).sum()),
            "b_only_right": int((~ra & rb).sum()), "both_wrong": int((~ra & ~rb).sum()),
            "errors_a": int((~ra).sum()), "errors_b": int((~rb).sum())}


def exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def mcnemar(y_true, pred_a, pred_b) -> dict:
    d = discordant(y_true, pred_a, pred_b)
    d["p"] = exact_p(d["a_only_right"], d["b_only_right"])
    return d


def holm(pvalues) -> list:
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def _f1s(ys, ps):
    tp = ((ys == 1) & (ps == 1)).sum(axis=-1)
    fp = ((ys == 0) & (ps == 1)).sum(axis=-1)
    fn = ((ys == 1) & (ps == 0)).sum(axis=-1)
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)


def f1_difference(y_true, pred_a, pred_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    y, a = _preds(y_true, pred_a)
    _, b = _preds(y_true, pred_b)
    def f1(p):
        return f1_from(int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()),
                       int(((p == 0) & (y == 1)).sum()))
    out = {"f1_a": f1(a), "f1_b": f1(b)}
    out["difference"] = out["f1_a"] - out["f1_b"]
    if len(y) == 0:
        return dict(out, lo=math.nan, hi=math.nan)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    diff = _f1s(y[idx], a[idx]) - _f1s(y[idx], b[idx])
    lo, hi = np.nanpercentile(diff, [2.5, 97.5]) if not np.isnan(diff).all() else (math.nan, math.nan)
    return dict(out, lo=float(lo), hi=float(hi))
