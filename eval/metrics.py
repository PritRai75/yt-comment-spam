"""Measures for every system. Spam is the positive class.

At a threshold (spam when score >= threshold):
  precision, recall and false-positive rate, each with a Wilson 95% interval
    (precision k = true positives of n = flagged; recall k = true positives of n = spam;
    false-positive rate k = false positives of n = ham);
  F1, with a seeded percentile bootstrap 95% interval (comments resampled with
    replacement). F1 is not a single proportion, so no Wilson interval exists for it
    (decision 36). Every interval here describes one system and is not a test between
    systems (those are the paired McNemar tests); each covers resampling comments
    within the scored videos, not variation across new videos.
Over all thresholds:
  the precision-recall curve, one point per distinct score, and its area as average
    precision: the sum over points of (gain in recall) x (precision there), the
    step-wise area scikit-learn's `average_precision_score` gives (no interpolation,
    which would overstate the area).
By group:
  recall by subtype (from `eval/labels.py`) and by video, each with a Wilson interval.
Running cost:
  cost and time per 1,000 comments.

An undefined value (precision with nothing flagged, a rate over an empty group) is NaN,
never 0. The functions take plain arrays; `evaluate` applies the label set, the
splits, the scoring exclusions and the share-text option from `eval/labels.py`.
"""
import math

import numpy as np
import pandas as pd

from eval.labels import labels, scored

Z95 = 1.959963984540054  # two-sided 95% normal quantile
SEED = 20261125
N_BOOT = 2000


def wilson(k: int, n: int, z: float = Z95) -> tuple:
    """Wilson score interval for k successes in n trials; (nan, nan) when n is 0."""
    if n == 0:
        return (math.nan, math.nan)
    if not 0 <= k <= n:
        raise ValueError(f"k = {k} outside 0..{n}")
    p = k / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(k, n):
    return k / n if n else math.nan


def _check(y_true, other, name):
    y = np.asarray(y_true, dtype=int)
    o = np.asarray(other)
    if y.shape != o.shape or y.ndim != 1:
        raise ValueError(f"labels and {name} must be 1-d and the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y, o


def _preds(y_true, y_pred):
    y, p = _check(y_true, y_pred, "predictions")
    if not np.isin(p, (0, 1)).all():
        raise ValueError("predictions must be 0 or 1")
    return y, p.astype(int)


def counts(y_true, y_pred) -> dict:
    y, p = _preds(y_true, y_pred)
    return {
        "tp": int(((p == 1) & (y == 1)).sum()),
        "fp": int(((p == 1) & (y == 0)).sum()),
        "fn": int(((p == 0) & (y == 1)).sum()),
        "tn": int(((p == 0) & (y == 0)).sum()),
    }


def f1_from(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else math.nan


def f1_interval(y_true, y_pred, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """Percentile bootstrap 95% interval for F1, resampling comments with replacement."""
    y, p = _preds(y_true, y_pred)
    n = len(y)
    if n == 0:
        return (math.nan, math.nan)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    ys, ps = y[idx], p[idx]
    tp = ((ys == 1) & (ps == 1)).sum(axis=1)
    fp = ((ys == 0) & (ps == 1)).sum(axis=1)
    fn = ((ys == 1) & (ps == 0)).sum(axis=1)
    d = 2 * tp + fp + fn
    f1 = np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
    if np.isnan(f1).all():
        return (math.nan, math.nan)
    lo, hi = np.nanpercentile(f1, [2.5, 97.5])
    return (float(lo), float(hi))


def binary_metrics(y_true, y_pred, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    c = counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    out = dict(c, n=tp + fp + fn + tn, spam=tp + fn, ham=fp + tn, flagged=tp + fp)
    for name, k, n in (("precision", tp, tp + fp), ("recall", tp, tp + fn), ("fpr", fp, fp + tn)):
        lo, hi = wilson(k, n)
        out.update({name: _ratio(k, n), f"{name}_lo": lo, f"{name}_hi": hi})
    out["f1"] = f1_from(tp, fp, fn)
    out["f1_lo"], out["f1_hi"] = f1_interval(y_true, y_pred, n_boot, seed)
    return out


def predict(score, threshold: float) -> np.ndarray:
    """Spam (1) when the score is at least the threshold."""
    return (np.asarray(score, dtype=float) >= threshold).astype(int)


def pr_curve(y_true, score) -> pd.DataFrame:
    """One row per distinct score, highest first: flagging every comment scored at or
    above `threshold` gives these counts, precision and recall."""
    y, s = _check(y_true, score, "scores")
    s = s.astype(float)
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # last index of each tie group
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    total = int(y.sum())
    return pd.DataFrame({
        "threshold": s[last],
        "tp": tp,
        "fp": fp,
        "precision": tp / (tp + fp),
        "recall": tp / total if total else np.full(len(last), np.nan),
    })


def average_precision(y_true, score) -> float:
    """Area under the precision-recall curve, step-wise: sum of recall gain x precision."""
    curve = pr_curve(y_true, score)
    if curve.recall.isna().any():
        return math.nan
    gain = np.diff(np.r_[0.0, curve.recall.to_numpy()])
    return float((gain * curve.precision.to_numpy()).sum())


def recall_by(y_true, y_pred, groups) -> pd.DataFrame:
    """Recall within each group, over the spam comments only, with a Wilson interval."""
    y, p = _preds(y_true, y_pred)
    g = np.asarray(groups, dtype=object)
    if g.shape != y.shape:
        raise ValueError("groups must be the same length as the labels")
    rows = []
    for name in sorted(set(g[y == 1])):
        m = (y == 1) & (g == name)
        k, n = int((p[m] == 1).sum()), int(m.sum())
        lo, hi = wilson(k, n)
        rows.append({"group": name, "spam": n, "caught": k, "recall": _ratio(k, n), "recall_lo": lo, "recall_hi": hi})
    return pd.DataFrame(rows, columns=["group", "spam", "caught", "recall", "recall_lo", "recall_hi"])


def per_1000(total: float, n_comments: int) -> float:
    if n_comments <= 0:
        raise ValueError("n_comments must be positive")
    return total * 1000 / n_comments


def cost_per_1000(n_comments: int, input_tokens: int, output_tokens: int,
                  usd_per_mtok_input: float, usd_per_mtok_output: float) -> float:
    """US dollars per 1,000 comments, from the tokens of every call made for them
    (all five votes for the LLM systems) and the per-million-token prices."""
    usd = input_tokens / 1e6 * usd_per_mtok_input + output_tokens / 1e6 * usd_per_mtok_output
    return per_1000(usd, n_comments)


def time_per_1000(n_comments: int, seconds: float) -> float:
    """Wall-clock seconds per 1,000 comments."""
    return per_1000(seconds, n_comments)


def evaluate(table: pd.DataFrame, score: pd.Series, threshold: float, *, which: str = "corrected",
             splits=("dev",), drop_share_text: bool = False, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Every measure for one system on one label set.

    `table` is `eval.labels.label_table()`; `score` is the system's score by row_id and
    must cover every scored row."""
    rows = scored(table, splits, drop_share_text)
    missing = rows.index.difference(score.index)
    if len(missing):
        raise ValueError(f"no score for {len(missing)} scored rows, e.g. {list(missing[:3])}")
    s = score.reindex(rows.index).to_numpy(dtype=float)
    y = labels(rows, which)
    p = predict(s, threshold)
    return {
        "which": which,
        "splits": list(rows.split.unique()),
        "drop_share_text": drop_share_text,
        "threshold": threshold,
        "metrics": binary_metrics(y, p, n_boot, seed),
        "average_precision": average_precision(y, s),
        "recall_by_subtype": recall_by(y, p, rows.review.to_numpy()),
        "recall_by_video": recall_by(y, p, rows.video.to_numpy()),
    }
