"""The two enforcement thresholds of every system, chosen on dev and frozen before test.

Each comment lands in one of three tiers by its score:

  remove   score >= auto_remove     taken down without a person
  review   allow_below <= score < auto_remove   sent to a human reviewer
  allow    score < allow_below      left up

How the thresholds are chosen, on the dev split and the corrected labels (decision 46;
the same thresholds are then applied unchanged to the original labels):

  auto_remove (decisions 7 and 45). Rank dev comments by score. A band is every comment
    at or above one distinct score. Walk the bands from the top down; the chosen band is
    the lowest one such that it and every band above it have precision >= 98%. The
    threshold is never below the system's own spam cut-off (FLOORS: rules >= 1 hit,
    logistic regression and the tree >= 0.5, the LLM >= 3 of 5 votes), so a system never
    auto-removes a comment it judges not spam. No band qualifies -> no auto-remove (None).
  allow_below (decision 8). The highest threshold at which remove + review still catch
    >= 95% of dev spam; never above auto_remove (when auto-remove alone catches 95%, the
    review tier is empty).

Where a threshold sits: halfway between the lowest dev score in the tier above it and the
next dev score below, so no dev comment's score lies on it and rounding differences
between machines cannot move a dev comment across it; then raised to the floor if below
it. A tier reaching the lowest dev score uses that score itself.

Bands are whole distinct scores: comments with equal scores are never split, whatever
their order.

    python -m eval.thresholds           # write eval/thresholds.json
    python -m eval.thresholds --check   # compare with the file on disk
"""
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from data.fetch import ROOT
from data.load import load_comments
from eval.labels import label_table, labels, scored
from eval.metrics import wilson

PRECISION = 0.98  # decision 7
RECALL = 0.95  # decision 8
LABEL_SET = "corrected"  # decision 46
FLOORS = {"rules": 1.0, "logreg": 0.5, "tree": 0.5, "llm_v1": 3.0, "llm_v2": 3.0}  # decision 45
SYSTEMS = list(FLOORS)
PATH = ROOT / "eval" / "thresholds.json"


# The rule on plain arrays

def levels(score, y) -> pd.DataFrame:
    """One row per distinct score, highest first: comments and spam at that score, and the
    band at or above it (flagged, spam, precision, share of all spam)."""
    d = (pd.DataFrame({"score": np.asarray(score, dtype=float), "y": np.asarray(y, dtype=int)})
         .groupby("score").agg(n=("y", "size"), spam=("y", "sum")).iloc[::-1].reset_index())
    d["band_n"] = d.n.cumsum()
    d["band_spam"] = d.spam.cumsum()
    d["precision"] = d.band_spam / d.band_n
    total = int(d.spam.sum())
    d["recall"] = d.band_spam / total if total else np.nan
    return d


def _place(d: pd.DataFrame, i: int) -> float:
    """Threshold for the band ending at level i: halfway to the next lower score."""
    if i + 1 < len(d):
        return float((d.score[i] + d.score[i + 1]) / 2)
    return float(d.score[i])


def auto_remove(score, y, floor: float = -math.inf, precision: float = PRECISION):
    """Decision 7 with the floor of decision 45; None when no band qualifies."""
    d = levels(score, y)
    ok = (d.precision >= precision).to_numpy()
    stop = int(np.argmin(ok)) if not ok.all() else len(d)  # first band below the target
    if stop == 0 or d.score[0] < floor:
        return None
    return max(_place(d, stop - 1), float(floor))


def allow_below(score, y, auto=None, recall: float = RECALL) -> float:
    """Decision 8: highest threshold at which remove + review catch >= `recall` of spam."""
    d = levels(score, y)
    i = int(np.argmax((d.recall >= recall - 1e-12).to_numpy()))
    t = _place(d, i)
    return t if auto is None else min(t, auto)


def tiers(score, auto, allow) -> np.ndarray:
    """'remove', 'review' or 'allow' for each score."""
    s = np.asarray(score, dtype=float)
    remove = s >= auto if auto is not None else np.zeros(len(s), dtype=bool)
    return np.where(remove, "remove", np.where(s >= allow, "review", "allow"))


def summary(score, y, auto, allow) -> dict:
    """Counts per tier on one label set, with the auto-remove precision's Wilson interval."""
    t = tiers(score, auto, allow)
    y = np.asarray(y, dtype=int)
    out = {"comments": len(y), "spam": int(y.sum())}
    for name in ("remove", "review", "allow"):
        m = t == name
        out[name] = {"n": int(m.sum()), "spam": int(y[m].sum())}
    k, n = out["remove"]["spam"], out["remove"]["n"]
    lo, hi = wilson(k, n) if n else (math.nan, math.nan)
    out["remove"].update(precision=k / n if n else None, precision_lo=lo if n else None,
                         precision_hi=hi if n else None)
    caught = out["remove"]["spam"] + out["review"]["spam"]
    out["remove_recall"] = k / y.sum() if y.sum() else None
    out["remove_or_review_recall"] = caught / y.sum() if y.sum() else None
    return out


# The five systems on dev

def dev_scores() -> pd.DataFrame:
    """Every system's score for each dev comment (train-fitted baselines; LLM votes from
    the committed caches). Nothing outside dev is scored."""
    from models import llm, logreg, rules, tree
    df = load_comments()
    train, dev = df[df.split == "train"], df[df.split == "dev"]
    out = pd.DataFrame(index=pd.Index(dev.row_id, name="row_id"))
    out["rules"] = [float(rules.rule_score(c)) for c in dev.content]
    out["logreg"] = logreg.spam_proba(logreg.fit(train.content, train.label), dev.content)
    out["tree"] = tree.spam_proba(tree.fit(train.content, train.label), dev.content)
    rows = dev.reset_index(drop=True)
    for v in ("v1", "v2"):
        out[f"llm_{v}"] = llm.scores(rows, llm.load_prompt(v)).score.reindex(out.index).to_numpy()
    return out


def choose(scores: pd.DataFrame = None) -> dict:
    from models import llm
    scores = dev_scores() if scores is None else scores
    rows = scored(label_table(), "dev")
    s = scores.reindex(rows.index)
    if s.isna().any().any():
        raise ValueError("a dev comment has no score")
    y = labels(rows, LABEL_SET)
    result = {
        "split": "dev",
        "label_set": LABEL_SET,
        "precision_target": PRECISION,
        "recall_target": RECALL,
        "llm_configs": {f"llm_{v}": llm.make_config(llm.load_prompt(v)).name for v in ("v1", "v2")},
        "systems": {},
    }
    for name in SYSTEMS:
        x = s[name].to_numpy()
        a = auto_remove(x, y, FLOORS[name])
        b = allow_below(x, y, a)
        result["systems"][name] = {
            "floor": FLOORS[name],
            "auto_remove": a,
            "allow_below": b,
            "dev_corrected": summary(x, labels(rows, "corrected"), a, b),
            "dev_original": summary(x, labels(rows, "original"), a, b),
        }
    return result


def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def dumps(result: dict) -> str:
    return json.dumps(result, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the file on disk; write nothing")
    args = ap.parse_args(argv)
    result = choose()
    rel = PATH.relative_to(ROOT).as_posix()
    if args.check:
        if not PATH.exists():
            print(f"MISSING: {rel}")
            return 1
        same = _same(json.loads(PATH.read_text(encoding="utf-8")), json.loads(dumps(result)))
        print(f"{'OK' if same else 'DIFFERS'}: {rel}")
        return 0 if same else 1
    PATH.write_text(dumps(result), encoding="utf-8", newline="\n")
    for name, r in result["systems"].items():
        c = r["dev_corrected"]
        print(f"{name}: auto_remove {r['auto_remove']}, allow_below {r['allow_below']}; dev corrected "
              f"remove {c['remove']['n']} ({c['remove']['spam']} spam), review {c['review']['n']} "
              f"({c['review']['spam']}), allow {c['allow']['n']} ({c['allow']['spam']})")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
