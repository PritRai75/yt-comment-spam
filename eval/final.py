"""The pre-registered analysis (eval/preregistration.md): every result for one split.

    python -m eval.final --split dev                 # rehearsal on dev
    python -m eval.final --split dev --check         # compare with the committed files
    python -m eval.final --split test --final        # the single test run (needs the open lock)

Writes results/final/<split>.json and results/final/<split>_predictions.csv (IDs,
labels, scores, predictions and tiers; no comment text).

For each label set (corrected is primary, decision 46; original beside it) and each
comment set (all scored comments; without the share-text family, decision 14; without
the comments the policy quotes, decision 42):
  - each system at its own spam cut-off (decision 48: rules >= 1 hit, logistic
    regression and the tree >= 0.5, the LLM >= 3 of 5 votes): counts, precision,
    recall and false-positive rate with Wilson intervals, F1 with a bootstrap
    interval, average precision;
  - H1 (LLM v2 vs v1) and H2 (LLM v2 vs logistic regression): exact McNemar on the
    paired predictions, Holm across the two (decision 49), and the F1 difference
    with a paired bootstrap interval.
On all scored comments: recall by subtype and by video, and the enforcement tiers
from eval/thresholds.json (share of spam auto-removed, auto-removal precision with
its Wilson interval, recall of remove + review, review queue per 1,000 comments).
For the LLMs: tokens and model time per 1,000 comments, from the vote caches.
"""
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from data.fetch import ROOT
from data.load import load_comments
from eval import lock, mcnemar, thresholds
from eval.labels import label_table, labels, scored
from eval.metrics import average_precision, binary_metrics, per_1000, predict, recall_by

SYSTEMS = list(thresholds.SYSTEMS)
CUTOFF = dict(thresholds.FLOORS)  # decision 48: each system's own cut-off (also the floors, decision 45)
LABEL_SETS = ["corrected", "original"]  # primary first (decision 46)
HYPOTHESES = {"H1": ("llm_v2", "llm_v1"), "H2": ("llm_v2", "logreg")}
ALPHA = 0.05
QUOTED = {  # comments the policy text quotes (decision 42)
    "dev": ["Shakira:294", "Shakira:151"],
    "test": ["LMFAO:349", "LMFAO:278", "LMFAO:360", "LMFAO:181"],
}
OUT = ROOT / "results" / "final"


def split_scores(split: str) -> pd.DataFrame:
    """Every system's score for each comment of `split` (all rows, scored or not)."""
    from models import llm, logreg, rules, tree
    df = load_comments()
    train, part = df[df.split == "train"], df[df.split == split]
    out = pd.DataFrame(index=pd.Index(part.row_id, name="row_id"))
    out["rules"] = [float(rules.rule_score(c)) for c in part.content]
    out["logreg"] = logreg.spam_proba(logreg.fit(train.content, train.label), part.content)
    out["tree"] = tree.spam_proba(tree.fit(train.content, train.label), part.content)
    rows = part.reset_index(drop=True)
    for v in ("v1", "v2"):
        out[f"llm_{v}"] = llm.scores(rows, llm.load_prompt(v)).score.reindex(out.index).to_numpy()
    return out


def llm_usage(split: str) -> dict:
    from models import llm
    n = int((load_comments().split == split).sum())
    out = {}
    for v in ("v1", "v2"):
        config = llm.make_config(llm.load_prompt(v))
        u = llm.usage(llm.Cache(llm.run_dir(config) / llm.cache_name(split)))
        out[f"llm_{v}"] = dict(u, comments=n, input_tokens_per_1000=per_1000(u["input_tokens"], n),
                               output_tokens_per_1000=per_1000(u["output_tokens"], n),
                               model_seconds_per_1000=per_1000(u["latency_s"], n))
    return out


def comment_sets(rows: pd.DataFrame, split: str) -> dict:
    missing = sorted(set(QUOTED[split]) - set(rows.index))
    if missing:
        raise ValueError(f"quoted comments not in the scored rows: {missing}")
    return {"all": rows, "no_share_text": rows[~rows.share_text], "no_quoted": rows.drop(index=QUOTED[split])}


def _records(df: pd.DataFrame) -> list:
    return df.to_dict(orient="records")


def hypotheses(y, pred) -> dict:
    out = {}
    for h, (a, b) in HYPOTHESES.items():
        out[h] = dict(a=a, b=b, **mcnemar.mcnemar(y, pred[a], pred[b]),
                      f1=mcnemar.f1_difference(y, pred[a], pred[b]))
    adj = mcnemar.holm([out[h]["p"] for h in HYPOTHESES])
    for h, p in zip(HYPOTHESES, adj):
        out[h]["p_holm"] = p
        out[h]["reject_at_alpha"] = bool(p < ALPHA)
    return out


def analyse(split: str, scores: pd.DataFrame = None) -> tuple:
    rows = scored(label_table(), split)
    scores = split_scores(split) if scores is None else scores
    s = scores.reindex(rows.index)
    if s.isna().any().any():
        raise ValueError("a scored comment has no score")
    th = json.loads(thresholds.PATH.read_text(encoding="utf-8"))["systems"]
    result = {"split": split, "comments_scored": len(rows), "primary_label_set": LABEL_SETS[0],
              "cutoffs": CUTOFF, "alpha": ALPHA, "results": {}}
    for which in LABEL_SETS:
        result["results"][which] = {}
        for name, part in comment_sets(rows, split).items():
            y = labels(part, which)
            sp = s.loc[part.index]
            pred = {k: predict(sp[k].to_numpy(), CUTOFF[k]) for k in SYSTEMS}
            block = {"comments": len(part), "spam": int(y.sum()), "systems": {}, "hypotheses": hypotheses(y, pred)}
            for k in SYSTEMS:
                entry = {"metrics": binary_metrics(y, pred[k]), "average_precision": average_precision(y, sp[k].to_numpy())}
                if name == "all":
                    entry["recall_by_subtype"] = _records(recall_by(y, pred[k], part.review.to_numpy()))
                    entry["recall_by_video"] = _records(recall_by(y, pred[k], part.video.to_numpy()))
                    t = th[k]
                    tier = thresholds.summary(sp[k].to_numpy(), y, t["auto_remove"], t["allow_below"])
                    tier["review_per_1000"] = per_1000(tier["review"]["n"], len(part))
                    tier["remove_per_1000"] = per_1000(tier["remove"]["n"], len(part))
                    entry["enforcement"] = tier
                block["systems"][k] = entry
            result["results"][which][name] = block
    result["llm_usage"] = llm_usage(split)
    preds = rows[["split", "video", "label_original", "label_corrected", "review", "share_text"]].copy()
    for k in SYSTEMS:
        preds[f"{k}_score"] = s[k]
        preds[f"{k}_pred"] = predict(s[k].to_numpy(), CUTOFF[k])
        preds[f"{k}_tier"] = thresholds.tiers(s[k].to_numpy(), th[k]["auto_remove"], th[k]["allow_below"])
    return _plain(result), preds.reset_index()


def _plain(x):
    """JSON-safe: numpy numbers to Python, NaN to None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    return x


def dumps(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=True) + "\n"


def preds_csv(preds: pd.DataFrame) -> str:
    return preds.to_csv(index=False, lineterminator="\n")


def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--final", action="store_true", help="the single test run (needs the open lock)")
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    args = ap.parse_args(argv)
    if args.split == "test":
        try:
            lock.require(args.final)
        except lock.TestLocked as e:
            print(e, file=sys.stderr)
            return 2
    result, preds = analyse(args.split)
    js, csv = OUT / f"{args.split}.json", OUT / f"{args.split}_predictions.csv"
    if args.check:
        bad = 0
        for path, ok in ((js, lambda: _same(json.loads(js.read_text(encoding="utf-8")), json.loads(dumps(result)))),
                         (csv, lambda: _same_csv(csv, preds))):
            rel = path.relative_to(ROOT).as_posix()
            same = path.exists() and ok()
            print(f"{'OK' if same else 'DIFFERS'}: {rel}")
            bad += not same
        return 1 if bad else 0
    OUT.mkdir(parents=True, exist_ok=True)
    js.write_text(dumps(result), encoding="utf-8", newline="\n")
    csv.write_text(preds_csv(preds), encoding="utf-8", newline="\n")
    primary = result["results"]["corrected"]["all"]
    for h, r in primary["hypotheses"].items():
        print(f"{h} {r['a']} vs {r['b']}: discordant {r['a_only_right']}/{r['b_only_right']}, p {r['p']:.4g}, "
              f"Holm {r['p_holm']:.4g}; F1 difference {r['f1']['difference']:+.3f} "
              f"({r['f1']['lo']:+.3f} to {r['f1']['hi']:+.3f})")
    print(f"wrote {js.relative_to(ROOT).as_posix()} and {csv.relative_to(ROOT).as_posix()}")
    return 0


def _same_csv(path, preds: pd.DataFrame) -> bool:
    old = pd.read_csv(path, keep_default_na=False)
    new = pd.read_csv(__import__("io").StringIO(preds_csv(preds)), keep_default_na=False)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return False
    for c in old.columns:
        if pd.api.types.is_float_dtype(old[c]) and pd.api.types.is_float_dtype(new[c]):
            if not np.allclose(old[c], new[c], rtol=1e-9, atol=1e-12):
                return False
        elif not (old[c].astype(str) == new[c].astype(str)).all():
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
