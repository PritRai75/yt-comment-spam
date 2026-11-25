"""Gradient-boosting baseline on readable features (spec addendum 1, decision 29).

scikit-learn's HistGradientBoostingClassifier on the 27 features of `models/features.py`,
trained on the train split with the dataset's labels, as the other two baselines are.
The score is the predicted probability of spam; the label is spam when it is at least 0.5.

Settings were chosen on train only, before the tree was scored on dev or test:
  - a grid of 12 (learning rate 0.05 or 0.1; at most 7, 15 or 31 leaves per tree;
    100 or 300 trees), every other setting at its default, no early stopping;
  - each scored by 5-fold stratified cross-validation on train (shuffled, seed
    20261125) by average precision; the highest mean wins, ties going to fewer
    leaves, then fewer trees, then the lower learning rate;
  - leave-one-video-out within train (fit on Psy, score KatyPerry, and the reverse)
    is reported beside it as the cross-video check; it does not choose.
The choice is frozen in CHOSEN; a test re-runs the grid and checks it still wins.

Train comments are scored out of fold, one video at a time, with the frozen settings
(as in `models/logreg.py`); dev and test by the model fitted on all of train.

Feature importance is permutation importance on dev: each feature's column is shuffled
30 times (seed 20261125) and the drop in average precision recorded, mean and
standard deviation. It needs no impurity counts, so it does not favour features with
many distinct values. Features that carry the same signal (a web address counts
towards `n_urls` and `rule_link`) share it, so each one alone can look less important
than the pair. The importances are descriptive and change nothing in the model.

    python -m models.tree cv            # write results/tree_cv.csv
    python -m models.tree importance    # write results/tree_importance.csv
    python -m models.tree cv --check    # compare with the files on disk
"""
import argparse
import io
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from data.fetch import ROOT
from data.load import load_comments
from eval.labels import label_table, labels, scored
from eval.metrics import average_precision
from models.features import FEATURES, feature_frame

SEED = 20261125
GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": trees}
    for lr, leaves, trees in itertools.product([0.05, 0.1], [7, 15, 31], [100, 300])
]
CHOSEN = {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 300}  # from `python -m models.tree cv` (train only)
N_REPEATS = 30
RESULTS = ROOT / "results"


def make_model(params: dict = CHOSEN) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(early_stopping=False, random_state=0, **params)


def fit(texts, labels, params: dict = CHOSEN):
    model = make_model(params)
    model.fit(feature_frame(texts), np.asarray(labels, dtype=int))
    return model


def spam_proba(model, texts) -> np.ndarray:
    return model.predict_proba(feature_frame(texts))[:, 1]


def score_all(df: pd.DataFrame, params: dict = CHOSEN) -> pd.Series:
    """Spam probability for every row: out of fold by video on train, full-train model elsewhere."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    train = df[df.split == "train"]
    for video in sorted(train.video.unique()):
        fit_on = train[train.video != video]
        score_on = train[train.video == video]
        out.loc[score_on.index] = spam_proba(fit(fit_on.content, fit_on.label, params), score_on.content)
    full = fit(train.content, train.label, params)
    rest = df[df.split != "train"]
    out.loc[rest.index] = spam_proba(full, rest.content)
    return out


def _key(p: dict) -> str:
    return f"lr={p['learning_rate']} leaves={p['max_leaf_nodes']} trees={p['max_iter']}"


def cv_table(df: pd.DataFrame = None) -> pd.DataFrame:
    """The grid on train: 5-fold CV average precision (mean, sd, each fold) and the
    leave-one-video-out average precision, one row per setting, best first."""
    df = load_comments() if df is None else df
    train = df[df.split == "train"].reset_index(drop=True)
    X = feature_frame(train.content)
    y = train.label.to_numpy(dtype=int)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    rows = []
    for p in GRID:
        fold_ap = []
        for fit_idx, score_idx in folds:
            m = make_model(p).fit(X.iloc[fit_idx], y[fit_idx])
            fold_ap.append(average_precision(y[score_idx], m.predict_proba(X.iloc[score_idx])[:, 1]))
        lovo = {}
        for video in sorted(train.video.unique()):
            on, off = (train.video == video).to_numpy(), (train.video != video).to_numpy()
            m = make_model(p).fit(X[off], y[off])
            lovo[video] = average_precision(y[on], m.predict_proba(X[on])[:, 1])
        row = dict(p, setting=_key(p), cv_ap_mean=float(np.mean(fold_ap)), cv_ap_sd=float(np.std(fold_ap, ddof=1)))
        row.update({f"cv_ap_fold{i + 1}": a for i, a in enumerate(fold_ap)})
        row.update({f"lovo_ap_{v}": a for v, a in lovo.items()})
        row["lovo_ap_mean"] = float(np.mean(list(lovo.values())))
        rows.append(row)
    t = pd.DataFrame(rows)
    t = t.sort_values(["cv_ap_mean", "max_leaf_nodes", "max_iter", "learning_rate"],
                      ascending=[False, True, True, True], kind="mergesort").reset_index(drop=True)
    t.insert(0, "rank", range(1, len(t) + 1))
    t["chosen"] = t.setting == _key(CHOSEN)
    return t


def best(table: pd.DataFrame) -> dict:
    r = table.iloc[0]
    return {"learning_rate": float(r.learning_rate), "max_leaf_nodes": int(r.max_leaf_nodes), "max_iter": int(r.max_iter)}


def importance_table(which: str = "corrected", df: pd.DataFrame = None) -> pd.DataFrame:
    """Permutation importance on dev (scoring exclusions applied), by average precision."""
    df = load_comments() if df is None else df
    train = df[df.split == "train"]
    model = fit(train.content, train.label)
    dev = scored(label_table(), "dev")
    X = feature_frame(df.set_index("row_id").loc[dev.index, "content"])
    y = labels(dev, which)
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=N_REPEATS, random_state=SEED)
    t = pd.DataFrame({"feature": FEATURES, "ap_drop_mean": r.importances_mean, "ap_drop_sd": r.importances_std})
    t = t.sort_values(["ap_drop_mean", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    t.insert(0, "rank", range(1, len(t) + 1))
    t.insert(1, "labels", which)
    return t


def _write(t: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    t.to_csv(RESULTS / name, index=False, lineterminator="\n", float_format="%.6f")


def _check(t: pd.DataFrame, name: str, atol: float = 2e-6) -> int:
    path = RESULTS / name
    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT).as_posix()}")
        return 1
    old = pd.read_csv(path)
    fresh = pd.read_csv(io.StringIO(t.to_csv(index=False, float_format="%.6f")))
    same = list(old.columns) == list(fresh.columns) and len(old) == len(fresh)
    if same:
        for c in old.columns:
            if pd.api.types.is_float_dtype(old[c]) and pd.api.types.is_float_dtype(fresh[c]):
                same &= bool(np.allclose(old[c], fresh[c], atol=atol, rtol=0))
            else:
                same &= bool((old[c].astype(str) == fresh[c].astype(str)).all())
    print(f"{'OK' if same else 'DIFFERS'}: {path.relative_to(ROOT).as_posix()}")
    return 0 if same else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["cv", "importance"])
    ap.add_argument("--check", action="store_true", help="compare with the file on disk; write nothing")
    args = ap.parse_args(argv)
    if args.what == "cv":
        t, name = cv_table(), "tree_cv.csv"
        if best(t) != CHOSEN:
            print(f"the grid now picks {best(t)}, not the frozen {CHOSEN}")
            return 1
    else:
        t, name = pd.concat([importance_table("corrected"), importance_table("original")], ignore_index=True), "tree_importance.csv"
    if args.check:
        return _check(t, name)
    _write(t, name)
    print(f"wrote {(RESULTS / name).relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
