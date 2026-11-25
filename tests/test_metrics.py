import math

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from statsmodels.stats.proportion import proportion_confint

from data.load import load_comments
from eval import metrics as M
from eval.labels import label_table
from models import logreg, rules

T = label_table()
DF = load_comments().set_index("row_id")
DEV = T[T.split == "dev"]


def _random(seed, n=400, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.random(n) + 0.4 * y
    if ties:
        s = np.round(s, 1)
    return y, s


# Wilson intervals

@pytest.mark.parametrize("k,n", [(0, 1), (0, 10), (3, 10), (10, 10), (68, 481), (26, 200), (1, 100), (199, 200), (5, 7)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = M.wilson(k, n)
    slo, shi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(slo, abs=1e-12)
    assert hi == pytest.approx(shi, abs=1e-12)


def test_wilson_reproduces_the_audit_figures():
    lo, hi = M.wilson(68, 481)
    assert (round(lo, 3), round(hi, 3)) == (0.113, 0.175)
    lo, hi = M.wilson(26, 200)
    assert (round(lo, 3), round(hi, 3)) == (0.090, 0.184)


def test_wilson_edges():
    assert all(math.isnan(x) for x in M.wilson(0, 0))
    assert M.wilson(0, 5)[0] == 0.0 and M.wilson(5, 5)[1] == 1.0
    with pytest.raises(ValueError):
        M.wilson(6, 5)


# Counts and rates against scikit-learn

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_binary_metrics_match_sklearn(seed):
    y, s = _random(seed)
    p = M.predict(s, 0.7)
    got = M.binary_metrics(y, p, n_boot=200)
    tn, fp, fn, tp = skm.confusion_matrix(y, p, labels=[0, 1]).ravel()
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (tp, fp, fn, tn)
    assert got["precision"] == pytest.approx(skm.precision_score(y, p))
    assert got["recall"] == pytest.approx(skm.recall_score(y, p))
    assert got["f1"] == pytest.approx(skm.f1_score(y, p))
    assert got["fpr"] == pytest.approx(fp / (fp + tn))
    assert got["precision_lo"] <= got["precision"] <= got["precision_hi"]


def test_nothing_flagged_gives_nan_precision_not_zero():
    got = M.binary_metrics([1, 0, 1], [0, 0, 0], n_boot=50)
    assert math.isnan(got["precision"]) and got["recall"] == 0.0 and got["f1"] == 0.0


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        M.counts([1, 0], [1])
    with pytest.raises(ValueError):
        M.counts([2, 0], [1, 0])
    with pytest.raises(ValueError):
        M.counts([1, 0], [0.5, 0])


def test_predict_is_at_least_threshold():
    assert list(M.predict([0.2, 0.5, 0.7], 0.5)) == [0, 1, 1]


# F1 bootstrap

def test_f1_interval_is_seeded_and_brackets_f1():
    y, s = _random(4)
    p = M.predict(s, 0.7)
    a = M.f1_interval(y, p, n_boot=500, seed=7)
    assert a == M.f1_interval(y, p, n_boot=500, seed=7)
    assert a != M.f1_interval(y, p, n_boot=500, seed=8)
    assert a[0] < skm.f1_score(y, p) < a[1]


def test_f1_interval_narrows_with_more_comments():
    y, s = _random(5, n=200)
    Y, S = _random(5, n=5000)
    w_small = np.subtract(*M.f1_interval(y, M.predict(s, 0.7), n_boot=300)[::-1])
    w_large = np.subtract(*M.f1_interval(Y, M.predict(S, 0.7), n_boot=300)[::-1])
    assert w_large < w_small


# Precision-recall curve and average precision

@pytest.mark.parametrize("seed,ties", [(1, False), (2, True), (3, True)])
def test_pr_curve_matches_sklearn(seed, ties):
    y, s = _random(seed, ties=ties)
    got = M.pr_curve(y, s)
    prec, rec, thr = skm.precision_recall_curve(y, s)
    # scikit-learn lists thresholds ascending and appends the point (precision 1, recall 0)
    assert np.allclose(got.threshold.to_numpy()[::-1], thr)
    assert np.allclose(got.precision.to_numpy()[::-1], prec[:-1])
    assert np.allclose(got.recall.to_numpy()[::-1], rec[:-1])


@pytest.mark.parametrize("seed,ties", [(1, False), (2, True), (6, True)])
def test_average_precision_matches_sklearn(seed, ties):
    y, s = _random(seed, ties=ties)
    assert M.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)


def test_average_precision_on_the_baselines_dev_scores():
    tr = DF[DF.split == "train"]
    dev = DF.loc[DEV.index]
    y = DEV.label_corrected.to_numpy()
    lr = logreg.spam_proba(logreg.fit(tr.content, tr.label), dev.content)
    rs = dev.content.map(rules.rule_score).to_numpy()  # integers 0-6: heavy ties
    for s in (lr, rs):
        assert M.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)


def test_pr_curve_rejects_nan():
    with pytest.raises(ValueError):
        M.pr_curve([1, 0], [0.5, np.nan])


# Groups and running cost

def test_recall_by_group():
    y = [1, 1, 1, 0, 1, 1]
    p = [1, 0, 1, 1, 0, 1]
    g = ["a", "a", "b", "b", "c", "c"]
    got = M.recall_by(y, p, g).set_index("group")
    assert got.spam.to_dict() == {"a": 2, "b": 1, "c": 2}
    assert got.caught.to_dict() == {"a": 1, "b": 1, "c": 1}
    assert got.recall["a"] == 0.5
    assert (got.recall_lo["a"], got.recall_hi["a"]) == M.wilson(1, 2)


def test_cost_and_time_per_1000():
    # 2,000 comments, 3M input and 0.5M output tokens at $1 and $5 per million
    assert M.cost_per_1000(2000, 3_000_000, 500_000, 1.0, 5.0) == pytest.approx(2.75)
    assert M.time_per_1000(500, 60) == pytest.approx(120)
    with pytest.raises(ValueError):
        M.per_1000(1, 0)


# evaluate()

def _rule_scores():
    return DF.content.map(rules.rule_score).astype(float)


def test_evaluate_on_dev():
    r = M.evaluate(T, _rule_scores(), 1, n_boot=200)
    m = r["metrics"]
    assert (m["n"], m["spam"], m["ham"]) == (370, 157, 213)
    y = DEV.label_corrected.to_numpy()
    p = (_rule_scores().loc[DEV.index] >= 1).astype(int).to_numpy()
    assert m["precision"] == pytest.approx(skm.precision_score(y, p))
    assert m["recall"] == pytest.approx(skm.recall_score(y, p))
    sub = r["recall_by_subtype"]
    assert set(sub.group) == {"scam", "link", "self_promotion", "bulk"}
    assert sub.spam.sum() == 157 and sub.caught.sum() == m["tp"]
    vid = r["recall_by_video"]
    assert list(vid.group) == ["Shakira"]


def test_evaluate_original_labels_show_the_label_errors_as_a_group():
    r = M.evaluate(T, _rule_scores(), 1, which="original", n_boot=50)
    sub = r["recall_by_subtype"].set_index("group")
    assert sub.spam["not_spam"] == 17
    assert r["metrics"]["spam"] == 174


def test_evaluate_applies_exclusions_and_share_text():
    # a constant score: row counts only, so no system is scored on test before the pre-registration
    flat = pd.Series(0.0, index=T.index)
    assert M.evaluate(T, flat, 1, splits=("test",), n_boot=50)["metrics"]["n"] == 885
    s = _rule_scores()
    assert M.evaluate(T, s, 1, drop_share_text=True, n_boot=50)["metrics"]["n"] == 360


def test_evaluate_needs_a_score_for_every_scored_row():
    s = _rule_scores().drop("Shakira:000")
    with pytest.raises(ValueError, match="no score"):
        M.evaluate(T, s, 1, n_boot=50)
