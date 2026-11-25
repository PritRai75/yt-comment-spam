import numpy as np
import pytest

from data.load import load_comments
from models import tree

DF = load_comments()
TR = DF[DF.split == "train"]


@pytest.fixture(scope="module")
def cv():
    return tree.cv_table(DF)


def test_grid_has_12_settings():
    assert len(tree.GRID) == 12
    assert len({tree._key(p) for p in tree.GRID}) == 12
    assert tree.CHOSEN in tree.GRID


def test_grid_still_picks_the_frozen_setting(cv):
    assert tree.best(cv) == tree.CHOSEN
    assert list(cv.chosen) == [True] + [False] * 11
    assert (np.diff(cv.cv_ap_mean.to_numpy()) <= 0).all()


def test_cv_table_reports_folds_and_videos(cv):
    folds = [f"cv_ap_fold{i}" for i in range(1, 6)]
    assert np.allclose(cv[folds].mean(axis=1), cv.cv_ap_mean)
    assert np.allclose(cv[folds].std(axis=1, ddof=1), cv.cv_ap_sd)
    assert {"lovo_ap_KatyPerry", "lovo_ap_Psy", "lovo_ap_mean"} <= set(cv.columns)


def test_committed_cv_results_regenerate(cv):
    assert tree._check(cv, "tree_cv.csv") == 0


def test_committed_importance_results_regenerate():
    assert tree.main(["importance", "--check"]) == 0


def test_tree_is_deterministic():
    a = tree.spam_proba(tree.fit(TR.content, TR.label), DF.content.head(50))
    b = tree.spam_proba(tree.fit(TR.content, TR.label), DF.content.head(50))
    assert np.array_equal(a, b)


def test_train_scored_out_of_fold_by_video():
    scores = tree.score_all(DF)
    assert not scores.isna().any()
    assert ((scores >= 0) & (scores <= 1)).all()
    kat = TR[TR.video == "KatyPerry"]
    psy = TR[TR.video == "Psy"]
    expected = tree.spam_proba(tree.fit(kat.content, kat.label), psy.content)
    assert np.allclose(scores.loc[psy.index].values, expected)
    dev = DF[DF.split == "dev"]
    full = tree.fit(TR.content, TR.label)
    assert np.allclose(scores.loc[dev.index].values, tree.spam_proba(full, dev.content))


def test_no_early_stopping():
    m = tree.make_model()
    assert m.early_stopping is False
    assert (m.learning_rate, m.max_leaf_nodes, m.max_iter) == (0.1, 15, 300)
