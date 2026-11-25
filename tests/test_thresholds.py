import json

import numpy as np
import pytest

from eval import thresholds as T
from models import llm


# The rule on small hand-made rankings

def test_levels_keep_ties_together():
    d = T.levels([3, 3, 2, 1, 1, 1], [1, 0, 1, 1, 0, 0])
    assert list(d.score) == [3, 2, 1]
    assert list(d.n) == [2, 1, 3]
    assert list(d.band_n) == [2, 3, 6]
    assert list(d.band_spam) == [1, 2, 3]


def test_auto_remove_takes_the_lowest_band_with_every_band_above_at_98_percent():
    # 60 spam at the top, then one ham (61 flagged, precision 60/61 = 98.4%), then 10 spam
    # (71 flagged, 70/71 = 98.6%), then 5 ham (76 flagged, 92.1%)
    score = list(range(100, 40, -1)) + [40] + list(range(39, 29, -1)) + [5] * 5
    y = [1] * 60 + [0] + [1] * 10 + [0] * 5
    t = T.auto_remove(score, y)
    assert t == pytest.approx((30 + 5) / 2)  # halfway between the lowest kept score and the next
    assert (np.asarray(score) >= t).sum() == 71


def test_auto_remove_stops_at_the_first_band_below_target_even_if_a_lower_one_recovers():
    score = [10, 9] + list(range(8, -42, -1))
    y = [1, 0] + [1] * 50  # band of 2 has precision 50%, the full band 51/52 = 98.1%
    t = T.auto_remove(score, y)
    assert t == pytest.approx(9.5)  # only the top comment


def test_no_auto_remove_when_the_top_band_misses_the_target():
    assert T.auto_remove([5, 4, 3], [0, 1, 1]) is None


def test_a_tie_group_is_never_split():
    # the top score holds 49 spam and 1 ham (98.0%): flagged together or not at all
    score = [9] * 50 + [8] * 5
    y = [1] * 49 + [0] + [0] * 5
    assert T.auto_remove(score, y) == pytest.approx(8.5)
    assert T.auto_remove(score, y, precision=0.99) is None


def test_floor_keeps_the_threshold_at_or_above_the_systems_own_cut_off():
    score = [6] * 99 + [0.4] + [0.1] * 200
    y = [1] * 99 + [1] + [0] * 200  # without a floor the band reaches 0.4 at 100%
    assert T.auto_remove(score, y) == pytest.approx(0.25)
    assert T.auto_remove(score, y, floor=3.0) == 3.0
    assert T.auto_remove([2, 1], [1, 1], floor=3.0) is None  # nothing at or above the floor


def test_allow_below_is_the_highest_threshold_catching_95_percent():
    score = list(range(100, 0, -1))
    y = [1] * 90 + [0, 1, 1, 1, 1, 1, 0, 0, 1, 1]  # 97 spam; 95% needs 93, reached at rank 94
    t = T.allow_below(score, y)
    assert t == pytest.approx(6.5)  # rank 94 holds score 7
    assert ((np.asarray(score) >= t) & (np.asarray(y) == 1)).sum() == 93


def test_allow_below_never_above_auto_remove():
    score = [9] * 10 + [1] * 10
    y = [1] * 10 + [0] * 10
    a = T.auto_remove(score, y)
    assert T.allow_below(score, y, a) == a


def test_allow_below_reaching_the_lowest_score_reviews_everything():
    t = T.allow_below([2, 1, 0, 0], [1, 0, 1, 0])
    assert t == 0.0
    assert set(T.tiers([2, 1, 0, 0], None, t)) == {"review"}


def test_tiers():
    assert list(T.tiers([0.9, 0.7, 0.2], 0.8, 0.5)) == ["remove", "review", "allow"]
    assert list(T.tiers([0.9], None, 0.5)) == ["review"]


def test_summary_counts_and_interval():
    s = T.summary([0.9, 0.9, 0.6, 0.1], [1, 0, 1, 1], 0.8, 0.5)
    assert s["remove"]["n"] == 2 and s["remove"]["spam"] == 1 and s["remove"]["precision"] == 0.5
    assert s["review"] == {"n": 1, "spam": 1} and s["allow"] == {"n": 1, "spam": 1}
    assert s["remove_or_review_recall"] == pytest.approx(2 / 3)
    assert T.summary([0.1], [1], None, 0.5)["remove"]["precision"] is None


# The LLM floor is the vote cut-off

def test_llm_floor_of_3_is_3_of_5_votes_whatever_the_confidence():
    def vote(label, conf):
        return {"valid": True, "label": label, "confidence": conf}
    low3 = llm.vote_score([vote("spam", 50)] * 3 + [vote("ham", 100)] * 2)["score"]
    high2 = llm.vote_score([vote("spam", 100)] * 2 + [vote("ham", 50)] * 3)["score"]
    assert low3 >= T.FLOORS["llm_v1"] > high2


# The committed file

COMMITTED = json.loads(T.PATH.read_text(encoding="utf-8"))


def test_committed_thresholds_match_a_fresh_choice():
    assert T._same(COMMITTED, json.loads(T.dumps(T.choose())))


def test_committed_thresholds_respect_the_rule():
    assert COMMITTED["label_set"] == "corrected" and COMMITTED["split"] == "dev"
    assert set(COMMITTED["systems"]) == set(T.SYSTEMS)
    for name, r in COMMITTED["systems"].items():
        c = r["dev_corrected"]
        assert r["auto_remove"] is not None and r["auto_remove"] >= r["floor"]
        assert r["allow_below"] <= r["auto_remove"]
        assert c["remove"]["precision"] >= T.PRECISION
        assert c["remove_or_review_recall"] >= T.RECALL
        assert c["remove"]["n"] + c["review"]["n"] + c["allow"]["n"] == c["comments"] == 370


def test_without_the_floor_the_llm_would_auto_remove_comments_it_voted_ham():
    scores = T.dev_scores()
    from eval.labels import label_table, labels, scored
    rows = scored(label_table(), "dev")
    y = labels(rows, "corrected")
    for v in ("llm_v1", "llm_v2"):
        assert T.auto_remove(scores[v].reindex(rows.index).to_numpy(), y) < 1.0  # inside the 0-vote group
        assert COMMITTED["systems"][v]["auto_remove"] == 3.0
