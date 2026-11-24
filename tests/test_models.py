import numpy as np
import pytest

from data.load import load_comments
from models import logreg, rules

DF = load_comments()


@pytest.mark.parametrize("text,family", [
    ("visit http://example.com", "link"),
    ("go to example.com now", "link"),
    ("mysite dot com", "link"),
    ("youtu.be/abc", "link"),
    ("please subscribe", "subscribe"),
    ("sub to me", "subscribe"),
    ("i will sub back", "subscribe"),
    ("check out my channel", "self_promo"),
    ("listen to my song", "self_promo"),
    ("free gift cards", "money"),
    ("earn $50 a day", "money"),
    ("please donate", "money"),
    ("like my pic", "engagement"),
    ("vote for my idol", "engagement"),
    ("add me on facebook", "off_platform"),
])
def test_rule_families_fire(text, family):
    assert family in rules.rule_hits(text)


@pytest.mark.parametrize("text", [
    "I love this song",
    "this song. me too",
    "Like if you're watching in 2015",
    "subtle beat",
    "the video is great",
])
def test_rules_quiet_on_plain_comments(text):
    assert rules.rule_hits(text) == []


def test_rule_score_counts_families():
    assert rules.rule_score("check out my channel http://x.com free stuff") == 3
    assert rules.rule_predict("nice") == 0


def test_rules_on_train():
    tr = DF[DF.split == "train"]
    pred = tr.content.map(rules.rule_predict)
    tp = int(((pred == 1) & (tr.label == 1)).sum())
    fp = int(((pred == 1) & (tr.label == 0)).sum())
    fn = int(((pred == 0) & (tr.label == 1)).sum())
    assert (tp, fp, fn) == (325, 18, 25)


def test_logreg_is_deterministic():
    tr = DF[DF.split == "train"]
    a = logreg.spam_proba(logreg.fit(tr.content, tr.label), DF.content.head(50))
    b = logreg.spam_proba(logreg.fit(tr.content, tr.label), DF.content.head(50))
    assert np.array_equal(a, b)


def test_train_scored_out_of_fold_by_video():
    scores = logreg.score_all(DF)
    assert not scores.isna().any()
    tr = DF[DF.split == "train"]
    kat = tr[tr.video == "KatyPerry"]
    psy = tr[tr.video == "Psy"]
    expected = logreg.spam_proba(logreg.fit(kat.content, kat.label), psy.content)
    assert np.allclose(scores.loc[psy.index].values, expected)
    full = logreg.fit(tr.content, tr.label)
    dev = DF[DF.split == "dev"]
    assert np.allclose(scores.loc[dev.index].values, logreg.spam_proba(full, dev.content))
