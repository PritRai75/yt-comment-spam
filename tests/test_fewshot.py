import pandas as pd
import pytest

from audit.flags import read
from data.clean import clean_text, dup_key
from data.load import load_comments
from models import fewshot as F

DF = load_comments()
POOL = F.pool(DF)
ORDER = F.ordered(POOL)
HELD = DF[DF.split != "train"]
HELD_RUNS = set().union(*(F.word_runs(t) for t in HELD.content))


def test_pool_is_train_only_and_unflagged():
    assert (POOL.split == "train").all()
    assert not set(POOL.row_id) & set(read("flags.csv").row_id)
    assert (len(POOL), int(POOL.label.sum())) == (528, 209)


def test_pool_shares_no_text_with_dev_or_test():
    keys = {dup_key(clean_text(t)) for t in HELD.content}
    for t in POOL.content:
        assert dup_key(clean_text(t)) not in keys
        assert not F.word_runs(t) & HELD_RUNS


def test_order_is_seeded_and_complete():
    assert ORDER == F.ordered(POOL)
    assert sorted(ORDER[1]) == sorted(POOL[POOL.label == 1].row_id)
    assert sorted(ORDER[0]) == sorted(POOL[POOL.label == 0].row_id)
    assert ORDER[1][:2] == ["Psy:057", "Psy:179"] and ORDER[0][:2] == ["Psy:321", "KatyPerry:307"]


def test_candidates_hide_the_label_and_mix_both():
    c = F.candidates(df=DF)
    assert list(c.candidate) == [f"C{i:02d}" for i in range(1, 13)]
    assert "label" not in c.columns
    assert set(c.row_id) == set(ORDER[1][:6]) | set(ORDER[0][:6])


def _answers(overrides=None, n=6):
    rows = [(rid, "self_promotion") for rid in ORDER[1][:n]] + [(rid, "not_spam") for rid in ORDER[0][:n]]
    a = dict(rows)
    a.update(overrides or {})
    return pd.DataFrame({"row_id": list(a), "answer": list(a.values())})


def test_examples_take_the_first_four_that_agree():
    ex = F.examples(_answers(), df=DF)
    assert len(ex) == 8 and (ex.label == "spam").sum() == 4
    assert set(ex[ex.label == "spam"].row_id) == set(ORDER[1][:4])
    assert set(ex[ex.label == "ham"].row_id) == set(ORDER[0][:4])
    assert (ex[ex.label == "ham"].subtype == "none").all()


def test_a_disagreeing_answer_skips_to_the_next():
    ex = F.examples(_answers({ORDER[1][1]: "not_spam", ORDER[0][0]: "bulk"}), df=DF)
    spam = set(ex[ex.label == "spam"].row_id)
    assert ORDER[1][1] not in spam and ORDER[1][4] in spam
    assert ORDER[0][0] not in set(ex.row_id) and ORDER[0][4] in set(ex.row_id)


def test_not_enough_answers_is_an_error():
    with pytest.raises(ValueError, match="no answer"):
        F.examples(_answers({rid: "not_spam" for rid in ORDER[1][:3]}), df=DF)
    with pytest.raises(ValueError, match="answer"):
        F.examples(_answers({ORDER[1][0]: "spam"}), df=DF)


def test_examples_show_the_normalised_text_and_reviewer_subtype():
    ex = F.examples(_answers({ORDER[1][0]: "bulk"}), df=DF)
    row = ex.set_index("row_id").loc[ORDER[1][0]]
    assert row.subtype == "bulk"
    assert row.text == F.shown(DF.set_index("row_id").content[ORDER[1][0]])
