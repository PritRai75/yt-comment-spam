import pandas as pd
import pytest

from audit.flags import read
from eval import labels as L

T = L.label_table()
DT = T[T.split != "train"]


def test_one_row_per_comment():
    assert len(T) == 1956
    assert T.index.is_unique


def test_label_counts():
    orig = T.groupby("split").label_original.sum().to_dict()
    corr = T.groupby("split").label_corrected.sum().to_dict()
    assert orig == {"dev": 174, "test": 481, "train": 350}
    assert corr == {"dev": 157, "test": 414, "train": 348}


def test_corrections_are_the_only_differences():
    changed = T[T.label_original != T.label_corrected]
    assert changed.groupby("split").size().to_dict() == {"dev": 17, "test": 69, "train": 2}
    assert set(changed.index) == set(read("corrections.csv").row_id)
    assert int((changed.label_corrected == 0).sum()) == 87


def test_every_dev_and_test_spam_was_reviewed():
    assert (DT[DT.label_original == 1].review != "").all()
    assert (DT[DT.label_corrected == 1].review != "").all()


def test_scored_spam_has_a_subtype():
    spam = DT[(DT.label_corrected == 1) & DT.scored]
    assert spam.review.isin(L.SUBTYPES).all()
    assert len(spam) == 157 + 413


def test_corrected_ham_is_never_given_a_subtype():
    assert not DT[DT.label_corrected == 0].review.isin(L.SUBTYPES + ["unsure"]).any()


def test_only_unsure_is_the_excluded_row():
    assert list(T[T.review == "unsure"].index) == ["Eminem:269"]
    assert list(T[~T.scored].index) == ["Eminem:269"]
    assert int(L.scored(T, "test").shape[0]) == 885


def test_review_agrees_with_corrections():
    for r in read("corrections.csv").itertuples():
        expected = r.subtype_after if r.label_after == "spam" else "not_spam"
        assert T.review[r.row_id] == expected, r.row_id


def test_second_pass_replaces_first_unsure():
    # U02: KatyPerry:247 was unsure on the flags, not_spam on the second pass
    assert T.review["KatyPerry:247"] == "not_spam"
    assert T.label_corrected["KatyPerry:247"] == 0
    assert T.review["Eminem:361"] == "self_promotion"  # T327, the one ham -> spam
    assert T.label_corrected["Eminem:361"] == 1


def test_review_counts_on_dev_and_test():
    got = DT[DT.scored].review.value_counts().to_dict()
    assert got == {"": 401, "self_promotion": 299, "not_spam": 284, "bulk": 159, "scam": 62, "link": 50}


def test_share_text_family():
    assert T.share_text.groupby(T.split).sum().to_dict() == {"dev": 10, "test": 153, "train": 1}
    assert len(L.scored(T, ["dev", "test"], drop_share_text=True)) == 370 + 885 - 10 - 153


def test_labels_picks_a_set():
    dev = L.scored(T, "dev")
    assert L.labels(dev, "original").sum() == 174
    assert L.labels(dev, "corrected").sum() == 157
    with pytest.raises(ValueError):
        L.labels(dev, "gold")


@pytest.mark.parametrize("current,verdict,subtype,expected", [
    ("spam", "keep", "link", "link"),
    ("spam", "correct", "", "not_spam"),
    ("ham", "keep", "", "not_spam"),
    ("ham", "correct", "scam", "scam"),
    ("spam", "unsure", "bulk", "unsure"),
])
def test_flag_answer(current, verdict, subtype, expected):
    assert L._flag_answer(current, verdict, subtype) == expected


def _patched_read(monkeypatch, name, edit):
    real = L.read

    def fake(n):
        df = real(n)
        return edit(df.copy()) if n == name else df
    monkeypatch.setattr(L, "read", fake)


def test_disagreeing_reviews_stop_the_loader(monkeypatch):
    # LMFAO:181 is a kept flag (bulk) and in the subtype sample (bulk)
    def edit(df):
        df.loc[df.row_id == "LMFAO:181", "subtype"] = "link"
        return df
    _patched_read(monkeypatch, "subtype_labels.csv", edit)
    with pytest.raises(L.LabelError, match="LMFAO:181"):
        L.review_answers()


def test_second_pass_on_a_settled_comment_stops_the_loader(monkeypatch):
    def edit(df):
        extra = pd.DataFrame([{"unsure_id": "U99", "source_id": "F999", "row_id": "LMFAO:181",
                               "answer": "bulk", "note": "x"}])
        return pd.concat([df, extra], ignore_index=True)
    _patched_read(monkeypatch, "unsure_resolution.csv", edit)
    with pytest.raises(L.LabelError, match="U99"):
        L.review_answers()
