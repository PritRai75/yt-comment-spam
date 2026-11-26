import pandas as pd

from data.load import load_comments
from eval.labels import label_table
from release import hf_dataset as H

DF = load_comments().set_index("row_id")
T = label_table()


def _read(name):
    return pd.read_csv(H.OUT / "data" / f"{name}.csv", dtype=str, keep_default_na=False)


def test_committed_files_match_the_builder():
    assert H.main(["--check"]) == 0


def test_every_comment_once_in_its_split_with_text_unchanged():
    parts = {name: _read(name) for name in H.SPLITS.values()}
    allrows = pd.concat(parts.values(), ignore_index=True)
    assert list(allrows.columns) == H.COLUMNS
    assert allrows.row_id.is_unique and set(allrows.row_id) == set(DF.index)
    for split, name in H.SPLITS.items():
        assert set(parts[name].split) == {split}
        assert set(parts[name].row_id) == set(DF.index[DF.split == split])
    a = allrows.set_index("row_id")
    assert (a.content == DF.content.loc[a.index]).all()  # byte for byte through the CSV
    assert (a.comment_id == DF.comment_id.loc[a.index]).all()
    assert "author" not in a.columns and "date" not in a.columns


def test_labels_and_corrections_match_the_audit_record():
    a = pd.concat([_read(n) for n in H.SPLITS.values()]).set_index("row_id")
    assert (a.label_original.astype(int) == T.label_original.loc[a.index]).all()
    assert (a.label_corrected.astype(int) == T.label_corrected.loc[a.index]).all()
    corrected = a.corrected == "True"
    assert corrected.sum() == 88
    assert (corrected == (a.label_original != a.label_corrected)).all()
    assert (a.correction_reason[corrected] != "").all() and (a.correction_reason[~corrected] == "").all()
    assert (a.review == T.review.loc[a.index]).all()
    assert set(a.index[a.scored == "False"]) == {"Eminem:269"}


def test_card_front_matter_and_numbers():
    card = (H.OUT / "README.md").read_text(encoding="utf-8")
    head = card.split("---\n")[1]
    for line in ("license: cc-by-4.0", "- split: validation", "path: data/test.csv", "- text-classification"):
        assert line in head
    assert "**88 corrections** (87 spam → ham, 1 ham → spam)" in card
    assert "68 of the 481 test comments labelled spam are not spam (14.1%" in card
    assert "https://doi.org/10.24432/C58885" in card
