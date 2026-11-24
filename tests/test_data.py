import json

import pytest

from data import load

DF = load.load_comments()
SPLITS_FILE = json.loads(load.SPLITS.read_text(encoding="utf-8"))

# (spam, ham) per video, as drawn from the pinned files
VIDEO_COUNTS = {
    "Psy": (175, 175),
    "KatyPerry": (175, 175),
    "LMFAO": (236, 202),
    "Eminem": (245, 203),
    "Shakira": (174, 196),
}


def test_total_rows_and_classes():
    assert len(DF) == 1956
    assert int(DF.label.sum()) == 1005
    assert int((DF.label == 0).sum()) == 951
    assert set(DF.label.unique()) == {0, 1}


@pytest.mark.parametrize("video,counts", VIDEO_COUNTS.items())
def test_counts_by_video(video, counts):
    sub = DF[DF.video == video]
    assert (int(sub.label.sum()), int((sub.label == 0).sum())) == counts


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_counts_by_split(split):
    exp = SPLITS_FILE["expected_counts"][split]
    sub = DF[DF.split == split]
    assert len(sub) == exp["rows"]
    assert int(sub.label.sum()) == exp["spam"]
    assert int((sub.label == 0).sum()) == exp["ham"]


def test_splits_are_disjoint_and_cover_every_video():
    vids = [v for vs in load.load_splits().values() for v in vs]
    assert len(vids) == len(set(vids)) == 5
    assert set(vids) == set(DF.video.unique())


def test_row_ids_unique_and_comment_ids_not():
    assert DF.row_id.is_unique
    # Three comment IDs appear twice in the source, each time as an identical row.
    dup = DF[DF.comment_id.duplicated(keep=False)]
    assert dup.comment_id.nunique() == 3
    assert len(dup) == 6
    cols = ["comment_id", "author", "date", "content", "label", "video"]
    assert len(dup[cols].drop_duplicates()) == 3


def test_no_empty_content():
    assert (DF.content.str.strip() != "").all()


def test_video_of():
    assert load.video_of("Youtube02-KatyPerry.csv") == "KatyPerry"
    assert load.video_of("folder/Youtube05-Shakira.csv") == "Shakira"
