import hashlib

import pytest

from data.load import load_comments
from data.share_text import after_share_text, is_share_text

DF = load_comments()
FAMILY = DF[DF.content.map(is_share_text)]


@pytest.mark.parametrize("raw", [
    "Check out this video on YouTube:",
    "Check out this video on YouTube:﻿",
    "check out this playlist on youtube:",
    "Check out this video on YouTube",
    "Take a look at this video on YouTube:",
    "Check out this video on YouTu",
    "Take a look at this video on YouTub<br />You",
    "Check out this video on YouTube:<br /><br />Eminem is amazing.",
])
def test_share_text_variants_match(raw):
    assert is_share_text(raw)


@pytest.mark.parametrize("raw", [
    "Check out my video on YouTube:",
    "check out this video on youtubers channel",
    "Love this song. Check out this video on YouTube:",
    "Check out this video",
    "",
])
def test_other_text_does_not_match(raw):
    assert not is_share_text(raw)


def test_after_share_text():
    assert after_share_text("Check out this video on YouTube:﻿") == ""
    assert after_share_text("Check out this playlist on YouTube:<br /><br />Qq") == "Qq"
    with pytest.raises(ValueError):
        after_share_text("great song")


def test_family_counts_on_pinned_data():
    # decision 14a: 164 comments (dev 10, test 153, train 1), all labelled spam
    assert len(FAMILY) == 164
    assert FAMILY.groupby("split").size().to_dict() == {"dev": 10, "test": 153, "train": 1}
    assert set(FAMILY.label) == {1}
    bare = FAMILY.content.map(after_share_text) == ""
    assert int(bare.sum()) == 133
    assert FAMILY[bare].groupby("split").size().to_dict() == {"dev": 8, "test": 125}


def test_family_membership_frozen():
    ids = "\n".join(sorted(FAMILY.row_id))
    assert hashlib.sha256(ids.encode()).hexdigest() == (
        "7d5800585ce7ba880263ec83da26be3f0f2583133a71685846882208ce7ac65d")
