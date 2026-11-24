import re

import pytest

from data.clean import clean_text, dup_key
from data.load import load_comments

DF = load_comments()
CLEAN = DF.content.map(clean_text)
TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^>]*>")


@pytest.mark.parametrize("raw,expected", [
    ("great song﻿", "great song"),
    ("line one<br />line two", "line one\nline two"),
    ("I&#39;m here &amp; there", "I'm here & there"),
    ("OPPA &lt;3﻿", "OPPA <3"),
    ('<a href="http://www.youtube.com/watch?v=x&amp;t=2m19s">2:19</a> best part', "2:19 best part"),
    ('<a href="https://plus.google.com/s/%23Eminem">#Eminem</a>', "#Eminem"),
    ('visit <a href="http://example.com/x">http://example.com/x</a> now', "visit http://example.com/x now"),
    ('subscribe to <span class="proflinkWrapper"><span class="proflinkPrefix">+</span>'
     '<a class="proflink" href="https://plus.google.com/1" oid="1">SomeName</a></span>', "subscribe to +SomeName"),
    ("This video deserves <b>1B</b> views", "This video deserves 1B views"),
    ("&lt;b&gt;not a tag&lt;/b&gt;", "<b>not a tag</b>"),
    ("  spaced  ﻿", "spaced"),
])
def test_clean_text_examples(raw, expected):
    assert clean_text(raw) == expected


def test_no_byte_order_marks_left():
    assert not CLEAN.str.contains("﻿").any()


def test_only_escaped_html_remains():
    # One spam comment shows HTML code as literal, entity-escaped text; viewers saw the code.
    left = CLEAN[CLEAN.str.contains(TAG)]
    assert len(left) == 1
    assert "&lt;script&gt;" in DF.content[left.index[0]]


def test_cleaning_counts():
    assert int((CLEAN != DF.content).sum()) == 1677
    assert int((CLEAN == "").sum()) == 0


def test_dup_key():
    assert dup_key("Check out this video on YouTube:﻿") == "check out this video on youtube"
    assert dup_key("WOW!!") == dup_key("wow") == "wow"
    assert dup_key("café") == dup_key("café")
    assert dup_key("a​b") == "ab"
