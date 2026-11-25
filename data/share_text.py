"""YouTube's automatic share text, as defined in the spam policy (edge case 4, P8).

A comment belongs to the share-text family when its cleaned text starts with
"Check out this video on YouTube", "Check out this playlist on YouTube" or
"Take a look at this video on YouTube", with or without the colon, including the
truncated endings "on YouTu" and "on YouTub". On the pinned data the family is
164 comments (dev 10, test 153, train 1), all labelled spam.

The pre-registered second result drops the whole family (decision 14a). Whether a
comment in the family is spam is still decided by the policy: bare share text, or
share text followed by a tail that says nothing on its own (P10), is bulk; share
text followed by the commenter's own words is judged by those words.
"""
import re

from data.clean import clean_text

SHARE_TEXT = re.compile(
    r"^(?:check out|take a look at) this (?:video|playlist) on youtu(?:be?)?(?![a-z])\s*:?",
    re.IGNORECASE,
)


def is_share_text(raw: str) -> bool:
    """True when the comment, as a viewer saw it, starts with YouTube's share text."""
    return SHARE_TEXT.match(clean_text(raw)) is not None


def after_share_text(raw: str) -> str:
    """The commenter's own text after the share text ("" for bare share text).

    Raises ValueError for a comment outside the family."""
    s = clean_text(raw)
    m = SHARE_TEXT.match(s)
    if m is None:
        raise ValueError("not share text")
    return s[m.end():].strip()
