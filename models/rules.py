"""Keyword and regex baseline.

Six rule families, written from the spam policy and the train split only, then frozen
before any score on dev or test was looked at. Each family that matches the cleaned,
case-folded text adds 1 to the score (0 to 6); the rules call a comment spam when the
score is at least 1.
"""
import re

from data.clean import clean_text

RULES = {
    # an address to another site: scheme, www., or a bare domain with a common ending
    "link": re.compile(
        r"https?://|\bwww\.|\b[a-z0-9-]+(?:\.|\s*(?:\[dot\]|\(dot\)|\bdot\b)\s*)"
        r"(?:com|net|org|ly|be|info|tk|io|tv)\b"
    ),
    # subscription requests, including "sub to me", "sub back", "subs"
    "subscribe": re.compile(r"\bsub(?:scrib\w*|s|bed|bing)?\b"),
    # promoting the commenter's own content
    "self_promo": re.compile(
        r"\bcheck\s*(?:out|this|it)\b|\bmy\s+(?:channel|video|vids?|music|page|song|songs|band|playlist|cover|rap|blog|account)\b"
        r"|\bvisit\s+(?:my|our|this)\b|\bour\s+(?:channel|video|music|band|page)\b|\bfollow\s+me\b"
    ),
    # money, prizes and scams
    "money": re.compile(
        r"\bfree\b|\bwin\b|\bgiveaway\b|\bgift\s*cards?\b|\$\s?\d|\bearn\w*\b|\bmoney\b|\bcash\b|\bpaypal\b"
        r"|\bbitcoins?\b|\bprize\w*\b|\bdonat\w+\b|\bhack\w*\b|\bgofundme\b"
    ),
    # engagement requests aimed at something outside the thread
    "engagement": re.compile(
        r"\b(?:like|share|vote\s+for|watch)\s+(?:my|our|this\s+(?:pic|page|video|link))\b|\bthumbs?\s+up\b.*\bmy\b"
    ),
    # other platforms and contact handles
    "off_platform": re.compile(r"\bfacebook\b|\binstagram\b|\big\b|\btwitter\b|\bkik\b|\bskype\b|\bsnapchat\b|\bwhatsapp\b|\bsoundcloud\b"),
}


def rule_hits(raw: str) -> list:
    """Names of the rule families that match a comment as served."""
    s = clean_text(raw).casefold()
    return [name for name, rx in RULES.items() if rx.search(s)]


def rule_score(raw: str) -> int:
    return len(rule_hits(raw))


def rule_predict(raw: str) -> int:
    return int(rule_score(raw) >= 1)
