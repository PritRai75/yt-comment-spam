"""Readable features for the gradient-boosting baseline (spec addendum 1, decision 29).

27 numbers per comment, computed from the cleaned text (`data/clean.py`), never from
the author, date or video. Written from the spam policy and the train split only,
and fixed before the tree was scored on dev or test.

Shape (8):   characters, words, mean word length, lines, share of letters in capitals,
             share of characters that are digits, share that are punctuation or symbols,
             longest run of one repeated character
Marks (8):   "!" count, "?" count, emoji and pictographs, web addresses, hashtags,
             mentions ("+Name", "@name"), a run of 7+ digits (a phone number),
             a currency sign
Words (5):   first person singular (i, me, my, mine, myself), first person plural
             (we, us, our, ours), second person (you, your, u, ur), request verbs
             (subscribe, check, visit, share, ...), song words (song, lyrics, beat, ...)
Rules (6):   one 0/1 column per rule family in `models/rules.py`

Word counts are over case-folded word tokens. Counts, not presence, except where the
name says `has_`.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from data.clean import clean_text
from models.rules import RULES

WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
URL = re.compile(
    r"https?://\S+|\bwww\.\S+|\b[a-z0-9-]+\.(?:com|net|org|ly|be|info|tk|io|tv)\b(?:/\S*)?",
    re.IGNORECASE,
)
HASHTAG = re.compile(r"(?<![\w&])#\w+")
MENTION = re.compile(r"(?<![\w.])[+@][^\W\d_]\w*")
LONG_NUMBER = re.compile(r"\d(?:[\s.-]?\d){6,}")
CURRENCY = re.compile(r"[$€£¥₹]")

FIRST_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}
FIRST_PLURAL = {"we", "us", "our", "ours", "we're", "we've"}
SECOND = {"you", "your", "yours", "u", "ur", "you're", "ya"}
REQUEST = {"subscribe", "sub", "subs", "check", "visit", "share", "follow", "like", "likes",
           "click", "watch", "go", "help", "support", "add", "join", "vote", "listen", "please", "plz", "pls"}
SONG = {"song", "songs", "lyrics", "beat", "voice", "dance", "dancing", "chorus", "singer", "sing", "sings"}

SHAPE = ["n_chars", "n_words", "mean_word_len", "n_lines", "caps_share", "digit_share",
         "punct_share", "max_char_run"]
MARKS = ["n_exclaim", "n_question", "n_emoji", "n_urls", "n_hashtags", "n_mentions",
         "has_long_number", "has_currency"]
WORDS = ["first_singular", "first_plural", "second_person", "request_verbs", "song_words"]
RULE_COLS = [f"rule_{name}" for name in RULES]
FEATURES = SHAPE + MARKS + WORDS + RULE_COLS


def _is_emoji(ch: str) -> bool:
    return unicodedata.category(ch) == "So" or ord(ch) >= 0x1F000


def _max_run(s: str) -> int:
    best = run = 0
    prev = None
    for ch in s:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def features(raw: str) -> dict:
    s = clean_text(raw)
    folded = s.casefold()
    words = WORD.findall(folded)
    letters = [c for c in s if c.isalpha()]
    n = len(s)
    out = {
        "n_chars": n,
        "n_words": len(words),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "n_lines": s.count("\n") + 1 if s else 0,
        "caps_share": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "digit_share": sum(c.isdigit() for c in s) / n if n else 0.0,
        "punct_share": sum(not (c.isalnum() or c.isspace()) for c in s) / n if n else 0.0,
        "max_char_run": _max_run(s),
        "n_exclaim": s.count("!"),
        "n_question": s.count("?"),
        "n_emoji": sum(_is_emoji(c) for c in s),
        "n_urls": len(URL.findall(s)),
        "n_hashtags": len(HASHTAG.findall(s)),
        "n_mentions": len(MENTION.findall(s)),
        "has_long_number": int(LONG_NUMBER.search(s) is not None),
        "has_currency": int(CURRENCY.search(s) is not None),
        "first_singular": sum(w in FIRST_SINGULAR for w in words),
        "first_plural": sum(w in FIRST_PLURAL for w in words),
        "second_person": sum(w in SECOND for w in words),
        "request_verbs": sum(w in REQUEST for w in words),
        "song_words": sum(w in SONG for w in words),
    }
    for name, rx in RULES.items():
        out[f"rule_{name}"] = int(rx.search(folded) is not None)
    return out


def feature_frame(texts) -> pd.DataFrame:
    """One row per comment, columns in FEATURES order, float."""
    texts = list(texts)
    return pd.DataFrame([features(t) for t in texts], columns=FEATURES, dtype=float)
