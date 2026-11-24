"""Turn a comment as served into the text a viewer would have seen.

Applied identically before every classifier (rules, logistic regression, both LLM
prompts). It removes collection artefacts only; it is not the adversarial
normalisation that LLM v2 adds.

1. `<br />` becomes a newline.
2. A link becomes its visible text. In this dataset a link's visible text is either the
   URL itself, or an auto-link YouTube and Google+ made (a timestamp such as "2:19",
   a hashtag, or a "+Name" mention), so no off-site URL is hidden by this step.
3. Any other tag (`<span>`, `<i>`, `<b>`) is dropped, keeping its text.
4. HTML entities are decoded (`&#39;` -> `'`, `&lt;3` -> `<3`). Tags are handled
   before decoding, so a decoded `<` is never read as a tag.
5. U+FEFF is removed: it ends 1,548 of 1,956 comments and its presence tracks the
   label on some videos, so it is a collection artefact a model must not learn.
6. Leading and trailing whitespace is trimmed.
"""
import html
import re
import unicodedata

BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
LINK = re.compile(r"<a\b[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^>]*>")
BOM = "﻿"


def clean_text(raw: str) -> str:
    s = BR.sub("\n", raw)
    s = LINK.sub(lambda m: m.group(1), s)
    s = TAG.sub("", s)
    s = html.unescape(s)
    s = s.replace(BOM, "")
    return s.strip()


ZERO_WIDTH = re.compile("[​-‏⁠﻿]")
NON_WORD = re.compile(r"[^\w\s]")


def dup_key(raw: str) -> str:
    """Key for "same text" in duplicate and leakage checks: the cleaned text after NFKC,
    zero-width removal, case folding and punctuation removal, with whitespace collapsed."""
    s = unicodedata.normalize("NFKC", clean_text(raw))
    s = ZERO_WIDTH.sub("", s)
    s = NON_WORD.sub(" ", s.casefold())
    return " ".join(s.split())
