"""Text normalisation used by LLM v2 (and, in the red-team step, in front of every system).

Runs after `data.clean.clean_text`. It handles exactly four things, in this order:

1. Unicode compatibility forms: NFKC, so full-width letters, ligatures, circled and
   "mathematical" letters (U+1D400 block, e.g. bold or script "free") become plain ones;
   no-break spaces become spaces and "…" becomes "...". The acute accent used as an
   apostrophe ("it´s") is made an apostrophe first, since NFKC would split it off.
2. Invisible format characters: every character in Unicode category Cf is removed
   (zero-width space, joiners, word joiner, soft hyphen, byte-order mark, direction marks).
   Emoji joined by a zero-width joiner fall apart into their parts; no word changes.
3. Look-alike letters: Cyrillic and Greek letters that look like Latin ones (the table
   LOOKALIKE) are replaced by the Latin letter, but only inside a word that also has a
   Latin letter. A word written wholly in Cyrillic or Greek is left alone.
4. Spaced-out links: "h t t p : / /" and "w w w" are closed up, and a dot written with
   spaces or as "dot", "(dot)" or "[dot]" between a name and a top-level domain in TLDS
   becomes "." ("bit . ly", "site dot com" -> "bit.ly", "site.com"). Domains that are
   also common words (in, co, me, us, be, tv) are left out of TLDS, so "your choice. in fact"
   stays as it is.

It does not handle, so the red team still measures them: leetspeak ("fr33 g1ft"),
letters spaced out outside links ("f r e e"), look-alikes from other scripts or from
symbols, links split by other characters, a benign sentence wrapped around spam, and
notes addressed to the moderator. Text that needs none of the four comes back unchanged.
"""
import re
import unicodedata

LOOKALIKE = {
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c",
    "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ӏ": "l", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W",
    # Greek
    "α": "a", "ο": "o", "ρ": "p", "ι": "i", "κ": "k", "ν": "v", "υ": "u", "τ": "t",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
TLDS = ["com", "net", "org", "info", "biz", "ly", "tk", "io", "uk"]

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z]")
_HTTP = re.compile(r"\bh\s+t\s+t\s+p(\s*s)?\s*:\s*/\s*/", re.IGNORECASE)
_WWW = re.compile(r"\bw\s+w\s+w\b", re.IGNORECASE)
_DOT = re.compile(
    r"\b([a-z0-9-]+)(?:\s+\.\s*|\s*\.\s+|\s+dot\s+|\s*[(\[]\s*dot\s*[)\]]\s*)(" + "|".join(TLDS) + r")\b",
    re.IGNORECASE,
)


def _fix_lookalikes(word: str) -> str:
    if not _LATIN.search(word) or not any(ch in LOOKALIKE for ch in word):
        return word
    return "".join(LOOKALIKE.get(ch, ch) for ch in word)


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text.replace("\u00b4", "'"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = _WORD.sub(lambda m: _fix_lookalikes(m.group(0)), s)
    s = _HTTP.sub(lambda m: "https://" if m.group(1) else "http://", s)
    s = _WWW.sub("www", s)
    s = _DOT.sub(lambda m: f"{m.group(1)}.{m.group(2)}", s)
    return s
