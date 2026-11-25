"""Few-shot examples for LLM v2 (decision 9, with the policy check of decision 44).

Pool: train comments that are not audit flags, whose text does not equal any dev or
test text under `data.clean.dup_key`, and that share no run of 4 words with any dev or
test comment. Words are case-folded word tokens of the text after `clean_text` and
`normalize`, the form the prompt shows.

Order: the pool's spam and ham, each sorted by row_id and shuffled with the seed stream
[20261125, 9]. The reviewer answers the candidates under the policy without seeing the
dataset label (`candidates` mixes the first 6 of each in a seeded order); the answers are
kept in `models/prompts/v2_examples.csv` (candidate, row_id, video, answer). The examples
are the first 4 spam and the first 4 ham whose dataset label the answer agrees with; a
spam example shows the answer's subtype. In the prompt the 8 appear in a seeded order.

Reviewed 25 Nov 2026: 12 candidates, all 12 answers agree with the dataset label.

    python -m models.fewshot candidates   # print the candidates for review
"""
import re
import sys

import numpy as np
import pandas as pd

from audit.flags import read
from data.clean import clean_text, dup_key
from data.fetch import ROOT
from data.load import load_comments
from data.normalize import normalize

SEED = [20261125, 9]
N_EACH = 4
N_SHOWN = 6  # candidates of each label shown for review at first
PROMPTS = ROOT / "models" / "prompts"
ANSWERS = PROMPTS / "v2_examples.csv"
SUBTYPES = ["scam", "link", "self_promotion", "bulk"]

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def shown(text: str) -> str:
    return normalize(clean_text(text))


def word_runs(text: str, n: int = 4) -> set:
    w = _WORD.findall(shown(text).casefold())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def pool(df: pd.DataFrame = None) -> pd.DataFrame:
    df = load_comments() if df is None else df
    held = df[df.split != "train"]
    held_runs = set().union(*(word_runs(t) for t in held.content))
    held_keys = {dup_key(clean_text(t)) for t in held.content}
    flags = set(read("flags.csv").row_id)
    train = df[df.split == "train"].copy()
    train["flagged"] = train.row_id.isin(flags)
    train["same_text"] = train.content.map(lambda t: dup_key(clean_text(t)) in held_keys)
    train["shared_run"] = train.content.map(lambda t: bool(word_runs(t) & held_runs))
    return train[~train.flagged & ~train.same_text & ~train.shared_run]


def ordered(p: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for label in (1, 0):  # spam first, so the draw does not depend on how many ham there are
        ids = sorted(p[p.label == label].row_id)
        out[label] = [ids[i] for i in rng.permutation(len(ids))]
    return out


def candidates(n_each: int = N_SHOWN, df: pd.DataFrame = None) -> pd.DataFrame:
    """The first n_each spam and ham in pool order, mixed in a seeded order, labels hidden."""
    df = load_comments() if df is None else df
    order = ordered(pool(df))
    rows = [(rid, 1, k) for k, rid in enumerate(order[1][:n_each])] + [(rid, 0, k) for k, rid in enumerate(order[0][:n_each])]
    rng = np.random.default_rng(SEED + [1])
    rows = [rows[i] for i in rng.permutation(len(rows))]
    by_id = df.set_index("row_id")
    return pd.DataFrame([{"candidate": f"C{i + 1:02d}", "row_id": rid, "video": by_id.video[rid],
                          "text": shown(by_id.content[rid])} for i, (rid, _, _) in enumerate(rows)])


def examples(answers: pd.DataFrame, df: pd.DataFrame = None) -> pd.DataFrame:
    """The 8 examples from the reviewer's answers (row_id, answer), in prompt order."""
    df = load_comments() if df is None else df
    order = ordered(pool(df))
    by_id = df.set_index("row_id")
    got = dict(zip(answers.row_id, answers.answer))
    picked = []
    for label in (1, 0):
        n = 0
        for rid in order[label]:
            if n == N_EACH:
                break
            if rid not in got:
                raise ValueError(f"no answer for {rid}, needed before {N_EACH} {'spam' if label else 'ham'} agree")
            a = got[rid]
            if a not in SUBTYPES + ["not_spam"]:
                raise ValueError(f"{rid}: answer {a!r}")
            if (a != "not_spam") == bool(label):
                picked.append({"row_id": rid, "video": by_id.video[rid], "text": shown(by_id.content[rid]),
                               "label": "spam" if label else "ham", "subtype": a if label else "none"})
                n += 1
        if n < N_EACH:
            raise ValueError(f"only {n} {'spam' if label else 'ham'} examples agree; review more candidates")
    rng = np.random.default_rng(SEED + [2])
    return pd.DataFrame([picked[i] for i in rng.permutation(len(picked))]).reset_index(drop=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["candidates"]:
        print(__doc__)
        return 2
    for r in candidates().itertuples():
        print(f"{r.candidate}  {r.row_id}  [{r.video}]  {r.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
