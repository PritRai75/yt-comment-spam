"""The two label sets every system is scored on, with each comment's review answer.

One row per comment (all 1,956), indexed by `row_id`:

  split, video        as in `data/load.py`
  label_original      the dataset's label (1 spam, 0 ham)
  label_corrected     the label after `audit/corrections.csv`
  review              the policy answer from the label audit: `scam`, `link`,
                      `self_promotion`, `bulk`, `not_spam`, `unsure`, or "" for a comment
                      nobody reviewed (unsampled ham, most of train)
  scored              False for the comments in `audit/scoring_exclusions.csv`
  share_text          in YouTube's share-text family (`data/share_text.py`)

`review` is taken from the audit record, in this order:
  1. the second pass on an unsure answer (`unsure_resolution.csv`) replaces the first;
  2. otherwise the single answer from the flags (`review_verdicts.csv`: the subtype for
     a spam final label, `not_spam` for a ham one, `unsure`), the subtype sample
     (`subtype_labels.csv`) or the full review (`full_review_labels.csv`).
A comment answered in more than one of these must have the same answer in each; the
loader stops otherwise. Every dev and test comment labelled spam in the dataset was
reviewed once, so on dev and test `review` is filled for every original spam and every
corrected spam.

Recall by subtype is read from `review`: on the corrected labels every scored spam
comment has one of the four subtypes; on the original labels the spam comments the
review judged not spam form their own group, `not_spam`.
"""
import numpy as np
import pandas as pd

from audit.apply_review import final_label
from audit.flags import read
from data.load import load_comments
from data.share_text import is_share_text

SUBTYPES = ["scam", "link", "self_promotion", "bulk"]
ANSWERS = SUBTYPES + ["not_spam", "unsure"]


class LabelError(Exception):
    pass


def _flag_answer(current: str, verdict: str, subtype_if_spam: str) -> str:
    final = final_label(current, verdict)
    if final is None:
        return "unsure"
    return subtype_if_spam if final == "spam" else "not_spam"


def review_answers() -> pd.Series:
    """The final review answer for every reviewed comment, by row_id."""
    first = []
    for r in read("review_verdicts.csv").itertuples():
        first.append((r.row_id, _flag_answer(r.current_label, r.verdict, r.subtype_if_spam), f"flag {r.flag_id}"))
    for r in read("subtype_labels.csv").itertuples():
        first.append((r.row_id, r.subtype, f"sample {r.sample_id}"))
    for r in read("full_review_labels.csv").itertuples():
        first.append((r.row_id, r.subtype, f"full review {r.review_id}"))
    second = {r.row_id: (r.answer, r.unsure_id) for r in read("unsure_resolution.csv").itertuples()}

    out, seen = {}, {}
    for row_id, answer, where in first:
        if answer not in ANSWERS:
            raise LabelError(f"{row_id}: {where} has answer {answer!r}")
        if row_id in out and out[row_id] != answer:
            raise LabelError(f"{row_id}: {where} says {answer}, {seen[row_id]} says {out[row_id]}")
        out[row_id], seen[row_id] = answer, where
    for row_id, (answer, unsure_id) in second.items():
        if out.get(row_id) != "unsure":
            raise LabelError(f"{row_id}: second pass {unsure_id} on a comment not answered unsure")
        if answer not in ANSWERS:
            raise LabelError(f"{row_id}: second pass {unsure_id} has answer {answer!r}")
        out[row_id] = answer
    return pd.Series(out, name="review", dtype=str).sort_index()


def label_table() -> pd.DataFrame:
    df = load_comments().set_index("row_id")
    out = df[["split", "video"]].copy()
    out["label_original"] = df.label.astype(int)
    corrected = out.label_original.copy()
    corr = read("corrections.csv")
    for r in corr.itertuples():
        if out.label_original[r.row_id] != int(r.label_before == "spam"):
            raise LabelError(f"{r.row_id}: corrections.csv label_before disagrees with the data")
        corrected[r.row_id] = int(r.label_after == "spam")
    out["label_corrected"] = corrected
    review = review_answers()
    unknown = sorted(set(review.index) - set(out.index))
    if unknown:
        raise LabelError(f"review answers for unknown rows: {unknown[:5]}")
    out["review"] = review.reindex(out.index).fillna("")
    out["scored"] = ~out.index.isin(read("scoring_exclusions.csv").row_id)
    out["share_text"] = df.content.map(is_share_text).astype(bool)
    return out


def scored(table: pd.DataFrame, splits, drop_share_text: bool = False) -> pd.DataFrame:
    """The rows a system is scored on: the given splits, without the scoring exclusions,
    and without the share-text family for the pre-registered second result."""
    if isinstance(splits, str):
        splits = [splits]
    keep = table.split.isin(splits) & table.scored
    if drop_share_text:
        keep &= ~table.share_text
    return table[keep]


def labels(table: pd.DataFrame, which: str) -> np.ndarray:
    if which not in ("original", "corrected"):
        raise ValueError("which must be 'original' or 'corrected'")
    return table[f"label_{which}"].to_numpy(dtype=int)
