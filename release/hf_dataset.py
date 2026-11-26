"""The audited labels as a Hugging Face dataset: every comment with its original and
corrected label, the policy answer from the review and the reason for each correction.

Writes release/huggingface/ (uploaded as is to the Hugging Face Hub):

  data/train.csv, data/validation.csv, data/test.csv
      One row per comment of the YouTube Spam Collection, split by video as in this
      repo (validation = the dev video). Columns in COLUMNS. Author names and dates are
      left out; `comment_id` joins back to the original files for them.
  README.md
      The dataset card. Every number in it is computed here from the audit record.

    python -m release.hf_dataset            # write the files
    python -m release.hf_dataset --check    # compare with the committed files

The files come from `data/load.py` (the pinned UCI zip) and `eval/labels.py` (the label
audit), so they change only if the audit record changes.
"""
import argparse
import sys

from audit.flags import read
from data.fetch import ROOT
from data.load import load_comments
from eval.labels import label_table
from eval.metrics import wilson

OUT = ROOT / "release" / "huggingface"
REPO_ID = "PritRai75/youtube-spam-collection-audited"
GITHUB = "https://github.com/PritRai75/yt-comment-spam"
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
COLUMNS = ["row_id", "comment_id", "video", "split", "content", "label_original", "label_corrected",
           "corrected", "review", "correction_source", "correction_reason", "share_text", "scored"]


def table():
    df = load_comments().set_index("row_id")
    t = label_table()
    corr = read("corrections.csv").set_index("row_id")
    out = df[["comment_id", "video", "split", "content"]].copy()
    out["label_original"] = t.label_original
    out["label_corrected"] = t.label_corrected
    out["corrected"] = out.label_original != out.label_corrected
    out["review"] = t.review
    out["correction_source"] = corr.source.reindex(out.index).fillna("")
    out["correction_reason"] = corr.reason.reindex(out.index).fillna("")
    out["share_text"] = t.share_text
    out["scored"] = t.scored
    if set(corr.index) != set(out.index[out.corrected]):
        raise ValueError("corrections.csv and the corrected labels disagree")
    return out.reset_index()[COLUMNS]


def csv_text(df) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def _n(x) -> str:
    return f"{x:,}"


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def card(t) -> str:
    by = t.groupby("split")
    split_rows = []
    for s, name in SPLITS.items():
        g = t[t.split == s]
        split_rows.append(f"| {name} | {', '.join(sorted(g.video.unique()))} | {_n(len(g))} | "
                          f"{_n(int(g.label_original.sum()))} | {_n(int(g.label_corrected.sum()))} | "
                          f"{_n(int(g.corrected.sum()))} |")
    c = t[t.corrected]
    s2h, h2s = int((c.label_original == 1).sum()), int((c.label_original == 0).sum())
    test_spam = t[(t.split == "test") & (t.label_original == 1)]
    k, n = int((test_spam.label_corrected == 0).sum()), len(test_spam)
    lo, hi = wilson(k, n)
    reviewed = t[(t.review != "") & t.split.isin(["dev", "test"])]
    ham_rev = reviewed[reviewed.label_original == 0]
    rc = read("recheck_verdicts.csv")
    same = int((rc.same_label.astype(str).str.lower() == "true").sum())
    subtypes = t[(t.label_corrected == 1) & t.split.isin(["dev", "test"]) & t.scored].review.value_counts()
    sources = c.correction_source.str.split(";").explode().value_counts()
    cited = c.correction_reason.str.extractall(r"\b(EC\d+|P\d+)\b")[0].groupby(level=0).unique().explode().value_counts()
    ec3 = int(cited.get("EC3", 0))
    top = ", ".join(f"{k} {v}" for k, v in cited.head(6).items())
    first_unsure = int(((rc.same_label.astype(str).str.lower() != "true") & (rc.first_verdict == "unsure")).sum())
    excluded = t[~t.scored]
    return f"""---
license: cc-by-4.0
language:
- en
pretty_name: YouTube Spam Collection, audited labels
task_categories:
- text-classification
size_categories:
- 1K<n<10K
tags:
- spam
- youtube
- content-moderation
- label-noise
- data-quality
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
  - split: validation
    path: data/validation.csv
  - split: test
    path: data/test.csv
---

# YouTube Spam Collection, audited labels

The {_n(len(t))} comments of the [YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/) (Alberto & Lochter, UCI) with their original spam/ham labels, a corrected label for each, the answer a hand review under a written spam policy gave, and the reason for every one of the **{_n(len(c))} corrections** ({_n(s2h)} spam → ham, {_n(h2s)} ham → spam).

The audit was done for the project [yt-comment-spam]({GITHUB}), which compares keyword rules, two classical classifiers and an LLM on these comments. The policy, the review sheets, the code that builds this dataset and the tests that check it are all in that repository.

## Why corrected labels

Under the project's policy (`policy/spam_policy.md` in the repository), {k} of the {n} test comments labelled spam are not spam ({_pct(k / n)}, 95% Wilson interval {_pct(lo)}–{_pct(hi)}). A spam detector scored on the original labels is marked wrong for calling those comments ham.

The rule cited most in the correction reasons is edge case 3: asking for likes on the comment itself ("1 like = 1 prayer") is ham, since it promotes nothing outside the thread ({ec3} of the {_n(len(c))} reasons). Rules cited, by number of corrections: {top} (see the policy).

The corrected labels follow one policy, written for this project. They are not ground truth for other definitions of spam: keep `label_original` beside them, and read `correction_reason` before relying on a correction.

## Splits

Split by video, so a model is evaluated on videos it never saw. `validation` is the project's dev split.

| Split | Videos | Comments | Spam (original) | Spam (corrected) | Corrections |
|---|---|---:|---:|---:|---:|
{chr(10).join(split_rows)}

## Columns

| Column | Meaning |
|---|---|
| `row_id` | `<video>:<row>`, the row's 0-based position in its CSV as published; unique (`comment_id` is not: three rows appear twice) |
| `comment_id` | `COMMENT_ID` from the original files; join on it (with the video) for author and date, left out here |
| `video` | Psy, KatyPerry, LMFAO, Eminem or Shakira |
| `split` | train, dev or test, as in the project (dev is this dataset's `validation`) |
| `content` | The comment exactly as published, HTML and invisible characters included |
| `label_original` | 1 spam, 0 ham, as published |
| `label_corrected` | 1 spam, 0 ham, after the audit |
| `corrected` | `label_corrected` differs from `label_original` |
| `review` | The reviewer's answer under the policy: `scam`, `link`, `self_promotion`, `bulk`, `not_spam`, `unsure`, or empty if the comment was not reviewed |
| `correction_source` | Which part of the audit led to the correction (below); empty if not corrected |
| `correction_reason` | The reviewer's reason, citing the policy's edge cases (`EC…`) and rules (`P…`) |
| `share_text` | The comment is in YouTube's "Check out this video on YouTube" share-text family |
| `scored` | False for the {_n(len(excluded))} row(s) the project leaves out of scoring ({", ".join(excluded.row_id)}: one source row holding several comments) |

## How the labels were audited

One reviewer, against the written policy, before any LLM was run on the comments, in three parts:

1. **Flags:** comments where a keyword-rule baseline and a logistic-regression baseline both disagreed with the label.
2. **Subtype sample:** 200 test comments labelled spam, drawn at random, each given a subtype or `not_spam`.
3. **Full review:** every remaining dev and test comment labelled spam, plus 100 random ham comments from each of dev and test, with the label hidden.

Every dev and test comment labelled spam was reviewed; {_n(len(reviewed))} dev and test comments were reviewed in all. Of the {_n(len(ham_rev))} reviewed ham, {_n(int((ham_rev.label_corrected == 1).sum()))} was judged spam. Unsure answers had a second pass under the final policy. Corrections by source (a correction can have two): {", ".join(f"{k} {v}" for k, v in sources.items())}.

Spam subtypes on the corrected labels of the scored dev and test comments: {", ".join(f"{k} {v}" for k, v in subtypes.items())}.

**Consistency:** a blind re-check of {len(rc)} flags two days after the first review gave the same label as the first answers on {same} of {len(rc)}; {first_unsure} of the other {len(rc) - same} had been answered unsure the first time. This is one reviewer's consistency, not agreement between reviewers.

## Limitations

- One reviewer, who also wrote the policy. No second labeller.
- Train comments were not reviewed apart from the flags, so train keeps almost all of its original labels.
- Ham was sampled (100 per split), not reviewed in full, so spam mislabelled as ham is estimated, not corrected everywhere.
- Five music videos from 2013–2015, English only; today's spam looks different.
- The policy makes choices other definitions would not, for example likes asked for the comment itself are ham, and YouTube's share text is spam (bulk).

## Licence and credit

CC BY 4.0, as the original dataset. Original data: Alberto, T. & Lochter, J. (2015). YouTube Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58885. The comment text here is unmodified.

The corrected labels, review answers and reasons: Prit Raithatha, {GITHUB}.
"""


def build() -> dict:
    t = table()
    files = {f"data/{name}.csv": csv_text(t[t.split == s]) for s, name in SPLITS.items()}
    files["README.md"] = card(t)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    args = ap.parse_args(argv)
    bad = 0
    for rel, text in build().items():
        path = OUT / rel
        if args.check:
            same = path.exists() and path.read_bytes() == text.encode("utf-8")
            print(f"{'OK' if same else 'DIFFERS'}: {path.relative_to(ROOT).as_posix()}")
            bad += not same
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
