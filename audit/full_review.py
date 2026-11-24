"""Full label review of dev and test (decision 27, option A).

Every dev and test spam comment not already reviewed (not an audit flag and not in the
200-comment subtype sample), plus 100 random ham comments from each of dev and test,
labelled blind: the reviewer sees the video and the text, never the current label.
Spam and ham are shuffled together within each split.

    python -m audit.full_review           write audit/full_review.csv
    python -m audit.full_review --check   regenerate and compare with the file on disk

The workbooks for labelling are built from the CSV by `python -m audit.workbook`.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from audit.flags import AUDIT, SEED, read
from data.clean import clean_text
from data.load import load_comments

HAM_PER_SPLIT = 100
PREFIX = {"dev": "D", "test": "T"}
FILE = "full_review.csv"


def build() -> pd.DataFrame:
    df = load_comments()
    reviewed = set(read("flags.csv").row_id) | set(read("subtype_sample.csv").row_id)
    # own stream, so this draw never shifts the S1 audit files
    rng = np.random.default_rng([SEED, 27])
    parts = []
    for split in ["dev", "test"]:
        sub = df[(df.split == split) & ~df.row_id.isin(reviewed)].reset_index(drop=True)
        spam = sub[sub.label == 1]
        ham = sub[sub.label == 0].reset_index(drop=True)
        pick = np.sort(rng.choice(len(ham), size=HAM_PER_SPLIT, replace=False))
        both = pd.concat([spam, ham.iloc[pick]]).reset_index(drop=True)
        both = both.iloc[rng.permutation(len(both))].reset_index(drop=True)
        parts.append(pd.DataFrame({
            "review_id": [f"{PREFIX[split]}{i + 1:03d}" for i in range(len(both))],
            "row_id": both.row_id,
            "split": split,
            "text": both.content.map(clean_text),
        }))
    return pd.concat(parts).reset_index(drop=True)


def write(t: pd.DataFrame) -> None:
    t.to_csv(AUDIT / FILE, index=False, encoding="utf-8-sig", lineterminator="\n")


def check(t: pd.DataFrame) -> int:
    same = read(FILE).equals(t.astype(str))
    print(("OK      " if same else "DIFFERS ") + f"audit/{FILE} ({len(t)} rows)")
    return 0 if same else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    t = build()
    if args.check:
        return check(t)
    write(t)
    print(f"{len(t)} comments: " + ", ".join(f"{k} {v}" for k, v in t.split.value_counts().sort_index().items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
