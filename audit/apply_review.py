"""Read the filled labelling workbooks back, check them, and write the audit record.

    python -m audit.apply_review --audit A.xlsx --subtypes S.xlsx
        [--full-dev D.xlsx --full-test T.xlsx] [--recheck R.xlsx] [--check]

Checks, before anything is written (any failure stops with a list of problems):
  - every sheet has the rows and columns it was built with, in the same order, and the
    given columns (IDs, video, text, current label) are unchanged;
  - every answer is filled in and is one of the drop-down values;
  - audit: a spam final label (keep on spam, correct on ham) or an unsure needs a
    subtype; a ham final label has none; every verdict has a reason;
  - subtype sample and full review: an answer that disagrees with the current label,
    and every unsure, needs a note;
  - the comments in both S1 sheets agree (spam verdict -> same subtype, ham -> not_spam,
    unsure -> unsure).

Writes, in audit/:
  review_verdicts.csv      the 63 audit answers
  subtype_labels.csv       the 200 subtype answers
  full_review_labels.csv   the option A answers (with --full-dev and --full-test)
  recheck_verdicts.csv     the blind re-check answers beside the first ones (with --recheck)
  corrections.csv          one row per comment whose label the review changes
  scoring_exclusions.csv   comments left out of scoring (decision 28)

An unsure answer never changes a label. With --check nothing is written; the fresh
tables are compared with the files on disk.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from audit import workbook
from audit.flags import AUDIT, read
from data.load import load_comments

SPAM_SUBTYPES = workbook.SUBTYPES
EXCLUSIONS = pd.DataFrame({
    "row_id": ["Eminem:269"],
    "reason": ["source quoting error merged six comments into one row (decision 28)"],
})
OUTPUTS = ["review_verdicts.csv", "subtype_labels.csv", "full_review_labels.csv",
           "recheck_verdicts.csv", "corrections.csv", "scoring_exclusions.csv"]


class ReviewError(Exception):
    def __init__(self, problems):
        super().__init__("\n".join(problems))
        self.problems = problems


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """All cells as stripped text; fully empty rows (Excel formatting leftovers) dropped."""
    ws = load_workbook(path, read_only=True)[sheet]
    rows = [["" if v is None else str(v) for v in r] for r in ws.iter_rows(values_only=True)]
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    df = pd.DataFrame(body, columns=header)
    return df


def _given(df: pd.DataFrame, expected: pd.DataFrame, given: list, name: str) -> list:
    if list(df.columns) != list(expected.columns):
        return [f"{name}: columns are {list(df.columns)}, expected {list(expected.columns)}"]
    if len(df) != len(expected):
        return [f"{name}: {len(df)} rows, expected {len(expected)}"]
    bad = []
    for col in given:
        diff = df[col].values != expected[col].astype(str).values
        for i in diff.nonzero()[0][:5]:
            bad.append(f"{name} row {i + 2}: {col} changed")
    return bad


def _answers(df, id_col, col, allowed, name, required=True) -> list:
    bad = []
    for i, v in zip(df[id_col], df[col].str.strip()):
        if v == "" and required:
            bad.append(f"{name} {i}: {col} is blank")
        elif v != "" and v not in allowed:
            bad.append(f"{name} {i}: {col} '{v}' is not one of {allowed}")
    return bad


def audit_answers(path: Path) -> tuple:
    df = read_sheet(path, "review")
    exp = workbook.review_table()
    bad = _given(df, exp, ["flag_id", "row_id", "video", "text", "current_label", "subtype_sample_id"], "audit")
    if bad:
        return df, bad
    for c in ["verdict", "subtype_if_spam", "reason"]:
        df[c] = df[c].str.strip()
    bad += _answers(df, "flag_id", "verdict", workbook.VERDICTS, "audit")
    bad += _answers(df, "flag_id", "subtype_if_spam", SPAM_SUBTYPES, "audit", required=False)
    for r in df.itertuples():
        final = final_label(r.current_label, r.verdict)
        if (final == "spam" or r.verdict == "unsure") and r.subtype_if_spam == "":
            bad.append(f"audit {r.flag_id}: final label {final or 'unsure'} needs a subtype")
        if final == "ham" and r.subtype_if_spam != "":
            bad.append(f"audit {r.flag_id}: final label ham but subtype '{r.subtype_if_spam}'")
        if r.reason == "":
            bad.append(f"audit {r.flag_id}: reason is blank")
    return df, bad


def final_label(current: str, verdict: str):
    """The label after an audit verdict; None for unsure."""
    if verdict == "keep":
        return current
    if verdict == "correct":
        return "ham" if current == "spam" else "spam"
    return None


def _labelled(df, id_col, answer, notes, current, name) -> list:
    """Checks shared by sheets answered with a subtype or not_spam."""
    df[answer] = df[answer].str.strip()
    bad = _answers(df, id_col, answer, workbook.SUBTYPE_SHEET_CHOICES, name)
    for i, a, n, cur in zip(df[id_col], df[answer], df[notes].str.strip(), current):
        says = {"not_spam": "ham", "unsure": None}.get(a, "spam")
        if (a == "unsure" or says != cur) and n == "":
            bad.append(f"{name} {i}: '{a}' on a {cur} comment needs a note")
    return bad


def subtype_answers(path: Path) -> tuple:
    df = read_sheet(path, "subtypes")
    exp = workbook.subtype_table()
    bad = _given(df, exp, ["sample_id", "row_id", "video", "text", "flag_id"], "subtypes")
    if bad:
        return df, bad
    return df, _labelled(df, "sample_id", "subtype", "notes", ["spam"] * len(df), "subtypes")


def full_answers(dev: Path, test: Path) -> tuple:
    current = load_comments().set_index("row_id").label.map({1: "spam", 0: "ham"})
    frames, bad = [], []
    for split, path in [("dev", dev), ("test", test)]:
        df = read_sheet(path, "review")
        exp = workbook.full_tables()[split]
        b = _given(df, exp, ["review_id", "row_id", "video", "text"], f"review_{split}")
        if not b:
            b = _labelled(df, "review_id", "subtype", "notes", list(df.row_id.map(current)), f"review_{split}")
        bad += b
        frames.append(df.assign(split=split))
    return pd.concat(frames).reset_index(drop=True), bad


def recheck_answers(path: Path) -> tuple:
    df = read_sheet(path, "recheck")
    exp = workbook.recheck_table()
    bad = _given(df, exp, ["recheck_id", "video", "text", "current_label"], "recheck")
    if bad:
        return df, bad
    for c in ["verdict", "subtype_if_spam", "reason"]:
        df[c] = df[c].str.strip()
    bad += _answers(df, "recheck_id", "verdict", workbook.VERDICTS, "recheck")
    bad += _answers(df, "recheck_id", "subtype_if_spam", SPAM_SUBTYPES, "recheck", required=False)
    return df, bad


def cross_check(audit: pd.DataFrame, subtypes: pd.DataFrame) -> list:
    sub = subtypes.set_index("sample_id").subtype
    bad = []
    for r in audit[audit.subtype_sample_id != ""].itertuples():
        final = final_label(r.current_label, r.verdict)
        want = {"spam": r.subtype_if_spam, "ham": "not_spam", None: "unsure"}[final]
        if sub[r.subtype_sample_id] != want:
            bad.append(f"{r.flag_id} / {r.subtype_sample_id}: audit says {want}, subtypes say {sub[r.subtype_sample_id]}")
    return bad


def corrections(audit, subtypes, full=None) -> pd.DataFrame:
    df = load_comments().set_index("row_id")
    found = {}  # row_id -> (label_after, subtype_after, [(source, id, reason)])

    def add(row_id, after, subtype, source, sid, reason):
        if row_id in found and found[row_id][:2] != (after, subtype):
            raise ReviewError([f"{row_id}: {source} {sid} disagrees with an earlier correction"])
        prev = found.get(row_id, (after, subtype, []))[2]
        found[row_id] = (after, subtype, prev + [(source, sid, reason)])

    for r in audit.itertuples():
        final = final_label(r.current_label, r.verdict)
        if final is not None and final != r.current_label:
            add(r.row_id, final, r.subtype_if_spam if final == "spam" else "", "audit_flag", r.flag_id, r.reason)
    for r in subtypes.itertuples():
        if r.subtype == "not_spam":
            add(r.row_id, "ham", "", "subtype_sample", r.sample_id, r.notes)
    if full is not None:
        for r in full.itertuples():
            before = "spam" if df.label[r.row_id] == 1 else "ham"
            if r.subtype == "unsure":
                continue
            after = "ham" if r.subtype == "not_spam" else "spam"
            if after != before:
                add(r.row_id, after, r.subtype if after == "spam" else "", "full_review", r.review_id, r.notes)

    rows = []
    for row_id in sorted(found, key=lambda x: (df.split[x], x)):
        after, subtype, src = found[row_id]
        rows.append({
            "row_id": row_id,
            "split": df.split[row_id],
            "video": df.video[row_id],
            "label_before": "spam" if df.label[row_id] == 1 else "ham",
            "label_after": after,
            "subtype_after": subtype,
            "source": ";".join(s for s, _, _ in src),
            "source_id": ";".join(i for _, i, _ in src),
            "reason": src[0][2],
        })
    return pd.DataFrame(rows, columns=["row_id", "split", "video", "label_before", "label_after",
                                       "subtype_after", "source", "source_id", "reason"])


def recheck_table(recheck: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    first = audit.set_index("flag_id")
    ids = workbook.recheck_ids()
    out = pd.DataFrame({
        "recheck_id": recheck.recheck_id,
        "flag_id": recheck.recheck_id.map(ids),
        "verdict": recheck.verdict,
        "subtype_if_spam": recheck.subtype_if_spam,
        "reason": recheck.reason,
    })
    out["first_verdict"] = out.flag_id.map(first.verdict)
    out["first_subtype_if_spam"] = out.flag_id.map(first.subtype_if_spam)
    cur = out.flag_id.map(first.current_label)
    out["same_label"] = [str(final_label(c, a) == final_label(c, b)).lower()
                         for c, a, b in zip(cur, out.verdict, out.first_verdict)]
    return out


def build(args) -> dict:
    audit, bad = audit_answers(args.audit)
    subtypes, b = subtype_answers(args.subtypes)
    bad += b
    full = None
    if args.full_dev or args.full_test:
        if not (args.full_dev and args.full_test):
            bad.append("give both --full-dev and --full-test")
        else:
            full, b = full_answers(args.full_dev, args.full_test)
            bad += b
    recheck = None
    if args.recheck:
        recheck, b = recheck_answers(args.recheck)
        bad += b
    if not bad:
        bad += cross_check(audit, subtypes)
    if bad:
        raise ReviewError(bad)

    tables = {
        "review_verdicts.csv": audit[["flag_id", "row_id", "current_label", "verdict", "subtype_if_spam", "reason"]],
        "subtype_labels.csv": subtypes[["sample_id", "row_id", "subtype", "notes"]],
        "corrections.csv": corrections(audit, subtypes, full),
        "scoring_exclusions.csv": EXCLUSIONS,
    }
    if full is not None:
        tables["full_review_labels.csv"] = full[["review_id", "row_id", "split", "subtype", "notes"]]
    if recheck is not None:
        tables["recheck_verdicts.csv"] = recheck_table(recheck, audit)
    return tables


def summary(tables: dict) -> list:
    c = tables["corrections.csv"]
    moves = c.groupby(["label_before", "label_after"]).size()
    lines = [f"corrections: {len(c)} (" + ", ".join(f"{b}->{a} {n}" for (b, a), n in moves.items()) + ")",
             "  by split: " + ", ".join(f"{k} {v}" for k, v in c.split.value_counts().sort_index().items()),
             "audit verdicts: " + ", ".join(f"{k} {v}" for k, v in
                                            tables["review_verdicts.csv"].verdict.value_counts().items()),
             "subtypes: " + ", ".join(f"{k} {v}" for k, v in
                                      tables["subtype_labels.csv"].subtype.value_counts().items())]
    if "full_review_labels.csv" in tables:
        f = tables["full_review_labels.csv"]
        lines.append("full review: " + ", ".join(f"{k} {v}" for k, v in f.subtype.value_counts().items()))
    if "recheck_verdicts.csv" in tables:
        r = tables["recheck_verdicts.csv"]
        lines.append(f"blind re-check: same final label on {(r.same_label == 'true').sum()} of {len(r)}")
    return lines


def write(tables: dict) -> None:
    for name, t in tables.items():
        t.to_csv(AUDIT / name, index=False, encoding="utf-8-sig", lineterminator="\n")


def check(tables: dict) -> int:
    bad = 0
    for name, t in tables.items():
        same = (AUDIT / name).exists() and read(name).equals(t.astype(str).reset_index(drop=True))
        print(("OK      " if same else "DIFFERS ") + f"audit/{name} ({len(t)} rows)")
        bad += not same
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--subtypes", type=Path, required=True)
    ap.add_argument("--full-dev", type=Path)
    ap.add_argument("--full-test", type=Path)
    ap.add_argument("--recheck", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    try:
        tables = build(args)
    except ReviewError as e:
        print(f"{len(e.problems)} problem(s); nothing written:", file=sys.stderr)
        for p in e.problems:
            print("  " + p, file=sys.stderr)
        return 1
    if args.check:
        return check(tables)
    write(tables)
    for line in summary(tables):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
