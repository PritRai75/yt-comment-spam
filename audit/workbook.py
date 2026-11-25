"""Excel working copies for hand labelling.

Every cell is written as text (so "+Name" or "=..." is never read as a formula), long
text wraps, and the answer columns have drop-down lists. Each row shows the video the
comment was posted under, and each workbook has a "videos" sheet with the five videos'
YouTube IDs, so a link can be checked against "this video" without leaving the sheet.
Model scores are never shown.

Sets:
  s1       audit_review.xlsx (63 flags) and subtype_labels.xlsx (200 test spam); comments
           in both carry the other sheet's ID
  full     review_dev.xlsx and review_test.xlsx (option A, audit/full_review.csv): current
           label hidden, spam and ham shuffled together
  recheck  blind_recheck.xlsx: the 20 flags in audit/blind_recheck.csv in a new order,
           with new IDs (R01-R20) and no first answers
  unsure   unsure.xlsx: a second pass on every unsure answer (audit flags and full review),
           current label hidden, first-pass note shown

The CSVs in audit/ stay the record; the workbooks are filled in by hand and read back by
audit/apply_review.py. An existing workbook is never overwritten.

    python -m audit.workbook <folder> --set {s1,full,recheck,unsure}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from audit.flags import SEED, read
from data.fetch import ROOT

VERDICTS = ["keep", "correct", "unsure"]
SUBTYPES = ["scam", "link", "self_promotion", "bulk"]
SUBTYPE_SHEET_CHOICES = SUBTYPES + ["not_spam", "unsure"]
VIDEOS = json.loads((ROOT / "data" / "videos.json").read_text(encoding="utf-8"))["videos"]


def video_label(video: str) -> str:
    v = VIDEOS[video]
    return f"{v['artist']} – {v['song']}"


def review_table() -> pd.DataFrame:
    sheet, flags, sample = read("review_sheet.csv"), read("flags.csv"), read("subtype_sample.csv")
    row_of = dict(zip(flags.flag_id, flags.row_id))
    video_of = dict(zip(flags.flag_id, flags.video))
    sample_of = dict(zip(sample.row_id, sample.sample_id))
    return pd.DataFrame({
        "flag_id": sheet.flag_id,
        "row_id": sheet.flag_id.map(row_of),
        "video": sheet.flag_id.map(lambda f: video_label(video_of[f])),
        "text": sheet.text,
        "current_label": sheet.current_label,
        "verdict": sheet.verdict,
        "subtype_if_spam": sheet.subtype_if_spam,
        "reason": sheet.reason,
        "subtype_sample_id": sheet.flag_id.map(lambda f: sample_of.get(row_of[f], "")),
    })


def subtype_table() -> pd.DataFrame:
    sample, flags = read("subtype_sample.csv"), read("flags.csv")
    flag_of = dict(zip(flags.row_id, flags.flag_id))
    return pd.DataFrame({
        "sample_id": sample.sample_id,
        "row_id": sample.row_id,
        "video": sample.row_id.map(lambda r: video_label(r.split(":")[0])),
        "text": sample.text,
        "subtype": sample.subtype,
        "notes": sample.notes,
        "flag_id": sample.row_id.map(lambda r: flag_of.get(r, "")),
    })


def full_tables() -> dict:
    """Option A review, one table per split; the current label is not a column."""
    t = read("full_review.csv")
    out = {}
    for split in ["dev", "test"]:
        sub = t[t.split == split]
        out[split] = pd.DataFrame({
            "review_id": sub.review_id.values,
            "row_id": sub.row_id.values,
            "video": [video_label(r.split(":")[0]) for r in sub.row_id],
            "text": sub.text.values,
            "subtype": "",
            "notes": "",
        })
    return out


def recheck_table() -> pd.DataFrame:
    """The blind re-check flags, re-ordered, under new IDs; R-id to flag_id is recheck_ids()."""
    review = review_table().set_index("flag_id")
    ids = recheck_ids()
    return pd.DataFrame({
        "recheck_id": list(ids),
        "video": [review.video[f] for f in ids.values()],
        "text": [review.text[f] for f in ids.values()],
        "current_label": [review.current_label[f] for f in ids.values()],
        "verdict": "",
        "subtype_if_spam": "",
        "reason": "",
    })


def recheck_ids() -> dict:
    flags = list(read("blind_recheck.csv").flag_id)
    order = np.random.default_rng([SEED, 6]).permutation(len(flags))
    return {f"R{i + 1:02d}": flags[j] for i, j in enumerate(order)}


def unsure_table() -> pd.DataFrame:
    """Second pass on every unsure answer (audit flags, then the full review), blind to the
    current label, with the first-pass note beside it. The first-pass record is unchanged."""
    review = review_table().set_index("flag_id")
    verdicts = read("review_verdicts.csv")
    flags = verdicts[verdicts.verdict == "unsure"]
    full = read("full_review_labels.csv")
    full = full[full.subtype == "unsure"]
    text_of = dict(zip(read("full_review.csv").review_id, read("full_review.csv").text))
    rows = [(f, r, review.text[f], n) for f, r, n in zip(flags.flag_id, flags.row_id, flags.reason)]
    rows += [(i, r, text_of[i], n) for i, r, n in zip(full.review_id, full.row_id, full.notes)]
    return pd.DataFrame({
        "unsure_id": [f"U{i + 1:02d}" for i in range(len(rows))],
        "source_id": [r[0] for r in rows],
        "row_id": [r[1] for r in rows],
        "video": [video_label(r[1].split(":")[0]) for r in rows],
        "text": [r[2] for r in rows],
        "first_note": [r[3] for r in rows],
        "answer": "",
        "note": "",
    })


def _save(wb, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists; a filled workbook is never overwritten")
    wb.save(path)
    return path


def _prepare(out_dir: Path, names: list) -> None:
    """Create the folder, and refuse before writing anything if any target exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    taken = [n for n in names if (out_dir / n).exists()]
    if taken:
        raise FileExistsError(f"{', '.join(taken)} already in {out_dir}; a filled workbook is never overwritten")


def _sheet(ws, df, widths, choices):
    ws.append(list(df.columns))
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in df.itertuples(index=False):
        ws.append([str(v) for v in row])
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.data_type = "s"
            c.alignment = Alignment(wrap_text=True, vertical="top")
    last = ws.max_row
    for col, options in choices.items():
        dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")
    ws.freeze_panes = "A2"


def _videos_sheet(wb):
    ws = wb.create_sheet("videos")
    ws.append(["video", "artist", "song", "youtube_id", "links that count as this video"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for key, v in VIDEOS.items():
        ws.append([key, v["artist"], v["song"], v["youtube_id"],
                   f"youtube.com/watch?v={v['youtube_id']} or youtu.be/{v['youtube_id']} (any timestamp)"])
    for col, w in {"A": 11, "B": 12, "C": 44, "D": 14, "E": 62}.items():
        ws.column_dimensions[col].width = w


def _book(title, df, widths, choices, path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = title
    _sheet(ws, df, widths, choices)
    _videos_sheet(wb)
    return _save(wb, path)


def build(out_dir: Path) -> list:
    """Set s1: the two session 1 workbooks."""
    _prepare(out_dir, ["audit_review.xlsx", "subtype_labels.xlsx"])
    return [
        _book("review", review_table(),
              {"A": 8, "B": 12, "C": 24, "D": 60, "E": 13, "F": 10, "G": 16, "H": 40, "I": 11},
              {"F": VERDICTS, "G": SUBTYPES}, out_dir / "audit_review.xlsx"),
        _book("subtypes", subtype_table(),
              {"A": 9, "B": 12, "C": 24, "D": 60, "E": 16, "F": 40, "G": 8},
              {"E": SUBTYPE_SHEET_CHOICES}, out_dir / "subtype_labels.xlsx"),
    ]


def build_full(out_dir: Path) -> list:
    _prepare(out_dir, ["review_dev.xlsx", "review_test.xlsx"])
    return [_book("review", t, {"A": 10, "B": 12, "C": 24, "D": 60, "E": 16, "F": 40},
                  {"E": SUBTYPE_SHEET_CHOICES}, out_dir / f"review_{split}.xlsx")
            for split, t in full_tables().items()]


def build_recheck(out_dir: Path) -> list:
    _prepare(out_dir, ["blind_recheck.xlsx"])
    return [_book("recheck", recheck_table(),
                  {"A": 10, "B": 24, "C": 60, "D": 13, "E": 10, "F": 16, "G": 40},
                  {"E": VERDICTS, "F": SUBTYPES}, out_dir / "blind_recheck.xlsx")]


def build_unsure(out_dir: Path) -> list:
    _prepare(out_dir, ["unsure.xlsx"])
    return [_book("unsure", unsure_table(),
                  {"A": 9, "B": 9, "C": 14, "D": 24, "E": 60, "F": 45, "G": 16, "H": 40},
                  {"G": SUBTYPE_SHEET_CHOICES}, out_dir / "unsure.xlsx")]


SETS = {"s1": build, "full": build_full, "recheck": build_recheck, "unsure": build_unsure}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--set", required=True, choices=list(SETS), dest="which")
    args = ap.parse_args(argv)
    try:
        paths = SETS[args.which](args.out_dir)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        return 1
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
