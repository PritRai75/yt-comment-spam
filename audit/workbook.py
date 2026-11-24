"""Excel working copies of the review sheet and the subtype sample, for hand labelling.

Every cell is written as text (so "+Name" or "=..." is never read as a formula), long
text wraps, and the answer columns have drop-down lists. Each row shows the video the
comment was posted under, and each workbook has a "videos" sheet with the five videos'
YouTube IDs, so a link can be checked against "this video" without leaving the sheet.
Comments in both samples carry the other sheet's ID. Model scores are never shown.

The CSVs in audit/ stay the record; these workbooks are filled in by hand and read back
by a later script.

    python -m audit.workbook <folder>
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from audit.flags import read
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


def build(out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    _sheet(ws, review_table(),
           {"A": 8, "B": 12, "C": 24, "D": 60, "E": 13, "F": 10, "G": 16, "H": 40, "I": 11},
           {"F": VERDICTS, "G": SUBTYPES})
    _videos_sheet(wb)
    p = out_dir / "audit_review.xlsx"
    wb.save(p)
    paths.append(p)

    wb = Workbook()
    ws = wb.active
    ws.title = "subtypes"
    _sheet(ws, subtype_table(),
           {"A": 9, "B": 12, "C": 24, "D": 60, "E": 16, "F": 40, "G": 8},
           {"E": SUBTYPE_SHEET_CHOICES})
    _videos_sheet(wb)
    p = out_dir / "subtype_labels.xlsx"
    wb.save(p)
    paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)
    for p in build(args.out_dir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
