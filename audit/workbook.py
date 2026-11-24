"""Excel working copies of the review sheet and the subtype sample, for hand labelling.

Every cell is written as text (so "+Name" or "=..." is never read as a formula), long
text wraps, and the answer columns have drop-down lists. The CSVs in audit/ stay the
record; these workbooks are filled in by hand and read back by a later script.

    python -m audit.workbook Files
"""
import argparse
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from audit.flags import read

VERDICTS = ["keep", "correct", "unsure"]
SUBTYPES = ["scam", "link", "self_promotion", "bulk"]


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


def build(out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    _sheet(ws, read("review_sheet.csv"),
           {"A": 9, "B": 70, "C": 14, "D": 11, "E": 17, "F": 45},
           {"D": VERDICTS, "E": SUBTYPES})
    p = out_dir / "audit_review.xlsx"
    wb.save(p)
    paths.append(p)

    wb = Workbook()
    ws = wb.active
    ws.title = "subtypes"
    _sheet(ws, read("subtype_sample.csv"),
           {"A": 9, "B": 14, "C": 70, "D": 17, "E": 40},
           {"D": SUBTYPES + ["not_spam"]})
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
