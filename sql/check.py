"""Check the SQL layer against the Python analysis.

`sql/confusion.sql` and `sql/per_video.sql` are written to run unchanged in BigQuery and
in SQLite. This script:

  1. loads results/final/<split>_predictions.csv for every split that has one into an
     in-memory SQLite database, as the table yt_comment_spam.predictions;
  2. runs both queries and compares them with results/final/<split>.json (counts
     exactly, rates within 1e-9);
  3. if the BigQuery results were saved as results/sql/confusion.csv and
     results/sql/per_video.csv, compares them with the SQLite results the same way.

    python -m sql.check                    # SQLite vs Python, and BigQuery files if present
    python -m sql.check --require-bigquery # also fail when the BigQuery files are missing
"""
import argparse
import json
import math
import sqlite3
import sys

import pandas as pd

from data.fetch import ROOT

SQL = ROOT / "sql"
FINAL = ROOT / "results" / "final"
EXPORTS = ROOT / "results" / "sql"
QUERIES = ["confusion", "per_video"]
KEYS = {"confusion": ["split", "label_set", "system"], "per_video": ["split", "video", "system"]}
RATES = {"precision", "recall", "f1", "fpr"}
INT_COLUMNS = ["label_original", "label_corrected", "rules_pred", "logreg_pred", "tree_pred", "llm_v1_pred",
               "llm_v2_pred"]


def splits() -> list:
    return [s for s in ("dev", "test") if (FINAL / f"{s}_predictions.csv").exists()]


def predictions() -> pd.DataFrame:
    frames = [pd.read_csv(FINAL / f"{s}_predictions.csv", keep_default_na=False) for s in splits()]
    df = pd.concat(frames, ignore_index=True)
    df[INT_COLUMNS] = df[INT_COLUMNS].astype(int)
    return df


def run_sqlite(df: pd.DataFrame = None) -> dict:
    """Both queries on the predictions in SQLite, as DataFrames."""
    df = predictions() if df is None else df
    con = sqlite3.connect(":memory:")
    con.execute("ATTACH DATABASE ':memory:' AS yt_comment_spam")
    df.to_sql("loaded", con, index=False)
    con.execute("CREATE TABLE yt_comment_spam.predictions AS SELECT * FROM loaded")
    out = {q: pd.read_sql_query((SQL / f"{q}.sql").read_text(encoding="utf-8"), con) for q in QUERIES}
    con.close()
    return out


def python_tables(split_list=None) -> dict:
    """The same numbers as the queries, read from results/final/<split>.json."""
    conf, vid = [], []
    for s in split_list or splits():
        r = json.loads((FINAL / f"{s}.json").read_text(encoding="utf-8"))["results"]
        for label_set, sets in r.items():
            for system, e in sets["all"]["systems"].items():
                m = e["metrics"]
                conf.append({"split": s, "label_set": label_set, "system": system, "tp": m["tp"], "fp": m["fp"],
                             "fn": m["fn"], "tn": m["tn"], "precision": m["precision"], "recall": m["recall"],
                             "f1": m["f1"], "fpr": m["fpr"]})
                if label_set == "corrected":
                    for v in e["recall_by_video"]:
                        vid.append({"split": s, "video": v["group"], "system": system, "spam": v["spam"],
                                    "caught": v["caught"], "recall": v["recall"]})
    return {"confusion": pd.DataFrame(conf), "per_video": pd.DataFrame(vid)}


def _num(x):
    if x is None or (isinstance(x, str) and x == ""):
        return math.nan
    return float(x)


def compare(name: str, a: pd.DataFrame, b: pd.DataFrame, what: str) -> list:
    """Problems where b differs from a on the columns b has, matched on the keys."""
    keys = KEYS[name]
    a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
    out = []
    if list(a.index) != list(b.index):
        return [f"{name}: {what} has rows {len(b)} with different keys from {len(a)}"]
    for col in b.columns:
        if col not in a.columns:
            out.append(f"{name}: {what} has an extra column {col}")
            continue
        for key, x, y in zip(a.index, a[col], b[col]):
            x, y = _num(x), _num(y)
            if math.isnan(x) and math.isnan(y):
                continue
            ok = math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9) if col in RATES else x == y
            if not ok:
                out.append(f"{name} {key} {col}: {x} vs {what} {y}")
    return out


def check(require_bigquery: bool = False, log=print) -> int:
    lite = run_sqlite()
    py = python_tables()
    problems = []
    for q in QUERIES:
        p = compare(q, lite[q], py[q], "Python")
        log(f"{'OK' if not p else 'DIFFERS'}: {q}.sql in SQLite vs results/final ({', '.join(splits())})")
        problems += p
    for q in QUERIES:
        path = EXPORTS / f"{q}.csv"
        rel = path.relative_to(ROOT).as_posix()
        if not path.exists():
            log(f"{'MISSING' if require_bigquery else 'not yet'}: {rel} (BigQuery results)")
            problems += [f"{rel} missing"] if require_bigquery else []
            continue
        bq = pd.read_csv(path, keep_default_na=False)
        lite_q = lite[q][lite[q].split.isin(set(bq.split))]
        p = compare(q, lite_q, bq, "BigQuery") if set(bq.columns) == set(lite[q].columns) else \
            [f"{rel}: columns {list(bq.columns)} differ from the query's {list(lite[q].columns)}"]
        log(f"{'OK' if not p else 'DIFFERS'}: {rel} vs SQLite ({', '.join(sorted(set(bq.split)))})")
        problems += p
    for p in problems[:20]:
        log("  " + p)
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--require-bigquery", action="store_true", help="fail if the BigQuery result files are missing")
    args = ap.parse_args(argv)
    return check(args.require_bigquery)


if __name__ == "__main__":
    sys.exit(main())
