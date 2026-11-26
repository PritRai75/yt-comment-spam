# SQL layer (BigQuery)

The confusion counts and per-video metrics of the final analysis, computed in SQL from
the committed predictions and checked against the Python results.

| File | What it computes |
|---|---|
| `confusion.sql` | TP, FP, FN, TN, precision, recall, F1 and false-positive rate per split, label set and system, each at its own spam cut-off |
| `per_video.sql` | Comments, spam, flagged, caught, recall and precision per split, video and system, on the corrected labels |
| `check.py` | Runs both queries in SQLite on `results/final/<split>_predictions.csv`, compares them with `results/final/<split>.json`, and compares the BigQuery results in `results/sql/` with them |

The queries use only portable SQL (no `COUNTIF`, `SAFE_DIVIDE` or backticks), so the
same text runs in BigQuery and SQLite, and the tests run it on every commit.

## Running it in BigQuery (sandbox)

1. In the BigQuery console, in the sandbox project, create a dataset named
   `yt_comment_spam` (any location; sandbox tables expire after 60 days, which is fine:
   the results are kept in this repo).
2. Create a table in it from an upload: file `results/final/dev_predictions.csv`, format
   CSV, table name `predictions`, schema auto-detected. Once the test results exist, upload
   `results/final/test_predictions.csv` to the same table with write preference *Append
   to table*; the queries group by split.
3. Open a query tab, paste `sql/confusion.sql`, run it, then *Save results* → *CSV (local
   file)*; save it as `results/sql/confusion.csv`. The same for `sql/per_video.sql` →
   `results/sql/per_video.csv`.
4. `python -m sql.check` must print OK for all four comparisons.
