import re

import pytest

from sql import check as C


def test_sql_in_sqlite_matches_the_python_results():
    assert C.check(log=lambda *_: None) == 0


@pytest.mark.parametrize("q", C.QUERIES)
def test_queries_use_only_the_one_table_and_portable_sql(q):
    text = (C.SQL / f"{q}.sql").read_text(encoding="utf-8")
    assert set(re.findall(r"\bFROM\s+([\w.]+)", text)) <= {"yt_comment_spam.predictions", "long", "labelled",
                                                           "counts"}
    for dialect_only in ("COUNTIF", "SAFE_DIVIDE", "IIF(", "IFNULL", "::", "`"):
        assert dialect_only not in text.upper()


def test_bigquery_files_are_compared(monkeypatch, tmp_path):
    lite = C.run_sqlite()
    monkeypatch.setattr(C, "EXPORTS", tmp_path)
    monkeypatch.setattr(C, "ROOT", tmp_path.parent)
    assert C.check(require_bigquery=True, log=lambda *_: None) == 1  # missing
    for q in C.QUERIES:
        lite[q].to_csv(tmp_path / f"{q}.csv", index=False)
    assert C.check(require_bigquery=True, log=lambda *_: None) == 0
    bad = lite["confusion"].copy()
    bad.loc[0, "tp"] += 1
    bad.to_csv(tmp_path / "confusion.csv", index=False)
    lines = []
    assert C.check(log=lines.append) == 1
    assert any("tp" in l for l in lines)
