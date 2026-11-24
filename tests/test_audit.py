import pytest

from audit import flags, workbook

TABLES = flags.build()


def test_committed_audit_files_regenerate():
    assert flags.check(TABLES) == 0


def test_flag_counts():
    f = TABLES["flags.csv"]
    assert len(f) == 63
    assert f.reason.value_counts().to_dict() == {"both_baselines_wrong": 63}
    assert f.split.value_counts().to_dict() == {"test": 38, "dev": 14, "train": 11}
    assert f.gold.value_counts().to_dict() == {"spam": 60, "ham": 3}
    assert f.flag_id.is_unique and f.row_id.is_unique


def test_flags_are_exactly_both_wrong():
    f = TABLES["flags.csv"]
    rules_pred = (f.rules_score >= 1).map(int)
    lr_pred = (f.lr_proba >= 0.5).map(int)
    gold = (f.gold == "spam").map(int)
    assert ((rules_pred != gold) & (lr_pred != gold)).all()


def test_review_sheet_hides_model_scores():
    sheet = TABLES["review_sheet.csv"]
    assert list(sheet.columns) == ["flag_id", "text", "current_label", "verdict", "subtype_if_spam", "reason"]
    assert list(sheet.flag_id) == list(TABLES["flags.csv"].flag_id)


def test_blind_recheck_is_drawn_from_flags():
    blind = TABLES["blind_recheck.csv"]
    assert len(blind) == 20 and blind.flag_id.is_unique
    assert set(blind.flag_id) <= set(TABLES["flags.csv"].flag_id)


def test_subtype_sample_is_200_test_spam():
    from data.load import load_comments
    df = load_comments().set_index("row_id")
    s = TABLES["subtype_sample.csv"]
    assert len(s) == 200 and s.row_id.is_unique
    assert (df.loc[s.row_id, "split"] == "test").all()
    assert (df.loc[s.row_id, "label"] == 1).all()


def test_workbooks_write_text_cells(tmp_path):
    from openpyxl import load_workbook
    paths = workbook.build(tmp_path)
    assert [p.name for p in paths] == ["audit_review.xlsx", "subtype_labels.xlsx"]
    wb = load_workbook(paths[1])
    assert wb.sheetnames == ["subtypes", "videos"]
    ws = wb["subtypes"]
    assert ws.max_row == 201
    assert [c.value for c in ws[1]] == ["sample_id", "row_id", "video", "text", "subtype", "notes", "flag_id"]
    texts = [ws.cell(row=r, column=4).value for r in range(2, ws.max_row + 1)]
    assert texts == list(TABLES["subtype_sample.csv"].text)
    assert all(ws.cell(row=r, column=4).data_type == "s" for r in range(2, ws.max_row + 1))
    assert load_workbook(paths[0])["videos"].max_row == 6


def test_review_workbook_columns_hide_scores():
    t = workbook.review_table()
    assert list(t.columns) == ["flag_id", "row_id", "video", "text", "current_label", "verdict",
                               "subtype_if_spam", "reason", "subtype_sample_id"]
    assert len(t) == 63
    assert not any("proba" in c or "score" in c or "hits" in c for c in t.columns)


def test_cross_references_between_workbooks():
    r, s = workbook.review_table(), workbook.subtype_table()
    pairs_r = {(f, x) for f, x in zip(r.flag_id, r.subtype_sample_id) if x}
    pairs_s = {(f, x) for x, f in zip(s.sample_id, s.flag_id) if f}
    assert pairs_r == pairs_s
    assert len(pairs_r) == 11
    rows = dict(zip(r.flag_id, r.row_id))
    srows = dict(zip(s.sample_id, s.row_id))
    assert all(rows[f] == srows[x] for f, x in pairs_r)


def test_videos_file_covers_every_video():
    from data.load import load_comments
    assert set(workbook.VIDEOS) == set(load_comments().video.unique())
    assert all(len(v["youtube_id"]) == 11 for v in workbook.VIDEOS.values())
    assert workbook.video_label("Eminem") == "Eminem – Love the Way You Lie (ft. Rihanna)"
