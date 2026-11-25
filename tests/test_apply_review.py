import argparse

import pandas as pd
import pytest
from openpyxl import load_workbook

from audit import apply_review, full_review, workbook
from audit.flags import read
from data.load import load_comments

LABEL = load_comments().set_index("row_id").label.map({1: "spam", 0: "ham"})


def fill(path, sheet, id_col, answers: dict):
    """Write answers {id: {column: value}} into a built workbook, as a labeller would."""
    wb = load_workbook(path)
    ws = wb[sheet]
    hdr = [c.value for c in ws[1]]
    for r in range(2, ws.max_row + 1):
        for col, v in answers.get(ws.cell(r, hdr.index(id_col) + 1).value, {}).items():
            ws.cell(r, hdr.index(col) + 1).value = v or None
    wb.save(path)


def s1_books(tmp_path, audit_edit=None, subtype_edit=None):
    a = read("review_verdicts.csv").set_index("flag_id")
    s = read("subtype_labels.csv").set_index("sample_id")
    if audit_edit:
        audit_edit(a)
    if subtype_edit:
        subtype_edit(s)
    pa, ps = workbook.build(tmp_path)
    fill(pa, "review", "flag_id", a[["verdict", "subtype_if_spam", "reason"]].to_dict("index"))
    fill(ps, "subtypes", "sample_id", s[["subtype", "notes"]].to_dict("index"))
    return pa, ps


def args(pa, ps, **kw):
    return argparse.Namespace(audit=pa, subtypes=ps, full_dev=kw.get("dev"), full_test=kw.get("test"),
                              recheck=kw.get("recheck"), unsure=kw.get("unsure"), check=False)


# --- the committed record (S1 labels, with the P9 edits) ---

def test_review_verdicts_match_flags():
    v, f = read("review_verdicts.csv"), read("flags.csv")
    assert list(v.flag_id) == list(f.flag_id) and list(v.row_id) == list(f.row_id)
    assert v.verdict.value_counts().to_dict() == {"correct": 38, "keep": 16, "unsure": 9}


def test_subtype_labels_after_p9():
    s = read("subtype_labels.csv").set_index("sample_id")
    assert s.subtype.value_counts().to_dict() == {
        "self_promotion": 74, "bulk": 69, "not_spam": 26, "scam": 18, "link": 12, "unsure": 1}
    assert set(s.subtype[["S106", "S146"]]) == {"scam"}
    assert read("review_verdicts.csv").set_index("flag_id").subtype_if_spam["F034"] == "scam"
    assert s.subtype["S163"] == "unsure" and s.row_id["S163"] == "Eminem:269"


def test_corrections_record():
    c = read("corrections.csv")
    assert len(c) == 85 and c.row_id.is_unique
    assert c.groupby("split").size().to_dict() == {"dev": 16, "test": 68, "train": 1}
    assert c.groupby(["label_before", "label_after"]).size().to_dict() == {("ham", "spam"): 1, ("spam", "ham"): 84}
    assert (c.label_before == c.row_id.map(LABEL)).all()
    assert c.source.value_counts().to_dict() == {
        "full_review": 30, "audit_flag": 29, "subtype_sample": 17, "audit_flag;subtype_sample": 9}
    to_spam = c[c.label_after == "spam"]
    assert list(zip(to_spam.row_id, to_spam.subtype_after, to_spam.source_id)) == [("Eminem:361", "self_promotion", "T327")]
    # random-sample estimate of test spam that is not spam (decision 21): 26 of 200
    assert (read("subtype_labels.csv").subtype == "not_spam").sum() == 26


def test_full_review_labels_record():
    f = read("full_review_labels.csv")
    assert list(f.review_id) == list(read("full_review.csv").review_id)
    assert list(f.row_id) == list(read("full_review.csv").row_id)
    says = f.subtype.map(lambda a: {"not_spam": "ham", "unsure": "unsure"}.get(a, "spam"))
    assert pd.crosstab([f.split, f.row_id.map(LABEL)], says).to_dict() == {
        "ham": {("dev", "ham"): 100, ("dev", "spam"): 11, ("test", "ham"): 99, ("test", "spam"): 18},
        "spam": {("dev", "ham"): 0, ("dev", "spam"): 146, ("test", "ham"): 1, ("test", "spam"): 233},
        "unsure": {("dev", "ham"): 0, ("dev", "spam"): 3, ("test", "ham"): 0, ("test", "spam"): 3},
    }
    # with the S1 reviews, every test spam comment has now been reviewed once: 67 of 481 judged not spam
    c = read("corrections.csv")
    assert ((c.split == "test") & (c.label_before == "spam")).sum() == 67


def test_scoring_exclusions():
    e = read("scoring_exclusions.csv")
    assert list(e.row_id) == ["Eminem:269"]
    assert "Eminem:269" not in set(read("corrections.csv").row_id)


def test_round_trip_through_workbooks(tmp_path):
    pa, ps = s1_books(tmp_path / "s1")
    f = read("full_review_labels.csv").set_index("review_id")
    dev, test = workbook.build_full(tmp_path / "full")
    answers = f[["subtype", "notes"]].to_dict("index")
    fill(dev, "review", "review_id", answers)
    fill(test, "review", "review_id", answers)
    tables = apply_review.build(args(pa, ps, dev=dev, test=test))
    assert set(tables) == {"review_verdicts.csv", "subtype_labels.csv", "full_review_labels.csv",
                           "corrections.csv", "scoring_exclusions.csv"}
    for name, t in tables.items():
        assert apply_review.differences(name, t) == [], name


def test_check_lists_differing_cells(tmp_path, capsys):
    pa, ps = s1_books(tmp_path, audit_edit=lambda a: a.__setitem__(
        "subtype_if_spam", a.subtype_if_spam.where(a.index != "F040", "scam")))   # F040 is not in the subtype sample
    assert apply_review.main(["--audit", str(pa), "--subtypes", str(ps), "--check"]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS audit/review_verdicts.csv" in out
    assert "F040 subtype_if_spam: 'link' -> 'scam'" in out


# --- validation ---

@pytest.mark.parametrize("edit,message", [
    (lambda a: a.__setitem__("verdict", a.verdict.where(a.index != "F001", "")), "F001: verdict is blank"),
    (lambda a: a.__setitem__("verdict", a.verdict.where(a.index != "F001", "maybe")), "'maybe' is not one of"),
    (lambda a: a.__setitem__("subtype_if_spam", a.subtype_if_spam.where(a.index != "F002", "")),
     "F002: final label spam needs a subtype"),
    (lambda a: a.__setitem__("subtype_if_spam", a.subtype_if_spam.where(a.index != "F001", "bulk")),
     "F001: final label ham but subtype"),
    (lambda a: a.__setitem__("reason", a.reason.where(a.index != "F005", "")), "F005: reason is blank"),
])
def test_audit_problems_are_caught(tmp_path, edit, message):
    pa, ps = s1_books(tmp_path, audit_edit=edit)
    with pytest.raises(apply_review.ReviewError) as e:
        apply_review.build(args(pa, ps))
    assert any(message in p for p in e.value.problems)


def test_not_spam_needs_a_note(tmp_path):
    pa, ps = s1_books(tmp_path, subtype_edit=lambda s: s.__setitem__("notes", s.notes.where(s.index != "S005", "")))
    with pytest.raises(apply_review.ReviewError, match="S005"):
        apply_review.build(args(pa, ps))


def test_overlap_disagreement_is_caught(tmp_path):
    # F034 and S106 are the same comment; change only one of them
    pa, ps = s1_books(tmp_path, subtype_edit=lambda s: s.__setitem__("subtype", s.subtype.where(s.index != "S106", "link")))
    with pytest.raises(apply_review.ReviewError, match="F034 / S106"):
        apply_review.build(args(pa, ps))


def test_changed_text_is_caught(tmp_path):
    pa, ps = s1_books(tmp_path)
    wb = load_workbook(ps)
    wb["subtypes"].cell(3, 4).value = "edited"
    wb.save(ps)
    with pytest.raises(apply_review.ReviewError, match="text changed"):
        apply_review.build(args(pa, ps))


def test_cli_writes_nothing_on_problems(tmp_path, capsys):
    pa, ps = s1_books(tmp_path, audit_edit=lambda a: a.__setitem__("verdict", ""))
    assert apply_review.main(["--audit", str(pa), "--subtypes", str(ps)]) == 1
    assert "nothing written" in capsys.readouterr().err


# --- option A review ---

def test_full_review_list():
    t = read("full_review.csv")
    assert full_review.build().equals(t)
    assert len(t) == 614 and t.row_id.is_unique
    label = t.row_id.map(LABEL)
    assert pd.crosstab(t.split, label).to_dict() == {"ham": {"dev": 100, "test": 100}, "spam": {"dev": 160, "test": 254}}
    reviewed = set(read("flags.csv").row_id) | set(read("subtype_sample.csv").row_id)
    assert not set(t.row_id) & reviewed
    # with the S1 reviews, every dev and test spam comment is reviewed exactly once
    df = load_comments()
    spam = set(df.row_id[(df.split != "train") & (df.label == 1)])
    assert spam <= set(t.row_id) | reviewed


def test_full_workbooks_hide_the_label(tmp_path):
    paths = workbook.build_full(tmp_path)
    assert [p.name for p in paths] == ["review_dev.xlsx", "review_test.xlsx"]
    ws = load_workbook(paths[1])["review"]
    assert [c.value for c in ws[1]] == ["review_id", "row_id", "video", "text", "subtype", "notes"]
    assert ws.max_row == 355
    assert all(ws.cell(r, 4).data_type == "s" for r in range(2, ws.max_row + 1))


def full_books(tmp_path, notes=True):
    t = read("full_review.csv").set_index("review_id")
    pa, ps = s1_books(tmp_path / "s1")
    dev, test = workbook.build_full(tmp_path / "full")
    answers = {}
    for rid, row in t.iterrows():
        answers[rid] = {"subtype": "bulk" if LABEL[row.row_id] == "spam" else "not_spam", "notes": ""}
    answers["D001"] = {"subtype": "link", "notes": "test: ham marked spam" if notes else ""}   # a ham row
    spam_rid = next(r for r, row in t.iterrows() if LABEL[row.row_id] == "spam" and r.startswith("T"))
    answers[spam_rid] = {"subtype": "not_spam", "notes": "test: spam marked ham"}
    fill(dev, "review", "review_id", answers)
    fill(test, "review", "review_id", answers)
    return pa, ps, dev, test, spam_rid


def test_full_review_corrections(tmp_path):
    pa, ps, dev, test, spam_rid = full_books(tmp_path)
    tables = apply_review.build(args(pa, ps, dev=dev, test=test))
    c = tables["corrections.csv"].set_index("source_id")
    assert len(c) == 55 + 2
    assert (c.label_before["D001"], c.label_after["D001"], c.subtype_after["D001"]) == ("ham", "spam", "link")
    assert (c.label_before[spam_rid], c.label_after[spam_rid]) == ("spam", "ham")
    assert len(tables["full_review_labels.csv"]) == 614


def test_full_review_disagreement_needs_a_note(tmp_path):
    pa, ps, dev, test, _ = full_books(tmp_path, notes=False)
    with pytest.raises(apply_review.ReviewError, match="D001"):
        apply_review.build(args(pa, ps, dev=dev, test=test))


# --- blind re-check ---

def test_recheck_workbook_is_blind(tmp_path):
    t = workbook.recheck_table()
    assert list(t.columns) == ["recheck_id", "video", "text", "current_label", "verdict", "subtype_if_spam", "reason"]
    assert len(t) == 20 and (t.verdict == "").all()
    ids = workbook.recheck_ids()
    assert sorted(ids.values()) == sorted(read("blind_recheck.csv").flag_id)
    assert list(ids.values()) != list(read("blind_recheck.csv").flag_id)
    assert workbook.build_recheck(tmp_path)[0].name == "blind_recheck.xlsx"


def test_recheck_agreement(tmp_path):
    pa, ps = s1_books(tmp_path / "s1")
    [rc] = workbook.build_recheck(tmp_path / "rc")
    first = read("review_verdicts.csv").set_index("flag_id")
    answers = {r: first.loc[f, ["verdict", "subtype_if_spam", "reason"]].to_dict()
               for r, f in workbook.recheck_ids().items()}
    flip = next(r for r, f in workbook.recheck_ids().items() if first.verdict[f] == "keep")
    answers[flip]["verdict"] = "correct"
    answers[flip]["subtype_if_spam"] = ""
    fill(rc, "recheck", "recheck_id", answers)
    r = apply_review.build(args(pa, ps, recheck=rc))["recheck_verdicts.csv"]
    assert (r.same_label == "true").sum() == 19


# --- workbook safety ---

def test_workbooks_are_never_overwritten(tmp_path):
    workbook.build_full(tmp_path)
    (tmp_path / "review_dev.xlsx").unlink()
    with pytest.raises(FileExistsError):
        workbook.build_full(tmp_path)
    assert not (tmp_path / "review_dev.xlsx").exists()
    assert workbook.main([str(tmp_path), "--set", "full"]) == 1


# --- unsure second pass ---

def test_unsure_workbook():
    t = workbook.unsure_table()
    assert list(t.columns) == ["unsure_id", "source_id", "row_id", "video", "text", "first_note", "answer", "note"]
    assert list(t.unsure_id) == [f"U{i:02d}" for i in range(1, 16)]
    assert list(t.source_id) == ["F003", "F007", "F009", "F016", "F032", "F042", "F049", "F053", "F057",
                                 "D029", "D090", "D121", "T030", "T059", "T299"]
    assert "Eminem:269" not in set(t.row_id)
    assert (t.first_note != "").all() and (t.answer == "").all()


def unsure_book(tmp_path, answers):
    pa, ps = s1_books(tmp_path / "s1")
    f = read("full_review_labels.csv").set_index("review_id")
    dev, test = workbook.build_full(tmp_path / "full")
    full = f[["subtype", "notes"]].to_dict("index")
    fill(dev, "review", "review_id", full)
    fill(test, "review", "review_id", full)
    [u] = workbook.build_unsure(tmp_path / "u")
    fill(u, "unsure", "unsure_id", answers)
    return args(pa, ps, dev=dev, test=test, unsure=u)


def test_unsure_resolution_corrections(tmp_path):
    answers = {f"U{i:02d}": {"answer": "bulk", "note": "test"} for i in range(1, 16)}
    answers["U02"] = {"answer": "not_spam", "note": "P1: view campaign for this artist"}   # F007, labelled spam
    answers["U15"] = {"answer": "unsure", "note": "still unsure"}
    tables = apply_review.build(unsure_book(tmp_path, answers))
    c = tables["corrections.csv"].set_index("source_id")
    assert len(c) == 86
    assert (c.row_id["U02"], c.label_after["U02"], c.source["U02"]) == ("KatyPerry:247", "ham", "unsure_resolution")
    assert len(tables["unsure_resolution.csv"]) == 15
    # first-pass verdicts are untouched
    assert apply_review.differences("review_verdicts.csv", tables["review_verdicts.csv"]) == []


def test_unsure_needs_a_note(tmp_path):
    answers = {f"U{i:02d}": {"answer": "bulk", "note": "test"} for i in range(1, 16)}
    answers["U05"] = {"answer": "link", "note": ""}
    with pytest.raises(apply_review.ReviewError, match="U05: note is blank"):
        apply_review.build(unsure_book(tmp_path, answers))
