import json

import pytest

from data.load import load_comments
from eval import report
from eval.metrics import cost_per_1000

DEV = json.loads((report.FINAL / "dev.json").read_text(encoding="utf-8"))


def test_dev_report_matches_the_committed_file():
    assert report.main(["--split", "dev", "--check"]) == 0


def test_readme_blocks_match_the_result_files():
    assert report.main(["--readme", "--check"]) == 0


def test_results_block_carries_the_numbers_in_the_result_file():
    block = report.render_results("dev")
    main = DEV["results"]["corrected"]["all"]
    for k, e in main["systems"].items():
        m = e["metrics"]
        assert f"{m['tp']} / {m['fp']} / {m['fn']}" in block
        assert f"{m['f1']:.3f} ({m['f1_lo']:.3f}–{m['f1_hi']:.3f})" in block
        assert f"{DEV['results']['original']['all']['systems'][k]['metrics']['f1']:.3f}" in block
    for h, x in main["hypotheses"].items():
        assert f"{x['a_only_right']} / {x['b_only_right']}" in block
    for u in DEV["llm_usage"].values():
        usd = cost_per_1000(u["comments"], u["input_tokens"], u["output_tokens"], 1.00, 5.00)
        assert f"${usd:.2f}" in block


def test_redteam_block_carries_the_numbers_in_the_result_file():
    r = json.loads((report.ADVERSARIAL / "dev.json").read_text(encoding="utf-8"))
    block = report.render_redteam("dev")
    for col, b in r["results"].items():
        for t in b["by_transform"].values():
            assert f"{t['evaded']}/{t['variants']}" in block
        a = b["all_transforms"]
        assert f"{100 * a['rate']:.1f}% ({100 * a['lo']:.1f}%–{100 * a['hi']:.1f}%)" in block


def test_missing_files_render_as_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "FINAL", tmp_path)
    monkeypatch.setattr(report, "ADVERSARIAL", tmp_path)
    monkeypatch.setattr(report, "ROOT", tmp_path.parent)
    assert report.render_results("test").startswith("_Pending: the single test run")
    assert report.render_redteam("test").startswith("_Pending: the red-team run")


def test_fill_replaces_each_block_once():
    text = "a\n" + "".join(f"<!-- report:{b} -->\nOLD-BODY\n<!-- /report:{b} -->\n" for b in report.BLOCKS)
    out = report.fill(text, split="dev")
    assert "OLD-BODY" not in out and out.count("<!-- report:") == 3
    assert report.fill(out, split="dev") == out
    with pytest.raises(ValueError):
        report.fill("no blocks here")
    with pytest.raises(ValueError):
        report.fill(text + text)


def test_readme_data_table_matches_the_data():
    df = load_comments()
    readme = report.README.read_text(encoding="utf-8")
    for split in ("train", "dev", "test"):
        d = df[df.split == split]
        row = f"| {split.capitalize()} | "
        spam, ham = int(d.label.astype(int).sum()), int((d.label.astype(int) == 0).sum())
        line = next(l for l in readme.splitlines() if l.startswith(row))
        assert line.endswith(f"| {len(d):,} | {spam:,} | {ham:,} |")


def test_price_is_stated_with_its_source():
    assert report.PRICE["usd_per_mtok_input"] == 1.00 and report.PRICE["usd_per_mtok_output"] == 5.00
    assert report.PRICE["source"].startswith("Anthropic model pricing list") and report.PRICE["read_on"]
