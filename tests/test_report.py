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
    assert report.render_example("test").startswith("_Pending: the single test run")


def test_fill_replaces_each_block_once():
    text = "a\n" + "".join(f"<!-- report:{b} -->\nOLD-BODY\n<!-- /report:{b} -->\n" for b in report.BLOCKS)
    out = report.fill(text, split="dev")
    assert "OLD-BODY" not in out and out.count("<!-- report:") == len(report.BLOCKS)
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
        # the data table's row for this split: its last three cells are counts
        lines = [l for l in readme.splitlines() if l.startswith(row) and l.rsplit("|", 4)[1].strip().replace(",", "").isdigit()]
        assert len(lines) == 1, lines
        assert lines[0].endswith(f"| {len(d):,} | {spam:,} | {ham:,} |")


def test_price_is_stated_with_its_source():
    assert report.PRICE["usd_per_mtok_input"] == 1.00 and report.PRICE["usd_per_mtok_output"] == 5.00
    assert report.PRICE["source"].startswith("Anthropic model pricing list") and report.PRICE["read_on"]


# The example comment: chosen by a fixed rule, and its numbers come from the result files

def test_pick_example_takes_the_first_row_the_rule_names():
    import pandas as pd
    p = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "label_original": [1, 1, 1, 0],
                      "label_corrected": [1, 0, 0, 1], "rules_pred": [1, 1, 1, 0], "llm_v2_pred": [0, 1, 0, 1]})
    assert report.pick_example(p, []).row_id == "c"   # a: label not corrected; b: the two agree
    assert report.pick_example(p, ["c"]).row_id == "d"
    assert report.pick_example(p, ["c", "d"]) is None


@pytest.mark.parametrize("split", ["dev", "test"])
def test_example_matches_the_predictions_and_the_audit(split):
    import pandas as pd
    from eval.final import QUOTED
    p = pd.read_csv(report.FINAL / f"{split}_predictions.csv", keep_default_na=False)
    row = report.pick_example(p, QUOTED[split])
    earlier = p.iloc[:int(p.index[p.row_id == row.row_id][0])]
    assert report.pick_example(earlier, QUOTED[split]) is None  # nothing earlier meets the rule
    assert row.label_original != row.label_corrected and row.rules_pred != row.llm_v2_pred
    assert row.row_id not in QUOTED[split]
    block = report.render_example(split)
    assert f"`{row.row_id}`" in block
    reason = report.read("corrections.csv").set_index("row_id").loc[row.row_id, "reason"]
    assert f"(audit reason: {reason})" in block
    for k in report.NAMES:
        line = next(l for l in block.splitlines() if l.startswith(f"| {report.NAMES[k]} |"))
        assert f"| **{report.LABEL[int(row[k + '_pred'])]}** |" in line
