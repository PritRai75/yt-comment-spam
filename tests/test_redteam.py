import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from adversarial import redteam as R
from adversarial import transforms as T
from data.clean import dup_key
from data.fetch import ROOT
from data.load import load_comments
from eval.final import QUOTED
from models import llm

DEV_PREDS = pd.read_csv(R.predictions_path("dev"), keep_default_na=False)


def test_pool_and_sample_follow_the_design():
    p = R.pool("dev", DEV_PREDS)
    preds = DEV_PREDS.set_index("row_id").loc[p.row_id]
    assert (preds.label_corrected == 1).all()
    for k in R.ATTACKED:
        assert (preds[f"{k}_pred"] == 1).all()
    assert not set(p.row_id) & set(QUOTED["dev"])
    df = load_comments().set_index("row_id")
    keys = df.loc[p.row_id].content.map(dup_key)
    assert keys.is_unique
    order = list(df.index)
    assert [order.index(r) for r in p.row_id] == sorted(order.index(r) for r in p.row_id)
    s = R.sample("dev", preds=DEV_PREDS)
    assert len(s) == min(R.N_SAMPLE, len(p)) and set(s.row_id) <= set(p.row_id)
    assert s.row_id.tolist() == R.sample("dev", preds=DEV_PREDS).row_id.tolist()


def test_sample_takes_100_when_the_pool_is_larger():
    big = pd.DataFrame({"row_id": [f"x{i}" for i in range(300)]})
    take = np.sort(np.random.default_rng(R.SEED).permutation(len(big))[:R.N_SAMPLE])
    assert len(take) == 100 and len(set(take)) == 100


def test_variant_table_has_a_row_per_variant():
    s = R.sample("dev", preds=DEV_PREDS)
    vt = R.variant_table(s, originals=True)
    assert (vt["transform"] == "none").sum() == len(s)
    for name in T.NAMES:
        n = sum(T.apply(name, t) is not None for t in s.text)
        assert (vt["transform"] == name).sum() == n


def test_test_split_needs_committed_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "predictions_path", lambda split: tmp_path / f"{split}_predictions.csv")
    with pytest.raises(R.NotReady, match="missing"):
        R.require_predictions("test")
    (tmp_path / "test_predictions.csv").write_text("x\n")
    monkeypatch.setattr(R, "ROOT", tmp_path)

    def git(*args, tracked=(), changed=()):
        out = tracked if args[0] == "ls-files" else changed
        return subprocess.CompletedProcess(args, 0, stdout="\n".join(out), stderr="")

    monkeypatch.setattr(R.lock, "_git", lambda *a: git(*a))
    with pytest.raises(R.NotReady, match="not committed"):
        R.require_predictions("test")
    rel = "test_predictions.csv"
    monkeypatch.setattr(R.lock, "_git", lambda *a: git(*a, tracked=[rel], changed=[rel]))
    with pytest.raises(R.NotReady, match="uncommitted"):
        R.require_predictions("test")
    monkeypatch.setattr(R.lock, "_git", lambda *a: git(*a, tracked=[rel]))
    R.require_predictions("test")


def test_cli_refuses_test_without_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "predictions_path", lambda split: tmp_path / f"{split}_predictions.csv")
    assert R.main(["sample", "--split", "test"]) == 2


def test_llm_run_reuses_the_runs_votes_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "FAKE_LLM_OUT", tmp_path / "adv")
    monkeypatch.setattr(llm, "FAKE_RUNS", tmp_path / "runs")
    s = R.sample("dev", preds=DEV_PREDS).head(3)
    vt = R.variant_table(s, originals=True)
    # the split's own run already voted on the three comments as sent
    _, prompt, config, _, original = R._llm_setup("dev", fake=True)
    for r in s.itertuples():
        text = prompt.text(r.text)
        for v in range(llm.N_VOTES):
            original.add(dict(llm.FakeClient.default_answer(text, v), valid=True, attempts=1, error="",
                              input_tokens=1, output_tokens=1, latency_ms=1, replicate=0, vote=v,
                              key=llm.call_key(config, r.video, text, v), row_id=r.row_id, config=config.hash))
    plan = R.llm_plan(vt, "dev", fake=True)
    assert plan["reused_from_run"] == 15  # the originals; their zero-width variants are the same texts
    out = R.llm_run(vt, "dev", fake=True, log=lambda *_: None)
    assert out["calls_made"] == plan["to_make"] > 0
    assert R.llm_run(vt, "dev", fake=True, log=lambda *_: None)["calls_made"] == 0
    scores, usage = R.llm_scores(vt, "dev", fake=True)
    assert len(scores) == len(vt) and usage["calls"] == plan["to_make"]
    z = vt["transform"].to_numpy() == "zero_width"
    o = vt["transform"].to_numpy() == "none"
    assert np.array_equal(scores[z], scores[o])
    recs = [json.loads(l) for l in out_path(tmp_path, config).read_text().splitlines()]
    assert all("|" in r["row_id"] for r in recs) and not any("text" in r for r in recs)


def out_path(tmp_path, config):
    return tmp_path / "adv" / config.name / "calls-dev.jsonl"


def test_llm_scores_need_every_vote(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "FAKE_LLM_OUT", tmp_path / "adv")
    monkeypatch.setattr(llm, "FAKE_RUNS", tmp_path / "runs")
    vt = R.variant_table(R.sample("dev", preds=DEV_PREDS).head(1))
    with pytest.raises(R.NotReady):
        R.llm_scores(vt, "dev", fake=True)


def test_pooled_rate_and_bootstrap():
    ids = np.repeat(np.arange(10), 3)
    assert R._pooled(ids, np.ones(30, dtype=bool)) == {"variants": 30, "comments": 10, "evaded": 30,
                                                        "rate": 1.0, "lo": 1.0, "hi": 1.0}
    ev = np.zeros(30, dtype=bool)
    ev[:6] = True
    p = R._pooled(ids, ev)
    assert p["evaded"] == 6 and p["rate"] == pytest.approx(0.2) and p["lo"] < 0.2 < p["hi"]
    assert p == R._pooled(ids, ev)


def test_check_originals_flags_a_difference():
    s = R.sample("dev", preds=DEV_PREDS).head(4)
    vt = R.variant_table(s, originals=True)
    scored, _ = R.score_variants(vt, "dev", with_llm=False)
    assert R.check_originals(s, scored, R.BASELINES) == []
    bad = s.copy()
    bad["rules_tier_test_run"] = "allow"
    assert any("rules" in p for p in R.check_originals(bad, scored, R.BASELINES))


def test_dev_rehearsal_matches_the_committed_files():
    assert R.main(["analyse", "--split", "dev", "--no-llm", "--check"]) == 0


def test_dev_rehearsal_shape():
    r = json.loads((R.OUT / "dev.json").read_text(encoding="utf-8"))
    assert r["systems"] == R.BASELINES and r["seed"] == R.SEED
    assert set(r["results"]) == {f"{k}_{c}" for k in R.BASELINES for c in ("as_is", "normalised")}
    for block in r["results"].values():
        assert list(block["by_transform"]) == T.NAMES
        assert block["all_transforms"]["variants"] == sum(b["variants"] for b in block["by_transform"].values())
    v = pd.read_csv(R.OUT / "dev_variants.csv", keep_default_na=False)
    assert "text" not in v.columns and v.text_sha256.str.len().eq(64).all()
