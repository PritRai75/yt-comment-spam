"""Red team (step 7): how often six evasion transforms turn caught spam into missed spam.

Design, fixed before any test variant was scored (adversarial/README.md):

  - Systems attacked: rules, logistic regression, the tree and LLM v2 (decision 51).
    LLM v1 is not attacked.
  - Sample: spam under the corrected labels that all four caught at their own
    cut-offs in the committed predictions (results/final/<split>_predictions.csv),
    without the comments the policy quotes, one comment per text (`dup_key`, keeping
    the first in data order); 100 drawn with seed 20261125, or all if fewer.
  - Variants: the six transforms of `adversarial/transforms.py` on the cleaned text;
    a transform with nothing to change in a comment has no variant there.
  - Each baseline scores every variant twice: as is, and with `data/normalize.py` in
    front. LLM v2 normalises every comment itself, so it has one condition. The
    baselines are the same fits as in the test run (train split, frozen settings).
  - Evaded = a variant the system scores below its own spam cut-off (rules >= 1 hit,
    logistic regression and the tree >= 0.5, the LLM >= 3 of 5 votes). Per transform:
    evaded / variants with a Wilson interval. Over all six: the pooled share, with a
    percentile bootstrap that resamples comments (2,000, seed 20261125), since one
    comment's six variants are not independent. Also: variants that leave the
    auto-remove tier (eval/thresholds.json) though their comment was auto-removed.
  - LLM v2's votes for a variant come from the same configuration and the same five
    vote indices as the test run. Where a variant, after cleaning and normalisation,
    is exactly the text the test run already sent for its comment, the request would
    be the same one, so the test run's votes are reused instead of asking again.
    New votes go to results/adversarial/llm/<config>/calls-<split>.jsonl.

    python -m adversarial.redteam sample --split test          # pool, sample, variants, calls needed
    python -m adversarial.redteam run --split test             # LLM v2 votes for the variants
    python -m adversarial.redteam analyse --split test         # write results/adversarial/test.*
    python -m adversarial.redteam analyse --split test --check # compare with the committed files
    python -m adversarial.redteam analyse --split dev --no-llm # rehearsal, baselines only

The test split needs results/final/test_predictions.csv committed and unchanged, so
the sample comes from the pre-registered test run and nothing else scores test
comments. Output files hold IDs, transform names, SHA-256 of the variant texts, scores
and predictions; never comment text.
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from adversarial import transforms
from data.clean import clean_text, dup_key
from data.fetch import ROOT
from data.load import load_comments
from data.normalize import normalize
from eval import lock, thresholds
from eval.final import QUOTED, _plain, _same, _same_csv, dumps, preds_csv
from eval.metrics import predict, wilson

SEED = 20261125
N_SAMPLE = 100
N_BOOT = 2000
BASELINES = ["rules", "logreg", "tree"]
LLM = "llm_v2"
ATTACKED = BASELINES + [LLM]
CUTOFF = {k: thresholds.FLOORS[k] for k in ATTACKED}
CONDITIONS = {"rules": ["as_is", "normalised"], "logreg": ["as_is", "normalised"],
              "tree": ["as_is", "normalised"], LLM: ["normalised"]}
OUT = ROOT / "results" / "adversarial"
LLM_OUT = OUT / "llm"
FAKE_LLM_OUT = ROOT / "results" / "llm-fake" / "adversarial"  # offline runs, not committed


class NotReady(Exception):
    pass


# The sample

def predictions_path(split: str):
    return ROOT / "results" / "final" / f"{split}_predictions.csv"


def require_predictions(split: str) -> None:
    """The test sample is drawn only from the committed, unchanged test predictions."""
    path = predictions_path(split)
    rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    if not path.exists():
        raise NotReady(f"{rel} is missing: run the pre-registered analysis first")
    if split != "test":
        return
    tracked = lock._git("ls-files", "--", rel).stdout.split()
    changed = lock._git("diff", "--name-only", "HEAD", "--", rel).stdout.split()
    if rel not in tracked:
        raise NotReady(f"{rel} is not committed")
    if rel in changed:
        raise NotReady(f"{rel} has uncommitted changes")


def pool(split: str, preds: pd.DataFrame = None) -> pd.DataFrame:
    """Eligible comments in data order: corrected spam caught by all four, not quoted by
    the policy, first comment of each text. Columns row_id, video, text (cleaned), and
    the four systems' scores and tiers in the test run."""
    if preds is None:
        require_predictions(split)
        preds = pd.read_csv(predictions_path(split), keep_default_na=False)
    p = preds.set_index("row_id")
    caught = (p.label_corrected.astype(int) == 1)
    for k in ATTACKED:
        caught &= p[f"{k}_pred"].astype(int) == 1
    caught &= ~p.index.isin(QUOTED[split])
    df = load_comments().set_index("row_id")
    rows = df.loc[[r for r in df.index if r in set(p.index[caught])]]  # data order
    out = pd.DataFrame({"row_id": rows.index, "video": rows.video.to_numpy(),
                        "text": rows.content.map(clean_text).to_numpy(),
                        "key": rows.content.map(dup_key).to_numpy()})
    out = out[~out.key.duplicated()].drop(columns="key").reset_index(drop=True)
    for k in ATTACKED:
        out[f"{k}_score_test_run"] = p.loc[out.row_id, f"{k}_score"].to_numpy(dtype=float)
        out[f"{k}_tier_test_run"] = p.loc[out.row_id, f"{k}_tier"].to_numpy()
    return out


def pool_counts(split: str, preds: pd.DataFrame = None) -> dict:
    if preds is None:
        preds = pd.read_csv(predictions_path(split), keep_default_na=False)
    p = preds.set_index("row_id")
    spam = p.label_corrected.astype(int) == 1
    caught = spam.copy()
    for k in ATTACKED:
        caught &= p[f"{k}_pred"].astype(int) == 1
    return {"corrected_spam": int(spam.sum()), "caught_by_all_four": int(caught.sum()),
            "quoted_by_policy": int((caught & p.index.isin(QUOTED[split])).sum())}


def sample(split: str, n: int = N_SAMPLE, seed: int = SEED, preds: pd.DataFrame = None) -> pd.DataFrame:
    """The seeded sample, in data order."""
    p = pool(split, preds)
    take = np.sort(np.random.default_rng(seed).permutation(len(p))[:n])
    return p.iloc[take].reset_index(drop=True)


def variant_table(s: pd.DataFrame, originals: bool = False) -> pd.DataFrame:
    """One row per (comment, transform) with a variant; text is the cleaned variant.
    With `originals`, each sampled comment first, as transform "none"."""
    rows = []
    for r in s.itertuples():
        if originals:
            rows.append({"row_id": r.row_id, "video": r.video, "transform": "none", "text": r.text})
        for name, v in transforms.variants(r.text).items():
            if v is not None and clean_text(v) != v:
                raise ValueError(f"{r.row_id} {name}: the systems' cleaning would change the variant")
            if v is not None:
                rows.append({"row_id": r.row_id, "video": r.video, "transform": name, "text": v})
    return pd.DataFrame(rows, columns=["row_id", "video", "transform", "text"])


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Scoring

def baseline_scores(texts) -> dict:
    """The three baselines' scores, fitted on train as in the test run."""
    from models import logreg, rules, tree
    train = load_comments().query("split == 'train'")
    texts = list(texts)
    return {"rules": np.array([float(rules.rule_score(t)) for t in texts]),
            "logreg": logreg.spam_proba(logreg.fit(train.content, train.label), texts),
            "tree": tree.spam_proba(tree.fit(train.content, train.label), texts)}


def _llm_setup(split: str, fake: bool):
    from models import llm
    prompt = llm.load_prompt("v2")
    config = llm.make_config(prompt)
    base = FAKE_LLM_OUT if fake else LLM_OUT
    cache = llm.Cache(base / config.name / f"calls-{split}.jsonl")
    original = llm.Cache((llm.FAKE_RUNS if fake else llm.RUNS) / config.name / "r0" / llm.cache_name(split))
    return llm, prompt, config, cache, original


def llm_plan(vt: pd.DataFrame, split: str, fake: bool = False) -> dict:
    """Votes the variants need: already in a cache (this run's or the test run's) or new."""
    llm, prompt, config, cache, original = _llm_setup(split, fake)
    new = reused = done = 0
    seen = set()
    for r in vt.itertuples():
        text = prompt.text(r.text)
        for v in range(llm.N_VOTES):
            key = llm.call_key(config, r.video, text, v)
            if key in seen:
                continue
            seen.add(key)
            if key in cache:
                done += 1
            elif key in original:
                reused += 1
            else:
                new += 1
    return {"config": config.name, "votes_needed": len(seen), "reused_from_run": reused,
            "already_made": done, "to_make": new}


def llm_run(vt: pd.DataFrame, split: str, client=None, fake: bool = False, rpm: float = None,
            limit: int = None, log=print) -> dict:
    """Ask LLM v2 for every vote the variants still need (five per distinct text as sent)."""
    llm, prompt, config, cache, original = _llm_setup(split, fake)
    client = client or (llm.FakeClient() if fake else llm.make_client())
    pacer = llm.Pacer(0 if fake else (rpm or llm.RPM))
    if limit is not None:
        vt = vt.head(limit)
    made = invalid = 0
    started = time.monotonic()
    for i, r in enumerate(vt.itertuples(), 1):
        text = prompt.text(r.text)
        for v in range(llm.N_VOTES):
            key = llm.call_key(config, r.video, text, v)
            if key in cache or key in original:
                continue
            pacer.wait()
            rec = llm.one_vote(client, config, prompt, r.video, text)
            rec.update(key=key, row_id=f"{r.row_id}|{r.transform}", vote=v, replicate=0, config=config.hash)
            cache.add(rec)
            made += 1
            invalid += not rec["valid"]
        if i % 25 == 0:
            log(f"{i}/{len(vt)} variants, {made} calls made, {invalid} invalid")
    return {"config": config.name, "variants": len(vt), "calls_made": made, "invalid": invalid,
            "seconds": round(time.monotonic() - started, 1), "path": str(cache.path)}


def llm_scores(vt: pd.DataFrame, split: str, fake: bool = False) -> tuple:
    """LLM v2's score for each variant, and usage of the votes this red team made."""
    llm, prompt, config, cache, original = _llm_setup(split, fake)
    scores, keys = [], set()
    for r in vt.itertuples():
        text = prompt.text(r.text)
        recs = []
        for v in range(llm.N_VOTES):
            key = llm.call_key(config, r.video, text, v)
            rec = cache.records.get(key) or original.records.get(key)
            if rec is None:
                raise NotReady(f"{r.row_id} {r.transform}: LLM v2 votes missing; run first")
            if key in cache.records:
                keys.add(key)
            recs.append(rec)
        scores.append(llm.vote_score(recs)["score"])
    made = [cache.records[k] for k in keys]
    usage = {"config": config.name, "calls": len(made), "attempts": sum(r["attempts"] for r in made),
             "invalid": sum(not r["valid"] for r in made),
             "input_tokens": sum(r["input_tokens"] for r in made),
             "output_tokens": sum(r["output_tokens"] for r in made),
             "latency_s": round(sum(r["latency_ms"] for r in made) / 1000, 1)}
    return np.array(scores, dtype=float), usage


# Analysis

def _rate(evaded: int, n: int) -> dict:
    lo, hi = wilson(evaded, n)
    return {"variants": n, "evaded": evaded, "rate": evaded / n if n else None,
            "lo": lo if n else None, "hi": hi if n else None}


def _pooled(row_ids: np.ndarray, evaded: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Share evaded over all variants, with a bootstrap over comments."""
    ids, inv = np.unique(row_ids, return_inverse=True)
    per_ev = np.bincount(inv, weights=evaded.astype(float), minlength=len(ids))
    per_n = np.bincount(inv, minlength=len(ids)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    boot = per_ev[idx].sum(axis=1) / per_n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"variants": int(len(evaded)), "comments": int(len(ids)), "evaded": int(evaded.sum()),
            "rate": float(evaded.mean()), "lo": float(lo), "hi": float(hi)}


def score_variants(vt: pd.DataFrame, split: str, with_llm: bool = True, fake: bool = False) -> tuple:
    """Every attacked system's score, prediction and tier for every row of `vt` (made with
    originals=True, so the sampled comments are scored too, as a check on the test run)."""
    th = json.loads(thresholds.PATH.read_text(encoding="utf-8"))["systems"]
    table = vt
    out = table[["row_id", "video", "transform"]].copy()
    out["text_sha256"] = table.text.map(sha256)
    texts = {"as_is": table.text.tolist(), "normalised": [normalize(t) for t in table.text]}
    for cond in ("as_is", "normalised"):
        sc = baseline_scores(texts[cond])
        for k in BASELINES:
            out[f"{k}_{cond}_score"] = sc[k]
    usage = None
    if with_llm:
        out[f"{LLM}_normalised_score"], usage = llm_scores(table, split, fake)
    systems = ATTACKED if with_llm else BASELINES
    for k in systems:
        for cond in CONDITIONS[k]:
            col = f"{k}_{cond}"
            out[f"{col}_pred"] = predict(out[f"{col}_score"].to_numpy(), CUTOFF[k])
            out[f"{col}_tier"] = thresholds.tiers(out[f"{col}_score"].to_numpy(), th[k]["auto_remove"],
                                                  th[k]["allow_below"])
    return out, usage


def check_originals(s: pd.DataFrame, scored: pd.DataFrame, systems) -> list:
    """Rescored sampled comments must match the test run: spam, same score and tier."""
    o = scored[scored["transform"] == "none"].set_index("row_id").loc[s.row_id]
    problems = []
    for k in systems:
        cond = "as_is" if k in BASELINES else "normalised"
        if not (o[f"{k}_{cond}_pred"] == 1).all():
            problems.append(f"{k}: a sampled comment is not caught when rescored")
        if not (o[f"{k}_{cond}_tier"].to_numpy() == s[f"{k}_tier_test_run"].to_numpy()).all():
            problems.append(f"{k}: a sampled comment's tier differs from the test run")
        if not np.allclose(o[f"{k}_{cond}_score"].to_numpy(), s[f"{k}_score_test_run"].to_numpy(),
                           rtol=1e-9, atol=1e-9):
            problems.append(f"{k}: a sampled comment's score differs from the test run")
    return problems


def analyse(split: str, with_llm: bool = True, fake: bool = False, preds: pd.DataFrame = None) -> tuple:
    s = sample(split, preds=preds)
    vt = variant_table(s, originals=True)
    scored, usage = score_variants(vt, split, with_llm, fake)
    systems = ATTACKED if with_llm else BASELINES
    problems = check_originals(s, scored, BASELINES if fake else systems)
    if problems:
        raise ValueError("; ".join(problems))
    orig_tier = scored[scored["transform"] == "none"].set_index("row_id")
    var = scored[scored["transform"] != "none"]
    result = {"split": split, "seed": SEED, "sample_size": N_SAMPLE,
              "pool": dict(pool_counts(split, preds), distinct_texts=len(pool(split, preds))),
              "sampled": len(s), "sampled_by_video": s.video.value_counts().sort_index().to_dict(),
              "systems": systems, "cutoffs": {k: CUTOFF[k] for k in systems},
              "variants_by_transform": var["transform"].value_counts().reindex(transforms.NAMES).fillna(0).astype(int).to_dict(),
              "results": {}}
    for k in systems:
        for cond in CONDITIONS[k]:
            col = f"{k}_{cond}"
            base_col = f"{k}_{'as_is' if k in BASELINES else 'normalised'}_tier"
            block = {"by_transform": {}}
            for name in transforms.NAMES:
                v = var[var["transform"] == name]
                ev = (v[f"{col}_pred"] == 0).to_numpy()
                was_removed = orig_tier.loc[v.row_id, base_col].to_numpy() == "remove"
                still = v[f"{col}_tier"].to_numpy() == "remove"
                block["by_transform"][name] = dict(_rate(int(ev.sum()), len(v)),
                                                   auto_removed_before=int(was_removed.sum()),
                                                   left_auto_remove=int((was_removed & ~still).sum()))
            ev = (var[f"{col}_pred"] == 0).to_numpy()
            block["all_transforms"] = _pooled(var.row_id.to_numpy(), ev)
            result["results"][col] = block
    if with_llm:
        result["llm_usage"] = dict(usage, **llm_plan(vt, split, fake))
    cols = ["row_id", "video", "transform", "text_sha256"] + [c for c in scored.columns
                                                            if c.endswith(("_score", "_pred", "_tier"))]
    return _plain(result), scored[cols].reset_index(drop=True)


# Command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["dev", "test"], required=True)
        p.add_argument("--fake", action="store_true", help="offline fake LLM client; writes under results/llm-fake/")
        if name == "run":
            p.add_argument("--rpm", type=float, default=None)
            p.add_argument("--limit", type=int, default=None, help="only the first N variants")
        if name == "analyse":
            p.add_argument("--no-llm", action="store_true", help="baselines only (the dev rehearsal)")
            p.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    args = ap.parse_args(argv)
    try:
        require_predictions(args.split)
        if args.cmd == "sample":
            s = sample(args.split)
            vt = variant_table(s, originals=True)
            print(json.dumps({"pool": dict(pool_counts(args.split), distinct_texts=len(pool(args.split))),
                              "sampled": len(s), "by_video": s.video.value_counts().sort_index().to_dict(),
                              "variants_by_transform": vt["transform"][vt["transform"] != "none"].value_counts().reindex(transforms.NAMES)
                              .fillna(0).astype(int).to_dict(),
                              "llm_v2": llm_plan(vt, args.split, args.fake)}, indent=2))
            return 0
        if args.cmd == "run":
            vt = variant_table(sample(args.split), originals=True)
            print(json.dumps(llm_run(vt, args.split, fake=args.fake, rpm=args.rpm, limit=args.limit)))
            print(json.dumps(llm_plan(vt, args.split, args.fake)))
            return 0
        result, table = analyse(args.split, with_llm=not args.no_llm, fake=args.fake)
    except NotReady as e:
        print(e, file=sys.stderr)
        return 2
    base = OUT if not args.fake else FAKE_LLM_OUT
    js, csv = base / f"{args.split}.json", base / f"{args.split}_variants.csv"
    if args.check:
        bad = 0
        for path, ok in ((js, lambda: _same(json.loads(js.read_text(encoding="utf-8")), json.loads(dumps(result)))),
                         (csv, lambda: _same_csv(csv, table))):
            same = path.exists() and ok()
            print(f"{'OK' if same else 'DIFFERS'}: {path.relative_to(ROOT).as_posix()}")
            bad += not same
        return 1 if bad else 0
    base.mkdir(parents=True, exist_ok=True)
    js.write_text(dumps(result), encoding="utf-8", newline="\n")
    csv.write_text(preds_csv(table), encoding="utf-8", newline="\n")
    for col, block in result["results"].items():
        a = block["all_transforms"]
        per = ", ".join(f"{n} {b['evaded']}/{b['variants']}" for n, b in block["by_transform"].items())
        print(f"{col}: evaded {a['evaded']}/{a['variants']} = {a['rate']:.1%} ({a['lo']:.1%} to {a['hi']:.1%}); {per}")
    print(f"wrote {js.relative_to(ROOT).as_posix()} and {csv.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
