"""The README's numbers, written from the result files so every one can be traced.

Four blocks, each rendered from files in the repo and nothing else:

  results  results/final/<split>.json (the pre-registered analysis): each system at its
           own cut-off on corrected labels with intervals; F1 on original labels and on
           the two sensitivity sets; H1 and H2; the enforcement tiers; LLM running cost,
           priced at PRICE.
  redteam  results/adversarial/<split>.json: evasion per transform and system, as is
           and with normalisation in front; LLM v2's red-team calls and their cost.
  example  one real comment as every system classified it, chosen by a fixed rule
           (EXAMPLE_RULE) from results/final/<split>_predictions.csv, with the audit's
           reason and each LLM's first stated reason from its vote cache.
  audit    the label audit record (audit/*.csv through eval/labels.py): corrections,
           the test spam the review judged not spam, the blind re-check.

A block whose result file is not there yet renders as a one-line "pending" note.

    python -m eval.report --split dev            # write results/final/dev_report.md (rehearsal)
    python -m eval.report --split dev --check    # compare with the committed file
    python -m eval.report --readme               # write the test blocks into README.md
    python -m eval.report --readme --check       # the README blocks match the result files

In README.md each block sits between `<!-- report:NAME -->` and `<!-- /report:NAME -->`.
"""
import argparse
import json
import re
import sys

import pandas as pd

from audit.flags import read
from data.fetch import ROOT
from eval.labels import label_table
from eval.metrics import cost_per_1000, wilson

README = ROOT / "README.md"
FINAL = ROOT / "results" / "final"
ADVERSARIAL = ROOT / "results" / "adversarial"
BLOCKS = ["results", "example", "redteam", "audit"]

NAMES = {
    "rules": "Keyword rules",
    "logreg": "TF-IDF + logistic regression",
    "tree": "Gradient boosting, readable features",
    "llm_v1": "LLM v1 (policy only)",
    "llm_v2": "LLM v2 (+ 8 examples, normalisation)",
}
CUTOFF_TEXT = {"rules": "≥ 1 rule", "logreg": "p ≥ 0.5", "tree": "p ≥ 0.5", "llm_v1": "≥ 3 of 5 votes",
               "llm_v2": "≥ 3 of 5 votes"}
TRANSFORM_TEXT = {
    "split_links": "Broken-up links", "leetspeak": "Leetspeak", "lookalikes": "Look-alike letters",
    "zero_width": "Zero-width characters", "benign_wrapper": "Benign wrapper", "injection": "Injection note",
}
# Price of the model the LLM systems ran on, for cost per 1,000 comments (pre-registration:
# the Bedrock price on the run date, with its source).
PRICE = {
    "usd_per_mtok_input": 1.00,
    "usd_per_mtok_output": 5.00,
    "what": "Claude Haiku 4.5 on Amazon Bedrock, Standard tier, global cross-region inference",
    "source": "Anthropic model pricing list (Amazon Bedrock section), effective 31 Aug 2026, "
              "https://assets.claude.com/9e03129acc36d31970777d336f969bb53dc84355.pdf",
    "read_on": "2026-11-26",
}


# Formatting

def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def ci3(m: dict, key: str) -> str:
    return f"{f3(m[key])} ({f3(m[key + '_lo'])}–{f3(m[key + '_hi'])})"


def num(n) -> str:
    return f"{n:,.0f}"


def pval(p: float) -> str:
    if p is None:
        return "–"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def table(header: list, rows: list, left: int = 1) -> str:
    """Markdown table: the first `left` columns left-aligned, the rest right-aligned (numbers)."""
    align = ["---"] * left + ["---:"] * (len(header) - left)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pending(what: str, path) -> str:
    return f"_Pending: {what} (`{path.relative_to(ROOT).as_posix()}` is not there yet)._"


# Blocks

def render_results(split: str) -> str:
    path = FINAL / f"{split}.json"
    if not path.exists():
        return pending("the single test run" if split == "test" else f"the {split} analysis", path)
    r = json.loads(path.read_text(encoding="utf-8"))
    res = r["results"]
    main = res["corrected"]["all"]
    out = [f"**{split.capitalize()}: {num(main['comments'])} scored comments from videos no system was trained on, "
           f"{num(main['spam'])} spam under the corrected labels.** Each system at its own spam cut-off; "
           "95% intervals (Wilson; F1 by bootstrap).", ""]
    rows = []
    for k, e in main["systems"].items():
        m = e["metrics"]
        rows.append([NAMES[k], CUTOFF_TEXT[k], ci3(m, "precision"), ci3(m, "recall"), ci3(m, "f1"),
                     ci3(m, "fpr"), f3(e["average_precision"]), f"{m['tp']} / {m['fp']} / {m['fn']}"])
    out.append(table(["System", "Spam if", "Precision", "Recall", "F1", "False-positive rate", "Avg. precision",
                      "TP / FP / FN"], rows))
    out += ["", "**F1 on other label and comment sets** (secondary; same cut-offs).", ""]
    sets = [("corrected", "all", "Corrected labels"), ("original", "all", "Original labels"),
            ("corrected", "no_share_text", "Corrected, without share text"),
            ("corrected", "no_quoted", "Corrected, without the comments the policy quotes")]
    header = ["System"] + [f"{t} ({num(res[w][s]['comments'])})" for w, s, t in sets]
    rows = [[NAMES[k]] + [f3(res[w][s]["systems"][k]["metrics"]["f1"]) for w, s, _ in sets] for k in main["systems"]]
    out.append(table(header, rows))
    out += ["", "**Hypotheses** (exact McNemar on paired predictions, Holm's adjustment across the two, "
            f"α = {r['alpha']}). Discordant: comments only the first system got right / only the second.", ""]
    rows = []
    for w in ("corrected", "original"):
        for h, x in res[w]["all"]["hypotheses"].items():
            f = x["f1"]
            rows.append([f"{h}{'' if w == 'corrected' else ' (original labels)'}",
                         f"{NAMES[x['a']]} vs {NAMES[x['b']]}", f"{x['a_only_right']} / {x['b_only_right']}",
                         pval(x["p"]), pval(x["p_holm"]), "yes" if x["reject_at_alpha"] else "no",
                         f"{f['difference']:+.3f} ({f['lo']:+.3f} to {f['hi']:+.3f})"])
    out.append(table(["Hypothesis", "Systems", "Discordant", "p", "Holm p", "Reject H0", "F1 difference (95% CI)"], rows))
    out += ["", "**Enforcement tiers** (thresholds chosen on dev, corrected labels; `eval/thresholds.json`).", ""]
    rows = []
    for k, e in main["systems"].items():
        t = e["enforcement"]
        rm = t["remove"]
        prec = "–" if rm["precision"] is None else (f"{f3(rm['precision'])} ({f3(rm['precision_lo'])}–"
                                                     f"{f3(rm['precision_hi'])})")
        rows.append([NAMES[k], num(rm["n"]), prec, pct(t["remove_recall"]), num(t["review"]["n"]),
                     f"{t['review_per_1000']:.0f}", pct(t["remove_or_review_recall"])])
    out.append(table(["System", "Auto-removed", "Auto-removal precision", "Spam auto-removed", "Sent to review",
                      "Review per 1,000 comments", "Spam removed or reviewed"], rows))
    out += ["", f"**LLM running cost** at US${PRICE['usd_per_mtok_input']:.2f} per million input and "
            f"US${PRICE['usd_per_mtok_output']:.2f} per million output tokens ({PRICE['what']}; "
            f"source: {PRICE['source']}, read {PRICE['read_on']}). Five votes per comment; comments with the same "
            "text under one video share their votes.", ""]
    rows = []
    for k, u in r["llm_usage"].items():
        usd = cost_per_1000(u["comments"], u["input_tokens"], u["output_tokens"], PRICE["usd_per_mtok_input"],
                            PRICE["usd_per_mtok_output"])
        rows.append([NAMES[k], num(u["calls"]), num(u["invalid"]), num(u["input_tokens_per_1000"]),
                     num(u["output_tokens_per_1000"]), f"{u['model_seconds_per_1000'] / 60:.0f}", f"${usd:.2f}"])
    out.append(table(["System", "Calls", "Invalid", "Input tokens per 1,000 comments",
                      "Output tokens per 1,000", "Model minutes per 1,000", "US$ per 1,000"], rows))
    return "\n".join(out)


def render_redteam(split: str) -> str:
    path = ADVERSARIAL / f"{split}.json"
    if not path.exists():
        return pending("the red-team run" if split == "test" else f"the {split} red-team rehearsal", path)
    r = json.loads(path.read_text(encoding="utf-8"))
    res = r["results"]
    pool = r["pool"]
    out = [f"**{split.capitalize()}: {r['sampled']} spam comments that every attacked system caught** "
           f"(drawn with seed {r['seed']} from {pool['distinct_texts']} distinct texts; "
           f"attacked: {'; '.join(NAMES[k] for k in r['systems'])}). Evaded = scored below the system's own cut-off. "
           "Cells: evaded / variants, as is → with normalisation in front.", ""]
    systems = r["systems"]

    def cell(k, name):
        def one(cond):
            b = res.get(f"{k}_{cond}")
            if b is None:
                return None
            b = b["all_transforms"] if name is None else b["by_transform"][name]
            return b
        a, n = one("as_is"), one("normalised")
        if a is None:
            return f"{n['evaded']}/{n['variants']} ({pct(n['rate'], 0)}), normalised by the system"
        return f"{a['evaded']}/{a['variants']} ({pct(a['rate'], 0)}) → {n['evaded']}/{n['variants']} ({pct(n['rate'], 0)})"

    rows = [[TRANSFORM_TEXT[t]] + [cell(k, t) for k in systems] for t in TRANSFORM_TEXT]
    rows.append(["**All six**"] + [cell(k, None) for k in systems])
    out.append(table(["Transform"] + [NAMES[k] for k in systems], rows))
    pooled = []
    for k in systems:
        for cond in ("as_is", "normalised"):
            b = res.get(f"{k}_{cond}")
            if b:
                a = b["all_transforms"]
                pooled.append(f"{NAMES[k]} {cond.replace('_', ' ')} {pct(a['rate'])} ({pct(a['lo'])}–{pct(a['hi'])})")
    out += ["", "Pooled evasion with a bootstrap 95% interval over comments: " + "; ".join(pooled) + "."]
    u = r.get("llm_usage")
    if u:
        cost = (u["input_tokens"] * PRICE["usd_per_mtok_input"] + u["output_tokens"] * PRICE["usd_per_mtok_output"]) / 1e6
        out += ["", f"LLM v2's red-team votes: {num(u['calls'])} new calls, {num(u['invalid'])} invalid (another "
                    f"{num(u['reused_from_run'])} reused from the test run), {num(u['input_tokens'])} input and "
                    f"{num(u['output_tokens'])} output tokens: US${cost:.2f} at the price above."]
    return "\n".join(out)


EXAMPLE_RULE = ("the first {split} comment, in data order, whose label the audit corrected and on which the "
                "keyword rules and LLM v2 disagree, leaving out the comments the policy quotes")
LABEL = {0: "ham", 1: "spam"}


def pick_example(preds: pd.DataFrame, quoted) -> pd.Series:
    """The row EXAMPLE_RULE names, or None. `preds` is in data order, as eval/final.py writes it."""
    hit = preds[(preds.label_original != preds.label_corrected) & (preds.rules_pred != preds.llm_v2_pred)
                & ~preds.row_id.isin(list(quoted))]
    return None if hit.empty else hit.iloc[0]


def _llm_votes(version: str, video: str, content: str, split: str) -> list:
    from models import llm
    prompt = llm.load_prompt(version)
    config = llm.make_config(prompt)
    cache = llm.Cache(llm.run_dir(config) / llm.cache_name(split))
    text = prompt.text(content)
    return [cache.records[llm.call_key(config, video, text, v)] for v in range(llm.N_VOTES)]


def render_example(split: str) -> str:
    path = FINAL / f"{split}_predictions.csv"
    if not path.exists():
        return pending("the single test run" if split == "test" else f"the {split} analysis", path)
    from data.clean import clean_text
    from data.load import load_comments
    from eval.final import QUOTED
    rule = EXAMPLE_RULE.format(split=split)
    row = pick_example(pd.read_csv(path, keep_default_na=False), QUOTED[split])
    if row is None:
        return f"_No {split} comment meets the rule ({rule})._"
    com = load_comments().set_index("row_id").loc[row.row_id]
    corr = read("corrections.csv").set_index("row_id").loc[row.row_id]
    out = [f"Chosen by a fixed rule: {rule}. `{row.row_id}`, as every system received it after cleaning:", "",
           "```", clean_text(com.content), "```", "",
           f"Dataset label **{LABEL[int(row.label_original)]}**; corrected label **{LABEL[int(row.label_corrected)]}** "
           f"(audit reason: {corr.reason}).", ""]
    rows = []
    for k in NAMES:
        score, pred = float(row[f"{k}_score"]), int(row[f"{k}_pred"])
        said = ""
        if k == "rules":
            from models.rules import rule_hits
            shown = f"{int(score)} rule famil{'y' if int(score) == 1 else 'ies'} ({', '.join(rule_hits(com.content))})"
        elif k in ("logreg", "tree"):
            shown = f"p = {score:.3f}" if score >= 0.001 else "p < 0.001"
        else:
            votes = _llm_votes(k.split("_")[1], com.video, com.content, split)
            spam = sum(v["valid"] and v["label"] == "spam" for v in votes)
            shown = f"{spam} of {len(votes)} votes spam"
            said = f"\"{votes[0]['reason']}\""
        rows.append([NAMES[k], CUTOFF_TEXT[k], shown, f"**{LABEL[pred]}**", said])
    out.append(table(["System", "Spam if", "Score", "Decision", "First vote's stated reason"], rows, left=5))
    return "\n".join(out)


def render_audit() -> str:
    t = label_table()
    corr = read("corrections.csv")
    by = corr.groupby(["split", "label_before", "label_after"]).size()
    spam_test = t[(t.split == "test") & (t.label_original == 1)]
    k, n = int((spam_test.label_corrected == 0).sum()), len(spam_test)
    lo, hi = wilson(k, n)
    ham = t[(t.label_original == 0) & (t.review != "") & t.split.isin(["dev", "test"])]
    rc = read("recheck_verdicts.csv")
    same = rc.same_label.astype(str).str.lower() == "true"
    unsure_first = (~same & (rc.first_verdict == "unsure")).sum()
    rows = [[s, num(by.get((s, "spam", "ham"), 0)), num(by.get((s, "ham", "spam"), 0))]
            for s in ("train", "dev", "test")]
    rows.append(["**All**", num((corr.label_before == "spam").sum()), num((corr.label_before == "ham").sum())])
    out = [f"**{num(len(corr))} label corrections** (`audit/corrections.csv`), each with the policy reason:", "",
           table(["Split", "Spam → ham", "Ham → spam"], rows), "",
           f"- Test comments labelled spam that the review judged not spam under the policy: {k} of {n} "
           f"({pct(k / n)}, 95% CI {pct(lo)}–{pct(hi)}). Every dev and test spam comment was reviewed.",
           f"- Ham comments reviewed (100 drawn from each of dev and test): {int((ham.label_corrected == 1).sum())} "
           f"of {len(ham)} judged spam.",
           f"- Blind re-check of {len(rc)} audit flags two days later: {int(same.sum())} of {len(rc)} gave the same "
           f"label as the first review; " + (f"the other {len(rc) - int(same.sum())} had all been answered unsure "
           "the first time" if int(unsure_first) == len(rc) - int(same.sum()) else
           f"of the other {len(rc) - int(same.sum())}, {int(unsure_first)} had been answered unsure the first time")
           + ". One labeller, so this is consistency, not agreement between raters."]
    return "\n".join(out)


def render(name: str, split: str) -> str:
    if name == "results":
        return render_results(split)
    if name == "redteam":
        return render_redteam(split)
    if name == "audit":
        return render_audit()
    if name == "example":
        return render_example(split)
    raise ValueError(name)


def report(split: str) -> str:
    titles = {"results": "Results", "example": "One comment, as each system classified it", "redteam": "Red team",
              "audit": "Label audit"}
    parts = [f"# Report rehearsal on {split}\n\nWritten by `python -m eval.report --split {split}` "
             "from the result files; the README's blocks are rendered the same way from the test files.\n"]
    for b in BLOCKS:
        parts.append(f"## {titles[b]}\n\n{render(b, split)}\n")
    return "\n".join(parts)


# README blocks

def _marker(name: str):
    return re.compile(rf"(<!-- report:{name} -->\n)(.*?)(\n<!-- /report:{name} -->)", re.DOTALL)


def fill(text: str, split: str = "test") -> str:
    for name in BLOCKS:
        rx = _marker(name)
        if len(rx.findall(text)) != 1:
            raise ValueError(f"README needs exactly one <!-- report:{name} --> block")
        body = render(name, split)
        text = rx.sub(lambda m: m.group(1) + body + m.group(3), text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--readme", action="store_true", help="write the test blocks into README.md")
    ap.add_argument("--check", action="store_true", help="compare; write nothing")
    args = ap.parse_args(argv)
    if args.readme == (args.split is not None):
        ap.error("give either --split dev|test or --readme")
    if args.readme:
        path = README
        old = path.read_text(encoding="utf-8")
        new = fill(old)
    else:
        path = FINAL / f"{args.split}_report.md"
        old = path.read_text(encoding="utf-8") if path.exists() else None
        new = report(args.split)
    rel = path.relative_to(ROOT).as_posix()
    if args.check:
        same = old == new
        print(f"{'OK' if same else 'DIFFERS'}: {rel}")
        return 0 if same else 1
    path.write_text(new, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
