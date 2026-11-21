"""Local vs. hosted: run every sample receipt through each provider and score it.

Usage:
  python compare.py                     # both providers
  python compare.py --providers ollama  # just one

Scores per provider:
  - extraction: share of numeric fields copied exactly (totals, tax, every line total)
  - audit match: share of receipts where the findings equal the findings from the correct data
  - avg seconds per receipt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import agent
from audit import audit_receipt
from schema import Receipt

SAMPLES = pathlib.Path(__file__).parent / "sample_data"


def field_scores(got: Receipt, truth: Receipt) -> tuple[int, int]:
    pairs = [(got.total, truth.total), (got.subtotal, truth.subtotal), (got.tax, truth.tax),
             (got.discounts, truth.discounts), (got.service_charge, truth.service_charge), (got.tip, truth.tip),
             (len(got.items), len(truth.items))]
    pairs += [(g.line_total, t.line_total) for g, t in zip(got.items, truth.items)]
    pairs += [(None, t.line_total) for t in truth.items[len(got.items):]]  # missed lines count as wrong
    ok = sum(1 for g, t in pairs if (g is None and t is None) or (g is not None and t is not None and abs(g - t) < 0.005))
    return ok, len(pairs)


def evaluate(provider: str) -> dict:
    llm = agent.get_llm(provider)
    ok = total = matches = 0
    secs = []
    rows = []
    for txt in sorted(SAMPLES.glob("*.txt")):
        truth = Receipt.model_validate(json.loads(txt.with_suffix(".json").read_text()))
        want = {f.code for f in audit_receipt(truth).findings}
        t0 = time.time()
        try:
            rep = agent.run(text=txt.read_text(), llm=llm, provider=None)
            secs.append(time.time() - t0)
            o, n = field_scores(rep.receipt, truth)
            got = {f.code for f in rep.findings}
        except Exception as e:  # a failed extraction counts as zero
            secs.append(time.time() - t0)
            o, n, got = 0, 7 + len(truth.items), {f"ERROR: {type(e).__name__}"}
        ok, total = ok + o, total + n
        matches += got == want
        rows.append((txt.stem, f"{o}/{n}", "yes" if got == want else f"no (got {sorted(got)})"))
    return {"provider": provider, "extraction": ok / total, "audit_match": matches / len(rows),
            "avg_seconds": sum(secs) / len(secs), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["ollama", "anthropic"])
    args = ap.parse_args()
    results = [evaluate(p) for p in args.providers]
    for r in results:
        print(f"\n== {r['provider']} ==")
        for name, fields, match in r["rows"]:
            print(f"  {name:22} fields {fields:>6}   findings match: {match}")
    print("\n| Provider | Fields copied exactly | Audit matches truth | Avg seconds |")
    print("| --- | --- | --- | --- |")
    for r in results:
        print(f"| {r['provider']} | {r['extraction']:.0%} | {r['audit_match']:.0%} | {r['avg_seconds']:.1f} |")


if __name__ == "__main__":
    main()
