#!/usr/bin/env python3
"""
Gemma Prompt-and-Select (Stage 5a / 5b inference)
=================================================
Reads a shortlist file (handoff 4 format: one JSON line per paper with `paper_id` and 50
`candidates` in the generator's order), shows each paper and its 50 candidates to Gemma, and
writes the §6 picks file: one JSON line per paper with the candidates as shown, Gemma's
ordered on-list picks, and every raw output line that matched no candidate.

Both arms use this script: 5a with the untouched checkpoint, 5b with `--adapter` pointing at
a saved LoRA adapter. The prompt, tokenisation and parser come from `gemma_common.py`; the
commit hash of the code is written into the summary before anything runs (WORK_PLAN Stage 5).

Development happens on the validation file only. With `--split validation` the script also
looks up the gold topics and reports P/R/F1, the gold∩shortlist ceiling and the emitted-vs-gold
cardinality on the papers it ran. With `--split test` no gold is read and no metric is
computed; the test file is run once per arm.

Usage (timing / smoke test, 20 validation papers, batch 1 then 8):
    python scripts/gemma_select.py --shortlist results/shortlists/standin_tfidf_top50_val.jsonl \
        --split validation --n 20 --batch-sizes 1,8 --arm smoke
Full 5a validation pass:
    python scripts/gemma_select.py --shortlist <scibert_full_top50_val.jsonl> --split validation --arm 5a
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from datasets import load_dataset

from scibert_dataset import clean_astronomy_text
from gemma_common import (
    N_CANDIDATES, render_prompt, parse_picks, stop_token_ids, load_quantized, git_provenance,
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_papers(split_name):
    """bibcode -> (title, abstract, gold names) for one split of the persisted data split."""
    split = json.load(open(DATA_DIR / "split.json", encoding="utf-8"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))
    wanted = set(split[split_name])
    ds = load_dataset(split["source"])
    papers = {}
    for part in ("train", "val"):
        for r in ds[part]:
            if r["bibcode"] in wanted:
                papers[r["bibcode"]] = {
                    "title": clean_astronomy_text(r["title"]),
                    "abstract": clean_astronomy_text(r["abstract"]),
                    "gold": [id_to_name[str(u)] for u in r["verified_uat_ids"] if str(u) in id_to_name],
                }
    return papers


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, stop_ids, max_new_tokens):
    """Left-padded batched greedy generation; returns (decoded texts, new-token counts)."""
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
    )
    new = out[:, enc["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
    n_new = [(row != tokenizer.pad_token_id).sum().item() for row in new]
    return decoded, n_new


def run_pass(model, tokenizer, records, papers, batch_size, stop_ids, max_new_tokens):
    """One inference pass over `records` at one batch size, length-grouped for padding
    efficiency; output order is restored to the input order."""
    prompts = [render_prompt(papers[r["paper_id"]]["title"], papers[r["paper_id"]]["abstract"], r["candidates"])
               for r in records]
    order = sorted(range(len(records)), key=lambda i: len(prompts[i]))
    results = [None] * len(records)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        texts, n_new = generate_batch(model, tokenizer, [prompts[i] for i in idx], stop_ids, max_new_tokens)
        for i, text, n in zip(idx, texts, n_new):
            r = records[i]
            picks, off_list = parse_picks(text, r["candidates"])
            results[i] = {
                "paper_id": r["paper_id"], "candidates_shown": r["candidates"],
                "picks": picks, "off_list": off_list, "raw_output": text,
                "n_new_tokens": n, "hit_max_new_tokens": n >= max_new_tokens,
            }
    torch.cuda.synchronize()
    return results, (time.time() - t0) / len(records)


def summarise(results, papers, with_gold):
    n = len(results)
    lines = sum(len(r["picks"]) + len(r["off_list"]) for r in results)
    off = sum(len(r["off_list"]) for r in results)
    s = {
        "papers": n,
        "parse_rate": sum(1 for r in results if r["picks"]) / n,
        "empty_output_rate": sum(1 for r in results if not r["picks"] and not r["off_list"]) / n,
        "off_list_rate_lines": off / lines if lines else 0.0,
        "papers_with_off_list": sum(1 for r in results if r["off_list"]) / n,
        "mean_picks": sum(len(r["picks"]) for r in results) / n,
        "hit_max_new_tokens_rate": sum(1 for r in results if r["hit_max_new_tokens"]) / n,
        "mean_new_tokens": sum(r["n_new_tokens"] for r in results) / n,
    }
    if with_gold:
        tp = fp = fn = 0; ceiling_hit = gold_total = 0; emitted = 0; gold_in_list = 0
        for r in results:
            gold = set(papers[r["paper_id"]]["gold"]); picks = set(r["picks"]); shown = set(r["candidates_shown"])
            tp += len(gold & picks); fp += len(picks - gold); fn += len(gold - picks)
            gold_total += len(gold); ceiling_hit += len(gold & shown)
            emitted += len(r["picks"]); gold_in_list += len(gold & shown)
        p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        s.update({
            "micro_precision": p, "micro_recall": rc, "micro_f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
            "gold_coverage_by_shortlist": ceiling_hit / gold_total if gold_total else None,
            "emitted_over_gold_in_list": emitted / gold_in_list if gold_in_list else None,
            "mean_gold": gold_total / n,
        })
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (5b); omitted = untouched model (5a)")
    ap.add_argument("--shortlist", required=True, help="handoff-4 jsonl: paper_id + 50 candidates in generator order")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--arm", required=True, help="name used in the output file, e.g. 5a, 5b, smoke")
    ap.add_argument("--n", type=int, default=None, help="run only the first n papers (sampled with --seed); dev only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-sizes", default="1", help="comma list; every size runs the same papers (timing); the last one's output is written")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "gemma_select"))
    args = ap.parse_args()

    if args.split == "test" and args.n is not None:
        sys.exit("refusing --n on the test split: the test file is run once, in full, per arm")

    provenance = git_provenance("scripts/gemma_select.py", "scripts/gemma_common.py")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_{args.split}"
    print(f"gemma_select: {args.model}{' + ' + args.adapter if args.adapter else ' (untouched)'} | {args.split} | "
          f"code {provenance.get('commit')}{' + UNCOMMITTED' if provenance.get('uncommitted_changes') else ''}", flush=True)

    records = read_jsonl(args.shortlist)
    assert all(len(r["candidates"]) == N_CANDIDATES for r in records), "every shortlist must have exactly 50 candidates"
    papers = load_papers(args.split)
    missing = [r["paper_id"] for r in records if r["paper_id"] not in papers]
    assert not missing, f"{len(missing)} shortlist paper_ids are not in the {args.split} split, e.g. {missing[:3]}"
    if args.n is not None:
        records = random.Random(args.seed).sample(records, args.n)
    print(f"{len(records)} papers; shortlist file {args.shortlist}", flush=True)

    t0 = time.time()
    tokenizer, model = load_quantized(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tokenizer.padding_side = "left"          # batched generation needs the prompt flush against the output
    stop_ids = stop_token_ids(tokenizer)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    with_gold = args.split == "validation"
    timing, results = {}, None
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        results, s_per_paper = run_pass(model, tokenizer, records, papers, bs, stop_ids, args.max_new_tokens)
        timing[bs] = {"sec_per_paper": s_per_paper, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3}
        summ = summarise(results, papers, with_gold)
        print(f"batch {bs:2d}: {s_per_paper:.2f} s/paper, peak {timing[bs]['peak_reserved_gb']:.2f} GB | parse {summ['parse_rate']:.2f}, "
              f"off-list lines {summ['off_list_rate_lines']:.3f}, empty {summ['empty_output_rate']:.2f}, mean picks {summ['mean_picks']:.1f}"
              + (f" (gold {summ['mean_gold']:.1f}) | P {summ['micro_precision']:.3f} R {summ['micro_recall']:.3f} F1 {summ['micro_f1']:.3f}, "
                 f"ceiling {summ['gold_coverage_by_shortlist']:.3f}, |y|/|gold∩list| {summ['emitted_over_gold_in_list']:.2f}" if with_gold else ""),
              flush=True)

    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "arm": args.arm, "split": args.split, "model": args.model, "adapter": args.adapter,
        "shortlist": args.shortlist, "n_papers": len(records), "sampled": args.n is not None,
        "code": provenance, "max_new_tokens": args.max_new_tokens,
        "timing_by_batch_size": timing, "metrics": summarise(results, papers, with_gold),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_dir / (tag + '.jsonl')} and {tag}_summary.json")


if __name__ == "__main__":
    main()
