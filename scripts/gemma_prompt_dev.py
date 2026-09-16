#!/usr/bin/env python3
"""
Gemma Prompt Development (Stage 5a) — validation split only
============================================================
Runs the untouched model over the same sample of validation papers with each prompt variant
below and prints one comparison table, so the 5a prompt is chosen from numbers on the
validation shortlists and never from the test set (WORK_PLAN.md Stage 5: "everything about
Gemma is developed on the validation papers"). The script has no --split flag: it can only
read the validation split.

Why this exists: the first prompt made untouched Gemma-3-4B emit a mean of 16.9 picks for
4.3 gold topics, walking down the shown list in order (results/gemma_select/smoke4b_*). The
training carve has 4.3 gold topics per paper on average; 95% of papers have <= 8 and 98% have
<= 10 — the "1 to 10" cap Alkan et al. gave DeepSeek. Each variant states that prior in a
different way. The winner is copied into gemma_common.PROMPT_TEMPLATE (one prompt for both
arms) and this file keeps the record of what was tried.

Outcome (16 Sept, 96 validation papers, TF-IDF stand-in lists, comparison.json): soft priors
barely move the model (13.0 and 11.7 picks); only the hard cap does (9.3, median 10 — it pins
to the cap). cap10f, the committed prompt: F1 0.29 -> 0.39, tail F1 0.19 -> 0.25, tail recall
0.62 -> 0.46 (the cap's cost), 0 parse failures, 47 output tokens/paper instead of 102 (twice
the inference speed), 3% off-list lines — all invented names, which the results table reports.
Re-run this on scibert_full_top50_val.jsonl when it lands: nothing decided on stand-in lists is
final.

Usage:
    python scripts/gemma_prompt_dev.py --n 96
    python scripts/gemma_prompt_dev.py --n 96 --variants prior,cap10      # re-check two on more papers
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import gemma_common
from gemma_common import N_CANDIDATES, stop_token_ids, load_quantized, git_provenance
from gemma_select import read_jsonl, load_papers, load_name_bands, run_pass, summarise

_HEAD = "You are indexing an astronomy paper with the Unified Astronomy Thesaurus (UAT).\n"
_BODY = "Title: {title}\n\nAbstract: {abstract}\n\nCandidate topics:\n{candidates}\n\n"

VARIANTS = {
    # the committed prompt, whatever it is
    "current": gemma_common.PROMPT_TEMPLATE,
    # the first prompt (commit e89c88c): no cardinality guidance at all
    "original": (
        _HEAD
        + "Read the title and abstract, then decide which of the candidate topics apply.\n\n"
        + _BODY
        + "Answer with the candidate topics that apply, most confident first, one per line, "
          "copied exactly as written above. If none apply, answer NONE."
    ),
    # states the cardinality prior and that most candidates are wrong
    "prior": (
        _HEAD
        + "Read the title and abstract, then decide which of the candidate topics apply. Most of the "
          "candidates do not: a paper is indexed with about 4 topics on average, rarely more than 8.\n\n"
        + _BODY
        + "Answer with only the candidate topics that apply, most confident first, one per line, "
          "copied exactly as written above. If none apply, answer NONE."
    ),
    # Alkan et al.'s instruction to DeepSeek: a hard 1-10 cap
    "cap10": (
        _HEAD
        + "Read the title and abstract, then choose between 1 and 10 of the candidate topics: the "
          "ones this paper should be indexed under.\n\n"
        + _BODY
        + "Answer with between 1 and 10 candidate topics, most confident first, one per line, "
          "copied exactly as written above. If none apply, answer NONE."
    ),
    # cap10 with the answer format spelled out: one paper in 96 answered with list numbers
    "cap10n": (
        _HEAD
        + "Read the title and abstract, then choose between 1 and 10 of the candidate topics: the "
          "ones this paper should be indexed under.\n\n"
        + _BODY
        + "Answer with the names of between 1 and 10 candidate topics, most confident first, one per "
          "line, each copied exactly as written above (the topic name, not its number). If none "
          "apply, answer NONE."
    ),
    # cap10n plus "no introduction": 13 of cap10n's 31 off-list lines were a preamble sentence
    "cap10f": (
        _HEAD
        + "Read the title and abstract, then choose between 1 and 10 of the candidate topics: the "
          "ones this paper should be indexed under.\n\n"
        + _BODY
        + "Answer with the names of between 1 and 10 candidate topics, most confident first, one per "
          "line, each copied exactly as written above (the topic name, not its number). Output only "
          "the list, with no introduction or commentary. If none apply, answer NONE."
    ),
    # the prior plus a rule for what "applies" means
    "strict": (
        _HEAD
        + "Read the title and abstract, then decide which of the candidate topics apply. A topic "
          "applies only if the paper is substantially about it, the way a librarian would index it, "
          "not if it is merely mentioned. Most candidates do not apply: a paper is indexed with about "
          "4 topics on average, rarely more than 8. Leave out a doubtful topic.\n\n"
        + _BODY
        + "Answer with only the candidate topics that apply, most confident first, one per line, "
          "copied exactly as written above. If none apply, answer NONE."
    ),
}

COLUMNS = [
    ("mean_picks", "picks"), ("median_picks", "med"), ("emitted_over_gold_in_list", "|y|/|g&l|"),
    ("micro_precision", "P"), ("micro_recall", "R"), ("micro_f1", "F1"),
    ("tail_precision", "tailP"), ("tail_recall", "tailR"), ("tail_f1", "tailF1"),
    ("first_pick_is_top1", "top1"), ("adjacent_in_shown_order", "inorder"),
    ("off_list_rate_lines", "offlist"), ("empty_output_rate", "empty"), ("hit_max_new_tokens_rate", "capped"),
    ("mean_new_tokens", "tokens"),
]


def fmt(v):
    if v is None:
        return "   -  "
    return f"{v:6.1f}" if v >= 10 else f"{v:6.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--shortlist", default=str(PROJECT_ROOT / "results" / "shortlists" / "standin_tfidf_top50_val.jsonl"),
                    help="a handoff-4 file for the VALIDATION papers (stand-in until scibert_full_top50_val.jsonl lands)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list of variant names to run")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "gemma_select" / "prompt_dev"))
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    assert not unknown, f"unknown variants {unknown}; known: {list(VARIANTS)}"

    provenance = git_provenance("scripts/gemma_prompt_dev.py", "scripts/gemma_select.py", "scripts/gemma_common.py")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"gemma_prompt_dev: {args.model} (untouched) | validation | code {provenance.get('commit')}"
          f"{' + UNCOMMITTED' if provenance.get('uncommitted_changes') else ''}", flush=True)

    records = read_jsonl(args.shortlist)
    assert all(len(r["candidates"]) == N_CANDIDATES for r in records), "every shortlist must have exactly 50 candidates"
    papers = load_papers("validation")
    missing = [r["paper_id"] for r in records if r["paper_id"] not in papers]
    assert not missing, f"{len(missing)} shortlist paper_ids are not in the validation split, e.g. {missing[:3]}"
    records = random.Random(args.seed).sample(records, args.n)
    bands = load_name_bands()
    print(f"{len(records)} validation papers (seed {args.seed}); shortlist {args.shortlist}; variants {names}", flush=True)

    t0 = time.time()
    tokenizer, model = load_quantized(args.model)
    model.eval()
    tokenizer.padding_side = "left"
    stop_ids = stop_token_ids(tokenizer)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    table = {}
    for name in names:
        gemma_common.PROMPT_TEMPLATE = VARIANTS[name]
        torch.cuda.reset_peak_memory_stats()
        results, s_per_paper = run_pass(model, tokenizer, records, papers, args.batch_size, stop_ids, args.max_new_tokens)
        summ = summarise(results, papers, True, bands)
        summ["sec_per_paper"] = s_per_paper
        summ["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3
        table[name] = summ
        with open(out_dir / f"{name}_validation.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:8s} {s_per_paper:.2f} s/paper | " + " ".join(f"{short} {fmt(summ.get(key))}" for key, short in COLUMNS), flush=True)
    gemma_common.PROMPT_TEMPLATE = VARIANTS["current"]

    print("\n" + " " * 9 + "".join(f"{short:>10s}" for _, short in COLUMNS))
    for name in names:
        print(f"{name:8s} " + "".join(f"{fmt(table[name].get(key)):>10s}" for key, _ in COLUMNS))
    gold = table[names[0]]
    print(f"\ngold per paper {gold['mean_gold']:.2f}; shortlist ceiling {gold['gold_coverage_by_shortlist']:.3f} "
          f"(tail {gold['tail_coverage_by_shortlist']:.3f}, {gold['tail_gold']} tail gold labels)")

    # Merge into the existing record so a partial re-run keeps every variant ever tried.
    path = out_dir / "comparison.json"
    record = json.load(open(path, encoding="utf-8")) if path.exists() else {"prompts": {}, "metrics": {}, "code": {}}
    if "commit" in record["code"]:  # first-sweep layout: one provenance for all variants
        record["code"] = {k: record["code"] for k in record["metrics"]}
    record.update({
        "model": args.model, "shortlist": args.shortlist, "n_papers": len(records), "seed": args.seed,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "gpu": torch.cuda.get_device_name(0),
    })
    for name in names:
        record["prompts"][name] = VARIANTS[name]
        record["metrics"][name] = table[name]
        record["code"][name] = provenance
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(f"wrote {path} and one <variant>_validation.jsonl per variant")


if __name__ == "__main__":
    main()
