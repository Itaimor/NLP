#!/usr/bin/env python3
"""
Gemma unconstrained generation (Stage 6)
========================================
Asks Gemma to name UAT topics for a paper with **no candidate list in the prompt and no
constrained decoding**, on ~200 validation papers, and measures what comes out:

--- out-of-vocabulary rate: emitted lines that match no UAT topic name
--- normalisation-recoverable near-misses vs true hallucinations, split out of those lines
--- duplicate rate: lines the model repeated inside one paper
--- mean predicted cardinality

These are the four numbers `project_full_specification.md:199` asks for, and the paragraph they
support is the justification for the shortlist design (`WORK_PLAN.md:392`, `:803`,
`SETTLED_QUESTIONS.md:341`). Both arms run through this script: the untouched checkpoint, and the
fine-tuned one with `--adapter`.

Three design choices worth stating, because each one could have gone the other way:

1. **The vocabulary is not in the prompt.** "Unconstrained" here means the model is given no list
   at all, which is the condition that makes an out-of-vocabulary rate meaningful. Pasting all
   1,864 names in would be a different experiment (Alkan et al.'s reported failure mode, where the
   model "became overwhelmed by the extensive vocabulary"), and at roughly 25k prompt tokens it
   does not fit beside a 4B model on an 8 GB card anyway.

2. **Decoding is not constrained to the vocabulary**, per `project_full_specification.md:199`:
   constraining it would delete the finding this stage exists to produce.

3. **`max_new_tokens` is 256, where Stage 5 used 160.** With no list to copy from, an over-emitting
   model can ramble, and truncation would silently understate both the duplicate rate and the
   cardinality. The truncation rate is reported so the choice can be checked.

The prompt below is a local copy, deliberately not added to `gemma_common.py`: the Stage 5 prompt
there is committed and frozen, and Stage 6 must not perturb it. It is otherwise the Stage 5 prompt
with the candidate block removed, so the only difference between the two conditions is the shortlist.

The in-vocabulary / out-of-vocabulary split uses the committed `parse_picks()` rule (exact match
after case-folding and list-marker stripping, never fuzzy). The near-miss classification is a
reporting layer applied afterwards to the lines that rule rejected, never a second chance to score.

Usage (smoke test, 8 papers):
    python scripts/gemma_free_generation.py --arm free-smoke --n 8
Untouched arm, the real pass:
    python scripts/gemma_free_generation.py --arm free-5a
Fine-tuned arm:
    python scripts/gemma_free_generation.py --arm free-5b \
        --adapter results/gemma_train/5b/checkpoints_kept/epoch-2/adapter
"""

import os
import re
import sys
import json
import time
import random
import argparse
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
# Same reason as gemma_select.py: without expandable segments the allocator fragments and
# Windows pages GPU memory to RAM, and the same batches run about 3x slower.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from datasets import load_dataset

# Windows consoles default to cp1252; keep progress lines printable everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scibert_dataset import clean_astronomy_text
from gemma_common import parse_picks, stop_token_ids, load_quantized, git_provenance

# The Stage 5 prompt with the candidate block and the "copied exactly as written above" clause
# removed. Everything else, including the "between 1 and 10" cap that Stage 5 settled on, is kept
# so cardinality is comparable across the two conditions.
FREE_PROMPT_TEMPLATE = (
    "You are indexing an astronomy paper with the Unified Astronomy Thesaurus (UAT).\n"
    "Read the title and abstract, then name between 1 and 10 UAT topics: the ones this paper "
    "should be indexed under.\n\n"
    "Title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Answer with the names of between 1 and 10 UAT topics, most confident first, one per line. "
    "Output only the list, with no introduction or commentary. If none apply, answer NONE."
)

# Mirrors gemma_common._LIST_MARKER_RE so raw-line accounting sees exactly what the parser saw.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_SEP_RE = re.compile(r"[-_/]+")


def render_free_prompt(title, abstract):
    return FREE_PROMPT_TEMPLATE.format(title=title, abstract=abstract)


def emitted_lines(text):
    """Every line the model emitted, marker-stripped, minus blanks and the NONE sentinel.
    parse_picks() de-duplicates; the duplicate rate needs the un-deduplicated list."""
    out = []
    for line in text.splitlines():
        line = _LIST_MARKER_RE.sub("", line, count=1).strip()
        if not line or line.upper() == "NONE":
            continue
        out.append(line)
    return out


def norm_punct(s):
    """Case, separator and punctuation fold: 'X-ray Binaries!' -> 'x ray binaries'."""
    s = _SEP_RE.sub(" ", s.lower())
    return " ".join(_PUNCT_RE.sub(" ", s).split())


def _depluralise(tok):
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 2 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def norm_plural(s):
    """norm_punct plus a per-token singular fold, so 'galaxy cluster' meets 'galaxy clusters'."""
    return " ".join(_depluralise(t) for t in norm_punct(s).split())


# Function words only. Dropping these turns 'the interstellar medium' into the vocabulary's
# 'interstellar medium' and 'clusters of galaxies' into 'galaxy clusters', both of which are
# naming differences rather than invented topics. No content word is ever dropped.
_STOPWORDS = frozenset({"a", "an", "the", "of", "in", "on", "for", "and", "to", "with"})


def norm_content(s):
    """norm_plural with function words removed, word order kept."""
    return " ".join(t for t in norm_plural(s).split() if t not in _STOPWORDS)


def build_vocab_indexes(names):
    """Four lookup tables over the 1,864 topic names, one per normalisation rung. Collisions are
    possible (two names folding together), so each table keeps a list and the count is reported."""
    punct, plural, content, wordset = {}, {}, {}, {}
    for n in names:
        punct.setdefault(norm_punct(n), []).append(n)
        plural.setdefault(norm_plural(n), []).append(n)
        content.setdefault(norm_content(n), []).append(n)
        wordset.setdefault(frozenset(norm_content(n).split()), []).append(n)
    return punct, plural, content, wordset


def classify_off_list(line, punct, plural, content, wordset):
    """Which rung, if any, recovers an out-of-vocabulary line. Rungs are tried strongest-first and
    the first hit wins, so a punctuation variant is never credited to a weaker rule.

    The ladder is deliberately conservative: every rung is a naming difference over the same
    content words, never a paraphrase and never a synonym. So `near_miss` is a lower bound on
    recoverable output and `hallucination` is an upper bound on invented topics, which is the
    direction that does not flatter the argument this stage supports.

    Returns (kind, matched_names) with kind one of:
    --- near_miss_punct:    differs only in case, separators or punctuation
    --- near_miss_plural:   differs only in singular/plural form
    --- near_miss_stopword: differs only by function words ('the interstellar medium')
    --- near_miss_wordset:  same content words, different order ('clusters of galaxies')
    --- hallucination:      matches no topic under any rung
    """
    key = norm_punct(line)
    if key in punct:
        return "near_miss_punct", punct[key]
    key = norm_plural(line)
    if key in plural:
        return "near_miss_plural", plural[key]
    key = norm_content(line)
    if key and key in content:
        return "near_miss_stopword", content[key]
    key = frozenset(norm_content(line).split())
    if key and key in wordset:
        return "near_miss_wordset", wordset[key]
    return "hallucination", []


def load_papers(split_name):
    """bibcode -> (title, abstract, gold names) for one split of the persisted data split.
    Same loader as gemma_select.py, so both stages read the papers identically."""
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


def load_vocab_and_bands():
    """The 1,864 topic names in label order, and name -> head/torso/tail under the primary
    (training-basis) band map."""
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))
    label_order = json.load(open(DATA_DIR / "label_order.json", encoding="utf-8"))
    names = [id_to_name[str(u)] for u in label_order]
    bands = json.load(open(DATA_DIR / "band_map_train.json", encoding="utf-8"))["bands"]
    name_band = {id_to_name[u]: b for u, b in bands.items() if u in id_to_name}
    return names, name_band


def sample_paper_ids(shortlist_path, n, seed):
    """Sample from the papers that also have a Stage 5 shortlist, so every Stage 6 paper has a
    prompt-and-select counterpart and the two conditions can be compared on the same papers."""
    ids = []
    with open(shortlist_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["paper_id"])
    ids.sort()  # the file's order is the generator's; sort first so the seed alone fixes the sample
    if n is not None and n < len(ids):
        ids = random.Random(seed).sample(ids, n)
    return ids


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


def run_pass(model, tokenizer, paper_ids, papers, vocab, indexes, batch_size, stop_ids, max_new_tokens):
    """One inference pass, length-grouped for padding efficiency; input order is restored."""
    prompts = [render_free_prompt(papers[p]["title"], papers[p]["abstract"]) for p in paper_ids]
    order = sorted(range(len(paper_ids)), key=lambda i: len(prompts[i]))
    results = [None] * len(paper_ids)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        texts, n_new = generate_batch(model, tokenizer, [prompts[i] for i in idx], stop_ids, max_new_tokens)
        for i, text, n in zip(idx, texts, n_new):
            raw = emitted_lines(text)
            picks, off_list = parse_picks(text, vocab)
            classified = []
            for line in off_list:
                kind, matched = classify_off_list(line, *indexes)
                classified.append({"line": line, "kind": kind, "matched": matched})
            counts = Counter(l.lower() for l in raw)
            # The fine-tuned checkpoint loops on a minority of papers (a decoding artefact the
            # 21 Sept ruling already recorded at 13-21 %). A loop repeats one term dozens of
            # times, so every line-weighted rate below is distorted by it in whichever direction
            # the looped term happens to fall. The unique-line counts are the honest denominator.
            uniq_oov = {c["line"].lower() for c in classified}
            results[i] = {
                "paper_id": paper_ids[i],
                "picks": picks,                      # in-vocabulary, de-duplicated, as ordered
                "off_list": off_list,                # raw lines the committed parser rejected
                "off_list_classified": classified,
                "n_emitted_lines": len(raw),
                "n_unique_lines": len(counts),
                "n_unique_oov_lines": len(uniq_oov),
                "n_duplicate_lines": len(raw) - len(counts),
                "raw_output": text,
                "n_new_tokens": n,
                "hit_max_new_tokens": n >= max_new_tokens,
            }
    torch.cuda.synchronize()
    return results, (time.time() - t0) / len(paper_ids)


def summarise(results, papers, name_band, with_gold):
    n = len(results)
    lines = sum(r["n_emitted_lines"] for r in results)
    uniq = sum(r["n_unique_lines"] for r in results)
    off = sum(len(r["off_list"]) for r in results)
    uniq_off = sum(r["n_unique_oov_lines"] for r in results)
    kinds = Counter(c["kind"] for r in results for c in r["off_list_classified"])
    near = sum(v for k, v in kinds.items() if k.startswith("near_miss"))
    # Same breakdown over unique lines: a looped term is counted once per paper, not once per line.
    kinds_u = Counter()
    for r in results:
        seen = set()
        for c in r["off_list_classified"]:
            k = c["line"].lower()
            if k not in seen:
                seen.add(k)
                kinds_u[c["kind"]] += 1
    near_u = sum(v for k, v in kinds_u.items() if k.startswith("near_miss"))
    loopers = [r for r in results if r["n_duplicate_lines"]]
    s = {
        "papers": n,
        "emitted_lines": lines,
        "unique_lines": uniq,
        # PRIMARY, unique-weighted: immune to the looping artefact. Report these.
        "mean_cardinality_unique": uniq / n,
        "median_cardinality_unique": sorted(r["n_unique_lines"] for r in results)[n // 2],
        "out_of_vocabulary_rate_unique_lines": uniq_off / uniq if uniq else 0.0,
        "near_miss_rate_of_oov_unique": near_u / uniq_off if uniq_off else None,
        "hallucination_rate_of_oov_unique": kinds_u.get("hallucination", 0) / uniq_off if uniq_off else None,
        "oov_breakdown_unique": dict(kinds_u),
        # Line-weighted equivalents, kept for comparison. Distorted wherever a paper looped, so
        # they are equal to the unique-weighted figures only when the duplicate rate is zero.
        "mean_cardinality_emitted": lines / n,
        "mean_cardinality_in_vocab": sum(len(r["picks"]) for r in results) / n,
        "median_cardinality_emitted": sorted(r["n_emitted_lines"] for r in results)[n // 2],
        "out_of_vocabulary_rate_lines": off / lines if lines else 0.0,
        "papers_with_any_oov": sum(1 for r in results if r["off_list"]) / n,
        "near_miss_rate_of_oov": near / off if off else None,
        "hallucination_rate_of_oov": kinds.get("hallucination", 0) / off if off else None,
        "near_miss_rate_lines": near / lines if lines else 0.0,
        "hallucination_rate_lines": kinds.get("hallucination", 0) / lines if lines else 0.0,
        "oov_breakdown": dict(kinds),
        # The looping artefact itself, reported rather than smoothed away.
        "duplicate_rate_lines": sum(r["n_duplicate_lines"] for r in results) / lines if lines else 0.0,
        "papers_with_duplicates": len(loopers) / n,
        "looper_mean_emitted": (sum(r["n_emitted_lines"] for r in loopers) / len(loopers)) if loopers else None,
        "looper_mean_unique": (sum(r["n_unique_lines"] for r in loopers) / len(loopers)) if loopers else None,
        # Output hygiene, same columns Stage 5 reports so the two are readable side by side.
        "empty_output_rate": sum(1 for r in results if not r["n_emitted_lines"]) / n,
        "any_in_vocab_rate": sum(1 for r in results if r["picks"]) / n,
        "hit_max_new_tokens_rate": sum(1 for r in results if r["hit_max_new_tokens"]) / n,
        "mean_new_tokens": sum(r["n_new_tokens"] for r in results) / n,
    }
    if with_gold:
        tp = fp = fn = gold_total = 0
        band_hits = Counter()
        for r in results:
            gold = set(papers[r["paper_id"]]["gold"])
            picks = set(r["picks"])
            tp += len(gold & picks); fp += len(picks - gold); fn += len(gold - picks)
            gold_total += len(gold)
            for p in picks:
                band_hits[name_band.get(p, "unbanded")] += 1
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        in_vocab_total = sum(band_hits.values())
        s.update({
            "micro_precision": p_, "micro_recall": r_,
            "micro_f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
            "mean_gold": gold_total / n,
            # Where free picks land: if they concentrate on the head, the shortlist is what puts
            # rare topics in front of the model at all.
            "in_vocab_band_profile": {k: v / in_vocab_total for k, v in band_hits.items()} if in_vocab_total else {},
            "in_vocab_band_counts": dict(band_hits),
        })
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omitted = untouched model")
    ap.add_argument("--arm", required=True, help="name used in the output file, e.g. free-5a, free-5b")
    ap.add_argument("--split", default="validation", choices=["validation"],
                    help="validation only: Stage 6 is a behavioural study and never touches the test split")
    ap.add_argument("--shortlist", default=str(PROJECT_ROOT / "results" / "scibert_full" / "scibert_full_top50_val.jsonl"),
                    help="paper-id pool; the tracked handoff file, so a fresh clone can reproduce the sample")
    ap.add_argument("--n", type=int, default=200, help="papers to run (spec 7.4 asks for ~200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-sizes", default="8", help="comma list; the last one's output is written")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "gemma_free"))
    args = ap.parse_args()

    provenance = git_provenance("scripts/gemma_free_generation.py", "scripts/gemma_common.py")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_{args.split}"
    print(f"gemma_free_generation: {args.model}{' + ' + args.adapter if args.adapter else ' (untouched)'} | "
          f"{args.split} | code {provenance.get('commit')}"
          f"{' + UNCOMMITTED' if provenance.get('uncommitted_changes') else ''}", flush=True)

    vocab, name_band = load_vocab_and_bands()
    indexes = build_vocab_indexes(vocab)
    collisions = {
        name: sum(len(v) - 1 for v in table.values() if len(v) > 1)
        for name, table in zip(("punct", "plural", "stopword", "wordset"), indexes)
    }
    print(f"vocabulary {len(vocab)} names; normalisation collisions {collisions}", flush=True)

    paper_ids = sample_paper_ids(args.shortlist, args.n, args.seed)
    papers = load_papers(args.split)
    missing = [p for p in paper_ids if p not in papers]
    assert not missing, f"{len(missing)} sampled paper_ids are not in the {args.split} split, e.g. {missing[:3]}"
    print(f"{len(paper_ids)} papers sampled (seed {args.seed}) from {args.shortlist}", flush=True)

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
        results, s_per_paper = run_pass(model, tokenizer, paper_ids, papers, vocab, indexes,
                                        bs, stop_ids, args.max_new_tokens)
        timing[bs] = {"sec_per_paper": s_per_paper, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3}
        summ = summarise(results, papers, name_band, with_gold)
        print(f"batch {bs:2d}: {s_per_paper:.2f} s/paper, peak {timing[bs]['peak_reserved_gb']:.2f} GB | "
              f"unique {summ['mean_cardinality_unique']:.1f}/paper (emitted {summ['mean_cardinality_emitted']:.1f}), "
              f"OOV {summ['out_of_vocabulary_rate_unique_lines']:.3f} "
              f"(near-miss {summ['near_miss_rate_of_oov_unique'] or 0:.2f} / halluc {summ['hallucination_rate_of_oov_unique'] or 0:.2f}), "
              f"loop {summ['papers_with_duplicates']:.3f} of papers, truncated {summ['hit_max_new_tokens_rate']:.2f}"
              + (f" | P {summ['micro_precision']:.3f} R {summ['micro_recall']:.3f} F1 {summ['micro_f1']:.3f}"
                 if with_gold else ""), flush=True)

    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "arm": args.arm, "split": args.split, "model": args.model, "adapter": args.adapter,
        "stage": 6, "prompt_template": FREE_PROMPT_TEMPLATE,
        "vocabulary_size": len(vocab), "normalisation_collisions": collisions,
        "paper_id_pool": args.shortlist, "n_papers": len(paper_ids), "seed": args.seed,
        "constrained_decoding": False, "vocabulary_in_prompt": False,
        "code": provenance, "max_new_tokens": args.max_new_tokens,
        "timing_by_batch_size": timing, "metrics": summarise(results, papers, name_band, with_gold),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_dir / (tag + '.jsonl')} and {tag}_summary.json")


if __name__ == "__main__":
    main()
