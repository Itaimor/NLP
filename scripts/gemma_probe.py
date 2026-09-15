#!/usr/bin/env python3
"""
Gemma QLoRA Training Probe (Stage 5 gate)
=========================================
Decides whether Gemma-5b fine-tunes at 4B or falls back to 1B (WORK_PLAN.md, Stage 5:
"Model size is decided by the probe, not chosen in advance").

Trains a Gemma-3 checkpoint in 4-bit (QLoRA) on the RTX 3060 Ti at 1,024 tokens, batch 1,
for at least 500 steps on 64 training papers, and records:
- peak GPU memory (allocated and reserved), seconds per step, gradient norm
- the loss at every step, checked finite; the run stops at the first NaN / inf
- per-example loss in eval mode (dropout off) before and after training, so "does it overfit
  the subset" is judged per paper and not on a mean over targets of very different lengths
- greedy generation on a random sample of the papers, before training (zero-shot control)
  and after, parsed with the Stage 5 rule
- trainable-parameter count (the compute-cost row, WORK_PLAN Stage 5 item 4)
- the git commit of the code that ran, and the saved LoRA adapter

Nothing tuned here is kept. The prompt has the real shape (title + abstract + 50 candidate
names) so the sequence length is honest; the 50 candidates are the paper's gold topics padded
with random topics from the 1,864-label space, because SciBERT's shortlists do not exist yet.
Random fillers are the easiest distractors there are: this probe says nothing about accuracy.

Usage:
    python scripts/gemma_probe.py                       # 4B, 500 steps, 64 papers
    python scripts/gemma_probe.py --model google/gemma-3-1b-it
    python scripts/gemma_probe.py --dry-run             # prompts + token stats, no model
"""

import os
import sys
import json
import time
import math
import random
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
import torch.nn.functional as F
from datasets import load_dataset

from scibert_dataset import clean_astronomy_text
from gemma_common import (
    N_CANDIDATES, render_prompt, encode_training_example, parse_picks, stop_token_ids,
    load_quantized, git_provenance,
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_probe_papers(n_papers, seed):
    """
    Samples n_papers bibcodes from the persisted 15,822-paper training carve and returns
    their dataset rows, plus the id -> name map for the 1,864-label space.
    """
    split = json.load(open(DATA_DIR / "split.json", encoding="utf-8"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))

    rng = random.Random(seed)
    chosen = set(rng.sample(split["train"], n_papers))

    ds = load_dataset(split["source"])["train"]
    rows = [r for r in ds if r["bibcode"] in chosen]
    rows.sort(key=lambda r: r["bibcode"])
    assert len(rows) == n_papers, f"expected {n_papers} rows, found {len(rows)}"
    return rows, id_to_name


def build_example(row, id_to_name, all_names, rng):
    """
    One paper -> prompt fields, candidates, gold, target. Candidates = gold + random fillers
    to N_CANDIDATES, shuffled (a stand-in for SciBERT's top-50). Target = gold names one per
    line in the dataset's order (the real 5b file orders them by SciBERT's score).
    """
    gold = [id_to_name[str(i)] for i in row["verified_uat_ids"] if str(i) in id_to_name]
    gold = gold[:N_CANDIDATES]
    gold_set = set(gold)
    fillers = rng.sample([n for n in all_names if n not in gold_set], N_CANDIDATES - len(gold))
    candidates = gold + fillers
    rng.shuffle(candidates)
    return {
        "bibcode": row["bibcode"],
        "title": clean_astronomy_text(row["title"]),
        "abstract": clean_astronomy_text(row["abstract"]),
        "candidates": candidates,
        "gold": gold,
        "target": "\n".join(gold) if gold else "NONE",
    }


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(model_name, attn_impl, lora_r, lora_alpha):
    from peft import LoraConfig, get_peft_model

    tokenizer, model = load_quantized(model_name, attn_impl)
    weights_gb = torch.cuda.memory_allocated() / 1024 ** 3

    # Manual k-bit prep. peft's prepare_model_for_kbit_training would also upcast every
    # bf16 parameter to fp32 -- for Gemma that is the 262k x 2560 embedding (tied lm_head),
    # +1.3 GB on an 8 GB card. Gemma3RMSNorm already computes in fp32 internally.
    for p in model.parameters():
        p.requires_grad_(False)
    set_train_mode(model, True)
    model.enable_input_require_grads()

    # The 4B checkpoint is multimodal (language_model + vision_tower); only the language
    # model's projections get adapters. The 1B checkpoint is text-only.
    proj = "(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    if any("language_model" in n for n, _ in model.named_modules()):
        target_modules = rf".*language_model.*\.{proj}$"
    else:
        target_modules = rf".*\.{proj}$"

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return tokenizer, model, weights_gb


def set_train_mode(model, training):
    """Training: gradient checkpointing on, KV cache off, dropout on. Eval / generation: the
    reverse. Kept in one place so the two are never left half-switched after a monitor."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if training:
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        model.eval()


def completion_loss(model, batch):
    """
    Cross-entropy over the completion positions only. Mathematically identical to the HF
    forward-with-labels loss (mean over non-ignored positions), but it runs lm_head on the
    ~20 target positions instead of all 1,024 -- Gemma's 262k vocabulary makes the full
    logit tensor ~1 GB in fp32 per sequence, which is what decides whether batch > 1 fits.
    """
    base = model.get_base_model()
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    shift_labels = batch["labels"][:, 1:]
    keep = shift_labels != -100
    logits = base.lm_head(hidden[:, :-1][keep]).float()
    cap = getattr(base.config.get_text_config(), "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return F.cross_entropy(logits, shift_labels[keep])


def hf_loss(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss


@torch.no_grad()
def eval_losses(model, encoded):
    """Per-example completion loss with dropout off and no gradient (the number a
    memorisation claim must be judged on; the training log is train-mode with dropout)."""
    set_train_mode(model, False)
    model.get_base_model().config.use_cache = False
    out = [completion_loss(model, e).item() for e in encoded]
    return out


@torch.no_grad()
def generate_picks(model, tokenizer, examples, encoded, idx, stop_ids):
    """Greedy generation on the selected examples; returns parsed records."""
    set_train_mode(model, False)
    recs = []
    for i in idx:
        ex, enc = examples[i], encoded[i]
        out = model.generate(
            input_ids=enc["prompt_ids"], attention_mask=torch.ones_like(enc["prompt_ids"]),
            max_new_tokens=128, do_sample=False,
            eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
        )
        new = out[0, enc["prompt_len"]:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        picks, off_list = parse_picks(text, ex["candidates"])
        gold = set(ex["gold"])
        recs.append({
            "bibcode": ex["bibcode"], "gold": ex["gold"], "output": text,
            "picks": picks, "off_list": off_list,
            "hits": len(gold & set(picks)), "n_picks": len(picks), "n_gold": len(gold),
            "n_new_tokens": int(new.numel()), "hit_max_new_tokens": int(new.numel()) >= 128,
            "exact_match": picks == ex["gold"] and not off_list,
        })
    return recs


def gen_summary(recs):
    hits = sum(r["hits"] for r in recs); picks = sum(r["n_picks"] for r in recs); gold = sum(r["n_gold"] for r in recs)
    return {
        "papers": len(recs), "recall": hits / gold if gold else None, "precision": hits / picks if picks else None,
        "mean_picks": picks / len(recs), "mean_gold": gold / len(recs),
        "off_list_lines": sum(len(r["off_list"]) for r in recs),
        "empty_outputs": sum(1 for r in recs if r["n_picks"] == 0 and not r["off_list"]),
        "exact_matches": sum(1 for r in recs if r["exact_match"]),
        "hit_max_new_tokens": sum(1 for r in recs if r["hit_max_new_tokens"]),
    }


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--n-papers", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0, help="max gradient norm")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-generate", type=int, default=8, help="papers (random sample) to generate on, before and after")
    ap.add_argument("--out", default=None, help="output dir (default results/gemma_probe/<model>)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and report token stats only")
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results" / "gemma_probe" / short
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    provenance = git_provenance("scripts/gemma_probe.py", "scripts/gemma_common.py")

    print("=" * 70)
    print(f"Gemma QLoRA probe: {args.model}   (code: {provenance.get('commit')}"
          f"{' + uncommitted changes' if provenance.get('uncommitted_changes') else ''})")
    print("=" * 70)

    # 1. Data -------------------------------------------------------------
    rows, id_to_name = load_probe_papers(args.n_papers, args.seed)
    all_names = sorted(id_to_name.values())
    rng = random.Random(args.seed)
    examples = [build_example(r, id_to_name, all_names, rng) for r in rows]
    print(f"papers: {len(examples)} from the 15,822 training carve; "
          f"gold topics per paper: mean {sum(len(e['gold']) for e in examples) / len(examples):.2f}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = [encode_training_example(tokenizer, ex["title"], ex["abstract"], ex["candidates"], ex["target"], args.max_len)
               for ex in examples]
    seq_lens = sorted(e["seq_len"] for e in encoded)
    prompt_lens = sorted(e["prompt_len"] for e in encoded)
    n_trunc = sum(e["truncated"] for e in encoded)
    token_stats = {
        "seq_len_min": seq_lens[0], "seq_len_median": seq_lens[len(seq_lens) // 2], "seq_len_max": seq_lens[-1],
        "prompt_len_min": prompt_lens[0], "prompt_len_median": prompt_lens[len(prompt_lens) // 2], "prompt_len_max": prompt_lens[-1],
        "label_tokens_mean": sum(e["n_label_tokens"] for e in encoded) / len(encoded),
        "abstracts_truncated_to_fit": n_trunc,
    }
    print(f"tokens per example (prompt + target): min {seq_lens[0]}, median {seq_lens[len(seq_lens)//2]}, "
          f"max {seq_lens[-1]}; {n_trunc} abstracts shortened to fit {args.max_len}")

    with open(out_dir / "probe_examples.json", "w", encoding="utf-8") as f:
        json.dump([{k: ex[k] for k in ("bibcode", "gold", "candidates", "target")}
                   | {"prompt": render_prompt(ex["title"], ex["abstract"], ex["candidates"])}
                   for ex in examples], f, indent=1, ensure_ascii=False)

    gen_idx = sorted(random.Random(args.seed).sample(range(len(examples)), min(args.n_generate, len(examples))))

    if args.dry_run:
        ex = examples[0]
        print("\n--- example prompt ---\n" + render_prompt(ex["title"], ex["abstract"], ex["candidates"])
              + "\n--- target ---\n" + ex["target"])
        print(f"\ndry run: wrote {out_dir / 'probe_examples.json'}")
        return

    # 2. Model ------------------------------------------------------------
    assert torch.cuda.is_available(), "the probe measures the CUDA card; no CUDA device found"
    device = torch.device("cuda:0")
    t0 = time.time()
    tokenizer, model, weights_gb = load_model(args.model, args.attn, args.lora_r, args.lora_alpha)
    load_s = time.time() - t0
    # peft's counter unpacks bnb 4-bit tensors (two weights per stored element); a plain
    # numel() sum would report the 4B model as ~2.5B.
    trainable, total = model.get_nb_trainable_parameters()
    stop_ids = stop_token_ids(tokenizer)
    print(f"model loaded in {load_s:.0f}s; weights on card: {weights_gb:.2f} GB; "
          f"trainable {trainable:,} / {total:,} params ({100 * trainable / total:.3f}%)")

    encoded = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in e.items()} for e in encoded]

    # One-off check that the lean loss matches the standard HF loss on the first example.
    loss_check = {}
    set_train_mode(model, False)
    model.get_base_model().config.use_cache = False
    with torch.no_grad():
        try:
            l_hf = hf_loss(model, encoded[0]).item()
            l_lean = completion_loss(model, encoded[0]).item()
            loss_check = {"hf_loss": l_hf, "completion_loss": l_lean, "abs_diff": abs(l_hf - l_lean)}
            print(f"loss check on example 0: HF {l_hf:.4f} vs completion-only {l_lean:.4f}")
        except torch.cuda.OutOfMemoryError:
            loss_check = {"hf_loss": None, "note": "standard HF loss (full 1,024 x 262k logits) OOM even without grad"}
            print("loss check: standard HF path OOM; continuing with completion-only loss")

    # Zero-shot controls: per-example eval-mode loss and greedy generation BEFORE training.
    # LoRA B is zero-initialised, so this is exactly the untouched model.
    t_gen = time.time()
    loss_before = eval_losses(model, encoded)
    gen_before = generate_picks(model, tokenizer, examples, encoded, gen_idx, stop_ids)
    gen_before_s = (time.time() - t_gen)
    print(f"zero-shot: eval loss median {sorted(loss_before)[len(loss_before)//2]:.3f}; generation on {len(gen_idx)} papers: "
          + ", ".join(f"{g['hits']}/{g['n_gold']} gold, {g['n_picks']} picks, {len(g['off_list'])} off-list" for g in gen_before))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 3. Train ------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        return args.lr * min(1.0, (step + 1) / max(1, args.warmup))

    order = list(range(len(encoded)))
    epoch_rng = random.Random(args.seed)
    log = []
    status = "completed"
    set_train_mode(model, True)
    print(f"\ntraining {args.steps} steps, batch 1, {args.max_len} tokens, lr {args.lr}, clip {args.clip}, LoRA r={args.lora_r}")
    t_train = time.time()
    for step in range(args.steps):
        if step % len(order) == 0:
            epoch_rng.shuffle(order)
        ex_i = order[step % len(order)]
        batch = encoded[ex_i]

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        torch.cuda.synchronize()
        t_step = time.time()
        try:
            loss = completion_loss(model, batch)
            if not torch.isfinite(loss):
                status = f"non-finite loss at step {step}: {loss.item()}"
                print("\n" + status)
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError as e:
            status = f"OOM at step {step}: {str(e).splitlines()[0][:200]}"
            print("\n" + status)
            break
        torch.cuda.synchronize()
        dt = time.time() - t_step

        entry = {
            "step": step, "example": ex_i, "loss": loss.item(), "grad_norm": grad_norm, "sec": dt, "lr": lr_at(step),
            "seq_len": batch["seq_len"], "n_label_tokens": batch["n_label_tokens"],
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
        }
        log.append(entry)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {entry['loss']:.4f}  gnorm {grad_norm:.2f}  {dt:.2f}s  "
                  f"peak alloc {entry['peak_alloc_gb']:.2f} GB  reserved {entry['peak_reserved_gb']:.2f} GB", flush=True)
        if step % 50 == 0:
            with open(out_dir / "probe_log.json", "w") as f:
                json.dump({"status": "running", "log": log}, f)
    train_s = time.time() - t_train

    # 4. After training: adapter, eval-mode losses, generation on the same papers --------
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    loss_after = eval_losses(model, encoded) if log else loss_before
    gen_after = generate_picks(model, tokenizer, examples, encoded, gen_idx, stop_ids) if log else []
    for g in gen_after:
        print(f"\n[{g['bibcode']}] gold {g['n_gold']} | picked {g['n_picks']} | hits {g['hits']} | off-list {len(g['off_list'])}"
              f"{' | EXACT' if g['exact_match'] else ''}\n{g['output'].strip()}")

    per_example = [{
        "bibcode": ex["bibcode"], "n_gold": len(ex["gold"]), "n_label_tokens": enc["n_label_tokens"],
        "eval_loss_before": lb, "eval_loss_after": la,
        "visits": sum(1 for e in log if e["example"] == i),
    } for i, (ex, enc, lb, la) in enumerate(zip(examples, encoded, loss_before, loss_after))]
    la_sorted = sorted(loss_after)
    eval_stats = {
        "before_median": sorted(loss_before)[len(loss_before) // 2],
        "after_median": la_sorted[len(la_sorted) // 2], "after_max": la_sorted[-1],
        "after_share_below_0.05": sum(1 for x in loss_after if x < 0.05) / len(loss_after),
        "after_share_below_0.10": sum(1 for x in loss_after if x < 0.10) / len(loss_after),
        "after_share_above_0.50": sum(1 for x in loss_after if x > 0.50) / len(loss_after),
    }

    # 5. Report -------------------------------------------------------------
    n_epoch = len(encoded)
    losses = [e["loss"] for e in log]
    timed = [e["sec"] for e in log[10:]] or [e["sec"] for e in log]
    gnorms = [e["grad_norm"] for e in log]
    summary = {
        "model": args.model,
        "status": status,
        "steps_completed": len(log),
        "code": provenance,
        "config": {k: v for k, v in vars(args).items() if k not in ("out", "dry_run")},
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
            "python": platform.python_version(), "os": platform.platform(),
        },
        "params": {"trainable": trainable, "total": total, "trainable_pct": 100 * trainable / total},
        "memory_gb": {
            "weights_after_load": weights_gb,
            "peak_allocated": max((e["peak_alloc_gb"] for e in log), default=None),
            "peak_reserved": max((e["peak_reserved_gb"] for e in log), default=None),
        },
        "timing": {
            "model_load_s": load_s, "train_s": train_s,
            "sec_per_step_mean": sum(timed) / len(timed) if timed else None,
            "sec_per_step_min": min(timed) if timed else None,
            "sec_per_step_max": max(timed) if timed else None,
            "zero_shot_eval_and_generate_s": gen_before_s,
        },
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "mean_first_pass": sum(losses[:n_epoch]) / len(losses[:n_epoch]) if losses else None,
            "mean_last_pass": sum(losses[-n_epoch:]) / len(losses[-n_epoch:]) if losses else None,
            "all_finite": all(math.isfinite(l) for l in losses),
            "grad_norm_max": max(gnorms) if gnorms else None,
            "grad_norm_mean_last_pass": sum(gnorms[-n_epoch:]) / len(gnorms[-n_epoch:]) if gnorms else None,
        },
        "eval_mode_loss": eval_stats,
        "loss_check": loss_check,
        "tokens": token_stats,
        "generation_before": {"summary": gen_summary(gen_before), "records": gen_before},
        "generation_after": {"summary": gen_summary(gen_after), "records": gen_after} if gen_after else None,
        "per_example": per_example,
    }
    with open(out_dir / "probe_log.json", "w") as f:
        json.dump({"summary": summary, "log": log}, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
        ax.plot(losses, lw=0.8, alpha=0.5, label="per step (train mode, dropout on)")
        if len(losses) >= n_epoch:
            roll = [sum(losses[i - n_epoch + 1:i + 1]) / n_epoch for i in range(n_epoch - 1, len(losses))]
            ax.plot(range(n_epoch - 1, len(losses)), roll, lw=2, label=f"rolling mean ({n_epoch} steps = one pass)")
        for b in range(n_epoch, len(losses), n_epoch):
            ax.axvline(b, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("step"); ax.set_ylabel("completion loss"); ax.set_ylim(bottom=0)
        ax.set_title(f"{short}: QLoRA on {n_epoch} papers, {args.max_len} tokens, batch 1")
        ax.legend()
        xs = [p["n_label_tokens"] for p in per_example]
        ax2.scatter(xs, [p["eval_loss_before"] for p in per_example], s=14, alpha=0.6, label="before training")
        ax2.scatter(xs, [p["eval_loss_after"] for p in per_example], s=14, alpha=0.9, label="after training")
        ax2.set_yscale("log"); ax2.set_xlabel("target tokens"); ax2.set_ylabel("eval-mode loss per example (log)")
        ax2.set_title("per-paper loss, dropout off"); ax2.legend()
        fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=130)
    except Exception as e:
        print(f"(loss plot skipped: {e})")

    mem = summary["memory_gb"]; tm = summary["timing"]; lo = summary["loss"]
    fits = status == "completed" and lo["all_finite"]
    verdict = ("FITS -- use this size for 5b" if fits else "DOES NOT FIT -- fall back to the smaller model")
    gb, ga = summary["generation_before"]["summary"], (summary["generation_after"] or {}).get("summary")
    lines = [
        f"# Gemma probe: `{args.model}`", "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Code**: commit `{provenance.get('commit')}`"
        + (" with uncommitted changes to " + ", ".join(provenance["uncommitted_changes"]) if provenance.get("uncommitted_changes") else "") + "  ",
        f"**GPU**: {summary['environment']['gpu']} ({summary['environment']['gpu_total_gb']:.1f} GB)  ",
        f"**Stack**: torch {torch.__version__} / transformers {summary['environment']['transformers']} / "
        f"peft {summary['environment']['peft']} / bitsandbytes {summary['environment']['bitsandbytes']}", "",
        f"## Verdict: {verdict}", "", f"Status: `{status}` after {len(log)} of {args.steps} steps.", "",
        "| metric | value |", "|---|---|",
        f"| sequence length / batch | {args.max_len} tokens / 1 |",
        f"| tokens per example (prompt+target) | min {token_stats['seq_len_min']}, median {token_stats['seq_len_median']}, "
        f"max {token_stats['seq_len_max']} ({n_trunc} abstracts shortened); label tokens mean {token_stats['label_tokens_mean']:.1f} |",
        f"| trainable params | {trainable:,} of {total:,} ({100 * trainable / total:.3f}%) -- LoRA r={args.lora_r}, alpha={args.lora_alpha} |",
        f"| weights on card after 4-bit load | {weights_gb:.2f} GB |",
    ]
    if log:
        lines += [
            f"| peak memory allocated / reserved | {mem['peak_allocated']:.2f} GB / {mem['peak_reserved']:.2f} GB |",
            f"| seconds per step (mean / min / max), padded to {args.max_len} | {tm['sec_per_step_mean']:.2f} / {tm['sec_per_step_min']:.2f} / {tm['sec_per_step_max']:.2f} |",
            f"| train-mode loss, first step -> last step | {lo['first']:.4f} -> {lo['last']:.4f} |",
            f"| train-mode mean loss, first pass -> last pass | {lo['mean_first_pass']:.4f} -> {lo['mean_last_pass']:.4f} |",
            f"| gradient norm, max / mean over last pass | {lo['grad_norm_max']:.2f} / {lo['grad_norm_mean_last_pass']:.2f} (clip {args.clip}) |",
            f"| all losses finite | {lo['all_finite']} |",
            f"| **eval-mode loss per paper, median before -> after** | {eval_stats['before_median']:.3f} -> {eval_stats['after_median']:.3f} (max after {eval_stats['after_max']:.3f}) |",
            f"| papers with eval loss < 0.05 / < 0.10 / > 0.50 after training | {eval_stats['after_share_below_0.05']:.0%} / {eval_stats['after_share_below_0.10']:.0%} / {eval_stats['after_share_above_0.50']:.0%} |",
        ]
    lines += [f"| model load time | {load_s:.0f} s |", f"| training wall-clock | {train_s / 60:.1f} min |", ""]
    if loss_check.get("hf_loss") is not None:
        lines += [f"Loss-path check on example 0: HF forward-with-labels {loss_check['hf_loss']:.4f} vs "
                  f"completion-only {loss_check['completion_loss']:.4f} (|diff| {loss_check['abs_diff']:.2e}).", ""]
    lines += [f"## Greedy generation on {len(gen_idx)} randomly sampled trained-on papers, before vs after training", "",
              "| | recall | precision | mean picks (gold) | off-list lines | empty | exact target match |", "|---|---|---|---|---|---|---|",
              f"| zero-shot (before) | {gb['recall']:.2f} | {gb['precision']:.2f} | {gb['mean_picks']:.1f} ({gb['mean_gold']:.1f}) | {gb['off_list_lines']} | {gb['empty_outputs']} | {gb['exact_matches']}/{gb['papers']} |"]
    if ga:
        lines += [f"| after training | {ga['recall']:.2f} | {ga['precision']:.2f} | {ga['mean_picks']:.1f} ({ga['mean_gold']:.1f}) | {ga['off_list_lines']} | {ga['empty_outputs']} | {ga['exact_matches']}/{ga['papers']} |"]
    lines += ["", "Candidates here are gold + random fillers, so these numbers describe format, stopping and "
              "memorisation only -- never accuracy.", "",
              f"Adapter saved to `adapter/`. Loss curve and per-paper eval loss: `loss_curve.png`. Full per-step log, "
              "per-example losses and generations: `probe_log.json`.", ""]
    (out_dir / "probe_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    if log:
        print(f"status: {status}; peak reserved {mem['peak_reserved']:.2f} GB; {tm['sec_per_step_mean']:.2f} s/step; "
              f"eval-mode loss median {eval_stats['before_median']:.3f} -> {eval_stats['after_median']:.3f}; "
              f"exact target match {ga['exact_matches']}/{ga['papers']} (zero-shot {gb['exact_matches']}/{gb['papers']})")
    print(f"report: {out_dir / 'probe_report.md'}")


if __name__ == "__main__":
    main()
