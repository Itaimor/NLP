#!/usr/bin/env python3
"""
Gemma QLoRA Trainer (Stage 5b)
==============================
Fine-tunes a Gemma-3 checkpoint in 4-bit to select topics from a 50-candidate shortlist.
Implements the Stage 5 recipe of WORK_PLAN.md exactly:
- input: a handoff-4b shortlist file — one JSON line per paper with `paper_id`, the 50
  `candidates` in the generator's score order, and `gold_in_list` (gold ∩ candidates,
  ordered by the generator's score, highest first)
- the candidate order is SHUFFLED in every training prompt (seeded per epoch and paper) and
  the shown order is recorded in `shown_order.jsonl`; validation and test lists are never
  shuffled (that is gemma_select.py's job)
- the target is `gold_in_list` verbatim, one per line, in its score order, or `NONE`
- the loss is over the completion only (gemma_common.completion_loss), dynamic right padding
  to the longest sequence in the batch, length-grouped batches
- every step logs loss, gradient norm, seconds, tokens and peak memory (the compute-cost row)
- checkpoints (adapter + optimizer + step + RNG) every --save-every steps and at every epoch
  end; `--resume` continues from the latest checkpoint in --out
- an end-of-epoch monitor (WORK_PLAN Stage 5, item 3): greedy generation on --monitor-n
  held-out training-style lists and --monitor-n validation lists, shown in the generator's
  order, reporting |ŷ| / |gold ∩ list|, recall of gold ∩ list, precision, off-list rate
- OOM or a non-finite loss skips the batch and is counted; three in a row saves and stops

Alternative B (train on the validation carve): `--split validation --train-shortlist <val
file>` holds --dev-holdout papers out first (written to dev_holdout_ids.json); the monitor's
"validation" lists then come from that hold-out.

Dress rehearsal on the TF-IDF stand-in lists (no real lists yet):
    python scripts/gemma_train.py --train-shortlist results/shortlists/standin_tfidf_top50_train.jsonl \
        --val-shortlist results/shortlists/standin_tfidf_top50_val.jsonl --split train \
        --max-steps 50 --save-every 50 --monitor-every 50 --monitor-n 16 --out results/gemma_train/smoke
    python scripts/gemma_train.py ... --resume --max-steps 100
"""

import os
import sys
import json
import time
import math
import random
import shutil
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
# Without this the caching allocator fragments after a long batch, reserved memory creeps
# towards the 8 GB card limit, and Windows silently pages GPU memory to RAM: the same
# batches ran 3x slower in the dress rehearsal (reserved 6.94 GB vs 5.65 GB with it).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scibert_dataset import clean_astronomy_text
from gemma_common import (
    N_CANDIDATES, render_prompt, encode_training_ids, collate_batch, parse_picks, stop_token_ids,
    load_quantized, attach_lora, set_train_mode, completion_loss, git_provenance,
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_papers(split_names):
    """bibcode -> title, abstract, gold names for the given persisted splits."""
    split = json.load(open(DATA_DIR / "split.json", encoding="utf-8"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))
    wanted = set()
    for s in split_names:
        wanted.update(split[s])
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


def target_text(gold_in_list):
    return "\n".join(gold_in_list) if gold_in_list else "NONE"


def shuffled_candidates(candidates, seed, epoch, paper_id):
    """Deterministic per (seed, epoch, paper) shuffle, so a resumed run shows the same lists."""
    rng = random.Random(f"{seed}|{epoch}|{paper_id}")
    perm = list(range(len(candidates)))
    rng.shuffle(perm)
    return [candidates[i] for i in perm], perm


def epoch_batches(n, batch_size, seed, epoch, lengths):
    """Length-grouped batches for one epoch: shuffle, cut into chunks of 32 batches, sort each
    chunk by prompt length, batch, then shuffle the batches. Deterministic per (seed, epoch)."""
    rng = random.Random(f"{seed}|epoch{epoch}")
    order = list(range(n))
    rng.shuffle(order)
    chunk = batch_size * 32
    batches = []
    for s in range(0, n, chunk):
        part = sorted(order[s:s + chunk], key=lambda i: lengths[i])
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------

def save_checkpoint(model, optimizer, state, out_dir, name, keep=2):
    ck = out_dir / "checkpoints" / name
    tmp = out_dir / "checkpoints" / (name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, tmp / "rng.pt")
    with open(tmp / "state.json", "w") as f:
        json.dump(state, f, indent=1)
    if ck.exists():
        shutil.rmtree(ck)
    tmp.rename(ck)
    (out_dir / "checkpoints" / "LATEST").write_text(name)
    # keep only the last `keep` step checkpoints; epoch checkpoints are kept
    steps = sorted((p for p in (out_dir / "checkpoints").glob("step-*") if p.is_dir()),
                   key=lambda p: int(p.name.split("-")[1]))
    for old in steps[:-keep]:
        shutil.rmtree(old)
    return ck


def latest_checkpoint(out_dir):
    marker = out_dir / "checkpoints" / "LATEST"
    if not marker.exists():
        return None
    ck = out_dir / "checkpoints" / marker.read_text().strip()
    return ck if (ck / "state.json").exists() else None


# ----------------------------------------------------------------------------
# Monitor
# ----------------------------------------------------------------------------

@torch.no_grad()
def monitor(model, tokenizer, records, papers, stop_ids, batch_size, max_new_tokens):
    """Greedy generation on shortlist records shown in the generator's order (never shuffled),
    like validation and test. Returns the Stage 5 monitor numbers."""
    set_train_mode(model, False)
    tokenizer.padding_side = "left"
    prompts = [render_prompt(papers[r["paper_id"]]["title"], papers[r["paper_id"]]["abstract"], r["candidates"])
               for r in records]
    order = sorted(range(len(records)), key=lambda i: len(prompts[i]))
    outs = [None] * len(records)
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                               add_generation_prompt=True) for i in idx]
        enc = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id)
        for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            outs[i] = tokenizer.decode(row, skip_special_tokens=True)
    sec = (time.time() - t0) / len(records)

    emitted = gold_in = hits = off = lines = empty = 0
    for r, text in zip(records, outs):
        picks, off_list = parse_picks(text, r["candidates"])
        gold = set(papers[r["paper_id"]]["gold"]) & set(r["candidates"])
        emitted += len(picks); gold_in += len(gold); hits += len(gold & set(picks))
        off += len(off_list); lines += len(picks) + len(off_list)
        empty += int(not picks and not off_list)
    set_train_mode(model, True)
    return {
        "papers": len(records), "sec_per_paper": sec,
        "emitted_over_gold_in_list": emitted / gold_in if gold_in else None,
        "recall_of_gold_in_list": hits / gold_in if gold_in else None,
        "precision": hits / emitted if emitted else None,
        "mean_picks": emitted / len(records),
        "off_list_rate_lines": off / lines if lines else 0.0,
        "empty_rate": empty / len(records),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--train-shortlist", required=True, help="handoff-4b jsonl with gold_in_list")
    ap.add_argument("--val-shortlist", default=None, help="handoff-4 validation jsonl (monitor lists)")
    ap.add_argument("--split", default="train", choices=["train", "validation"],
                    help="which persisted split the training papers come from (validation = Alternative B)")
    ap.add_argument("--dev-holdout", type=int, default=300, help="Alternative B only: papers held out of training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (total)")
    ap.add_argument("--max-hours", type=float, default=None, help="save and stop after this much training time")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--monitor-n", type=int, default=50, help="held-out training lists and validation lists per monitor")
    ap.add_argument("--monitor-every", type=int, default=0, help="steps between monitors (0 = epoch end only)")
    ap.add_argument("--monitor-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance("scripts/gemma_train.py", "scripts/gemma_common.py")
    print(f"gemma_train: {args.model} | code {provenance.get('commit')}"
          f"{' + UNCOMMITTED CHANGES' if provenance.get('uncommitted_changes') else ''} | out {out_dir}", flush=True)

    # ---- data ------------------------------------------------------------
    train_recs = read_jsonl(args.train_shortlist)
    assert all(len(r["candidates"]) == N_CANDIDATES for r in train_recs), "every shortlist must have exactly 50 candidates"
    papers = load_papers(["train", "validation"])
    missing = [r["paper_id"] for r in train_recs if r["paper_id"] not in papers]
    assert not missing, f"{len(missing)} training paper_ids not in the dataset, e.g. {missing[:3]}"
    # a handoff-4 file (Alternative B trains on the validation lists) carries no target; the
    # candidates are already in the generator's score order, so gold ∩ candidates in that
    # order is exactly the handoff-4b target
    filled = 0
    for r in train_recs:
        if "gold_in_list" not in r:
            gold = set(papers[r["paper_id"]]["gold"])
            r["gold_in_list"] = [c for c in r["candidates"] if c in gold]
            filled += 1
    if filled:
        print(f"gold_in_list filled from gold ∩ candidates (candidate order) for {filled} lists", flush=True)
    split = json.load(open(DATA_DIR / "split.json", encoding="utf-8"))
    allowed = set(split[args.split])
    outside = [r["paper_id"] for r in train_recs if r["paper_id"] not in allowed]
    assert not outside, f"{len(outside)} training papers are not in the '{args.split}' split (never train on test)"

    hold_rng = random.Random(args.seed)
    if args.split == "validation":                       # Alternative B
        dev_ids = set(hold_rng.sample([r["paper_id"] for r in train_recs], args.dev_holdout))
        with open(out_dir / "dev_holdout_ids.json", "w") as f:
            json.dump(sorted(dev_ids), f)
        val_monitor = [r for r in train_recs if r["paper_id"] in dev_ids]
        train_recs = [r for r in train_recs if r["paper_id"] not in dev_ids]
    else:
        val_monitor = read_jsonl(args.val_shortlist) if args.val_shortlist else []
    # held-out training-style lists for the monitor (never trained on)
    mon_ids = set(hold_rng.sample([r["paper_id"] for r in train_recs], min(args.monitor_n, len(train_recs))))
    train_monitor = [r for r in train_recs if r["paper_id"] in mon_ids]
    train_recs = [r for r in train_recs if r["paper_id"] not in mon_ids]
    val_monitor = random.Random(args.seed).sample(val_monitor, min(args.monitor_n, len(val_monitor))) if val_monitor else []
    n_none = sum(1 for r in train_recs if not r["gold_in_list"])
    print(f"training lists: {len(train_recs)} ({n_none} with NONE targets); monitor: {len(train_monitor)} held-out "
          f"training-style + {len(val_monitor)} validation lists", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # prompt length (unshuffled) only drives length grouping; cheap proxy: characters
    lengths = [len(papers[r["paper_id"]]["abstract"]) + len(papers[r["paper_id"]]["title"]) for r in train_recs]

    # ---- model -----------------------------------------------------------
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    t0 = time.time()
    tokenizer, model = load_quantized(args.model)
    stop_ids = stop_token_ids(tokenizer)
    ck = latest_checkpoint(out_dir) if args.resume else None
    if ck:
        from peft import PeftModel
        for p in model.parameters():
            p.requires_grad_(False)
        set_train_mode(model, True)
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, ck / "adapter", is_trainable=True)
        state = json.load(open(ck / "state.json"))
        print(f"resumed from {ck.name}: step {state['step']}, epoch {state['epoch']}, batch {state['batch_in_epoch']}", flush=True)
    else:
        model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "skipped": 0, "train_s": 0.0}
    trainable, total = model.get_nb_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if ck:
        optimizer.load_state_dict(torch.load(ck / "optimizer.pt"))
        rng = torch.load(ck / "rng.pt")
        torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"])
    else:
        torch.manual_seed(args.seed)
    print(f"model ready in {time.time() - t0:.0f}s; trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)", flush=True)

    steps_per_epoch = math.ceil(math.ceil(len(train_recs) / args.batch_size) / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    def lr_at(step):
        return args.lr * min(1.0, (step + 1) / max(1, args.warmup))

    cost = {
        "model": args.model, "code": provenance, "config": vars(args),
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__,
                        "bitsandbytes": __import__("bitsandbytes").__version__, "os": platform.platform()},
        "params": {"trainable": trainable, "total": total},
        "data": {"training_lists": len(train_recs), "none_targets": n_none, "steps_per_epoch": steps_per_epoch,
                 "monitor_train": len(train_monitor), "monitor_val": len(val_monitor)},
    }

    log_f = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")
    shown_f = open(out_dir / "shown_order.jsonl", "a", encoding="utf-8")
    mon_f = open(out_dir / "monitor.jsonl", "a", encoding="utf-8")

    def run_monitor(reason):
        rec = {"step": state["step"], "epoch": state["epoch"], "reason": reason}
        if train_monitor:
            rec["train_style"] = monitor(model, tokenizer, train_monitor, papers, stop_ids, args.monitor_batch, args.max_new_tokens)
        if val_monitor:
            rec["validation"] = monitor(model, tokenizer, val_monitor, papers, stop_ids, args.monitor_batch, args.max_new_tokens)
        mon_f.write(json.dumps(rec) + "\n"); mon_f.flush()
        for k in ("train_style", "validation"):
            if k in rec:
                m = rec[k]
                print(f"  monitor[{k}] @step {state['step']}: |y|/|gold&list| {m['emitted_over_gold_in_list']:.2f}, "
                      f"recall {m['recall_of_gold_in_list']:.2f}, precision {m['precision'] if m['precision'] is None else round(m['precision'], 2)}, "
                      f"mean picks {m['mean_picks']:.1f}, off-list {m['off_list_rate_lines']:.3f}, empty {m['empty_rate']:.2f}, "
                      f"{m['sec_per_paper']:.2f} s/paper", flush=True)
        set_train_mode(model, True)

    def checkpoint(name):
        state["train_s"] = state.get("train_s", 0.0)
        save_checkpoint(model, optimizer, state, out_dir, name)
        with open(out_dir / "cost.json", "w") as f:
            json.dump(cost | {"state": state, "step_seconds": step_stats()}, f, indent=1)
        print(f"  checkpoint {name} saved", flush=True)

    step_times, peak_alloc, peak_res = [], 0.0, 0.0

    def step_stats():
        t = step_times[10:] or step_times
        return {"n": len(step_times), "mean": sum(t) / len(t) if t else None, "max": max(t) if t else None,
                "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_res}

    # ---- train -----------------------------------------------------------
    set_train_mode(model, True)
    torch.cuda.reset_peak_memory_stats()
    consecutive_failures = 0
    stop_reason = "completed"
    t_start = time.time()
    print(f"training: {total_steps} steps ({steps_per_epoch}/epoch), batch {args.batch_size} x accum {args.grad_accum}, "
          f"max_len {args.max_len}, lr {args.lr} (warmup {args.warmup}), clip {args.clip}", flush=True)
    done = state["step"] >= total_steps
    while not done and state["epoch"] < args.epochs:
        epoch = state["epoch"]
        batches = epoch_batches(len(train_recs), args.batch_size, args.seed, epoch, lengths)
        micro = [batches[i:i + args.grad_accum] for i in range(0, len(batches), args.grad_accum)]
        for b_idx in range(state["batch_in_epoch"], len(micro)):
            if state["step"] >= total_steps:
                break
            torch.cuda.synchronize(); t_step = time.time()
            for g in optimizer.param_groups:
                g["lr"] = lr_at(state["step"])
            try:
                # encode the whole accumulation window once (shuffled lists, shown order logged)
                window, items_all = [], []
                for mb in micro[b_idx]:
                    items = []
                    for i in mb:
                        r = train_recs[i]; p = papers[r["paper_id"]]
                        cands, perm = shuffled_candidates(r["candidates"], args.seed, epoch, r["paper_id"])
                        shown_f.write(json.dumps({"epoch": epoch, "step": state["step"], "paper_id": r["paper_id"], "perm": perm}) + "\n")
                        ids, labels, _, _ = encode_training_ids(tokenizer, p["title"], p["abstract"], cands,
                                                                target_text(r["gold_in_list"]), args.max_len)
                        items.append((ids, labels)); items_all.append(len(ids))
                    window.append(items)
                n_tok = sum(sum(1 for l in lab if l != -100) for items in window for _, lab in items)
                # global token mean across the window, so grad_accum > 1 is exactly one larger batch
                losses = []
                for items in window:
                    batch = {k: v.to(device) for k, v in collate_batch(items, tokenizer.pad_token_id).items()}
                    l_sum, _ = completion_loss(model, batch, reduction="sum")
                    if not torch.isfinite(l_sum):
                        raise FloatingPointError(f"non-finite loss {l_sum.item()}")
                    (l_sum / n_tok).backward()
                    losses.append(l_sum.item())
                loss_val = sum(losses) / n_tok
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                consecutive_failures = 0
            except (torch.cuda.OutOfMemoryError, FloatingPointError) as e:
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                consecutive_failures += 1; state["skipped"] = state.get("skipped", 0) + 1
                msg = f"step {state['step']}: skipped batch ({type(e).__name__}: {str(e).splitlines()[0][:120]}); max tokens in batch {max(items_all) if items_all else '?'}"
                print(msg, flush=True)
                log_f.write(json.dumps({"step": state["step"], "epoch": epoch, "skipped": True, "error": msg}) + "\n")
                if consecutive_failures >= 3:
                    stop_reason = "three consecutive failed batches"
                    break
                state["batch_in_epoch"] = b_idx + 1
                continue
            torch.cuda.synchronize(); dt = time.time() - t_step
            step_times.append(dt)
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / 1024 ** 3)
            peak_res = max(peak_res, torch.cuda.max_memory_reserved() / 1024 ** 3)
            state["step"] += 1; state["batch_in_epoch"] = b_idx + 1
            state["train_s"] = state.get("train_s", 0.0) + dt
            entry = {"step": state["step"], "epoch": epoch, "loss": loss_val, "grad_norm": grad_norm, "sec": dt,
                     "label_tokens": n_tok, "max_seq_len": max(items_all), "lr": lr_at(state["step"] - 1),
                     "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_res}
            log_f.write(json.dumps(entry) + "\n"); log_f.flush()
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                recent = step_times[-args.log_every:]
                eta_h = (total_steps - state["step"]) * (sum(recent) / len(recent)) / 3600
                print(f"step {state['step']:6d}/{total_steps}  ep {epoch}  loss {loss_val:.4f}  gnorm {grad_norm:.2f}  "
                      f"{dt:.2f}s  len {max(items_all)}  peak {peak_alloc:.2f}/{peak_res:.2f} GB  ETA {eta_h:.2f} h", flush=True)
            if args.save_every and state["step"] % args.save_every == 0:
                checkpoint(f"step-{state['step']}")
            if args.monitor_every and state["step"] % args.monitor_every == 0:
                run_monitor("every")
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                stop_reason = f"max-hours {args.max_hours} reached"; break
        else:
            # epoch finished cleanly
            state["epoch"] += 1; state["batch_in_epoch"] = 0
            run_monitor("epoch_end")
            checkpoint(f"epoch-{state['epoch']}")
            continue
        break  # a break inside the batch loop (max steps / failure / hours) ends training

    if stop_reason == "completed" and state["step"] >= total_steps and args.max_steps is not None:
        stop_reason = f"max-steps {args.max_steps} reached"
    last = latest_checkpoint(out_dir)
    if last is None or json.load(open(last / "state.json"))["step"] != state["step"]:
        checkpoint(f"step-{state['step']}")
    cost["stop_reason"] = stop_reason
    with open(out_dir / "cost.json", "w") as f:
        json.dump(cost | {"state": state, "step_seconds": step_stats()}, f, indent=1)
    log_f.close(); shown_f.close(); mon_f.close()
    st = step_stats()
    print(f"\ndone: {stop_reason}; step {state['step']}, epoch {state['epoch']}, skipped {state.get('skipped', 0)}; "
          f"{st['mean'] and round(st['mean'], 2)} s/step, peak {peak_alloc:.2f}/{peak_res:.2f} GB; out {out_dir}", flush=True)


if __name__ == "__main__":
    main()
