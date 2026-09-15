#!/usr/bin/env python3
"""
Gemma QLoRA Training Probe (Stage 5 gate)
=========================================
Decides whether Gemma-5b fine-tunes at 4B or falls back to 1B (WORK_PLAN.md, Stage 5:
"Model size is decided by the probe, not chosen in advance").

Trains google/gemma-3-4b-it in 4-bit (QLoRA) on the RTX 3060 Ti at 1,024 tokens, batch 1,
for at least 500 steps on 64 training papers, and records:
- peak GPU memory (allocated and reserved)
- seconds per step
- the loss at every step, checked finite; the run stops at the first NaN / inf
- whether the model overfits the 64 papers (the sanity check the course guidelines ask to
  be shown; the loss curve is saved as a PNG)
- trainable-parameter count (the compute-cost row, WORK_PLAN Stage 5 item 4)

Nothing tuned here is kept. The prompt has the real shape (title + abstract + 50 candidate
names) so the sequence length is honest; the 50 candidates are the paper's gold topics padded
with random topics from the 1,864-label space, because SciBERT's shortlists do not exist yet.

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

N_CANDIDATES = 50

PROMPT_TEMPLATE = (
    "You are indexing an astronomy paper with the Unified Astronomy Thesaurus (UAT).\n"
    "Read the title and abstract, then decide which of the candidate topics apply.\n\n"
    "Title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Candidate topics:\n{candidates}\n\n"
    "Answer with the candidate topics that apply, most confident first, one per line, "
    "copied exactly as written above. If none apply, answer NONE."
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_probe_papers(n_papers, seed):
    """
    Samples n_papers bibcodes from the persisted 15,822-paper training carve and returns
    their dataset rows, plus the id -> name map for the 1,864-label space.

    Parameters:
    --- n_papers: int, how many training papers to sample
    --- seed: int

    Returns:
    --- rows: list of dataset rows (bibcode, title, abstract, verified_uat_ids, ...)
    --- id_to_name: dict str(uat_id) -> topic name
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
    Turns one paper into a dict with prompt fields, target, gold names and candidates.

    The candidate list is the gold topics plus random fillers to N_CANDIDATES, shuffled --
    a stand-in for SciBERT's top-50. The target is the gold names one per line (in the real
    5b file they are ordered by SciBERT's score; there is no such score yet).
    """
    gold = [id_to_name[str(i)] for i in row["verified_uat_ids"] if str(i) in id_to_name]
    gold = gold[:N_CANDIDATES]
    gold_set = set(gold)
    fillers = rng.sample([n for n in all_names if n not in gold_set], N_CANDIDATES - len(gold))
    candidates = gold + fillers
    rng.shuffle(candidates)

    target = "\n".join(gold) if gold else "NONE"
    return {
        "bibcode": row["bibcode"],
        "title": clean_astronomy_text(row["title"]),
        "abstract": clean_astronomy_text(row["abstract"]),
        "candidates": candidates,
        "gold": gold,
        "target": target,
    }


def render_prompt(ex, abstract_words=None):
    abstract = ex["abstract"]
    if abstract_words is not None:
        abstract = " ".join(abstract.split()[:abstract_words])
    cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(ex["candidates"]))
    return PROMPT_TEMPLATE.format(title=ex["title"], abstract=abstract, candidates=cands)


def encode_example(tokenizer, ex, max_len):
    """
    Chat-formats the prompt, appends the target, and returns padded tensors with the loss
    masked to the completion. If the sequence exceeds max_len the abstract is shortened
    until it fits (the candidate list and target are never cut).

    Returns:
    --- dict with input_ids, attention_mask, labels (each shape (1, max_len)),
        prompt_ids (unpadded, for generation), prompt_len, seq_len, truncated (bool)
    """
    target_ids = tokenizer(ex["target"] + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]

    words = len(ex["abstract"].split())
    truncated = False
    for _ in range(30):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": render_prompt(ex, words)}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(target_ids) <= max_len:
            break
        words = int(words * 0.9)
        truncated = True
    else:
        raise RuntimeError(f"{ex['bibcode']}: could not fit in {max_len} tokens")

    ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    seq_len = len(ids)
    pad = max_len - seq_len
    pad_id = tokenizer.pad_token_id

    return {
        "input_ids": torch.tensor([ids + [pad_id] * pad]),
        "attention_mask": torch.tensor([[1] * seq_len + [0] * pad]),
        "labels": torch.tensor([labels + [-100] * pad]),
        "prompt_ids": torch.tensor([prompt_ids]),
        "prompt_len": len(prompt_ids),
        "seq_len": seq_len,
        "truncated": truncated,
    }


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(model_name, attn_impl, lora_r, lora_alpha):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_impl,
    )
    weights_gb = torch.cuda.memory_allocated() / 1024 ** 3

    # Manual k-bit prep. peft's prepare_model_for_kbit_training would also upcast every
    # bf16 parameter to fp32 -- for Gemma that is the 262k x 2560 embedding (tied lm_head),
    # +1.3 GB on an 8 GB card. Gemma3RMSNorm already computes in fp32 internally.
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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


def completion_loss(model, batch):
    """
    Cross-entropy over the completion positions only. Mathematically identical to the HF
    forward-with-labels loss (mean over non-ignored positions), but it runs lm_head on the
    ~60 target positions instead of all 1,024 -- Gemma's 262k vocabulary makes the full
    logit tensor ~1 GB in fp32, which is the difference between fitting and not on 8 GB.
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


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------

def parse_picks(text, candidates):
    """Maps output lines to candidates by case-insensitive exact match (Stage 5 rule)."""
    by_lower = {c.lower(): c for c in candidates}
    picks, off_list = [], []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line or line.upper() == "NONE":
            continue
        if line.lower() in by_lower:
            picks.append(by_lower[line.lower()])
        else:
            off_list.append(line)
    return picks, off_list


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--n-papers", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-generate", type=int, default=4, help="papers to generate on after training")
    ap.add_argument("--out", default=None, help="output dir (default results/gemma_probe/<model>)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and report token stats only")
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results" / "gemma_probe" / short
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"Gemma QLoRA probe: {args.model}")
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
    encoded = [encode_example(tokenizer, ex, args.max_len) for ex in examples]
    seq_lens = sorted(e["seq_len"] for e in encoded)
    prompt_lens = sorted(e["prompt_len"] for e in encoded)
    n_trunc = sum(e["truncated"] for e in encoded)
    token_stats = {
        "seq_len_min": seq_lens[0], "seq_len_median": seq_lens[len(seq_lens) // 2], "seq_len_max": seq_lens[-1],
        "prompt_len_min": prompt_lens[0], "prompt_len_median": prompt_lens[len(prompt_lens) // 2], "prompt_len_max": prompt_lens[-1],
        "abstracts_truncated_to_fit": n_trunc,
    }
    print(f"tokens per example (prompt + target): min {seq_lens[0]}, median {seq_lens[len(seq_lens)//2]}, "
          f"max {seq_lens[-1]}; {n_trunc} abstracts shortened to fit {args.max_len}")

    with open(out_dir / "probe_examples.json", "w", encoding="utf-8") as f:
        json.dump([{k: ex[k] for k in ("bibcode", "gold", "candidates", "target")} | {"prompt": render_prompt(ex)}
                   for ex in examples], f, indent=1, ensure_ascii=False)

    if args.dry_run:
        print("\n--- example prompt ---\n" + render_prompt(examples[0]) + "\n--- target ---\n" + examples[0]["target"])
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
    print(f"model loaded in {load_s:.0f}s; weights on card: {weights_gb:.2f} GB; "
          f"trainable {trainable:,} / {total:,} params ({100 * trainable / total:.3f}%)")

    encoded = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in e.items()} for e in encoded]

    # One-off check that the lean loss matches the standard HF loss on the first example.
    loss_check = {}
    model.eval()
    with torch.no_grad():
        try:
            l_hf = hf_loss(model, encoded[0]).item()
            l_lean = completion_loss(model, encoded[0]).item()
            loss_check = {"hf_loss": l_hf, "completion_loss": l_lean, "abs_diff": abs(l_hf - l_lean)}
            print(f"loss check on example 0: HF {l_hf:.4f} vs completion-only {l_lean:.4f}")
        except torch.cuda.OutOfMemoryError:
            loss_check = {"hf_loss": None, "note": "standard HF loss (full 1,024 x 262k logits) OOM even without grad"}
            print("loss check: standard HF path OOM; continuing with completion-only loss")
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
    model.train()
    print(f"\ntraining {args.steps} steps, batch 1, {args.max_len} tokens, lr {args.lr}, LoRA r={args.lora_r}")
    t_train = time.time()
    for step in range(args.steps):
        if step % len(order) == 0:
            epoch_rng.shuffle(order)
        batch = encoded[order[step % len(order)]]

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
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError as e:
            status = f"OOM at step {step}: {str(e).splitlines()[0][:200]}"
            print("\n" + status)
            break
        torch.cuda.synchronize()
        dt = time.time() - t_step

        entry = {
            "step": step, "loss": loss.item(), "sec": dt, "lr": lr_at(step),
            "seq_len": batch["seq_len"],
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
        }
        log.append(entry)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {entry['loss']:.4f}  {dt:.2f}s  "
                  f"peak alloc {entry['peak_alloc_gb']:.2f} GB  reserved {entry['peak_reserved_gb']:.2f} GB",
                  flush=True)
        if step % 50 == 0:
            with open(out_dir / "probe_log.json", "w") as f:
                json.dump({"status": "running", "log": log}, f)
    train_s = time.time() - t_train

    # 4. Overfit check: does it reproduce the gold topics on papers it trained on? ------
    generations = []
    if log and args.n_generate > 0:
        model.eval()
        model.config.use_cache = True
        model.gradient_checkpointing_disable()
        with torch.no_grad():
            for ex, enc in list(zip(examples, encoded))[: args.n_generate]:
                out = model.generate(
                    input_ids=enc["prompt_ids"], attention_mask=torch.ones_like(enc["prompt_ids"]),
                    max_new_tokens=128, do_sample=False,
                )
                text = tokenizer.decode(out[0, enc["prompt_len"]:], skip_special_tokens=True)
                picks, off_list = parse_picks(text, ex["candidates"])
                gold = set(ex["gold"])
                generations.append({
                    "bibcode": ex["bibcode"], "gold": ex["gold"], "output": text,
                    "picks": picks, "off_list": off_list,
                    "hits": len(gold & set(picks)), "n_picks": len(picks), "n_gold": len(gold),
                })
                print(f"\n[{ex['bibcode']}] gold {len(gold)} | picked {len(picks)} | hits {len(gold & set(picks))}"
                      f" | off-list {len(off_list)}\n{text.strip()}")

    # 5. Report -------------------------------------------------------------
    n_epoch = len(encoded)
    losses = [e["loss"] for e in log]
    timed = [e["sec"] for e in log[10:]] or [e["sec"] for e in log]
    summary = {
        "model": args.model,
        "status": status,
        "steps_completed": len(log),
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
        },
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "mean_first_pass": sum(losses[:n_epoch]) / len(losses[:n_epoch]) if losses else None,
            "mean_last_pass": sum(losses[-n_epoch:]) / len(losses[-n_epoch:]) if losses else None,
            "all_finite": all(math.isfinite(l) for l in losses),
        },
        "loss_check": loss_check,
        "tokens": token_stats,
        "generations": generations,
    }
    with open(out_dir / "probe_log.json", "w") as f:
        json.dump({"summary": summary, "log": log}, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, lw=0.8, alpha=0.5, label="per step")
        if len(losses) >= n_epoch:
            roll = [sum(losses[i - n_epoch + 1:i + 1]) / n_epoch for i in range(n_epoch - 1, len(losses))]
            ax.plot(range(n_epoch - 1, len(losses)), roll, lw=2, label=f"rolling mean ({n_epoch} steps = one pass)")
        for b in range(n_epoch, len(losses), n_epoch):
            ax.axvline(b, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("step"); ax.set_ylabel("completion loss"); ax.set_ylim(bottom=0)
        ax.set_title(f"{short}: QLoRA on {n_epoch} papers, {args.max_len} tokens, batch 1")
        ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=130)
    except Exception as e:
        print(f"(loss plot skipped: {e})")

    mem = summary["memory_gb"]; tm = summary["timing"]; lo = summary["loss"]
    fits = status == "completed" and lo["all_finite"]
    verdict = ("FITS -- use this size for 5b" if fits else "DOES NOT FIT -- fall back to the smaller model")
    lines = [
        f"# Gemma probe: `{args.model}`", "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**GPU**: {summary['environment']['gpu']} ({summary['environment']['gpu_total_gb']:.1f} GB)  ",
        f"**Stack**: torch {torch.__version__} / transformers {summary['environment']['transformers']} / "
        f"peft {summary['environment']['peft']} / bitsandbytes {summary['environment']['bitsandbytes']}", "",
        f"## Verdict: {verdict}", "", f"Status: `{status}` after {len(log)} of {args.steps} steps.", "",
        "| metric | value |", "|---|---|",
        f"| sequence length / batch | {args.max_len} tokens / 1 |",
        f"| tokens per example (prompt+target) | min {token_stats['seq_len_min']}, median {token_stats['seq_len_median']}, "
        f"max {token_stats['seq_len_max']} ({n_trunc} abstracts shortened) |",
        f"| trainable params | {trainable:,} of {total:,} ({100 * trainable / total:.3f}%) -- LoRA r={args.lora_r}, alpha={args.lora_alpha} |",
        f"| weights on card after 4-bit load | {weights_gb:.2f} GB |",
    ]
    if log:
        lines += [
            f"| peak memory allocated / reserved | {mem['peak_allocated']:.2f} GB / {mem['peak_reserved']:.2f} GB |",
            f"| seconds per step (mean / min / max) | {tm['sec_per_step_mean']:.2f} / {tm['sec_per_step_min']:.2f} / {tm['sec_per_step_max']:.2f} |",
            f"| loss, first step -> last step | {lo['first']:.4f} -> {lo['last']:.4f} |",
            f"| mean loss, first pass -> last pass over the {n_epoch} papers | {lo['mean_first_pass']:.4f} -> {lo['mean_last_pass']:.4f} |",
            f"| all losses finite | {lo['all_finite']} |",
        ]
    lines += [f"| model load time | {load_s:.0f} s |", f"| training wall-clock | {train_s / 60:.1f} min |", ""]
    if loss_check.get("hf_loss") is not None:
        lines += [f"Loss-path check on example 0: HF forward-with-labels {loss_check['hf_loss']:.4f} vs "
                  f"completion-only {loss_check['completion_loss']:.4f} (|diff| {loss_check['abs_diff']:.2e}).", ""]
    if generations:
        lines += ["## Overfit check: greedy generation on trained-on papers", ""]
        for g in generations:
            lines += [f"- `{g['bibcode']}`: {g['hits']}/{g['n_gold']} gold recovered, {g['n_picks']} picks, "
                      f"{len(g['off_list'])} off-list"]
        lines.append("")
    lines += ["Loss curve: `loss_curve.png`. Full per-step log: `probe_log.json`.", ""]
    (out_dir / "probe_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    if log:
        print(f"status: {status}; peak reserved {mem['peak_reserved']:.2f} GB; "
              f"{tm['sec_per_step_mean']:.2f} s/step; loss {lo['first']:.3f} -> {lo['last']:.3f}")
    print(f"report: {out_dir / 'probe_report.md'}")


if __name__ == "__main__":
    main()
