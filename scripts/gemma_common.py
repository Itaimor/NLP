#!/usr/bin/env python3
"""
Gemma Shared Front End (Stage 5, both arms)
===========================================
The one prompt, one tokenisation and one parser that both Gemma arms use — 5a (untouched)
and 5b (fine-tuned) — so that the only difference between the arms is the weights. Imported by
`gemma_probe.py`, `gemma_select.py` and the 5b trainer. WORK_PLAN.md Stage 5 requires the prompt's
commit hash to be logged before any test run; `git_provenance()` is that hook.

Inputs (a shortlist record, handoff 4 / 4b format, WORK_PLAN §6):
    {"paper_id": bibcode, "candidates": [50 topic names in the generator's order],
     "gold_in_list": [...]}   # training file only, ordered by the generator's score
Outputs (a picks record, §6 Gemma format):
    {"paper_id", "candidates_shown", "picks" (ordered, on-list names), "off_list" (raw lines that
     matched no candidate), "raw_output"}
"""

import re
import subprocess
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_CANDIDATES = 50
END_OF_TURN = "<end_of_turn>"

# The 1-10 cap is the instruction Alkan et al. gave their LLM selector. Without it untouched
# Gemma-3-4B emits ~17 of the 50 candidates per paper (4.3 gold); soft priors ("about 4 on
# average") do not move it. Chosen on validation shortlists: scripts/gemma_prompt_dev.py,
# results/gemma_select/prompt_dev/comparison.json.
PROMPT_TEMPLATE = (
    "You are indexing an astronomy paper with the Unified Astronomy Thesaurus (UAT).\n"
    "Read the title and abstract, then choose between 1 and 10 of the candidate topics: the "
    "ones this paper should be indexed under.\n\n"
    "Title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Candidate topics:\n{candidates}\n\n"
    "Answer with the names of between 1 and 10 candidate topics, most confident first, one per "
    "line, each copied exactly as written above (the topic name, not its number). Output only "
    "the list, with no introduction or commentary. If none apply, answer NONE."
)

_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")


def render_prompt(title, abstract, candidates, abstract_words=None):
    """Fills PROMPT_TEMPLATE; `abstract_words` caps the abstract (used only when a sequence
    would not fit the training length)."""
    if abstract_words is not None:
        abstract = " ".join(abstract.split()[:abstract_words])
    cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return PROMPT_TEMPLATE.format(title=title, abstract=abstract, candidates=cands)


def chat_prompt_ids(tokenizer, prompt):
    """Chat-formats a user prompt with the generation prefix and returns token ids.
    The template already emits <bos>, so special tokens are not added again."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def target_ids(tokenizer, target):
    """The completion the model is trained to emit: the target lines, then <end_of_turn>.
    No trailing newline — generation stops at <end_of_turn>, so a token after it would be a
    label the model never has to produce."""
    return tokenizer(target + END_OF_TURN, add_special_tokens=False)["input_ids"]


def stop_token_ids(tokenizer):
    """<eos> and <end_of_turn>; passed explicitly to generate() rather than trusting the
    checkpoint's generation_config."""
    return [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(END_OF_TURN)]


def encode_training_ids(tokenizer, title, abstract, candidates, target, max_len):
    """
    Prompt + target as one unpadded id sequence with labels masked to the completion.
    If it exceeds max_len the abstract is shortened until it fits; the candidate list and the
    target are never cut.

    Returns:
    --- ids, labels (lists, same length), prompt_len, truncated (bool)
    """
    t_ids = target_ids(tokenizer, target)
    words = len(abstract.split())
    truncated = False
    for _ in range(30):
        p_ids = chat_prompt_ids(tokenizer, render_prompt(title, abstract, candidates, words))
        if len(p_ids) + len(t_ids) <= max_len:
            break
        words = int(words * 0.9)
        truncated = True
    else:
        raise RuntimeError(f"could not fit example in {max_len} tokens")
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids, len(p_ids), truncated


def collate_batch(items, pad_id, pad_to=None):
    """Right-pads a list of (ids, labels) to the longest in the batch (or to pad_to) and
    returns input_ids / attention_mask / labels tensors of shape (B, L). Right padding is
    correct for training with an attention mask; generation uses the tokenizer's own
    left padding instead (see gemma_select.py)."""
    L = pad_to or max(len(ids) for ids, _ in items)
    input_ids, attention, labels = [], [], []
    for ids, lab in items:
        pad = L - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention),
        "labels": torch.tensor(labels),
    }


def encode_training_example(tokenizer, title, abstract, candidates, target, max_len):
    """
    One example, right-padded to exactly max_len (the probe's worst-case memory setting).

    Returns:
    --- dict with input_ids, attention_mask, labels (each shape (1, max_len)),
        prompt_ids (unpadded, for generation), prompt_len, seq_len, n_label_tokens, truncated
    """
    ids, labels, prompt_len, truncated = encode_training_ids(tokenizer, title, abstract, candidates, target, max_len)
    batch = collate_batch([(ids, labels)], tokenizer.pad_token_id, pad_to=max_len)
    batch.update({
        "prompt_ids": torch.tensor([ids[:prompt_len]]),
        "prompt_len": prompt_len,
        "seq_len": len(ids),
        "n_label_tokens": len(ids) - prompt_len,
        "truncated": truncated,
    })
    return batch


def parse_picks(text, candidates):
    """
    Maps each output line to a candidate by case-insensitive exact match after stripping a
    list marker (Stage 5 rule). A line that matches nothing is returned in `off_list`, never
    dropped and never fuzzy-matched — "gravitational wave" is off-list even when
    "gravitational waves" is a candidate. Duplicate picks are kept once, first position wins.

    Returns:
    --- picks: ordered list of candidate names (most-confident-first as emitted)
    --- off_list: raw lines that matched no candidate (NONE and blank lines excluded)
    """
    by_lower = {c.lower(): c for c in candidates}
    picks, off_list, seen = [], [], set()
    for line in text.splitlines():
        line = _LIST_MARKER_RE.sub("", line, count=1).strip()
        if not line or line.upper() == "NONE":
            continue
        name = by_lower.get(line.lower())
        if name is None:
            off_list.append(line)
        elif name not in seen:
            seen.add(name)
            picks.append(name)
    return picks, off_list


def load_quantized(model_name, attn_impl="sdpa", device_index=0):
    """Loads a Gemma-3 checkpoint in 4-bit (NF4, double quant, bf16 compute) on one GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        device_map={"": device_index},
        attn_implementation=attn_impl,
    )
    return tokenizer, model


def attach_lora(model, r=16, alpha=32, dropout=0.05):
    """
    Freezes the quantized base and adds LoRA adapters to the language model's seven
    projections. Manual k-bit prep: peft's prepare_model_for_kbit_training would also
    upcast every bf16 parameter to fp32 -- for Gemma that is the 262k x 2560 embedding
    (tied lm_head), +1.3 GB on an 8 GB card. Gemma3RMSNorm already computes in fp32.
    The 4B checkpoint is multimodal (language_model + vision_tower); only the language
    model's projections get adapters. The 1B checkpoint is text-only.
    """
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    set_train_mode(model, True)
    model.enable_input_require_grads()

    proj = "(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    if any("language_model" in n for n, _ in model.named_modules()):
        target_modules = rf".*language_model.*\.{proj}$"
    else:
        target_modules = rf".*\.{proj}$"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=target_modules, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


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


def completion_loss(model, batch, reduction="mean"):
    """
    Cross-entropy over the completion positions only, batch-safe. Mathematically identical
    to the HF forward-with-labels loss (mean over non-ignored positions), but it runs lm_head
    on the ~20 target positions per sequence instead of all of them -- Gemma's 262k
    vocabulary makes the full logit tensor ~1 GB in fp32 per 1,024-token sequence, which is
    what decides whether batch > 1 fits on 8 GB.

    reduction="mean": token-mean over the batch. reduction="sum": (summed loss, n_tokens),
    for gradient accumulation with a correct global token mean.
    """
    import torch.nn.functional as F

    base = model.get_base_model()
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    shift_labels = batch["labels"][:, 1:]
    keep = shift_labels != -100
    logits = base.lm_head(hidden[:, :-1][keep]).float()
    cap = getattr(base.config.get_text_config(), "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    if reduction == "sum":
        return F.cross_entropy(logits, shift_labels[keep], reduction="sum"), int(keep.sum())
    return F.cross_entropy(logits, shift_labels[keep])


def git_provenance(*paths):
    """Commit hash of HEAD and whether the given files differ from it — logged with every run
    so a number in the paper can be traced to the code that produced it."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=PROJECT_ROOT, text=True).strip()
        return {"commit": head, "uncommitted_changes": dirty.splitlines()}
    except Exception as e:  # not a git checkout, git missing
        return {"commit": None, "error": str(e)}
