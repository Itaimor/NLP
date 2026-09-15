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

PROMPT_TEMPLATE = (
    "You are indexing an astronomy paper with the Unified Astronomy Thesaurus (UAT).\n"
    "Read the title and abstract, then decide which of the candidate topics apply.\n\n"
    "Title: {title}\n\n"
    "Abstract: {abstract}\n\n"
    "Candidate topics:\n{candidates}\n\n"
    "Answer with the candidate topics that apply, most confident first, one per line, "
    "copied exactly as written above. If none apply, answer NONE."
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


def encode_training_example(tokenizer, title, abstract, candidates, target, max_len):
    """
    Prompt + target as one right-padded sequence with the loss masked to the completion.
    If it exceeds max_len the abstract is shortened until it fits; the candidate list and the
    target are never cut.

    Returns:
    --- dict with input_ids, attention_mask, labels (each shape (1, max_len)),
        prompt_ids (unpadded, for generation), prompt_len, seq_len, truncated (bool)
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

    ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    seq_len = len(ids)
    pad = max_len - seq_len
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([ids + [pad_id] * pad]),
        "attention_mask": torch.tensor([[1] * seq_len + [0] * pad]),
        "labels": torch.tensor([labels + [-100] * pad]),
        "prompt_ids": torch.tensor([p_ids]),
        "prompt_len": len(p_ids),
        "seq_len": seq_len,
        "n_label_tokens": len(t_ids),
        "truncated": truncated,
    }


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


def git_provenance(*paths):
    """Commit hash of HEAD and whether the given files differ from it — logged with every run
    so a number in the paper can be traced to the code that produced it."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=PROJECT_ROOT, text=True).strip()
        return {"commit": head, "uncommitted_changes": dirty.splitlines()}
    except Exception as e:  # not a git checkout, git missing
        return {"commit": None, "error": str(e)}
