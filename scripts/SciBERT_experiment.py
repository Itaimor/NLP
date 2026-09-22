# This file implements the SciBERT part in our project

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

# Project imports
from scibert_dataset import (
    SciXDataset,
    build_dataloaders,
    build_topic_mapping,
    clean_astronomy_text,
    DEFAULT_MAX_LENGTH,
)


def build_scibert(num_labels, mode="full", lora_configuration_params=None):
    """
    Build SciBert model while adding a classification head on top of the existing model.
    When model is LORA, the model adds LORA's matrices with new weights.

    Assumptions:
    --- lora_configuration_params contains the following keys and should be as follows:
        lora_configuration_params = {
        "r": 8,                                # Adapter rank
        "lora_alpha": 16,                       # Update scaling: alpha / r
        "lora_dropout": 0.1,                    # LoRA dropout probability
        "target_modules": ["query", "value"],   # Projections receiving adapters
        }
    Inputs:
    --- num_labels: int
    --- mode: str
    --- lora_configuration_params: None or dict
    Output:
    --- model:
    """

    mode = mode.lower()
    if mode not in ["full", "lora"]:
        raise ValueError("Input 'mode' is invalid, must be 'full' or 'lora'.")

    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    if mode == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,      # Classifies the entire input text
            modules_to_save=["classifier"],  # Trains and saves the classification head
            bias="none",                     # Keeps base-model biases frozen
            **(lora_configuration_params or {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["query", "value"],
            }),
        )

        model = get_peft_model(model, config)     # Attach adapters and freeze base weights
        model.print_trainable_parameters()

    return model


def compute_coverage_at_k(probabilities: torch.Tensor, true_labels: torch.Tensor, k: int = 50) -> float:
    """
    Computes validation/test Coverage@k (Recall@k): the fraction of true labels
    that appear in the model's top-k predicted candidates.
    Matches the Gemma candidate pool quality metric.
    """
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities, dtype=torch.float32)
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels, dtype=torch.float32)

    topk_indices = torch.topk(probabilities, k=min(k, probabilities.size(1)), dim=-1).indices
    hits = 0
    total_pos = 0
    for i in range(len(true_labels)):
        pos = set(torch.where(true_labels[i] > 0.5)[0].tolist())
        total_pos += len(pos)
        hits += len(pos.intersection(set(topk_indices[i].tolist())))
    return hits / max(1, total_pos)


def evaluate(model, data_loader, device, pos_weight=None):
    """
    Evaluates SciBERT without updating its parameters for the validation and test sets.
    This implementation supports both full fine-tuning and LoRA.
    Runs the model once per batch and returns probabilities for all samples,
    allowing multiple tau thresholds to be tested without re-running the model.

    Assumptions:
    --- model has already been moved to device.
    --- data_loader yields tuples of: (input_ids, attention_mask, labels).
    --- labels are multi-hot tensors with shape: [batch_size, num_labels].
    Inputs:
    --- model: transformers model or peft.PeftModel
    --- data_loader: torch.utils.data.DataLoader
    --- device: torch.device.
    --- pos_weight: optional tensor of positive weights for BCEWithLogitsLoss
    Outputs:
    --- average_loss: float, mean batch loss
    --- probabilities: torch.Tensor with shape [num_samples, num_labels]
    --- true_labels: torch.Tensor with shape [num_samples, num_labels]
    """

    model.eval()

    total_loss = 0.0
    all_probabilities = []
    all_labels = []

    loss_fct = (
        torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else torch.nn.BCEWithLogitsLoss()
    )

    with torch.inference_mode():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device=device, dtype=torch.float32)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = loss_fct(logits, labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())

    average_loss = total_loss / max(1, len(data_loader))
    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return average_loss, all_probabilities, all_labels


def evaluate_thresholds(
    probabilities: torch.Tensor,
    true_labels: torch.Tensor,
    tau_candidates: Optional[List[float]] = None,
    rare_indices: Optional[List[int]] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Sweeps decision threshold tau over probabilities and computes
    per-label Macro-F1 and (optionally) Rare-topic Macro-F1.

    Inputs:
    --- probabilities: Tensor of shape [N, num_labels] (sigmoid outputs in [0, 1])
    --- true_labels: Tensor of shape [N, num_labels] (binary 0/1)
    --- tau_candidates: list of floats to sweep (default: 0.10 to 0.90 in steps of 0.05)
    --- rare_indices: optional list of column indices corresponding to rare/tail topics
    --- eps: epsilon to avoid division by zero

    Outputs:
    --- dict containing:
        - "best_tau": float, threshold giving highest target F1
        - "best_macro_f1": float
        - "best_rare_f1": float or None
        - "best_target_f1": float
        - "tau_sweep": list of per-tau summary dicts
    """
    if tau_candidates is None:
        tau_candidates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities, dtype=torch.float32)
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels, dtype=torch.float32)

    device = probabilities.device
    true_labels = true_labels.to(device)

    actual_pos = true_labels.sum(dim=0)  # shape: [num_labels]

    best_tau = tau_candidates[0]
    best_target_f1 = -1.0
    best_macro_f1 = 0.0
    best_rare_f1 = None
    tau_sweep = []

    for tau in tau_candidates:
        preds = (probabilities >= tau).to(torch.float32)
        true_pos = (preds * true_labels).sum(dim=0)
        pred_pos = preds.sum(dim=0)

        precision = true_pos / (pred_pos + eps)
        recall = true_pos / (actual_pos + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # Classes with 0 actual positives and 0 predicted positives receive F1 = 0.0
        f1 = torch.where((actual_pos == 0) & (pred_pos == 0), torch.zeros_like(f1), f1)

        macro_f1 = f1.mean().item()

        if rare_indices is not None and len(rare_indices) > 0:
            rare_f1 = f1[rare_indices].mean().item()
            # If rare_f1 is 0.0, fallback to macro_f1 so model selection does not latch on 0.0
            target_f1 = macro_f1 if rare_f1 == 0.0 else (0.5 * macro_f1 + 0.5 * rare_f1)
        else:
            rare_f1 = None
            target_f1 = macro_f1

        tau_sweep.append({
            "tau": tau,
            "macro_f1": macro_f1,
            "rare_f1": rare_f1,
            "mean_predictions_per_doc": (pred_pos.sum() / max(1, len(probabilities))).item(),
        })

        if target_f1 > best_target_f1:
            best_target_f1 = target_f1
            best_tau = tau
            best_macro_f1 = macro_f1
            best_rare_f1 = rare_f1

    return {
        "best_tau": best_tau,
        "best_macro_f1": best_macro_f1,
        "best_rare_f1": best_rare_f1,
        "best_target_f1": best_target_f1,
        "tau_sweep": tau_sweep,
    }


def export_top_k_candidates(
    probabilities: torch.Tensor,
    bibcodes: List[str],
    idx_to_topic: Dict[int, Any],
    output_path: Union[str, Path],
    k: int = 50,
    uat_id_to_label: Optional[Dict[Any, str]] = None,
) -> Dict[str, Any]:
    """
    Exports SciBERT top-k predicted candidate topics per paper to a JSON file.
    Matches Handoff #4 specification: SciBERT's top-50 candidate lists for Gemma.

    Inputs:
    --- probabilities: Tensor of shape [N, num_labels]
    --- bibcodes: list of length N with NASA ADS bibcode identifiers
    --- idx_to_topic: dict mapping label index -> UAT ID
    --- output_path: file path for exported JSON
    --- k: number of top candidates to keep (default: 50)
    --- uat_id_to_label: optional dict mapping UAT ID -> human-readable string label

    Outputs:
    --- candidates_dict: {bibcode: {"candidate_ids": [...], "scores": [...]}}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples, num_labels = probabilities.shape
    k_actual = min(k, num_labels)

    if len(bibcodes) != num_samples:
        raise ValueError(
            f"Length of bibcodes ({len(bibcodes)}) does not match probabilities ({num_samples})"
        )

    candidates_dict = {}

    for i in range(num_samples):
        scores, top_indices = torch.topk(probabilities[i], k=k_actual)
        top_ids = [idx_to_topic[idx.item()] for idx in top_indices]
        scores_list = [round(float(s.item()), 5) for s in scores]

        paper_entry = {
            "candidate_ids": top_ids,
            "scores": scores_list,
        }

        if uat_id_to_label is not None:
            paper_entry["candidate_labels"] = [
                uat_id_to_label.get(tid, str(tid)) for tid in top_ids
            ]

        candidates_dict[bibcodes[i]] = paper_entry

    with open(output_path, "w") as f:
        json.dump(candidates_dict, f, indent=2)

    print(f"Top-{k} candidate lists for {num_samples} papers saved to: {output_path}")
    return candidates_dict


### Subsection: Training ###

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    scheduler=None,
    pos_weight=None,
    max_steps=None,
):
    """
    Trains SciBert for one epoch (or up to max_steps) and return the average loss.
    This implementation works identically whether model's mode is full or lora.

    Assumptions:
    --- model has already moved to "device".
    --- train_loader yields tuple of (input_ids, attention_mask, labels).
    --- labels are multi-hot tensors of dtype torch.float32 with shape [batch_size, num_labels],
        as required by BCEWithLogitsLoss for multi-label classification.
    Inputs:
    --- model: HF model or PeftModel.
    --- train_loader: torch.utils.data.DataLoader
    --- optimizer: torch.optim.Optimizer
    --- device: torch.device.
    --- scheduler: optional LR scheduler, stepped once per batch
    --- pos_weight: optional tensor of positive class weights to counter extreme label imbalance
    --- max_steps: optional int, stop after this many batches for timing slice benchmarks
    Output:
    --- avg_loss: float, mean loss over the steps executed
    """

    model.train()
    total_loss = 0.0
    total_batches = len(train_loader)
    target_batches = min(total_batches, max_steps) if max_steps is not None else total_batches

    loss_fct = (
        torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else torch.nn.BCEWithLogitsLoss()
    )

    slice_start_time = time.time()
    steps_done = 0

    # Track progress across batches with periodic flushed log updates
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        if max_steps is not None and batch_idx >= max_steps:
            break

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        loss = loss_fct(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps_done += 1

        # Log batch progress every 100 batches and at the end of epoch/slice
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == target_batches:
            running_loss = total_loss / (batch_idx + 1)
            pct = 100.0 * (batch_idx + 1) / target_batches
            print(
                f"  [Batch {batch_idx + 1:4d}/{target_batches} ({pct:5.1f}%)] "
                f"Current Batch Loss: {loss.item():.4f} | Running Avg Loss: {running_loss:.4f}",
                flush=True,
            )

    if max_steps is not None and steps_done > 0:
        elapsed = time.time() - slice_start_time
        sec_per_step = elapsed / steps_done
        cuda_mem = ""
        if torch.cuda.is_available():
            cuda_mem = f" | Peak CUDA Allocated: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB"
        print(f"\n{'=' * 60}")
        print(f"[TIMING BENCHMARK] Completed {steps_done} steps in {elapsed:.2f}s ({sec_per_step:.4f} s/step){cuda_mem}")
        print(f"{'=' * 60}\n", flush=True)

    return total_loss / max(1, steps_done)


def get_model_state_dict(model):
    """
    Returns weights for saving:
    LoRA adapters and classification head for PEFT; all weights otherwise.

    Inputs:
    --- model: transformers model or peft.PeftModel
    Output:
    --- state_dict: dict
    """
    if isinstance(model, PeftModel):
        return get_peft_model_state_dict(model)
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    """
    Loads weights returned by _get_model_state_dict into an already constructed model,
     using PEFT loading for LoRA and standard loading otherwise.

    Inputs:
    --- model: transformers model or peft.PeftModel
    --- state_dict: dict, as returned by _get_model_state_dict
    Output: None
    """
    if isinstance(model, PeftModel):
        set_peft_model_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict)


def full_train_for_one_configuration(
    model,
    train_loader,
    validation_loader,
    optimizer,
    device,
    checkpoint_path,
    save_directory,
    num_epochs=8,
    scheduler=None,
    selection_metric="coverage",
    pos_weight=None,
    tau_candidates=None,
    rare_indices=None,
    max_steps=None,
):
    """
    Trains one configuration and supports resuming from a checkpoint.
    The best model is selected according to validation Macro-F1 (or rare F1),
    falling back to validation loss if selection_metric="loss".

    A temporary checkpoint is saved after training each epoch.
    A complete checkpoint is saved after validation.
    For a PeftModel (LoRA mode), only adapter + modules_to_save parameters are saved.

    Best weights and validation results are saved to "best_artifacts.pt"
    only when the selection metric improves. This file is saved before marking
    the epoch checkpoint complete, so the best results are available on resume.

    Assumptions:
    --- model has already been moved to device.
    --- loaders are not empty.
    --- num_epochs is a positive integer (default 8 per Alkan et al. replication).
    --- checkpoint_path belongs to this configuration and dataset.
    --- save_directory remains the same across resumes of a given checkpoint.
    --- when resuming, model, optimizer and scheduler are constructed
        with the same settings as the original run.
    --- checkpoints were created by this function.

    Inputs:
    --- model: transformers model or peft.PeftModel
    --- train_loader: torch.utils.data.DataLoader
    --- validation_loader: torch.utils.data.DataLoader
    --- optimizer: torch.optim.Optimizer
    --- device: torch.device
    --- checkpoint_path: str, path used for resuming training
    --- save_directory: str, directory for the selected model
    --- num_epochs: int, total number of epochs (default 8)
    --- scheduler: optional LR scheduler, stepped once per training batch
    --- selection_metric: str, "f1" (best F1 on validation) or "loss" (lowest loss)
    --- tau_candidates: optional list of float thresholds to sweep
    --- rare_indices: optional list of tail/rare topic column indices for targeted F1

    Outputs:
    --- model: model restored to the best validation epoch
    --- best_configuration: dict containing training information
    --- best_evaluation_results: dict containing validation probabilities,
        labels, best_tau, and best_macro_f1
    --- best_validation_loss: float
    """

    # Creates local directory for saving progress:
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    # Separate file for the best model weights and validation results:
    best_artifacts_path = os.path.join(save_directory, "best_artifacts.pt")

    # Variables used for selecting the best model:
    best_validation_loss = float("inf")
    best_target_score = -1.0 if selection_metric in ["f1", "coverage"] else float("inf")
    best_coverage_50 = 0.0
    best_model_state = None
    best_evaluation_results = None
    best_epoch = None
    best_tau = None
    best_macro_f1 = None
    best_rare_f1 = None

    # Default starting state:
    start_epoch = 0
    validation_completed = True
    saved_train_loss = None

    # Continues from an existing checkpoint:
    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        load_model_state_dict(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        saved_train_loss = checkpoint["train_loss"]
        saved_scheduler_state = checkpoint["scheduler_state_dict"]
        if (scheduler is None) != (saved_scheduler_state is None):
            raise ValueError("Scheduler setup must match the saved checkpoint.")
        if scheduler is not None:
            scheduler.load_state_dict(saved_scheduler_state)

        validation_completed = checkpoint["validation_completed"]
        if validation_completed:
            start_epoch = checkpoint["completed_epoch"] + 1
            print(f"Checkpoint loaded. Continuing from epoch {start_epoch + 1}.")
        else:
            start_epoch = checkpoint["completed_epoch"]
            print(f"Training for epoch {start_epoch + 1} was completed. Continuing with validation.")

        # Stops if a previously selected best model is missing.
        if (
            checkpoint.get("best_epoch") is not None
            and not os.path.exists(best_artifacts_path)
        ):
            raise FileNotFoundError(
                "Cannot resume: best_artifacts.pt is missing "
                "although the checkpoint records a best model."
            )

        del checkpoint

        # Restores the best model information from its separate file:
        if os.path.exists(best_artifacts_path):
            best_artifacts = torch.load(best_artifacts_path, map_location="cpu", weights_only=True)

            best_validation_loss = best_artifacts["best_validation_loss"]
            best_model_state = best_artifacts["best_model_state"]
            best_evaluation_results = best_artifacts["best_evaluation_results"]
            best_epoch = best_artifacts["best_epoch"]
            best_tau = best_artifacts.get("best_tau")
            best_macro_f1 = best_artifacts.get("best_macro_f1")
            best_rare_f1 = best_artifacts.get("best_rare_f1")
            best_coverage_50 = best_artifacts.get("best_coverage_50", 0.0)

            if selection_metric == "coverage":
                best_target_score = best_artifacts.get("best_target_score", best_coverage_50)
            elif selection_metric == "f1":
                best_target_score = best_artifacts.get("best_target_score", best_macro_f1 if best_macro_f1 is not None else -1.0)
            else:
                best_target_score = best_validation_loss

            del best_artifacts

    epoch = start_epoch
    while epoch < num_epochs:

        # Skips training if it was completed before the interruption:
        resume_before_validation = (epoch == start_epoch and not validation_completed)

        if resume_before_validation:
            train_loss = saved_train_loss
        else:
            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                pos_weight=pos_weight,
                max_steps=max_steps,
            )

        if max_steps is not None:
            training_info = {
                "original_best_epoch": 1,
                "best_tau": 0.4,
                "best_coverage_50": None,
                "best_macro_f1": None,
                "best_rare_f1": None,
                "max_steps_slice": max_steps,
            }
            return model, training_info, {}, train_loss

        # Saves the current training state before validation.
        checkpoint = {
            "completed_epoch": epoch,
            "validation_completed": False,
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "train_loss": train_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_target_score": best_target_score,
            "best_coverage_50": best_coverage_50,
            "best_tau": best_tau,
        }

        # Replaces the previous checkpoint only after writing succeeds:
        temporary_path = str(checkpoint_path) + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

        # Validates the model after the training epoch:
        validation_loss, probabilities, labels = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
            pos_weight=pos_weight,
        )

        # Save per-epoch validation probabilities for Shai's evaluate.py (WORK_PLAN Stage 3)
        val_probs_epoch_path = os.path.join(save_directory, f"val_probs_epoch_{epoch + 1}.pt")
        torch.save(probabilities.detach().cpu(), val_probs_epoch_path)

        # Compute Coverage@50 matching Gemma shortlist objective
        current_coverage_50 = compute_coverage_at_k(probabilities, labels, k=50)

        # Evaluates F1 across tau thresholds:
        threshold_eval = evaluate_thresholds(
            probabilities=probabilities,
            true_labels=labels,
            tau_candidates=tau_candidates,
            rare_indices=rare_indices,
        )
        current_macro_f1 = threshold_eval["best_macro_f1"]
        current_rare_f1 = threshold_eval["best_rare_f1"]
        current_tau = threshold_eval["best_tau"]
        current_target_score = threshold_eval["best_target_f1"]

        f1_str = f"Coverage@50={current_coverage_50*100:.2f}%, Macro-F1={current_macro_f1:.4f} (at tau={current_tau})"
        if current_rare_f1 is not None:
            f1_str += f", Rare-F1={current_rare_f1:.4f}"

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f} | {f1_str}",
            flush=True,
        )

        # Model selection decision:
        if selection_metric == "coverage":
            is_best = current_coverage_50 > best_target_score
        elif selection_metric == "f1":
            is_best = current_target_score > best_target_score
        else:
            is_best = validation_loss < best_validation_loss

        if is_best:
            if selection_metric == "coverage":
                best_target_score = current_coverage_50
            elif selection_metric == "f1":
                best_target_score = current_target_score
            else:
                best_target_score = validation_loss

            best_validation_loss = validation_loss
            best_coverage_50 = current_coverage_50
            best_epoch = epoch
            best_tau = current_tau
            best_macro_f1 = current_macro_f1
            best_rare_f1 = current_rare_f1

            best_evaluation_results = {
                "probabilities": probabilities,
                "true_labels": labels,
                "best_tau": best_tau,
                "best_macro_f1": best_macro_f1,
                "best_rare_f1": best_rare_f1,
                "best_coverage_50": best_coverage_50,
                "tau_sweep": threshold_eval["tau_sweep"],
            }

            # Saves an independent CPU copy of the best parameters:
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in get_model_state_dict(model).items()
            }

            # Saves the best artifacts before marking validation complete:
            best_artifacts = {
                "best_validation_loss": best_validation_loss,
                "best_target_score": best_target_score,
                "best_coverage_50": best_coverage_50,
                "best_tau": best_tau,
                "best_macro_f1": best_macro_f1,
                "best_rare_f1": best_rare_f1,
                "best_model_state": best_model_state,
                "best_evaluation_results": best_evaluation_results,
                "best_epoch": best_epoch,
            }

            temporary_best_path = str(best_artifacts_path) + ".tmp"
            torch.save(best_artifacts, temporary_best_path)
            os.replace(temporary_best_path, best_artifacts_path)

            del best_artifacts

        # Updates and saves the complete checkpoint after validation:
        checkpoint.update({
            "validation_completed": True,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "best_target_score": best_target_score,
            "best_coverage_50": best_coverage_50,
            "best_epoch": best_epoch,
            "best_tau": best_tau,
        })

        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

        del checkpoint

        print(f"Complete checkpoint saved after epoch {epoch + 1}.", flush=True)

        # From the next epoch onward, perform normal training:
        validation_completed = True
        saved_train_loss = None
        epoch += 1

    # All requested epochs have finished:
    if best_model_state is None:
        raise RuntimeError("Training ended without selecting a best model.")

    # Restores the selected model:
    load_model_state_dict(model, best_model_state)
    model.eval()

    training_info = {
        "original_best_epoch": best_epoch + 1,
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "selection_metric": selection_metric,
        "best_tau": best_tau,
        "best_macro_f1": best_macro_f1,
        "best_rare_f1": best_rare_f1,
        "best_coverage_50": best_coverage_50,
    }

    # Saves the selected model separately from the resume checkpoint:
    best_model_path = os.path.join(save_directory, "best_model.pt")

    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_configuration": training_info,
            "best_evaluation_results": best_evaluation_results,
            "best_validation_loss": best_validation_loss,
        },
        best_model_path,
    )

    print(
        f"\nBest model selected at epoch {best_epoch + 1}: "
        f"val_loss={best_validation_loss:.4f}, "
        f"Coverage@50={best_coverage_50*100:.2f}%, "
        f"Macro-F1={best_macro_f1:.4f} (tau={best_tau})"
    )

    return (
        model,
        training_info,
        best_evaluation_results,
        best_validation_loss,
    )


def run_SciBERT_experiment(
    mode: str,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    num_labels: int,
    device: Optional[torch.device] = None,
    save_directory: str = "artifacts/scibert",
    num_epochs: int = 8,
    learning_rate: Optional[float] = None,
    pos_weight: Optional[float] = 30.0,
    lora_params: Optional[dict] = None,
    rare_indices: Optional[List[int]] = None,
    selection_metric: str = "coverage",
    tau_candidates: Optional[List[float]] = None,
    test_bibcodes: Optional[List[str]] = None,
    val_bibcodes: Optional[List[str]] = None,
    idx_to_topic: Optional[Dict[int, Any]] = None,
    export_candidates_path: Optional[str] = None,
    export_val_candidates_path: Optional[str] = None,
    k_candidates: int = 50,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes an end-to-end SciBERT experiment arm (Full fine-tuning or LoRA).

    Steps:
    1. Sets up device (MPS / CUDA / CPU) and model architecture.
    2. Configures AdamW optimizer (with differential LR for full mode) and linear decay scheduler.
    3. Runs full_train_for_one_configuration with checkpointing and pos_weight loss.
    4. Evaluates winning model on test set with the tuned validation tau and computes Coverage@k.
    5. Optionally exports top-50 candidate lists for Gemma (Stage 4).

    Inputs:
    --- mode: "full" or "lora"
    --- dataloaders: dict with keys "train", "validation", and optionally "test"
    --- num_labels: int, number of target topics (e.g. 1,864)
    --- device: torch.device or None (auto-detects MPS)
    --- save_directory: directory path for saving checkpoints and best models
    --- num_epochs: int, training epochs (default 8)
    --- learning_rate: float, defaults to 2e-5 for full, 1e-4 for LoRA
    --- pos_weight: float or None, positive class weight for BCE loss (default 30.0)
    --- lora_params: LoRA hyperparameters dict (if mode="lora")
    --- rare_indices: list of column indices corresponding to rare topics
    --- selection_metric: "coverage", "f1", or "loss"
    --- tau_candidates: thresholds to test
    --- test_bibcodes: list of test set bibcodes for candidate export
    --- idx_to_topic: mapping from column index to UAT ID
    --- export_candidates_path: optional filepath to save top-k candidates JSON
    --- k_candidates: number of candidate topics per paper (default: 50)

    Outputs:
    --- results: dict containing winning configuration, test metrics, and runtime
    """
    mode = mode.lower()
    if mode not in ["full", "lora"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full' or 'lora'.")

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"\n{'=' * 60}")
    print(f"Starting SciBERT ({mode.upper()}) Experiment on {device}")
    print(f"{'=' * 60}")

    arm_dir = os.path.join(save_directory, mode)
    checkpoint_path = os.path.join(arm_dir, "checkpoint.pt")

    # 1. Hyperparameters matching specification
    if learning_rate is None:
        # Full: 2e-5 (Alkan et al. Table 5); LoRA: 1e-4
        learning_rate = 2e-5 if mode == "full" else 1e-4

    if mode == "lora" and lora_params is None:
        lora_params = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value"],
        }

    # 2. Build model
    model = build_scibert(
        num_labels=num_labels,
        mode=mode,
        lora_configuration_params=lora_params,
    )
    model.to(device)

    # 3. Optimizer & Scheduler
    if mode == "full":
        head_lr = max(learning_rate * 5, 1e-4)
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "score" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        optimizer_grouped_parameters = [
            {"params": backbone_params, "lr": learning_rate, "weight_decay": 0.01},
            {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        print(f"[SciBERT Setup] Differential LR: backbone={learning_rate:.2e}, classifier head={head_lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]
    total_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([float(pos_weight)], device=device)
        print(f"[SciBERT Setup] Using pos_weight={pos_weight:.1f} for BCEWithLogitsLoss to counter label imbalance.")

    start_time = time.time()

    # 4. Train with checkpointing & threshold tuning
    model, training_info, val_eval_results, best_val_loss = full_train_for_one_configuration(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_path=checkpoint_path,
        save_directory=arm_dir,
        num_epochs=num_epochs,
        scheduler=scheduler,
        selection_metric=selection_metric,
        pos_weight=pos_weight_tensor,
        tau_candidates=tau_candidates,
        rare_indices=rare_indices,
        max_steps=max_steps,
    )

    elapsed_time = time.time() - start_time
    best_tau = training_info["best_tau"]

    if max_steps is not None:
        print(f"\n[SciBERT Runner] max_steps={max_steps} timing benchmark slice completed in {elapsed_time:.2f}s.")
        return {
            "mode": mode,
            "device": str(device),
            "max_steps": max_steps,
            "training_time_seconds": round(elapsed_time, 2),
            "seconds_per_step": round(elapsed_time / max(1, max_steps), 4),
        }

    results = {
        "mode": mode,
        "device": str(device),
        "learning_rate": learning_rate,
        "pos_weight": pos_weight,
        "selection_metric": selection_metric,
        "num_epochs": num_epochs,
        "training_time_seconds": round(elapsed_time, 2),
        "best_epoch": training_info["original_best_epoch"],
        "best_val_loss": round(best_val_loss, 4),
        "best_tau": best_tau,
        "val_coverage_50": training_info.get("best_coverage_50"),
        "val_macro_f1": training_info.get("best_macro_f1"),
        "val_rare_f1": training_info.get("best_rare_f1"),
    }

    # 5. Evaluate on Test Set (if provided)
    if "test" in dataloaders and dataloaders["test"] is not None:
        print(f"\nEvaluating {mode.upper()} model on Test split using tuned tau={best_tau}...")
        test_loss, test_probs, test_labels = evaluate(
            model=model,
            data_loader=dataloaders["test"],
            device=device,
            pos_weight=pos_weight_tensor,
        )

        test_eval = evaluate_thresholds(
            probabilities=test_probs,
            true_labels=test_labels,
            tau_candidates=[best_tau],
            rare_indices=rare_indices,
        )

        test_coverage_50 = compute_coverage_at_k(test_probs, test_labels, k=k_candidates)

        results["test_loss"] = round(test_loss, 4)
        results["test_coverage_50"] = round(test_coverage_50, 4)
        results["test_macro_f1"] = test_eval["best_macro_f1"]
        results["test_rare_f1"] = test_eval["best_rare_f1"]

        print(
            f"Test Results: loss={test_loss:.4f}, "
            f"Coverage@50={test_coverage_50*100:.2f}%, "
            f"Macro-F1={results['test_macro_f1']:.4f}"
            + (f", Rare-F1={results['test_rare_f1']:.4f}" if results['test_rare_f1'] else "")
        )

        # Save test probabilities for Shai's evaluate.py (WORK_PLAN Stage 3)
        torch.save(test_probs.detach().cpu(), os.path.join(arm_dir, "test_probabilities.pt"))
        torch.save(test_labels.detach().cpu(), os.path.join(arm_dir, "test_labels.pt"))

        # 6. Export top-k candidates for Gemma on Test set (Handoff #4)
        if export_candidates_path and test_bibcodes and idx_to_topic:
            export_top_k_candidates(
                probabilities=test_probs,
                bibcodes=test_bibcodes,
                idx_to_topic=idx_to_topic,
                output_path=export_candidates_path,
                k=k_candidates,
            )
            results["test_candidates_file"] = str(export_candidates_path)

    # 7. Export top-k candidates for Gemma on Validation set (Handoff #4)
    if (
        export_val_candidates_path
        and val_bibcodes
        and idx_to_topic
        and "probabilities" in val_eval_results
        and len(val_bibcodes) == len(val_eval_results["probabilities"])
    ):
        print(f"\nExporting top-{k_candidates} candidate lists for Validation split...")
        export_top_k_candidates(
            probabilities=val_eval_results["probabilities"],
            bibcodes=val_bibcodes,
            idx_to_topic=idx_to_topic,
            output_path=export_val_candidates_path,
            k=k_candidates,
        )
        results["val_candidates_file"] = str(export_val_candidates_path)

    # Save experiment summary JSON
    summary_path = os.path.join(arm_dir, "experiment_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment complete for SciBERT-{mode.upper()}. Summary saved to {summary_path}")
    return results


if __name__ == "__main__":
    import argparse
    import sys
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "data"))
    sys.path.insert(0, str(project_root / "scripts"))
    from build_split import load_split

    parser = argparse.ArgumentParser(description="Run SciBERT training and evaluation.")
    parser.add_argument(
        "--mode",
        choices=["full", "lora"],
        default="full",
        help="Training mode: 'full' fine-tuning or 'lora' (default: 'full')",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=8,
        help="Number of epochs to train (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Validation/test batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5 for full, 1e-4 for lora)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=30.0,
        help="Positive class weight for BCE loss to counter 432:1 imbalance (default: 30.0)",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["coverage", "f1", "loss"],
        default="coverage",
        help="Metric for selecting best checkpoint: 'coverage' (Coverage@50), 'f1', or 'loss' (default: 'coverage')",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts/scibert",
        help="Root directory to save checkpoints and outputs (default: 'artifacts/scibert')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('mps', 'cuda', 'cpu'; auto-detects if None)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a 1-step verification pass on 32 papers to verify end-to-end functionality",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing checkpoints and start training fresh from epoch 1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps to execute before stopping (useful for 200-step timing slices on 3060 Ti)",
    )
    args = parser.parse_args()

    # 1. Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n[SciBERT Runner] Target device: {device}")
    print("[SciBERT Runner] Loading Stage 1 split data from data/ ...")
    split_data = load_split()
    train_df = split_data["train_df"]
    val_df = split_data["validation_df"]
    test_df = split_data["test_df"]
    topic_to_idx = split_data["topic_to_idx"]
    idx_to_topic = split_data["idx_to_topic"]
    rare_indices = split_data["rare_indices"]
    num_labels = split_data["num_labels"]

    if args.dry_run:
        args.save_dir = os.path.join(args.save_dir, "dry_run")
        print(f"[SciBERT Runner] Dry-run mode enabled: using isolated directory {args.save_dir} with 32 train, 16 val, 16 test papers.")
        train_df = train_df.iloc[:32].reset_index(drop=True)
        val_df = val_df.iloc[:16].reset_index(drop=True)
        test_df = test_df.iloc[:16].reset_index(drop=True)
        args.num_epochs = 1
        args.force_restart = True

    if args.force_restart:
        arm_dir = os.path.join(args.save_dir, args.mode)
        if os.path.exists(arm_dir):
            import shutil
            shutil.rmtree(arm_dir)
            print(f"[SciBERT Runner] Fresh start: cleared previous checkpoints in {arm_dir}")

    print("[SciBERT Runner] Building DataLoaders...")
    dataloaders = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        topic_to_idx=topic_to_idx,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    test_bibcodes = test_df["bibcode"].tolist()
    val_bibcodes = val_df["bibcode"].tolist()
    export_test_path = os.path.join(args.save_dir, args.mode, "gemma_candidates_test.json")
    export_val_path = os.path.join(args.save_dir, args.mode, "gemma_candidates_validation.json")

    results = run_SciBERT_experiment(
        mode=args.mode,
        dataloaders=dataloaders,
        num_labels=num_labels,
        device=device,
        save_directory=args.save_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        pos_weight=args.pos_weight,
        rare_indices=rare_indices,
        selection_metric=args.selection_metric,
        test_bibcodes=test_bibcodes,
        val_bibcodes=val_bibcodes,
        idx_to_topic=idx_to_topic,
        export_candidates_path=export_test_path,
        export_val_candidates_path=export_val_path,
        k_candidates=50,
        max_steps=args.max_steps,
    )