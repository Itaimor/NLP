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


def evaluate(model, data_loader, device):
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
    Outputs:
    --- average_loss: float, mean batch loss
    --- probabilities: torch.Tensor with shape [num_samples, num_labels]
    --- true_labels: torch.Tensor with shape [num_samples, num_labels]
    """

    model.eval()

    total_loss = 0.0
    all_probabilities = []
    all_labels = []

    with torch.inference_mode():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device=device, dtype=torch.float32)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
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
        tau_candidates = [round(0.10 + i * 0.05, 2) for i in range(17)]  # 0.10 to 0.90

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
            target_f1 = rare_f1
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

def train_one_epoch(model, train_loader, optimizer, device, scheduler=None):
    """
    Trains SciBert for one epoch and return the average loss.
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
    Output:
    --- avg_loss: float, mean loss over the epoch
    """

    model.train()
    total_loss = 0.0

    for input_ids, attention_mask, labels in train_loader:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


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
    selection_metric="f1",
    tau_candidates=None,
    rare_indices=None,
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
    best_target_score = -1.0 if selection_metric == "f1" else float("inf")
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

            if selection_metric == "f1":
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
            )

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
        )

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

        f1_str = f"Macro-F1={current_macro_f1:.4f} (at tau={current_tau})"
        if current_rare_f1 is not None:
            f1_str += f", Rare-F1={current_rare_f1:.4f}"

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f} | {f1_str}"
        )

        # Model selection decision:
        if selection_metric == "f1":
            is_best = current_target_score > best_target_score
        else:
            is_best = validation_loss < best_validation_loss

        if is_best:
            if selection_metric == "f1":
                best_target_score = current_target_score
            else:
                best_target_score = validation_loss

            best_validation_loss = validation_loss
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
            "best_epoch": best_epoch,
            "best_tau": best_tau,
        })

        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

        del checkpoint

        print(f"Complete checkpoint saved after epoch {epoch + 1}.")

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
    lora_params: Optional[dict] = None,
    rare_indices: Optional[List[int]] = None,
    selection_metric: str = "f1",
    tau_candidates: Optional[List[float]] = None,
    test_bibcodes: Optional[List[str]] = None,
    idx_to_topic: Optional[Dict[int, Any]] = None,
    export_candidates_path: Optional[str] = None,
    k_candidates: int = 50,
) -> Dict[str, Any]:
    """
    Executes an end-to-end SciBERT experiment arm (Full fine-tuning or LoRA).

    Steps:
    1. Sets up device (MPS / CUDA / CPU) and model architecture.
    2. Configures AdamW optimizer and linear decay scheduler.
    3. Runs full_train_for_one_configuration with checkpointing.
    4. Evaluates winning model on test set with the tuned validation tau.
    5. Optionally exports top-50 candidate lists for Gemma (Stage 4).

    Inputs:
    --- mode: "full" or "lora"
    --- dataloaders: dict with keys "train", "validation", and optionally "test"
    --- num_labels: int, number of target topics (e.g. 1,864)
    --- device: torch.device or None (auto-detects MPS)
    --- save_directory: directory path for saving checkpoints and best models
    --- num_epochs: int, training epochs (default 8)
    --- learning_rate: float, defaults to 2e-5 for full, 1e-4 for LoRA
    --- lora_params: LoRA hyperparameters dict (if mode="lora")
    --- rare_indices: list of column indices corresponding to rare topics
    --- selection_metric: "f1" or "loss"
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
        tau_candidates=tau_candidates,
        rare_indices=rare_indices,
    )

    elapsed_time = time.time() - start_time
    best_tau = training_info["best_tau"]

    results = {
        "mode": mode,
        "device": str(device),
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "training_time_seconds": round(elapsed_time, 2),
        "best_epoch": training_info["original_best_epoch"],
        "best_val_loss": round(best_val_loss, 4),
        "best_tau": best_tau,
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
        )

        test_eval = evaluate_thresholds(
            probabilities=test_probs,
            true_labels=test_labels,
            tau_candidates=[best_tau],
            rare_indices=rare_indices,
        )

        results["test_loss"] = round(test_loss, 4)
        results["test_macro_f1"] = test_eval["best_macro_f1"]
        results["test_rare_f1"] = test_eval["best_rare_f1"]

        print(
            f"Test Results: loss={test_loss:.4f}, "
            f"Macro-F1={results['test_macro_f1']:.4f}"
            + (f", Rare-F1={results['test_rare_f1']:.4f}" if results['test_rare_f1'] else "")
        )

        # 6. Export top-k candidates for Gemma (Handoff #4)
        if export_candidates_path and test_bibcodes and idx_to_topic:
            export_top_k_candidates(
                probabilities=test_probs,
                bibcodes=test_bibcodes,
                idx_to_topic=idx_to_topic,
                output_path=export_candidates_path,
                k=k_candidates,
            )
            results["candidates_file"] = str(export_candidates_path)

    # Save experiment summary JSON
    summary_path = os.path.join(arm_dir, "experiment_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment complete for SciBERT-{mode.upper()}. Summary saved to {summary_path}")
    return results