# This file implements the SciBERT part in our project

import torch
import os

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict



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
        problem_type="multi_label_classification"
    )

    if mode == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,      # Classifies the entire input text
            modules_to_save=["classifier"],  # Trains and saves the classification head
            bias="none",                     # Keeps base-model biases frozen
            **lora_configuration_params,
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

            logits = outputs.logits
            probabilities = torch.sigmoid(logits)

            loss = outputs.loss
            total_loss += loss.item()

            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())

    average_loss = total_loss / len(data_loader)

    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return average_loss, all_probabilities, all_labels


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

    return total_loss / len(train_loader)



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
        num_epochs=20,
        scheduler=None,
):
    """
    Trains one configuration and supports resuming from a checkpoint.
    The best model is selected according to validation loss.

    A temporary checkpoint is saved after training each epoch.
    A complete checkpoint is saved after validation.
    For a PeftModel (LoRA mode), only adapter + modules_to_save parameters are saved.

    Best weights and validation results are saved to "best_artifacts.pt"
    only when validation loss improves. This file is saved before marking
    the epoch checkpoint complete, so the best results are available on resume.

    Assumptions:
    --- model has already been moved to device.
    --- loaders are not empty.
    --- num_epochs is a positive integer.
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
    --- num_epochs: int, total number of epochs including completed ones
    --- scheduler: optional LR scheduler, stepped once per training batch

    Outputs:
    --- model: model restored to the best validation epoch
    --- best_configuration: dict containing training information
    --- best_evaluation_results: dict containing validation probabilities
        and labels for testing tau thresholds without rerunning the model
    --- best_validation_loss: float
    """

    # Creates local directory for saving progress:
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    # Separate file for the best model weights and validation results:
    best_artifacts_path = os.path.join(save_directory, "best_artifacts.pt")

    # Variables used for selecting the best model:
    best_validation_loss = float("inf")
    best_model_state = None
    best_evaluation_results = None
    best_epoch = None

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

            del best_artifacts


    epoch = start_epoch
    while epoch < num_epochs:

        # Skips training if it was completed before the interruption:
        resume_before_validation = ( epoch == start_epoch and not validation_completed )

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
        # Best weights and evaluation results remain in their separate file.
        checkpoint = {
            "completed_epoch": epoch,
            "validation_completed": False,
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "train_loss": train_loss,

            # Remembers the best model even before validation completes.
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
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

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}"
        )

        # Prefers lower validation loss:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

            best_evaluation_results = {
                "probabilities": probabilities,
                "true_labels": labels,
            }

            # Saves an independent CPU copy of the best parameters:
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in get_model_state_dict(model).items()
            }

            # Saves the best artifacts before marking validation complete:
            best_artifacts = {
                "best_validation_loss": best_validation_loss,
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
            "best_epoch": best_epoch,
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

    best_configuration = {
        "original_best_epoch": best_epoch + 1,
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size,
    }

    # Saves the selected model separately from the resume checkpoint:
    best_model_path = os.path.join(save_directory, "best_model.pt")

    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_configuration": best_configuration,
            "best_evaluation_results": best_evaluation_results,
            "best_validation_loss": best_validation_loss,
        },
        best_model_path,
    )

    print(
        f"\nBest model was found at epoch {best_epoch + 1}: "
        f"validation_loss={best_validation_loss:.4f}"
    )

    return (
        model,
        best_configuration,
        best_evaluation_results,
        best_validation_loss,
    )