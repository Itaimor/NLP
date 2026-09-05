# This file implements the SciBERT part in our project

import torch

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def build_scibert(num_labels, mode="full", lora_params=None):
    """
    Build SciBert model while adding a classification head on top of the existing model.
    When model is LORA, the model adds LORA's matrices with new weights.

    Assumptions:
    --- lora_param contains the following keys and should be as follows:
        lora_params = {
        "r": 8,                                # Adapter rank
        "lora_alpha": 16,                       # Update scaling: alpha / r
        "lora_dropout": 0.1,                    # LoRA dropout probability
        "target_modules": ["query", "value"],   # Projections receiving adapters
        }
    Inputs:
    --- num_labels: int
    --- mode: str
    --- lora_params: None or dict
    Output:
    --- model:
    """


    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    if mode.lower() == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,      # Classifies the entire input text
            modules_to_save=["classifier"],  # Trains and saves the classification head
            bias="none",                     # Keeps base-model biases frozen
            **lora_params,
        )

        model = get_peft_model(model, config)     # Attach adapters and freeze base weights
        model.print_trainable_parameters()

    return model


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



def evaluate(model, data_loader, device):
    """
    Evaluates SciBERT without updating its parameters for the validation and test sets.
    This implementation supports both full fine-tuning and LoRA.
    Runs the model once per batch and returns probabilities for all samples,
    allowing multiple tau thresholds to be tested without rerunning the model.

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