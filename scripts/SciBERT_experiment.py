# By Shai Habi, September 4th, 2026.
# This file implements the SciBERT part in our project

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def build_scibert(num_labels, mode="full", lora_params=None):
    """
    Build SciBert model while adding a classification head on top of the existing model.
    When model is LORA, the model adds LORA's matrices with new weights.

    Assumptions:
    --- lora_param contains the following keys:
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

    return model

