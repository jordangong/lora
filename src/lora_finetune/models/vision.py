"""Vision model utilities for ViT finetuning."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from ..config import ModelConfig

logger = logging.getLogger(__name__)


VISION_TARGET_MODULES = {
    "vit": ["query", "key", "value", "dense"],
    "swin": ["query", "key", "value", "dense"],
    "deit": ["query", "key", "value", "dense"],
    "beit": ["query", "key", "value", "dense"],
    "convnext": ["dwconv", "pwconv1", "pwconv2"],
    "resnet": ["conv1", "conv2", "conv3"],
    "clip": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "dinov2": ["query", "key", "value", "dense"],
    "default": ["query", "key", "value"],
}


def get_vision_target_modules(model_name_or_path: str) -> List[str]:
    """Get target modules for LoRA based on vision model architecture."""
    model_name_lower = model_name_or_path.lower()

    for key in VISION_TARGET_MODULES:
        if key in model_name_lower:
            return VISION_TARGET_MODULES[key]

    return VISION_TARGET_MODULES["default"]


def load_vision_model(
    config: ModelConfig,
    num_labels: int,
    device_map: Optional[str] = "auto",
) -> Tuple[PreTrainedModel, Any]:
    """Load vision model for image classification."""
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }

    if config.torch_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config.torch_dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif config.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch.float32

    if config.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2 for vision model")

    if config.load_in_4bit:
        compute_dtype = (
            torch.bfloat16 if config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        logger.info("Loading vision model in 4-bit quantization")
    elif config.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Loading vision model in 8-bit quantization")

    logger.info(f"Loading vision model: {config.model_name_or_path}")
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )

    return model, image_processor


def get_num_labels_from_dataset(dataset) -> int:
    """Get number of labels from dataset."""
    if hasattr(dataset, "features") and "label" in dataset.features:
        return dataset.features["label"].num_classes
    elif hasattr(dataset, "features") and "labels" in dataset.features:
        return dataset.features["labels"].num_classes
    else:
        unique_labels = set()
        for example in dataset:
            if "label" in example:
                unique_labels.add(example["label"])
            elif "labels" in example:
                unique_labels.add(example["labels"])
        return len(unique_labels)


def get_id2label(dataset) -> Dict[int, str]:
    """Get id2label mapping from dataset."""
    if hasattr(dataset, "features") and "label" in dataset.features:
        return {i: name for i, name in enumerate(dataset.features["label"].names)}
    return {}


def get_label2id(dataset) -> Dict[str, int]:
    """Get label2id mapping from dataset."""
    if hasattr(dataset, "features") and "label" in dataset.features:
        return {name: i for i, name in enumerate(dataset.features["label"].names)}
    return {}
