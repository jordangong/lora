"""Vision model utilities for ViT finetuning."""

import logging
from typing import Dict, List

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


def get_num_labels_from_dataset(dataset) -> int:
    """Get number of labels from dataset."""

    def _extract_num_classes(feature) -> int | None:
        num_classes = getattr(feature, "num_classes", None)
        if isinstance(num_classes, int) and num_classes > 0:
            return num_classes

        names = getattr(feature, "names", None)
        if names:
            return len(names)

        return None

    label_column = None
    if hasattr(dataset, "features") and "label" in dataset.features:
        label_column = "label"
    elif hasattr(dataset, "features") and "labels" in dataset.features:
        label_column = "labels"

    if label_column is not None:
        feature_num_classes = _extract_num_classes(dataset.features[label_column])
        if feature_num_classes is not None:
            return feature_num_classes

        if hasattr(dataset, "unique"):
            try:
                return len(dataset.unique(label_column))
            except Exception:
                pass

    unique_labels = set()
    for example in dataset:
        if "label" in example:
            unique_labels.add(example["label"])
        elif "labels" in example:
            unique_labels.add(example["labels"])

    if not unique_labels:
        raise ValueError("Could not infer number of labels from dataset")

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
