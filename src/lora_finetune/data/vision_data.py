"""Vision dataset utilities for ViT finetuning."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torchvision import transforms

from ..config import AugmentationConfig, DataConfig

logger = logging.getLogger(__name__)

# Default ImageNet normalization values
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def load_vision_dataset(config: DataConfig) -> DatasetDict:
    """Load vision dataset from HuggingFace Hub or local files."""
    if config.dataset_name:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            streaming=config.streaming,
        )
    elif config.train_file:
        data_files = {"train": config.train_file}
        if config.validation_file:
            data_files["validation"] = config.validation_file

        dataset = load_dataset("imagefolder", data_files=data_files)
    else:
        raise ValueError("Either dataset_name or train_file must be provided")

    return dataset


def extract_normalization_from_processor(
    image_processor: Any,
) -> Tuple[List[float], List[float]]:
    """Extract normalization mean and std from image processor."""
    mean = DEFAULT_MEAN
    std = DEFAULT_STD

    if image_processor is not None:
        if hasattr(image_processor, "image_mean") and image_processor.image_mean is not None:
            mean = list(image_processor.image_mean)
            logger.info(f"Using image_processor mean: {mean}")
        if hasattr(image_processor, "image_std") and image_processor.image_std is not None:
            std = list(image_processor.image_std)
            logger.info(f"Using image_processor std: {std}")

    return mean, std


def get_image_size_from_processor(image_processor: Any, default: int = 224) -> int:
    """Extract image size from image processor."""
    if image_processor is not None:
        if hasattr(image_processor, "size"):
            size = image_processor.size
            if isinstance(size, dict):
                return size.get("height", size.get("shortest_edge", default))
            elif isinstance(size, int):
                return size
        if hasattr(image_processor, "crop_size"):
            crop_size = image_processor.crop_size
            if isinstance(crop_size, dict):
                return crop_size.get("height", default)
            elif isinstance(crop_size, int):
                return crop_size
    return default


def build_train_transforms(
    image_size: int,
    aug_config: AugmentationConfig,
    mean: List[float],
    std: List[float],
) -> transforms.Compose:
    """Build training transforms from augmentation config."""
    transform_list = []

    # Spatial transforms (before ToTensor)
    if aug_config.random_resized_crop:
        transform_list.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=aug_config.random_resized_crop_scale,
                ratio=aug_config.random_resized_crop_ratio,
            )
        )
    else:
        transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.CenterCrop(image_size))

    if aug_config.random_horizontal_flip:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=aug_config.random_horizontal_flip_p)
        )

    if aug_config.random_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=aug_config.random_vertical_flip_p))

    if aug_config.random_rotation:
        transform_list.append(transforms.RandomRotation(degrees=aug_config.random_rotation_degrees))

    if aug_config.random_affine:
        transform_list.append(
            transforms.RandomAffine(
                degrees=aug_config.random_affine_degrees,
                translate=aug_config.random_affine_translate,
                scale=aug_config.random_affine_scale,
                shear=aug_config.random_affine_shear,
            )
        )

    # Color transforms (before ToTensor)
    if aug_config.color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=aug_config.color_jitter_brightness,
                contrast=aug_config.color_jitter_contrast,
                saturation=aug_config.color_jitter_saturation,
                hue=aug_config.color_jitter_hue,
            )
        )

    if aug_config.random_grayscale:
        transform_list.append(transforms.RandomGrayscale(p=aug_config.random_grayscale_p))

    # AutoAugment policies (before ToTensor)
    if aug_config.auto_augment:
        policy_map = {
            "imagenet": transforms.AutoAugmentPolicy.IMAGENET,
            "cifar10": transforms.AutoAugmentPolicy.CIFAR10,
            "svhn": transforms.AutoAugmentPolicy.SVHN,
        }
        policy = policy_map.get(aug_config.auto_augment.lower())
        if policy:
            transform_list.append(transforms.AutoAugment(policy=policy))

    if aug_config.rand_augment:
        transform_list.append(
            transforms.RandAugment(
                num_ops=aug_config.rand_augment_num_ops,
                magnitude=aug_config.rand_augment_magnitude,
            )
        )

    if aug_config.trivial_augment:
        transform_list.append(transforms.TrivialAugmentWide())

    # ToTensor
    transform_list.append(transforms.ToTensor())

    # Post-tensor transforms
    if aug_config.gaussian_blur:
        transform_list.append(
            transforms.GaussianBlur(
                kernel_size=aug_config.gaussian_blur_kernel_size,
                sigma=aug_config.gaussian_blur_sigma,
            )
        )

    if aug_config.random_erasing:
        transform_list.append(
            transforms.RandomErasing(
                p=aug_config.random_erasing_p,
                scale=aug_config.random_erasing_scale,
                ratio=aug_config.random_erasing_ratio,
            )
        )

    # Normalization (always last)
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def build_eval_transforms(
    image_size: int,
    aug_config: AugmentationConfig,
    mean: List[float],
    std: List[float],
) -> transforms.Compose:
    """Build evaluation transforms (no augmentation, just resize/crop/normalize)."""
    resize_size = int(image_size * aug_config.eval_resize_factor)

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_train_transforms(
    image_size: int = 224,
    aug_config: Optional[AugmentationConfig] = None,
    image_processor: Any = None,
) -> transforms.Compose:
    """Get training transforms for vision models.

    Args:
        image_size: Target image size
        aug_config: Augmentation configuration. If None, uses defaults.
        image_processor: HuggingFace image processor to extract normalization from.

    Returns:
        Composed transforms for training.
    """
    aug_config = aug_config or AugmentationConfig()

    # Get normalization values
    if aug_config.normalize_mean and aug_config.normalize_std:
        mean = aug_config.normalize_mean
        std = aug_config.normalize_std
    else:
        mean, std = extract_normalization_from_processor(image_processor)

    return build_train_transforms(image_size, aug_config, mean, std)


def get_eval_transforms(
    image_size: int = 224,
    aug_config: Optional[AugmentationConfig] = None,
    image_processor: Any = None,
) -> transforms.Compose:
    """Get evaluation transforms for vision models.

    Args:
        image_size: Target image size
        aug_config: Augmentation configuration (for resize factor and normalization).
        image_processor: HuggingFace image processor to extract normalization from.

    Returns:
        Composed transforms for evaluation.
    """
    aug_config = aug_config or AugmentationConfig()

    # Get normalization values
    if aug_config.normalize_mean and aug_config.normalize_std:
        mean = aug_config.normalize_mean
        std = aug_config.normalize_std
    else:
        mean, std = extract_normalization_from_processor(image_processor)

    return build_eval_transforms(image_size, aug_config, mean, std)


def preprocess_vision_example(
    example: Dict[str, Any],
    image_processor: Optional[Callable] = None,
    transform: Optional[transforms.Compose] = None,
    image_column: str = "image",
    label_column: str = "label",
) -> Dict[str, Any]:
    """Preprocess a single vision example."""
    image = example[image_column]

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")

    if image_processor is not None:
        inputs = image_processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
    elif transform is not None:
        pixel_values = transform(image)
    else:
        raise ValueError("Either image_processor or transform must be provided")

    result = {"pixel_values": pixel_values}

    if label_column in example:
        result["labels"] = example[label_column]

    return result


def make_transform_fn(
    transform: transforms.Compose,
    image_column: str,
    label_column: str,
) -> Callable:
    """Create a transform function for set_transform (online augmentation).

    Args:
        transform: Composed transforms to apply
        image_column: Column name for images
        label_column: Column name for labels

    Returns:
        Transform function compatible with dataset.set_transform()
    """

    def transform_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        images = examples[image_column]
        pixel_values = []

        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")

            pv = transform(image)
            pixel_values.append(pv)

        result = {"pixel_values": pixel_values}

        if label_column in examples:
            result["labels"] = examples[label_column]

        return result

    return transform_fn


def preprocess_vision_dataset(
    dataset: DatasetDict,
    image_processor: Optional[Callable] = None,
    config: Optional[DataConfig] = None,
) -> DatasetDict:
    """Prepare vision dataset for training with online augmentation.

    Uses set_transform for on-the-fly augmentation during training,
    which applies different random augmentations each epoch.

    Args:
        dataset: HuggingFace DatasetDict with train/validation splits
        image_processor: HuggingFace image processor (used to extract normalization)
        config: Data configuration including augmentation settings

    Returns:
        DatasetDict with transforms set for online augmentation
    """
    config = config or DataConfig()
    aug_config = config.augmentation

    # Validate that required columns exist in the dataset
    sample_split = config.train_split if config.train_split in dataset else list(dataset.keys())[0]
    available_columns = dataset[sample_split].column_names

    if config.image_column not in available_columns:
        raise ValueError(
            f"Image column '{config.image_column}' not found in dataset. "
            f"Available columns: {available_columns}. "
            f"Set data.image_column to the correct column name."
        )

    if config.label_column not in available_columns:
        raise ValueError(
            f"Label column '{config.label_column}' not found in dataset. "
            f"Available columns: {available_columns}. "
            f"Set data.label_column to the correct column name."
        )

    # Get image size - prefer from processor if available
    image_size = get_image_size_from_processor(image_processor, config.image_size)
    logger.info(f"Using image size: {image_size}")

    # Build transforms with augmentation config and processor normalization
    train_transform = get_train_transforms(
        image_size=image_size,
        aug_config=aug_config,
        image_processor=image_processor,
    )
    eval_transform = get_eval_transforms(
        image_size=image_size,
        aug_config=aug_config,
        image_processor=image_processor,
    )

    logger.info(f"Train transforms: {train_transform}")
    logger.info(f"Eval transforms: {eval_transform}")

    # Create transform functions for online augmentation
    train_transform_fn = make_transform_fn(
        train_transform,
        config.image_column,
        config.label_column,
    )
    eval_transform_fn = make_transform_fn(
        eval_transform,
        config.image_column,
        config.label_column,
    )

    processed_dataset = DatasetDict()

    if config.train_split in dataset:
        train_ds = dataset[config.train_split]
        if config.max_train_samples:
            train_ds = train_ds.select(range(min(config.max_train_samples, len(train_ds))))
        # Use set_transform for online augmentation (applied on-the-fly)
        train_ds.set_transform(train_transform_fn)
        processed_dataset[config.train_split] = train_ds

    if config.validation_split in dataset:
        eval_ds = dataset[config.validation_split]
        if config.max_eval_samples:
            eval_ds = eval_ds.select(range(min(config.max_eval_samples, len(eval_ds))))
        # Use set_transform for online transforms (no augmentation, just resize/normalize)
        eval_ds.set_transform(eval_transform_fn)
        processed_dataset[config.validation_split] = eval_ds

    return processed_dataset


def get_vision_collator() -> Callable:
    """Get data collator for vision tasks."""

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        batch = {"pixel_values": pixel_values}

        if "labels" in examples[0]:
            labels = torch.tensor([example["labels"] for example in examples])
            batch["labels"] = labels

        return batch

    return collate_fn
