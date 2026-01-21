"""Main training script for LoRA finetuning."""

import argparse
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

import torch
from transformers import set_seed as hf_set_seed

from .config import (
    AugmentationConfig,
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from .data.text_data import get_text_collator, load_text_dataset, preprocess_text_dataset
from .data.vision_data import get_vision_collator, load_vision_dataset, preprocess_vision_dataset
from .models.base import get_peft_model_with_lora, load_model_and_tokenizer
from .models.llm import get_llm_target_modules
from .models.vision import get_num_labels_from_dataset, get_vision_target_modules
from .trainer import create_trainer, prepare_model_for_training
from .utils import print_gpu_memory_usage, print_model_size, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _is_list_or_str_union(type_hint: Type) -> bool:
    """Check if type is Union[List[str], str] (possibly Optional)."""
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        non_none_args = [a for a in args if a is not type(None)]
        # Check if we have both list and str types
        has_list = any(get_origin(a) is list or a is list for a in non_none_args)
        has_str = any(a is str for a in non_none_args)
        return has_list and has_str
    return False


def _get_base_type(type_hint: Type) -> Type:
    """Extract the base type from Optional or other generic types."""
    origin = get_origin(type_hint)
    if origin is type(None):
        return str
    # Handle Optional[X] which is Union[X, None]
    if origin is not None:
        args = get_args(type_hint)
        # Filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return non_none_args[0]
    return type_hint


def _add_dataclass_args(
    parser: argparse.ArgumentParser,
    dataclass_type: Type,
    prefix: str = "",
    skip_fields: Optional[List[str]] = None,
) -> None:
    """Add arguments from a dataclass to the parser."""
    skip_fields = skip_fields or []

    for field_info in dataclasses.fields(dataclass_type):
        field_name = field_info.name
        if field_name in skip_fields:
            continue

        # Skip nested dataclass fields (handled separately)
        if dataclasses.is_dataclass(field_info.type):
            continue

        arg_name = f"--{prefix}{field_name}" if prefix else f"--{field_name}"
        field_type = _get_base_type(field_info.type)

        # Extract help text from field metadata
        help_text = field_info.metadata.get("help") if field_info.metadata else None

        # Get default value
        default_val = field_info.default
        if default_val is dataclasses.MISSING:
            default_val = None

        # Build help text with default value
        def _build_help(base_help: Optional[str], default: Any) -> str:
            if base_help and default is not None:
                return f"{base_help} (default: {default})"
            elif base_help:
                return base_help
            elif default is not None:
                return f"(default: {default})"
            return ""

        # Handle boolean fields - add both --flag and --no_flag (no_ hidden from help)
        if field_type is bool:
            if default_val is None:
                bool_help = help_text if help_text else ""
            else:
                default_state = "enabled" if default_val else "disabled"
                bool_help = (
                    f"{help_text} (default: {default_state})"
                    if help_text
                    else f"(default: {default_state})"
                )
            parser.add_argument(arg_name, action="store_true", default=None, help=bool_help)
            no_arg_name = f"--no_{prefix}{field_name}" if prefix else f"--no_{field_name}"
            parser.add_argument(
                no_arg_name, action="store_true", default=None, help=argparse.SUPPRESS
            )
        # Handle Union[List[str], str] fields (e.g., target_modules supporting regex)
        elif _is_list_or_str_union(field_info.type):
            parser.add_argument(
                arg_name,
                type=str,
                nargs="*",
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle list fields
        elif get_origin(field_type) is list or field_type is list:
            inner_type = get_args(field_type)
            item_type = inner_type[0] if inner_type else str
            parser.add_argument(
                arg_name,
                type=item_type,
                nargs="+",
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle tuple fields (as space-separated values)
        elif get_origin(field_type) is tuple or field_type is tuple:
            parser.add_argument(
                arg_name,
                type=float,
                nargs="+",
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle Literal types by extracting choices
        elif get_origin(field_type) is type(None):
            parser.add_argument(
                arg_name, type=str, default=None, help=_build_help(help_text, default_val)
            )
        elif hasattr(field_type, "__origin__") and str(field_type.__origin__) == "typing.Literal":
            choices = get_args(field_type)
            parser.add_argument(
                arg_name,
                type=str,
                choices=choices,
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle dict fields (skip - too complex for CLI)
        elif field_type is dict or get_origin(field_type) is dict:
            continue
        # Handle basic types
        elif field_type in (int, float, str):
            parser.add_argument(
                arg_name, type=field_type, default=None, help=_build_help(help_text, default_val)
            )
        else:
            # Try to use the type directly, fallback to str
            try:
                parser.add_argument(
                    arg_name, type=str, default=None, help=_build_help(help_text, default_val)
                )
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA Finetuning")

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Add arguments from all config dataclasses
    # LoRA config with "lora_" prefix for r, alpha, dropout to avoid conflicts
    _add_dataclass_args(parser, LoraConfig, prefix="lora_")

    # Model config
    _add_dataclass_args(parser, ModelConfig)

    # Data config (skip augmentation - handled separately)
    _add_dataclass_args(parser, DataConfig, skip_fields=["augmentation"])

    # Augmentation config with "aug_" prefix
    _add_dataclass_args(parser, AugmentationConfig, prefix="aug_")

    # Training config
    _add_dataclass_args(parser, TrainingConfig)

    return parser.parse_args()


def _apply_args_to_config(
    config_obj: Any,
    args_dict: Dict[str, Any],
    prefix: str = "",
) -> None:
    """Apply CLI arguments to a config object."""
    for field_info in dataclasses.fields(config_obj):
        field_name = field_info.name

        # Skip nested dataclass fields
        if dataclasses.is_dataclass(field_info.type):
            continue

        arg_name = f"{prefix}{field_name}" if prefix else field_name
        field_type = _get_base_type(field_info.type)

        # Handle --no_* flags for boolean fields
        if field_type is bool:
            no_arg_name = f"no_{arg_name}"
            if args_dict.get(no_arg_name):
                setattr(config_obj, field_name, False)
                continue

        if arg_name in args_dict and args_dict[arg_name] is not None:
            value = args_dict[arg_name]
            # Convert list to tuple if needed
            if (get_origin(field_type) is tuple or field_type is tuple) and isinstance(value, list):
                value = tuple(value)
            # For Union[List[str], str] fields, keep single value as string (for regex)
            if (
                _is_list_or_str_union(field_info.type)
                and isinstance(value, list)
                and len(value) == 1
            ):
                value = value[0]
            setattr(config_obj, field_name, value)


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from args and config file."""
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    args_dict = vars(args)

    # Apply args to each config section
    _apply_args_to_config(config.lora, args_dict, prefix="lora_")
    _apply_args_to_config(config.model, args_dict)
    _apply_args_to_config(config.data, args_dict)
    _apply_args_to_config(config.data.augmentation, args_dict, prefix="aug_")
    _apply_args_to_config(config.training, args_dict)

    return config


def train_llm(config: Config) -> None:
    """Train a language model with LoRA."""
    logger.info("Starting LLM training")
    logger.info(f"Model: {config.model.model_name_or_path}")

    model, tokenizer = load_model_and_tokenizer(config.model)

    if config.lora.target_modules is None or config.lora.target_modules == [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]:
        config.lora.target_modules = get_llm_target_modules(config.model.model_name_or_path)
        logger.info(f"Auto-detected target modules: {config.lora.target_modules}")

    is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
    model = get_peft_model_with_lora(
        model,
        config.lora,
        model_type=config.model.model_type,
        is_quantized=is_quantized,
    )

    model = prepare_model_for_training(model, config.training)

    print_model_size(model)
    if torch.cuda.is_available():
        print_gpu_memory_usage()

    logger.info("Loading dataset")
    dataset = load_text_dataset(config.data)

    logger.info("Preprocessing dataset")
    tokenized_dataset = preprocess_text_dataset(dataset, tokenizer, config.data)

    train_dataset = tokenized_dataset[config.data.train_split]
    eval_dataset = None
    if config.data.validation_split in tokenized_dataset:
        eval_dataset = tokenized_dataset[config.data.validation_split]

    data_collator = get_text_collator(tokenizer)

    trainer = create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    logger.info(f"Saving model to {config.training.output_dir}")
    trainer.save_model()

    if config.training.push_to_hub:
        logger.info("Pushing to Hub")
        trainer.push_to_hub()


def train_vision(config: Config) -> None:
    """Train a vision model with LoRA."""
    logger.info("Starting vision model training")
    logger.info(f"Model: {config.model.model_name_or_path}")

    # Vision training uses set_transform which needs the original image column
    # before it's transformed to pixel_values, so we must keep unused columns
    if config.training.remove_unused_columns:
        logger.info(
            "Disabling remove_unused_columns for vision training (required for set_transform)"
        )
        config.training.remove_unused_columns = False

    logger.info("Loading dataset")
    dataset = load_vision_dataset(config.data)

    num_labels = get_num_labels_from_dataset(dataset[config.data.train_split])
    logger.info(f"Number of labels: {num_labels}")

    model, image_processor = load_model_and_tokenizer(config.model, num_labels=num_labels)

    if config.lora.target_modules is None or config.lora.target_modules == [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]:
        config.lora.target_modules = get_vision_target_modules(config.model.model_name_or_path)
        logger.info(f"Auto-detected target modules: {config.lora.target_modules}")

    is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
    model = get_peft_model_with_lora(
        model,
        config.lora,
        model_type="vision",
        is_quantized=is_quantized,
    )

    model = prepare_model_for_training(model, config.training)

    print_model_size(model)
    if torch.cuda.is_available():
        print_gpu_memory_usage()

    logger.info("Preprocessing dataset")
    processed_dataset = preprocess_vision_dataset(dataset, image_processor, config.data)

    train_dataset = processed_dataset[config.data.train_split]
    eval_dataset = None
    if config.data.validation_split in processed_dataset:
        eval_dataset = processed_dataset[config.data.validation_split]

    data_collator = get_vision_collator()

    trainer = create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=None,
        data_collator=data_collator,
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    logger.info(f"Saving model to {config.training.output_dir}")
    trainer.save_model()

    if config.training.push_to_hub:
        logger.info("Pushing to Hub")
        trainer.push_to_hub()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = build_config(args)

    setup_logging()
    set_seed(config.training.seed)
    hf_set_seed(config.training.seed)

    logger.info("Configuration:")
    logger.info(f"  Model: {config.model.model_name_or_path}")
    logger.info(f"  Model type: {config.model.model_type}")
    logger.info(f"  LoRA r: {config.lora.r}, alpha: {config.lora.alpha}")
    logger.info(f"  Flash Attention 2: {config.model.use_flash_attention_2}")
    logger.info(f"  Gradient checkpointing: {config.training.gradient_checkpointing}")
    logger.info(f"  FSDP: {config.training.fsdp or 'disabled'}")
    logger.info(f"  Output dir: {config.training.output_dir}")

    if config.model.model_type == "vision":
        train_vision(config)
    else:
        train_llm(config)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
