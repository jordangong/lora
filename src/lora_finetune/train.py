"""Main training script for LoRA finetuning."""

import logging

import torch
from transformers import set_seed as hf_set_seed

from .cli import build_config, parse_args
from .config import Config
from .data.text_data import get_text_collator, load_text_dataset, preprocess_text_dataset
from .data.vision_data import get_vision_collator, load_vision_dataset, preprocess_vision_dataset
from .models.base import get_peft_model_with_lora, load_model_and_tokenizer
from .models.llm import get_llm_target_modules
from .models.vision import get_num_labels_from_dataset, get_vision_target_modules
from .trainer import (
    compute_metrics_for_classification,
    compute_metrics_for_lm,
    create_trainer,
    prepare_model_for_training,
)
from .utils import print_gpu_memory_usage, print_model_size, set_seed, setup_logging

logger = logging.getLogger(__name__)


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

    # Use perplexity metric for LLM evaluation if eval dataset exists
    compute_metrics = compute_metrics_for_lm if eval_dataset is not None else None

    trainer = create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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

    # Use accuracy metric for vision classification
    compute_metrics = compute_metrics_for_classification if eval_dataset is not None else None

    trainer = create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
