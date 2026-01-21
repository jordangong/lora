"""Main training script for LoRA finetuning."""

import argparse
import logging

import torch
from transformers import set_seed as hf_set_seed

from .config import Config
from .data.text_data import get_text_collator, load_text_dataset, preprocess_text_dataset
from .data.vision_data import get_vision_collator, load_vision_dataset, preprocess_vision_dataset
from .models.base import get_peft_model_with_lora, load_model_and_tokenizer
from .models.llm import get_llm_target_modules
from .models.vision import get_num_labels_from_dataset, get_vision_target_modules
from .trainer import create_trainer, prepare_model_for_training
from .utils import print_gpu_memory_usage, print_model_size, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA Finetuning")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--model_type", type=str, choices=["causal_lm", "seq2seq", "vision"])
    parser.add_argument("--dataset_name", type=str, help="Dataset name from HuggingFace Hub")
    parser.add_argument("--train_file", type=str, help="Path to training data file")
    parser.add_argument("--validation_file", type=str, help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", help="LoRA target modules")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per device")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation steps"
    )
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    parser.add_argument(
        "--use_flash_attention_2", action="store_true", help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--no_flash_attention_2", action="store_true", help="Disable Flash Attention 2"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing"
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument("--fsdp", type=str, help="FSDP configuration")
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed config file")
    parser.add_argument("--report_to", type=str, help="Report to (wandb, tensorboard, none)")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from args and config file."""
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    args_dict = vars(args)

    if args_dict.get("lora_r"):
        config.lora.r = args_dict["lora_r"]
    if args_dict.get("lora_alpha"):
        config.lora.alpha = args_dict["lora_alpha"]
    if args_dict.get("lora_dropout"):
        config.lora.dropout = args_dict["lora_dropout"]
    if args_dict.get("lora_target_modules"):
        config.lora.target_modules = args_dict["lora_target_modules"]

    if args_dict.get("no_flash_attention_2"):
        config.model.use_flash_attention_2 = False

    config.update_from_args(args_dict)

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
        tokenizer=tokenizer,
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
        tokenizer=None,
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
    logger.info(f"  FSDP: {config.training.fsdp}")
    logger.info(f"  Output dir: {config.training.output_dir}")

    if config.model.model_type == "vision":
        train_vision(config)
    else:
        train_llm(config)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
