"""Main training script for LoRA finetuning."""

import logging

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
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
from .utils import (
    print_model_size,
    set_seed,
    setup_logging,
    suppress_warnings,
)

console = Console()
logger = logging.getLogger(__name__)


def train_llm(config: Config) -> None:
    """Train a language model with LoRA."""
    with Status("[bold blue]Loading model...", console=console):
        model, tokenizer = load_model_and_tokenizer(config.model)

        if config.lora.target_modules is None or config.lora.target_modules == [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ]:
            config.lora.target_modules = get_llm_target_modules(config.model.model_name_or_path)

        is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
        model = get_peft_model_with_lora(
            model,
            config.lora,
            model_type=config.model.model_type,
            is_quantized=is_quantized,
        )

        model = prepare_model_for_training(model, config.training)

    print_model_size(model)

    with Status("[bold blue]Loading dataset...", console=console):
        dataset = load_text_dataset(config.data)
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

    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    with Status("[bold blue]Saving model...", console=console):
        trainer.save_model()

    if config.training.push_to_hub:
        with Status("[bold blue]Pushing to Hub...", console=console):
            trainer.push_to_hub()


def train_vision(config: Config) -> None:
    """Train a vision model with LoRA."""
    # Vision training uses set_transform which needs the original image column
    if config.training.remove_unused_columns:
        config.training.remove_unused_columns = False

    with Status("[bold blue]Loading dataset...", console=console):
        dataset = load_vision_dataset(config.data)
        num_labels = get_num_labels_from_dataset(dataset[config.data.train_split])

    with Status("[bold blue]Loading model...", console=console):
        model, image_processor = load_model_and_tokenizer(config.model, num_labels=num_labels)

        if config.lora.target_modules is None or config.lora.target_modules == [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ]:
            config.lora.target_modules = get_vision_target_modules(config.model.model_name_or_path)

        is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
        model = get_peft_model_with_lora(
            model,
            config.lora,
            model_type="vision",
            is_quantized=is_quantized,
        )

        model = prepare_model_for_training(model, config.training)

    print_model_size(model)

    with Status("[bold blue]Preprocessing dataset...", console=console):
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

    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    with Status("[bold blue]Saving model...", console=console):
        trainer.save_model()

    if config.training.push_to_hub:
        with Status("[bold blue]Pushing to Hub...", console=console):
            trainer.push_to_hub()


def main() -> None:
    """Main entry point."""
    suppress_warnings()

    args = parse_args()
    config = build_config(args)

    setup_logging(level="INFO" if args.verbose else "WARNING")
    set_seed(config.training.seed)
    hf_set_seed(config.training.seed)

    # Display configuration in a nice table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="cyan")

    # Model settings
    table.add_row("Model", config.model.model_name_or_path)
    table.add_row("Model type", config.model.model_type)
    table.add_row("Flash Attention 2", "✓" if config.model.use_flash_attention_2 else "✗")
    if config.model.load_in_4bit or config.model.load_in_8bit:
        table.add_row("Quantization", "4-bit" if config.model.load_in_4bit else "8-bit")
    table.add_row("", "")  # Separator

    # Data settings
    table.add_row("Dataset", config.data.dataset_name or config.data.train_file or "N/A")
    if config.data.dataset_config_name:
        table.add_row("Dataset config", config.data.dataset_config_name)
    if config.model.model_type != "vision":
        table.add_row("Max seq length", str(config.data.max_seq_length))
    if config.data.max_train_samples:
        table.add_row("Max train samples", str(config.data.max_train_samples))
    if config.data.max_eval_samples:
        table.add_row("Max eval samples", str(config.data.max_eval_samples))
    table.add_row("", "")  # Separator

    # LoRA settings
    table.add_row("LoRA rank (r)", str(config.lora.r))
    table.add_row("LoRA alpha", str(config.lora.alpha))
    table.add_row("LoRA dropout", str(config.lora.dropout))
    if config.lora.target_modules:
        modules = config.lora.target_modules
        if isinstance(modules, list):
            modules = ", ".join(modules[:4]) + ("..." if len(modules) > 4 else "")
        table.add_row("Target modules", str(modules))
    table.add_row("", "")  # Separator

    # Training settings
    table.add_row("Epochs", str(config.training.num_train_epochs))
    table.add_row("Batch size", str(config.training.per_device_train_batch_size))
    table.add_row("Learning rate", str(config.training.learning_rate))
    table.add_row("Gradient checkpointing", "✓" if config.training.gradient_checkpointing else "✗")
    table.add_row("FSDP", config.training.fsdp or "disabled")
    table.add_row("Output dir", config.training.output_dir)

    console.print(Panel(table, title="[bold blue]Configuration[/bold blue]", border_style="blue"))

    if config.model.model_type == "vision":
        train_vision(config)
    else:
        train_llm(config)


if __name__ == "__main__":
    main()
