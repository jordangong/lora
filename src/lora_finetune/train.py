"""Main training script for LoRA finetuning."""

import logging

from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from transformers import set_seed as hf_set_seed

from .cli import build_config, parse_args
from .config import BenchmarkEvalConfig, Config
from .data.text_data import (
    get_text_collator,
    load_text_dataset,
    prepare_text_dataset_for_trl,
    preprocess_text_dataset,
    requires_trl_native_dataset,
)
from .data.vision_data import (
    get_vision_collator,
    load_vision_dataset,
    preprocess_vision_dataset,
)
from .evaluators import LightEvalCallback, run_lighteval
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
    capture_runtime_output,
    console,
    get_method_display_name,
    print_model_size,
    set_seed,
    setup_logging,
    suppress_warnings,
    verbose_logging_enabled,
)

logger = logging.getLogger(__name__)


def run_benchmark_eval(model, model_name: str, eval_config: BenchmarkEvalConfig) -> None:
    """Run benchmark evaluation after training using lighteval."""
    console.print(Panel("[bold blue]Running Benchmark Evaluation[/bold blue]"))

    metrics = run_lighteval(
        model=model,
        model_name=model_name,
        tasks=eval_config.tasks,
        max_samples=eval_config.num_samples,
        batch_size=eval_config.batch_size,
        max_new_tokens=eval_config.max_new_tokens,
    )

    # Display results
    table = Table(title="Benchmark Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for metric_name, value in metrics.items():
        table.add_row(metric_name, f"{value:.4f}")
    console.print(table)


def train_llm(config: Config) -> None:
    """Train a language model with LoRA."""
    with Status("[bold blue]Loading model...", console=console):
        with capture_runtime_output(
            enabled=not verbose_logging_enabled(),
            rich_console=console,
        ):
            model, tokenizer = load_model_and_tokenizer(
                config.model,
                max_seq_length=config.data.max_seq_length,
            )

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
                use_unsloth=config.model.use_unsloth,
                use_gradient_checkpointing=config.training.gradient_checkpointing,
                random_state=config.training.seed,
                max_seq_length=config.data.max_seq_length,
            )

            model = prepare_model_for_training(model, config.training, tokenizer)

    print_model_size(model)

    with Status("[bold blue]Loading dataset...", console=console):
        dataset = load_text_dataset(config.data)
        use_trl_native_dataset = config.training.llm_trainer == "trl" and (
            config.data.append_eos_token or requires_trl_native_dataset(dataset)
        )
        if use_trl_native_dataset:
            prepared_dataset = prepare_text_dataset_for_trl(
                dataset,
                config.data,
                shuffle_seed=config.training.data_seed,
            )
        else:
            prepared_dataset = preprocess_text_dataset(
                dataset,
                tokenizer,
                config.data,
                shuffle_seed=config.training.data_seed,
            )

    train_dataset = prepared_dataset[config.data.train_split]
    eval_dataset = None
    if config.data.validation_split in prepared_dataset:
        eval_dataset = prepared_dataset[config.data.validation_split]

    data_collator = None if use_trl_native_dataset else get_text_collator(tokenizer)

    # Use perplexity metric for LLM evaluation if eval dataset exists
    compute_metrics = compute_metrics_for_lm if eval_dataset is not None else None

    # Create benchmark evaluation callback if enabled
    callbacks = []
    if config.benchmark_eval.enabled:
        eval_callback = LightEvalCallback(
            model_name=config.model.model_name_or_path,
            tasks=config.benchmark_eval.tasks,
            eval_steps=config.benchmark_eval.eval_steps,
            max_samples=config.benchmark_eval.num_samples,
            max_new_tokens=config.benchmark_eval.max_new_tokens,
            batch_size=config.benchmark_eval.batch_size,
        )
        callbacks.append(eval_callback)
        logger.info(
            f"Benchmark eval ({config.benchmark_eval.tasks}) enabled every {config.benchmark_eval.eval_steps} steps"
        )

    trainer = create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        data_config=config.data,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        lora_config=config.lora,
        callbacks=callbacks if callbacks else None,
    )

    with capture_runtime_output(
        enabled=not verbose_logging_enabled(),
        rich_console=console,
    ):
        trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    with Status("[bold blue]Saving model...", console=console):
        trainer.save_model()

    if config.training.push_to_hub:
        with Status("[bold blue]Pushing to Hub...", console=console):
            trainer.push_to_hub()

    # Run benchmark evaluation if enabled
    if config.benchmark_eval.enabled:
        run_benchmark_eval(model, config.model.model_name_or_path, config.benchmark_eval)


def train_vision(config: Config) -> None:
    """Train a vision model with LoRA."""
    # Vision training uses set_transform which needs the original image column
    if config.training.remove_unused_columns:
        config.training.remove_unused_columns = False

    with Status("[bold blue]Loading dataset...", console=console):
        dataset = load_vision_dataset(config.data)
        num_labels = get_num_labels_from_dataset(dataset[config.data.train_split])

    with Status("[bold blue]Loading model...", console=console):
        with capture_runtime_output(
            enabled=not verbose_logging_enabled(),
            rich_console=console,
        ):
            model, image_processor = load_model_and_tokenizer(config.model, num_labels=num_labels)

            if config.lora.target_modules is None or config.lora.target_modules == [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ]:
                config.lora.target_modules = get_vision_target_modules(
                    config.model.model_name_or_path
                )

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
        lora_config=config.lora,
    )

    with capture_runtime_output(
        enabled=not verbose_logging_enabled(),
        rich_console=console,
    ):
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
    if config.model.model_type == "causal_lm":
        table.add_row("Unsloth", "✓" if config.model.use_unsloth else "✗")
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

    # Finetuning method settings
    method_display = get_method_display_name(config.lora.method)
    table.add_row("Method", method_display)
    if config.lora.method != "full":
        if config.lora.method in ("lora", "dora", "adalora", "loraplus"):
            table.add_row("Rank (r)", str(config.lora.r))
            table.add_row("Alpha", str(config.lora.alpha))
            table.add_row("Dropout", str(config.lora.dropout))
        if config.lora.method == "adalora":
            table.add_row("Target rank", str(config.lora.target_r))
        if config.lora.method == "loraplus":
            table.add_row("LR ratio (B/A)", str(config.lora.loraplus_lr_ratio))
        if config.lora.method == "prefix_tuning":
            table.add_row("Virtual tokens", str(config.lora.num_virtual_tokens))
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
    if config.model.model_type == "causal_lm":
        table.add_row("LLM trainer", config.training.llm_trainer)
    table.add_row("FSDP", config.training.fsdp or "disabled")
    table.add_row("Output dir", config.training.output_dir)

    # Benchmark evaluation settings
    if config.benchmark_eval.enabled:
        table.add_row("", "")  # Separator
        table.add_row("Benchmark eval", "✓")
        table.add_row("Benchmark tasks", config.benchmark_eval.tasks)
        table.add_row("Bench eval steps", str(config.benchmark_eval.eval_steps))
        if config.benchmark_eval.num_samples:
            table.add_row("Bench samples", str(config.benchmark_eval.num_samples))

    console.print(Panel(table, title="[bold blue]Configuration[/bold blue]", border_style="blue"))

    if config.model.model_type == "vision":
        train_vision(config)
    else:
        train_llm(config)


if __name__ == "__main__":
    main()
