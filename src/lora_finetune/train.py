"""Main training script for LoRA finetuning."""

import logging

from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from .cli import build_config, parse_args
from .config import BenchmarkEvalConfig, Config
from .utils import (
    console,
    get_method_display_name,
    get_warning_handler,
    print_model_size,
    set_seed,
    setup_logging,
    suppress_warnings,
)

logger = logging.getLogger(__name__)

hf_set_seed = None
get_text_collator = None
load_text_dataset = None
prepare_text_dataset_for_trl = None
preprocess_text_dataset = None
requires_trl_native_dataset = None
get_vision_collator = None
load_vision_dataset = None
preprocess_vision_dataset = None
LightEvalCallback = None
run_lighteval = None
get_peft_model_with_lora = None
load_model_and_tokenizer = None
get_llm_target_modules = None
get_num_labels_from_dataset = None
get_vision_target_modules = None
compute_metrics_for_classification = None
compute_metrics_for_lm = None
create_trainer = None
prepare_model_for_training = None


def _ensure_runtime_imports():
    global hf_set_seed
    global get_text_collator, load_text_dataset, prepare_text_dataset_for_trl
    global preprocess_text_dataset, requires_trl_native_dataset
    global get_vision_collator, load_vision_dataset, preprocess_vision_dataset
    global LightEvalCallback, run_lighteval
    global get_peft_model_with_lora, load_model_and_tokenizer
    global get_llm_target_modules, get_num_labels_from_dataset, get_vision_target_modules
    global compute_metrics_for_classification, compute_metrics_for_lm
    global create_trainer, prepare_model_for_training

    if hf_set_seed is None:
        from transformers import set_seed as imported_hf_set_seed

        hf_set_seed = imported_hf_set_seed

    if any(
        value is None
        for value in (
            get_text_collator,
            load_text_dataset,
            prepare_text_dataset_for_trl,
            preprocess_text_dataset,
            requires_trl_native_dataset,
        )
    ):
        from .data.text_data import (
            get_text_collator as imported_get_text_collator,
        )
        from .data.text_data import (
            load_text_dataset as imported_load_text_dataset,
        )
        from .data.text_data import (
            prepare_text_dataset_for_trl as imported_prepare_text_dataset_for_trl,
        )
        from .data.text_data import (
            preprocess_text_dataset as imported_preprocess_text_dataset,
        )
        from .data.text_data import (
            requires_trl_native_dataset as imported_requires_trl_native_dataset,
        )

        if get_text_collator is None:
            get_text_collator = imported_get_text_collator
        if load_text_dataset is None:
            load_text_dataset = imported_load_text_dataset
        if prepare_text_dataset_for_trl is None:
            prepare_text_dataset_for_trl = imported_prepare_text_dataset_for_trl
        if preprocess_text_dataset is None:
            preprocess_text_dataset = imported_preprocess_text_dataset
        if requires_trl_native_dataset is None:
            requires_trl_native_dataset = imported_requires_trl_native_dataset

    if any(
        value is None
        for value in (get_vision_collator, load_vision_dataset, preprocess_vision_dataset)
    ):
        from .data.vision_data import (
            get_vision_collator as imported_get_vision_collator,
        )
        from .data.vision_data import (
            load_vision_dataset as imported_load_vision_dataset,
        )
        from .data.vision_data import (
            preprocess_vision_dataset as imported_preprocess_vision_dataset,
        )

        if get_vision_collator is None:
            get_vision_collator = imported_get_vision_collator
        if load_vision_dataset is None:
            load_vision_dataset = imported_load_vision_dataset
        if preprocess_vision_dataset is None:
            preprocess_vision_dataset = imported_preprocess_vision_dataset

    if LightEvalCallback is None or run_lighteval is None:
        from .evaluators import (
            LightEvalCallback as imported_LightEvalCallback,
        )
        from .evaluators import (
            run_lighteval as imported_run_lighteval,
        )

        if LightEvalCallback is None:
            LightEvalCallback = imported_LightEvalCallback
        if run_lighteval is None:
            run_lighteval = imported_run_lighteval

    if get_peft_model_with_lora is None or load_model_and_tokenizer is None:
        from .models.base import (
            get_peft_model_with_lora as imported_get_peft_model_with_lora,
        )
        from .models.base import (
            load_model_and_tokenizer as imported_load_model_and_tokenizer,
        )

        if get_peft_model_with_lora is None:
            get_peft_model_with_lora = imported_get_peft_model_with_lora
        if load_model_and_tokenizer is None:
            load_model_and_tokenizer = imported_load_model_and_tokenizer

    if get_llm_target_modules is None:
        from .models.llm import get_llm_target_modules as imported_get_llm_target_modules

        get_llm_target_modules = imported_get_llm_target_modules

    if get_num_labels_from_dataset is None or get_vision_target_modules is None:
        from .models.vision import (
            get_num_labels_from_dataset as imported_get_num_labels_from_dataset,
        )
        from .models.vision import (
            get_vision_target_modules as imported_get_vision_target_modules,
        )

        if get_num_labels_from_dataset is None:
            get_num_labels_from_dataset = imported_get_num_labels_from_dataset
        if get_vision_target_modules is None:
            get_vision_target_modules = imported_get_vision_target_modules

    if any(
        value is None
        for value in (
            compute_metrics_for_classification,
            compute_metrics_for_lm,
            create_trainer,
            prepare_model_for_training,
        )
    ):
        from .trainer import (
            compute_metrics_for_classification as imported_compute_metrics_for_classification,
        )
        from .trainer import (
            compute_metrics_for_lm as imported_compute_metrics_for_lm,
        )
        from .trainer import (
            create_trainer as imported_create_trainer,
        )
        from .trainer import (
            prepare_model_for_training as imported_prepare_model_for_training,
        )

        if compute_metrics_for_classification is None:
            compute_metrics_for_classification = imported_compute_metrics_for_classification
        if compute_metrics_for_lm is None:
            compute_metrics_for_lm = imported_compute_metrics_for_lm
        if create_trainer is None:
            create_trainer = imported_create_trainer
        if prepare_model_for_training is None:
            prepare_model_for_training = imported_prepare_model_for_training


def run_benchmark_eval(model, model_name: str, eval_config: BenchmarkEvalConfig) -> None:
    """Run benchmark evaluation after training using lighteval."""
    _ensure_runtime_imports()
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
    _ensure_runtime_imports()
    wh = get_warning_handler()
    if wh is not None:
        wh.start_buffering()
    with Status("[bold blue]Loading model...", console=console):
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
    if wh is not None:
        wh.flush_buffered()

    print_model_size(model)

    if wh is not None:
        wh.start_buffering()
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

    if wh is not None:
        wh.flush_buffered()

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
    _ensure_runtime_imports()
    # Vision training uses set_transform which needs the original image column
    if config.training.remove_unused_columns:
        config.training.remove_unused_columns = False

    wh = get_warning_handler()
    if wh is not None:
        wh.start_buffering()
    with Status("[bold blue]Loading dataset...", console=console):
        dataset = load_vision_dataset(config.data)
        num_labels = get_num_labels_from_dataset(dataset[config.data.train_split])
    if wh is not None:
        wh.flush_buffered()

    if wh is not None:
        wh.start_buffering()
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
    if wh is not None:
        wh.flush_buffered()

    print_model_size(model)

    if wh is not None:
        wh.start_buffering()
    with Status("[bold blue]Preprocessing dataset...", console=console):
        processed_dataset = preprocess_vision_dataset(dataset, image_processor, config.data)
    if wh is not None:
        wh.flush_buffered()

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

    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    with Status("[bold blue]Saving model...", console=console):
        trainer.save_model()

    if config.training.push_to_hub:
        with Status("[bold blue]Pushing to Hub...", console=console):
            trainer.push_to_hub()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = build_config(args)

    if config.model.use_unsloth:
        from ._optional_unsloth import ensure_unsloth_imported

        ensure_unsloth_imported()

    suppress_warnings()
    _ensure_runtime_imports()

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
