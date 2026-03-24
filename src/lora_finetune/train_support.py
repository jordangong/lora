import copy
import gc
import os
from typing import Any

import torch
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from transformers.trainer_callback import TrainerCallback

from .config import BenchmarkEvalConfig, Config
from .trainer_hpo import _iter_hpo_parameter_names, build_hpo_compute_objective


def _resolve_best_run_hpo_parameters(
    config: Config, hyperparameters: dict[str, Any]
) -> dict[str, Any]:
    resolved_parameters: dict[str, Any] = {}
    for parameter_name in _iter_hpo_parameter_names(config.hpo):
        if parameter_name in hyperparameters:
            resolved_parameters[parameter_name] = hyperparameters[parameter_name]
            continue

        if "." in parameter_name:
            section_name, field_name = parameter_name.split(".", 1)
        else:
            section_name, field_name = "training", parameter_name

        section_values = hyperparameters.get(section_name)
        if isinstance(section_values, dict) and field_name in section_values:
            resolved_parameters[parameter_name] = section_values[field_name]

    return resolved_parameters


def _log_metrics_to_wandb(
    metrics: dict[str, Any], namespace: str, global_step: int | None = None
) -> None:
    if not metrics:
        return

    prefixed_metrics = {f"{namespace}/{key}": value for key, value in metrics.items()}
    if global_step is not None:
        prefixed_metrics["train/global_step"] = global_step
    try:
        import wandb

        if wandb.run is None:
            return
        if global_step is None:
            wandb.log(prefixed_metrics)
        else:
            wandb.log(prefixed_metrics, step=global_step)
    except ImportError:
        pass


def _extract_prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    prefix_with_separator = f"{prefix}_"
    extracted_metrics = {}
    for key, value in metrics.items():
        if key.startswith(prefix_with_separator):
            extracted_metrics[key[len(prefix_with_separator) :]] = value
    return extracted_metrics


def _replace_metric_prefix(
    metrics: dict[str, Any], source_prefix: str, target_prefix: str
) -> dict[str, Any]:
    remapped_metrics = {}
    source_prefix_with_separator = f"{source_prefix}_"
    target_prefix_with_separator = f"{target_prefix}_"
    for key, value in metrics.items():
        if key.startswith(source_prefix_with_separator):
            remapped_metrics[
                target_prefix_with_separator + key[len(source_prefix_with_separator) :]
            ] = value
        else:
            remapped_metrics[key] = value
    return remapped_metrics


def _pop_wandb_callback(trainer):
    pop_callback = getattr(trainer, "pop_callback", None)
    if not callable(pop_callback):
        return None

    try:
        from transformers.integrations import WandbCallback
    except ImportError:
        return None

    return pop_callback(WandbCallback)


def run_benchmark_eval(
    module,
    model,
    model_name: str,
    eval_config: BenchmarkEvalConfig,
    trainer=None,
) -> None:
    module._ensure_runtime_imports()
    module.console.print(Panel("[bold blue]Running Benchmark Evaluation[/bold blue]"))

    metrics = module.run_lighteval(
        model=model,
        model_name=model_name,
        tasks=eval_config.tasks,
        max_samples=eval_config.num_samples,
        batch_size=eval_config.batch_size,
        max_new_tokens=eval_config.max_new_tokens,
    )
    global_step = getattr(getattr(trainer, "state", None), "global_step", None)
    _log_metrics_to_wandb(metrics, namespace="final/benchmark", global_step=global_step)

    # Display results
    table = Table(title="Benchmark Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for metric_name, value in metrics.items():
        table.add_row(metric_name, f"{value:.4f}")
    module.console.print(table)


def cleanup_trainer_callbacks(trainer) -> None:
    callback_handler = getattr(trainer, "callback_handler", None)
    if callback_handler is None:
        return

    for callback in getattr(callback_handler, "callbacks", []):
        cleanup = getattr(callback, "cleanup", None)
        if callable(cleanup):
            cleanup()


def run_final_trainer_evaluation(console, trainer):
    eval_dataset = getattr(trainer, "eval_dataset", None)
    evaluate = getattr(trainer, "evaluate", None)
    if eval_dataset is None or not callable(evaluate):
        return None

    global_step = getattr(getattr(trainer, "state", None), "global_step", None)
    wandb_callback = _pop_wandb_callback(trainer)
    try:
        with Status("[bold blue]Running final evaluation...", console=console):
            metrics = evaluate(metric_key_prefix="final_eval")
    finally:
        if wandb_callback is not None:
            add_callback = getattr(trainer, "add_callback", None)
            if callable(add_callback):
                add_callback(wandb_callback)

    _log_metrics_to_wandb(
        _extract_prefixed_metrics(metrics, prefix="final_eval"),
        namespace="final/eval",
        global_step=global_step,
    )
    return metrics


class _FinalEvaluationCallback(TrainerCallback):
    def __init__(self, trainer, final_evaluation_fn):
        self.trainer = trainer
        self.final_evaluation_fn = final_evaluation_fn
        self.ran = False

    def on_train_end(self, args, state, control, **kwargs):
        self.final_evaluation_fn(self.trainer)
        self.ran = True
        return control


def _attach_final_evaluation_callback(trainer, final_evaluation_fn):
    add_callback = getattr(trainer, "add_callback", None)
    if not callable(add_callback):
        return None

    callback = _FinalEvaluationCallback(trainer, final_evaluation_fn)
    add_callback(callback)
    return callback


def should_run_final_benchmark_eval(trainer, eval_config: BenchmarkEvalConfig) -> bool:
    global_step = getattr(getattr(trainer, "state", None), "global_step", None)
    if global_step is None or global_step <= 0:
        return True

    eval_steps = getattr(eval_config, "eval_steps", None)
    if eval_steps is None or eval_steps <= 0:
        return True

    return global_step % eval_steps != 0


def run_trainer_training(
    trainer,
    cleanup_trainer_callbacks_fn,
    resume_from_checkpoint=None,
    final_evaluation_fn=None,
    final_evaluation_enabled=True,
) -> None:
    final_evaluation_callback = None
    try:
        if final_evaluation_enabled and callable(final_evaluation_fn):
            final_evaluation_callback = _attach_final_evaluation_callback(
                trainer, final_evaluation_fn
            )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if (
            final_evaluation_enabled
            and callable(final_evaluation_fn)
            and (final_evaluation_callback is None or not final_evaluation_callback.ran)
        ):
            final_evaluation_fn(trainer)
    finally:
        cleanup_trainer_callbacks_fn(trainer)


def run_trainer_hpo(
    trainer,
    config: Config,
    run_hyperparameter_search_fn,
    cleanup_trainer_callbacks_fn,
    final_evaluation_fn=None,
) -> None:
    try:
        if getattr(trainer, "model_init", None) is not None:
            try:
                from accelerate.utils.memory import release_memory

                trainer.model_wrapped, trainer.model = release_memory(
                    getattr(trainer, "model_wrapped", None),
                    getattr(trainer, "model", None),
                )
            except ImportError:
                trainer.model_wrapped = None
                trainer.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if hasattr(trainer, "optimizer"):
                trainer.optimizer = None
            if hasattr(trainer, "lr_scheduler"):
                trainer.lr_scheduler = None
            if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
                trainer.accelerator.free_memory()

        if callable(final_evaluation_fn):
            compute_objective = build_hpo_compute_objective(config.training, config.hpo)

            def _run_hpo_trial_final_evaluation(current_trainer):
                metrics = final_evaluation_fn(current_trainer)
                if metrics:
                    current_trainer.objective = compute_objective(
                        _replace_metric_prefix(
                            metrics, source_prefix="final_eval", target_prefix="eval"
                        )
                    )
                return metrics

            _attach_final_evaluation_callback(trainer, _run_hpo_trial_final_evaluation)

        return run_hyperparameter_search_fn(trainer, config.training, config.hpo)
    finally:
        cleanup_trainer_callbacks_fn(trainer)


def display_hpo_best_run(console, best_run: Any) -> None:
    console.print(Panel("[bold blue]Hyperparameter Search Complete[/bold blue]"))
    table = Table(title="Best Hyperparameter Run")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run ID", str(getattr(best_run, "run_id", "N/A")))
    table.add_row("Objective", str(getattr(best_run, "objective", "N/A")))
    hyperparameters = getattr(best_run, "hyperparameters", {}) or {}
    for key, value in sorted(hyperparameters.items()):
        if key in {"assignments", "metric"}:
            continue
        table.add_row(key, str(value))
    sweep_id = getattr(best_run, "run_summary", None)
    if sweep_id is not None:
        table.add_row("Sweep ID", str(sweep_id))
    console.print(table)


def save_hpo_best_config(module, config: Config, best_run: Any) -> None:
    module._ensure_runtime_imports()
    hyperparameters = getattr(best_run, "hyperparameters", {}) or {}
    if not hyperparameters:
        return

    resolved_hyperparameters = _resolve_best_run_hpo_parameters(config, hyperparameters)
    if not resolved_hyperparameters:
        return

    best_config = copy.deepcopy(config)
    module.apply_hpo_parameters_to_config_sections(
        {
            "training": best_config.training,
            "lora": best_config.lora,
            "dpo": best_config.dpo,
            "grpo": best_config.grpo,
        },
        resolved_hyperparameters,
    )
    os.makedirs(best_config.training.output_dir, exist_ok=True)
    best_config_path = os.path.join(best_config.training.output_dir, "best_hpo_config.yaml")
    best_config.to_yaml(best_config_path)
    module.console.print(Panel(f"[bold blue]Saved best HPO config:[/bold blue] {best_config_path}"))


def run_with_status(get_warning_handler, console, status_message: str, fn):
    wh = get_warning_handler()
    if wh is not None:
        wh.start_buffering()
    with Status(status_message, console=console):
        result = fn()
    if wh is not None:
        wh.flush_buffered()
    return result


def resolve_default_target_modules(config: Config, resolver) -> None:
    if config.lora.target_modules is None or config.lora.target_modules == [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]:
        config.lora.target_modules = resolver(config.model.model_name_or_path)


def get_train_and_eval_datasets(dataset, data_config):
    train_dataset = dataset[data_config.train_split]
    eval_dataset = None
    if data_config.validation_split in dataset:
        eval_dataset = dataset[data_config.validation_split]
    return train_dataset, eval_dataset


def get_hpo_config_sections(config: Config) -> dict[str, Any]:
    return {
        "training": config.training,
        "lora": config.lora,
        "dpo": config.dpo,
        "grpo": config.grpo,
    }


def get_hpo_setup(config: Config, build_model, get_hpo_config_sections_fn):
    if not config.hpo.enabled:
        return None, None

    def model_init(_trial=None):
        rebuilt_model, _ = build_model()
        return rebuilt_model

    return get_hpo_config_sections_fn(config), model_init


def run_hpo_if_enabled(
    trainer,
    config: Config,
    run_trainer_hpo_fn,
    display_hpo_best_run_fn,
    save_hpo_best_config_fn,
) -> bool:
    if not config.hpo.enabled:
        return False

    best_run = run_trainer_hpo_fn(trainer, config)
    display_hpo_best_run_fn(best_run)
    save_hpo_best_config_fn(config, best_run)
    return True


def save_and_maybe_push_model(console, trainer, config: Config) -> None:
    with Status("[bold blue]Saving model...", console=console):
        trainer.save_model()

    if config.training.push_to_hub:
        with Status("[bold blue]Pushing to Hub...", console=console):
            trainer.push_to_hub()


def build_configuration_panel(config: Config, get_method_display_name) -> Panel:
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
    table.add_row(
        "Gradient checkpointing",
        "✓ (unsloth)"
        if config.training.gradient_checkpointing == "unsloth"
        else ("✓" if config.training.gradient_checkpointing else "✗"),
    )
    if config.model.model_type == "causal_lm":
        table.add_row("Trainer type", config.training.trainer_type)
        if config.training.trainer_type == "sft":
            table.add_row("LLM trainer", config.training.llm_trainer)
    table.add_row("FSDP", config.training.fsdp or "disabled")
    table.add_row("Output dir", config.training.output_dir)
    if config.hpo.enabled:
        table.add_row("HPO", "wandb")
        table.add_row("HPO trials", str(config.hpo.n_trials))
        if config.hpo.metric_name:
            table.add_row("HPO metric", config.hpo.metric_name)

    # Benchmark evaluation settings
    if config.benchmark_eval.enabled:
        table.add_row("", "")  # Separator
        table.add_row("Benchmark eval", "✓")
        table.add_row("Benchmark tasks", config.benchmark_eval.tasks)
        table.add_row("Bench eval steps", str(config.benchmark_eval.eval_steps))
        if config.benchmark_eval.num_samples:
            table.add_row("Bench samples", str(config.benchmark_eval.num_samples))

    return Panel(table, title="[bold blue]Configuration[/bold blue]", border_style="blue")
