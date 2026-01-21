"""Custom trainer with performance optimizations."""

import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, TrainerCallback

from .config import ModelConfig, TrainingConfig

console = Console()

logger = logging.getLogger(__name__)


class RichProgressCallback(TrainerCallback):
    """Rich-based progress display for nicer training output."""

    def __init__(self):
        self.progress = None
        self.train_task = None
        self.eval_task = None
        self.max_epochs = 1
        self.in_eval = False

    def _print_gpu_memory(self):
        """Print GPU memory usage."""
        try:
            import torch

            if not torch.cuda.is_available():
                return

            table = Table(title="GPU Memory", show_header=True, header_style="bold cyan")
            table.add_column("GPU", style="dim")
            table.add_column("Allocated", justify="right")
            table.add_column("Reserved", justify="right")
            table.add_column("Free", justify="right", style="green")
            table.add_column("Total", justify="right")

            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                free = total - reserved
                table.add_row(
                    f"gpu_{i}",
                    f"{allocated:.2f} GB",
                    f"{reserved:.2f} GB",
                    f"{free:.2f} GB",
                    f"{total:.2f} GB",
                )
            console.print(table)
        except Exception:
            pass

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize progress bar at training start."""
        # Show GPU memory now that model is on device
        self._print_gpu_memory()
        console.print(Panel("[bold green]Training Started[/bold green]", border_style="green"))

        self.max_epochs = args.num_train_epochs
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]({task.completed}/{task.total})[/dim]"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self.progress.start()
        self.train_task = self.progress.add_task(
            f"Training [dim](epochs: {self.max_epochs:.0f})[/dim]",
            total=state.max_steps,
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress bar on each step."""
        if self.progress and self.train_task is not None and not self.in_eval:
            epoch = int(state.epoch) + 1 if state.epoch else 1
            self.progress.update(
                self.train_task,
                completed=state.global_step,
                description=f"Training [dim](epoch {epoch}/{self.max_epochs:.0f})[/dim]",
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Display training metrics inline."""
        if logs is None or self.in_eval:
            return

        # Filter to only training metrics (not eval)
        train_logs = {k: v for k, v in logs.items() if not k.startswith(("eval_", "_"))}
        if not train_logs:
            return

        # Format as inline text (same style as eval metrics)
        parts = []
        for key, value in train_logs.items():
            if key == "epoch":
                continue
            if isinstance(value, float):
                if key in ["learning_rate", "total_flos"]:
                    parts.append(f"[cyan]{key}[/cyan]={value:.2e}")
                else:
                    parts.append(f"[cyan]{key}[/cyan]={value:.4f}")
            else:
                parts.append(f"[cyan]{key}[/cyan]={value}")

        epoch = logs.get("epoch", state.epoch or 0)
        if parts and self.progress:
            self.progress.console.print(
                f"  [bold]Train[/bold] @ epoch {epoch:.2f}: " + "  ".join(parts)
            )

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        """Update eval progress bar during evaluation."""
        if self.eval_task is not None and self.progress:
            self.progress.advance(self.eval_task)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Display evaluation results."""
        self.in_eval = False

        # Remove eval progress bar
        if self.eval_task is not None and self.progress:
            self.progress.remove_task(self.eval_task)
            self.eval_task = None

        if metrics is None:
            return

        # Show key metrics inline
        key_metrics = ["eval_loss", "eval_accuracy", "eval_perplexity"]
        parts = []
        for key in key_metrics:
            if key in metrics:
                value = metrics[key]
                name = key.replace("eval_", "")
                if isinstance(value, float):
                    parts.append(f"[green]{name}[/green]={value:.4f}")

        epoch = metrics.get("epoch", "?")
        if self.progress:
            self.progress.console.print(
                f"  [bold]Eval[/bold] @ epoch {epoch:.2f}: " + "  ".join(parts)
            )

    def on_evaluate_begin(self, args, state, control, **kwargs):
        """Add eval progress bar when evaluation starts."""
        pass  # We don't have access to dataloader length here

    def _start_eval_progress(self, num_steps: int):
        """Start eval progress bar with known steps."""
        if self.progress and self.eval_task is None:
            self.in_eval = True
            self.eval_task = self.progress.add_task(
                "[yellow]Evaluating[/yellow]",
                total=num_steps,
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up progress bar and show final stats."""
        if self.progress:
            self.progress.stop()

        # Build final stats table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right", style="cyan")

        # Format training time as hh:mm:ss
        if state.log_history:
            train_runtime = None
            total_flos = None
            train_loss = None
            train_samples_per_second = None

            # Get metrics from last log entry
            for log in reversed(state.log_history):
                if "train_runtime" in log:
                    train_runtime = log["train_runtime"]
                if "total_flos" in log:
                    total_flos = log["total_flos"]
                if "train_loss" in log:
                    train_loss = log["train_loss"]
                if "train_samples_per_second" in log:
                    train_samples_per_second = log["train_samples_per_second"]
                if all([train_runtime, total_flos]):
                    break

            if train_runtime:
                hours, remainder = divmod(int(train_runtime), 3600)
                minutes, seconds = divmod(remainder, 60)
                table.add_row("Training time", f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            if train_loss is not None:
                table.add_row("Final loss", f"{train_loss:.4f}")

            if train_samples_per_second:
                table.add_row("Samples/second", f"{train_samples_per_second:.2f}")

            if total_flos:
                table.add_row("Total FLOPs", f"{total_flos:.2e}")

        table.add_row("Total steps", str(state.global_step))
        table.add_row("Epochs completed", f"{state.epoch:.2f}")

        console.print(
            Panel(table, title="[bold green]✓ Training Complete[/bold green]", border_style="green")
        )


class WandbCallback(TrainerCallback):
    """Custom callback for enhanced wandb logging."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._wandb = None

    def setup(self, args, state, model, **kwargs):
        """Setup wandb if enabled."""
        if self.config.report_to == "wandb":
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log model config at training start."""
        if self._wandb is None or not self._wandb.run:
            return

        if isinstance(model, PeftModel):
            peft_config = model.peft_config
            self._wandb.config.update({"peft_config": str(peft_config)})

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhanced logging with additional metrics."""
        if self._wandb is None or not self._wandb.run or logs is None:
            return

        if "grad_norm" in logs:
            self._wandb.log({"train/grad_norm": logs["grad_norm"]}, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Log final metrics."""
        if self._wandb is None or not self._wandb.run:
            return

        self._wandb.log(
            {
                "train/total_steps": state.global_step,
                "train/total_epochs": state.epoch,
            }
        )


def get_training_arguments(
    config: TrainingConfig,
    model_config: ModelConfig,
) -> TrainingArguments:
    """Create TrainingArguments from config with performance optimizations."""
    fsdp_config = None
    if config.fsdp:
        fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer", "MistralDecoderLayer"],
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_forward_prefetch": True,
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": True,
        }
        if config.fsdp_config:
            fsdp_config.update(config.fsdp_config)

    gradient_checkpointing_kwargs = config.gradient_checkpointing_kwargs
    if config.gradient_checkpointing and gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": False}

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_steps=config.eval_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        optim=config.optim,
        seed=config.seed,
        data_seed=config.data_seed,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=config.remove_unused_columns,
        report_to=config.report_to if config.report_to else "none",
        run_name=config.run_name or config.wandb_run_name,
        fsdp=config.fsdp,
        fsdp_config=fsdp_config,
        deepspeed=config.deepspeed,
        local_rank=config.local_rank,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        resume_from_checkpoint=config.resume_from_checkpoint,
        hub_model_id=config.hub_model_id,
        push_to_hub=config.push_to_hub,
        hub_token=config.hub_token,
    )

    return training_args


def setup_wandb(config: TrainingConfig) -> None:
    """Setup wandb logging."""
    if config.report_to != "wandb":
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        return

    if wandb.run is None:
        wandb.init(
            project=config.wandb_project or "lora-finetune",
            name=config.wandb_run_name,
            config={
                "output_dir": config.output_dir,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_train_epochs,
                "batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
            },
        )

    if config.wandb_watch:
        os.environ["WANDB_WATCH"] = config.wandb_watch

    if config.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = "true"


class LoraTrainer(Trainer):
    """Custom trainer with LoRA-specific optimizations."""

    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        data_collator: Optional[DataCollator] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with optional label smoothing."""
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override to inject eval progress bar."""
        # Start eval progress bar via callback
        num_steps = len(dataloader)
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, RichProgressCallback):
                callback._start_eval_progress(num_steps)
                break

        return super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model, handling PEFT models correctly."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(output_dir)
            logger.info(f"PEFT model saved to {output_dir}")
        else:
            super().save_model(output_dir, _internal_call=_internal_call)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def compute_metrics_for_classification(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy metrics for classification tasks."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def compute_metrics_for_lm(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute perplexity for language modeling tasks."""
    # For causal LM, predictions are logits and labels are token ids
    # The loss is already computed by the model, so we compute perplexity from it
    # Note: This requires the trainer to have computed loss already
    logits, labels = eval_pred

    # Compute cross entropy loss manually
    # Shift logits and labels for causal LM
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Flatten
    shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.reshape(-1)

    # Filter out padding tokens (typically -100)
    mask = shift_labels != -100
    if mask.sum() == 0:
        return {"perplexity": float("inf")}

    # Compute softmax and log probabilities
    max_logits = np.max(shift_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits - max_logits)
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Get probabilities for correct tokens
    valid_labels = np.where(mask, shift_labels, 0).astype(np.int64)
    probs = softmax[np.arange(len(valid_labels)), valid_labels]

    # Compute cross entropy only for valid tokens
    log_probs = np.log(probs + 1e-10)
    loss = -np.sum(log_probs * mask) / mask.sum()

    perplexity = np.exp(loss)
    return {"perplexity": float(perplexity)}


def create_trainer(
    model: Union[PreTrainedModel, PeftModel],
    training_config: TrainingConfig,
    model_config: ModelConfig,
    train_dataset,
    eval_dataset=None,
    processing_class=None,
    data_collator: Optional[DataCollator] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
) -> LoraTrainer:
    """Create trainer with all optimizations configured."""
    logger.info(f"Creating trainer with output_dir={training_config.output_dir}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    setup_wandb(training_config)

    training_args = get_training_arguments(training_config, model_config)
    logger.info(
        f"Training args: epochs={training_args.num_train_epochs}, batch_size={training_args.per_device_train_batch_size}, lr={training_args.learning_rate}"
    )

    callbacks = [RichProgressCallback()]
    if training_config.report_to == "wandb":
        callbacks.append(WandbCallback(training_config))

    # Disable default transformers progress bar (we use Rich instead)
    training_args.disable_tqdm = True

    trainer = LoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # Remove default PrinterCallback (we use RichProgressCallback instead)
    trainer.remove_callback(PrinterCallback)

    return trainer


def enable_gradient_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
    """Enable gradient checkpointing for memory efficiency."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def prepare_model_for_training(
    model: PreTrainedModel,
    training_config: TrainingConfig,
) -> PreTrainedModel:
    """Prepare model for training with optimizations."""
    if training_config.gradient_checkpointing:
        model = enable_gradient_checkpointing(model)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()

    return model
