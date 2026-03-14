"""Custom trainer with performance optimizations."""

import inspect
import logging
import math
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

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

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    SFTConfig = None
    SFTTrainer = None

from .config import DataConfig, LoraConfig, ModelConfig, TrainingConfig

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

    @staticmethod
    def _format_epoch(epoch) -> str:
        if epoch is None:
            return "?"
        try:
            return f"{float(epoch):.2f}"
        except (TypeError, ValueError):
            return str(epoch)

    def _print_gpu_memory(self):
        """Print GPU memory usage."""
        try:
            import torch

            if not torch.cuda.is_available():
                return

            table = Table(
                title="GPU Memory", show_header=True, header_style="bold cyan"
            )
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
        console.print(
            Panel("[bold green]Training Started[/bold green]", border_style="green")
        )

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
            try:
                current_epoch = float(state.epoch or 0.0)
            except (TypeError, ValueError):
                current_epoch = 0.0
            max_epochs = max(1, int(round(self.max_epochs)))
            epoch = min(max_epochs, max(1, int(math.ceil(current_epoch))))
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
                f"  [bold]Train[/bold] @ epoch {self._format_epoch(epoch)}: "
                + "  ".join(parts)
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

        epoch = metrics.get("epoch", state.epoch if state is not None else "?")
        if self.progress:
            self.progress.console.print(
                f"  [bold]Eval[/bold] @ epoch {self._format_epoch(epoch)}: "
                + "  ".join(parts)
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
                table.add_row(
                    "Training time", f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                )

            if train_loss is not None:
                table.add_row("Final loss", f"{train_loss:.4f}")

            if train_samples_per_second:
                table.add_row("Samples/second", f"{train_samples_per_second:.2f}")

            if total_flos:
                table.add_row("Total FLOPs", f"{total_flos:.2e}")

        table.add_row("Total steps", str(state.global_step))
        table.add_row("Epochs completed", self._format_epoch(state.epoch))

        console.print(
            Panel(
                table,
                title="[bold green]✓ Training Complete[/bold green]",
                border_style="green",
            )
        )


def _get_training_arguments_kwargs(
    config: TrainingConfig,
    model_config: ModelConfig,
) -> Dict[str, Any]:
    fsdp_config = None
    if config.fsdp:
        fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": [
                "LlamaDecoderLayer",
                "MistralDecoderLayer",
            ],
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

    return {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "warmup_steps": config.warmup_steps,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "eval_steps": config.eval_steps,
        "eval_strategy": config.eval_strategy,
        "prediction_loss_only": config.prediction_loss_only,
        "save_strategy": config.save_strategy,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "tf32": config.tf32,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "optim": config.optim,
        "seed": config.seed,
        "data_seed": config.data_seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.dataloader_pin_memory,
        "remove_unused_columns": config.remove_unused_columns,
        "report_to": config.report_to if config.report_to else "none",
        "run_name": config.run_name or config.wandb_run_name,
        "fsdp": config.fsdp,
        "fsdp_config": fsdp_config,
        "deepspeed": config.deepspeed,
        "local_rank": config.local_rank,
        "ddp_find_unused_parameters": config.ddp_find_unused_parameters,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "hub_model_id": config.hub_model_id,
        "push_to_hub": config.push_to_hub,
        "hub_token": config.hub_token,
    }


def get_training_arguments(
    config: TrainingConfig,
    model_config: ModelConfig,
) -> TrainingArguments:
    """Create TrainingArguments from config with performance optimizations."""
    return TrainingArguments(**_get_training_arguments_kwargs(config, model_config))


def get_sft_training_arguments(
    config: TrainingConfig,
    model_config: ModelConfig,
    data_config: Optional[DataConfig] = None,
    skip_prepare_dataset: bool = True,
):
    if SFTConfig is None:
        raise ImportError(
            "TRL is required for causal LM finetuning. Install it with: pip install trl"
        )

    training_args = SFTConfig(**_get_training_arguments_kwargs(config, model_config))
    dataset_kwargs = getattr(training_args, "dataset_kwargs", None) or {}
    dataset_kwargs["skip_prepare_dataset"] = skip_prepare_dataset
    training_args.dataset_kwargs = dataset_kwargs

    if hasattr(training_args, "packing"):
        training_args.packing = False
    if hasattr(training_args, "eval_packing") and training_args.eval_packing is None:
        training_args.eval_packing = False
    if data_config is not None:
        if hasattr(training_args, "dataset_text_field"):
            training_args.dataset_text_field = data_config.text_column
        if hasattr(training_args, "dataset_num_proc"):
            training_args.dataset_num_proc = data_config.preprocessing_num_workers
        if hasattr(training_args, "max_length"):
            training_args.max_length = data_config.max_seq_length
        if hasattr(training_args, "completion_only_loss"):
            training_args.completion_only_loss = data_config.response_only_loss
        if hasattr(training_args, "assistant_only_loss"):
            training_args.assistant_only_loss = data_config.assistant_only_loss
        if hasattr(training_args, "eos_token") and data_config.eos_token is not None:
            training_args.eos_token = data_config.eos_token
    elif (
        hasattr(training_args, "dataset_text_field")
        and training_args.dataset_text_field is None
    ):
        training_args.dataset_text_field = "text"

    return training_args


def _dataset_is_pretokenized(dataset) -> bool:
    try:
        sample = next(iter(dataset))
    except Exception:
        return False
    return isinstance(sample, dict) and "input_ids" in sample


def _get_dataset_sample(dataset) -> Optional[Dict[str, Any]]:
    try:
        sample = next(iter(dataset))
    except Exception:
        return None
    return sample if isinstance(sample, dict) else None


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_wandb(config: TrainingConfig) -> Optional[str]:
    """Setup wandb logging and return run name for output directory."""
    if config.report_to != "wandb":
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        return None

    wandb_watch = (
        str(config.wandb_watch).strip().lower() if config.wandb_watch else "false"
    )
    os.environ["WANDB_WATCH"] = wandb_watch
    os.environ["WANDB_LOG_MODEL"] = "true" if config.wandb_log_model else "false"

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

    return wandb.run.name if wandb.run else None


class LoraTrainerMixin:
    def __init__(self, *args, lora_config: Optional[LoraConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with optional label smoothing."""
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )

    def create_optimizer(self):
        """Create optimizer with LoRA+ support for different learning rates."""
        if self.lora_config is not None and self.lora_config.method == "loraplus":
            return self._create_loraplus_optimizer()
        return super().create_optimizer()

    def _create_loraplus_optimizer(self):
        """Create LoRA+ optimizer with different LRs for A and B matrices."""
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        lr_ratio = self.lora_config.loraplus_lr_ratio
        base_lr = self.args.learning_rate

        # Separate parameters into groups based on LoRA A/B matrices
        optimizer_grouped_parameters = [
            # LoRA B matrices - higher learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and "lora_B" in n and n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
                "lr": base_lr * lr_ratio,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and "lora_B" in n and n not in decay_parameters
                ],
                "weight_decay": 0.0,
                "lr": base_lr * lr_ratio,
            },
            # LoRA A matrices and other params - base learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and "lora_B" not in n and n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
                "lr": base_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                    and "lora_B" not in n
                    and n not in decay_parameters
                ],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
        ]

        # Filter out empty parameter groups
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if len(group["params"]) > 0
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        # Remove lr from kwargs since we set it per group
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        logger.info(
            f"LoRA+ optimizer created with lr_ratio={lr_ratio} "
            f"(lr_A={base_lr}, lr_B={base_lr * lr_ratio})"
        )

        return self.optimizer

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

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """Save model, handling PEFT models correctly."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(output_dir)
            logger.info(f"PEFT model saved to {output_dir}")
        else:
            super().save_model(output_dir, _internal_call=_internal_call)

        processing_class = getattr(self, "processing_class", None)
        if processing_class is None:
            processing_class = getattr(self, "tokenizer", None)
        if processing_class is not None:
            processing_class.save_pretrained(output_dir)


class LoraTrainer(LoraTrainerMixin, Trainer):
    """Custom trainer with LoRA-specific optimizations."""


if SFTTrainer is not None:

    class LoraSFTTrainer(LoraTrainerMixin, SFTTrainer):
        pass

else:
    LoraSFTTrainer = None


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
    data_config: Optional[DataConfig] = None,
    eval_dataset=None,
    processing_class=None,
    data_collator: Optional[DataCollator] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    lora_config: Optional[LoraConfig] = None,
    callbacks: Optional[List] = None,
) -> Trainer:
    """Create trainer with all optimizations configured."""
    eval_strategy = str(training_config.eval_strategy).lower()
    if eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            "training.eval_strategy is set to evaluate, but no eval_dataset was provided. "
            "Provide a validation split (e.g. data.eval_split_ratio or data.eval_dataset_name), "
            "or set training.eval_strategy='no'."
        )

    wandb_run_name = setup_wandb(training_config)

    # Generate unique run identifier for output directory
    run_id = wandb_run_name if wandb_run_name else generate_run_id()
    training_config.output_dir = os.path.join(training_config.output_dir, run_id)

    logger.info(f"Creating trainer with output_dir={training_config.output_dir}")
    try:
        logger.info(f"Train dataset size: {len(train_dataset)}")
    except TypeError:
        logger.info("Train dataset size: unknown (streaming/iterable dataset)")
    if eval_dataset is not None:
        try:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
        except TypeError:
            logger.info("Eval dataset size: unknown (streaming/iterable dataset)")

    train_sample = _get_dataset_sample(train_dataset)
    use_trl_sft = (
        training_config.llm_trainer == "trl" and model_config.model_type == "causal_lm"
    )
    if use_trl_sft:
        if LoraSFTTrainer is None or SFTConfig is None:
            raise ImportError(
                "TRL is required for causal LM finetuning. Install it with: pip install trl"
            )
        training_args = get_sft_training_arguments(
            training_config,
            model_config,
            data_config=data_config,
            skip_prepare_dataset=bool(train_sample and "input_ids" in train_sample),
        )
        if (
            data_config is not None
            and train_sample is not None
            and data_config.response_only_loss
            and not data_config.assistant_only_loss
            and hasattr(training_args, "assistant_only_loss")
            and ("messages" in train_sample or "conversations" in train_sample)
        ):
            training_args.assistant_only_loss = True
        logger.info("Using TRL SFTTrainer for causal LM finetuning")
    else:
        training_args = get_training_arguments(training_config, model_config)
    logger.info(
        f"Training args: epochs={training_args.num_train_epochs}, batch_size={training_args.per_device_train_batch_size}, lr={training_args.learning_rate}"
    )

    rich_progress_callback = RichProgressCallback()
    trainer_callbacks = [rich_progress_callback]
    if callbacks:
        # Pass RichProgressCallback reference to LightEvalCallback for progress integration
        for cb in callbacks:
            if cb.__class__.__name__ == "LightEvalCallback":
                cb.rich_progress_callback = rich_progress_callback
        trainer_callbacks.extend(callbacks)

    # Disable default transformers progress bar (we use Rich instead)
    training_args.disable_tqdm = True

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "callbacks": trainer_callbacks,
        "compute_metrics": compute_metrics,
        "lora_config": lora_config,
    }

    if use_trl_sft:
        sft_signature = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in sft_signature:
            trainer_kwargs["processing_class"] = processing_class
        elif "tokenizer" in sft_signature:
            trainer_kwargs["tokenizer"] = processing_class
        trainer = LoraSFTTrainer(**trainer_kwargs)
    else:
        trainer_kwargs["processing_class"] = processing_class
        trainer = LoraTrainer(**trainer_kwargs)

    # Remove default PrinterCallback (we use RichProgressCallback instead)
    trainer.remove_callback(PrinterCallback)

    return trainer


def enable_gradient_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
    """Enable gradient checkpointing for memory efficiency."""
    if hasattr(model, "gradient_checkpointing_enable"):
        # Disable use_cache as it's incompatible with gradient checkpointing
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def prepare_model_for_training(
    model: PreTrainedModel,
    training_config: TrainingConfig,
    tokenizer=None,
) -> PreTrainedModel:
    """Prepare model for training with optimizations."""
    unsloth_managed_gradient_checkpointing = bool(
        getattr(model, "_lora_finetune_unsloth_managed_gradient_checkpointing", False)
    )

    if training_config.gradient_checkpointing and not unsloth_managed_gradient_checkpointing:
        model = enable_gradient_checkpointing(model)

    # Sync model config with tokenizer to avoid mismatch warnings (after PEFT wrapping)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()

    return model
