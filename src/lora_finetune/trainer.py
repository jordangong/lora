"""Custom trainer with performance optimizations."""

import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from peft import PeftModel
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from .config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


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
    setup_wandb(training_config)

    training_args = get_training_arguments(training_config, model_config)

    callbacks = []
    if training_config.report_to == "wandb":
        callbacks.append(WandbCallback(training_config))

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
