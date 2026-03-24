"""Custom trainer with performance optimizations."""

import copy
import inspect
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from peft import PeftModel
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import WandbCallback
from transformers.trainer_callback import PrinterCallback

from .config import (
    BenchmarkEvalConfig,
    DataConfig,
    HPOConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from .config import (
    DPOConfig as ProjectDPOConfig,
)
from .config import (
    GRPOConfig as ProjectGRPOConfig,
)
from .trainer_hpo import (
    _SUPPORTED_HPO_CONFIG_SECTIONS,
    _WANDB_HPO_RESERVED_KEYS,
    HPOWandbCallback,
    _coerce_config_value,
    _sync_wandb_config,
    apply_hpo_parameters_to_config_sections,
    setup_wandb,
    validate_hpo_parameter_support,
)
from .trainer_hpo import (
    run_hyperparameter_search as imported_run_hyperparameter_search,
)
from .trainer_metrics import (
    build_grpo_reward_functions,
)
from .trainer_metrics import (
    compute_metrics_for_classification as imported_compute_metrics_for_classification,
)
from .trainer_metrics import (
    compute_metrics_for_lm as imported_compute_metrics_for_lm,
)
from .trainer_progress import RichProgressCallback
from .utils import capture_stdout

TRL_SFT_IMPORT_ERROR = None
try:
    from trl import SFTConfig, SFTTrainer
except Exception as exc:
    TRL_SFT_IMPORT_ERROR = exc
    SFTConfig = None
    SFTTrainer = None

TRL_DPO_IMPORT_ERROR = None
try:
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer
except Exception as exc:
    TRL_DPO_IMPORT_ERROR = exc
    TRLDPOConfig = None
    DPOTrainer = None

TRL_GRPO_IMPORT_ERROR = None
try:
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer
except Exception as exc:
    TRL_GRPO_IMPORT_ERROR = exc
    TRLGRPOConfig = None
    GRPOTrainer = None

logger = logging.getLogger(__name__)

run_hyperparameter_search = imported_run_hyperparameter_search
compute_metrics_for_classification = imported_compute_metrics_for_classification
compute_metrics_for_lm = imported_compute_metrics_for_lm


def _raise_missing_trl_dependency(
    message: str, import_error: Optional[BaseException] = None
) -> None:
    if import_error is None:
        raise ImportError(message)
    raise ImportError(f"{message}. Original import error: {import_error}") from import_error


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
        "warmup_steps": config.warmup_steps if config.warmup_steps > 0 else config.warmup_ratio,
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
        _raise_missing_trl_dependency(
            "TRL is required for causal LM finetuning. Install it with: pip install trl",
            TRL_SFT_IMPORT_ERROR,
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
    elif hasattr(training_args, "dataset_text_field") and training_args.dataset_text_field is None:
        training_args.dataset_text_field = "text"

    return training_args


def _build_trl_config_kwargs(
    config: TrainingConfig,
    model_config: ModelConfig,
) -> Dict[str, Any]:
    return dict(_get_training_arguments_kwargs(config, model_config))


def _filter_init_kwargs(init_target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parameters = inspect.signature(init_target).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def get_dpo_training_arguments(
    config: TrainingConfig,
    model_config: ModelConfig,
    dpo_config: ProjectDPOConfig,
    data_config: Optional[DataConfig] = None,
):
    if TRLDPOConfig is None:
        _raise_missing_trl_dependency(
            "TRL is required for DPO finetuning. Install it with: pip install trl",
            TRL_DPO_IMPORT_ERROR,
        )

    kwargs = _build_trl_config_kwargs(config, model_config)
    kwargs.update(
        {
            "beta": dpo_config.beta,
            "max_prompt_length": dpo_config.max_prompt_length,
            "max_completion_length": dpo_config.max_completion_length,
            "max_length": dpo_config.max_length,
            "reference_free": dpo_config.reference_free,
            "label_smoothing": dpo_config.label_smoothing,
            "disable_dropout": dpo_config.disable_dropout,
        }
    )
    if data_config is not None:
        kwargs["dataset_num_proc"] = data_config.preprocessing_num_workers
        kwargs["max_length"] = dpo_config.max_length or data_config.max_seq_length

    return TRLDPOConfig(**_filter_init_kwargs(TRLDPOConfig.__init__, kwargs))


def get_grpo_training_arguments(
    config: TrainingConfig,
    model_config: ModelConfig,
    grpo_config: ProjectGRPOConfig,
):
    if TRLGRPOConfig is None:
        _raise_missing_trl_dependency(
            "TRL is required for GRPO finetuning. Install it with: pip install trl",
            TRL_GRPO_IMPORT_ERROR,
        )

    kwargs = _build_trl_config_kwargs(config, model_config)
    kwargs.update(
        {
            "remove_unused_columns": False,
            "max_prompt_length": grpo_config.max_prompt_length,
            "max_completion_length": grpo_config.max_completion_length,
            "num_generations": grpo_config.num_generations,
            "temperature": grpo_config.temperature,
            "top_p": grpo_config.top_p,
            "top_k": grpo_config.top_k,
            "beta": grpo_config.beta,
            "num_iterations": grpo_config.num_iterations,
            "epsilon": grpo_config.epsilon,
            "loss_type": grpo_config.loss_type,
            "scale_rewards": grpo_config.scale_rewards,
        }
    )

    return TRLGRPOConfig(**_filter_init_kwargs(TRLGRPOConfig.__init__, kwargs))


def resolve_trainer_type(training_config: TrainingConfig, model_config: ModelConfig) -> str:
    if model_config.model_type != "causal_lm":
        if training_config.trainer_type != "sft":
            raise ValueError(
                f"training.trainer_type='{training_config.trainer_type}' is only supported for causal_lm models"
            )
        return "transformers"

    if training_config.trainer_type == "sft":
        return "sft" if training_config.llm_trainer == "trl" else "transformers"

    if training_config.llm_trainer != "trl":
        raise ValueError(
            f"training.trainer_type='{training_config.trainer_type}' requires training.llm_trainer='trl'"
        )

    return training_config.trainer_type


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


class LoraTrainerMixin:
    def __init__(
        self,
        *args,
        lora_config: Optional[LoraConfig] = None,
        hpo_config: Optional[HPOConfig] = None,
        hpo_config_sections: Optional[Dict[str, Any]] = None,
        hpo_sync_callback: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        self.hpo_config = hpo_config
        self._hpo_config_sections = hpo_config_sections or {}
        self._hpo_base_config_sections = {
            name: copy.deepcopy(section)
            for name, section in self._hpo_config_sections.items()
            if section is not None
        }
        self._hpo_base_output_dir = getattr(self.args, "output_dir", None)
        self._hpo_trial_count = 0
        self._hpo_sync_callback = hpo_sync_callback

    def _reset_hpo_config_sections(self) -> None:
        for name, base_section in self._hpo_base_config_sections.items():
            current_section = self._hpo_config_sections.get(name)
            if current_section is None:
                continue
            current_section.__dict__.clear()
            current_section.__dict__.update(copy.deepcopy(base_section.__dict__))

    def _apply_hpo_config_parameter(self, parameter_name: str, value: Any) -> None:
        if "." in parameter_name:
            section_name, field_name = parameter_name.split(".", 1)
        else:
            section_name, field_name = "training", parameter_name
        if section_name not in _SUPPORTED_HPO_CONFIG_SECTIONS:
            raise ValueError(
                f"Unsupported HPO parameter '{parameter_name}'. Supported section prefixes: "
                f"{', '.join(sorted(_SUPPORTED_HPO_CONFIG_SECTIONS))}"
            )
        section = self._hpo_config_sections.get(section_name)
        if section is None:
            raise ValueError(
                f"HPO parameter '{parameter_name}' has no available config section to update"
            )
        if not hasattr(section, field_name):
            raise ValueError(
                f"HPO parameter '{parameter_name}' does not match any field on {section_name} config"
            )
        current_value = getattr(section, field_name)
        setattr(section, field_name, _coerce_config_value(current_value, value))

    def _apply_hpo_trial_parameters(self, params: Dict[str, Any]) -> None:
        self._reset_hpo_config_sections()
        apply_hpo_parameters_to_config_sections(self._hpo_config_sections, params)
        if callable(self._hpo_sync_callback):
            self._hpo_sync_callback()

    def _get_hpo_trial_output_dir(self) -> Optional[str]:
        base_output_dir = self._hpo_base_output_dir or getattr(self.args, "output_dir", None)
        if not base_output_dir:
            return None

        self._hpo_trial_count += 1
        trial_name = getattr(self.state, "trial_name", None)
        if not trial_name:
            trial_name = f"trial_{self._hpo_trial_count:03d}"

        safe_trial_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(trial_name)).strip("._")
        if not safe_trial_name:
            safe_trial_name = f"trial_{self._hpo_trial_count:03d}"

        return os.path.join(base_output_dir, safe_trial_name)

    def _set_hpo_trial_output_dir(self) -> None:
        trial_output_dir = self._get_hpo_trial_output_dir()
        if trial_output_dir is None:
            return
        self.args.output_dir = trial_output_dir
        os.makedirs(trial_output_dir, exist_ok=True)

    def _hp_search_setup(self, trial) -> None:
        if not (
            self.hpo_config is not None and self.hpo_config.enabled and isinstance(trial, dict)
        ):
            return super()._hp_search_setup(trial)

        trainer_params: Dict[str, Any] = {}
        config_params: Dict[str, Any] = {}
        training_section = self._hpo_config_sections.get("training")

        for key, value in trial.items():
            if key in _WANDB_HPO_RESERVED_KEYS:
                continue
            if "." in key:
                section_name, field_name = key.split(".", 1)
                config_params[key] = value
                if hasattr(self.args, field_name):
                    trainer_params[field_name] = value
            elif hasattr(self.args, key):
                trainer_params[key] = value
                if training_section is not None and hasattr(training_section, key):
                    config_params[f"training.{key}"] = value
            else:
                raise ValueError(
                    f"Unsupported HPO parameter '{key}'. Use dotted keys like 'lora.r' for non-training args."
                )

        self._set_hpo_trial_output_dir()
        super()._hp_search_setup(trainer_params)
        self._apply_hpo_trial_parameters(config_params)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with optional label smoothing."""
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

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
                    if p.requires_grad and "lora_B" not in n and n not in decay_parameters
                ],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
        ]

        # Filter out empty parameter groups
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if len(group["params"]) > 0
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
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


if DPOTrainer is not None:

    class LoraDPOTrainer(LoraTrainerMixin, DPOTrainer):
        pass

else:
    LoraDPOTrainer = None


if GRPOTrainer is not None:

    class LoraGRPOTrainer(LoraTrainerMixin, GRPOTrainer):
        pass

else:
    LoraGRPOTrainer = None


def create_trainer(
    model: Union[PreTrainedModel, PeftModel],
    training_config: TrainingConfig,
    model_config: ModelConfig,
    train_dataset,
    data_config: Optional[DataConfig] = None,
    dpo_config: Optional[ProjectDPOConfig] = None,
    grpo_config: Optional[ProjectGRPOConfig] = None,
    benchmark_eval_config: Optional[BenchmarkEvalConfig] = None,
    eval_dataset=None,
    processing_class=None,
    data_collator: Optional[DataCollator] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    lora_config: Optional[LoraConfig] = None,
    callbacks: Optional[List] = None,
    hpo_config: Optional[HPOConfig] = None,
    model_init: Optional[Callable[..., Union[PreTrainedModel, PeftModel]]] = None,
    hpo_config_sections: Optional[Dict[str, Any]] = None,
) -> Trainer:
    """Create trainer with all optimizations configured."""
    eval_strategy = str(training_config.eval_strategy).lower()
    if eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            "training.eval_strategy is set to evaluate, but no eval_dataset was provided. "
            "Provide a validation split (e.g. data.eval_split_ratio or data.eval_dataset_name), "
            "or set training.eval_strategy='no'."
        )

    trainer_type = resolve_trainer_type(training_config, model_config)
    hpo_enabled = hpo_config is not None and hpo_config.enabled
    if hpo_enabled and eval_dataset is None:
        raise ValueError(
            "hpo.enabled requires an eval_dataset so Trainer.hyperparameter_search can score each trial"
        )
    if hpo_enabled and model_init is None:
        raise ValueError(
            "hpo.enabled requires model_init so the model can be rebuilt for each trial"
        )
    if hpo_enabled and training_config.push_to_hub:
        raise ValueError("training.push_to_hub is not supported during hyperparameter search")
    if hpo_enabled and training_config.resume_from_checkpoint is not None:
        raise ValueError(
            "training.resume_from_checkpoint is not supported during hyperparameter search"
        )
    if hpo_enabled and benchmark_eval_config is not None and benchmark_eval_config.enabled:
        raise ValueError("benchmark_eval is not supported during hyperparameter search")
    if hpo_enabled and training_config.report_to != "wandb":
        raise ValueError("W&B hyperparameter search requires training.report_to='wandb'")
    if hpo_enabled:
        validate_hpo_parameter_support(
            trainer_type=trainer_type,
            training_config=training_config,
            lora_config=lora_config,
            dpo_config=dpo_config,
            grpo_config=grpo_config,
            hpo_config=hpo_config,
        )

    wandb_run_name = setup_wandb(
        training_config,
        model_config=model_config,
        data_config=data_config,
        lora_config=lora_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        hpo_config=hpo_config,
        benchmark_eval_config=benchmark_eval_config,
    )

    # Generate unique run identifier for output directory
    if not hpo_enabled:
        run_id = wandb_run_name if wandb_run_name else generate_run_id()
        training_config.output_dir = os.path.join(training_config.output_dir, run_id)

    def _sync_current_wandb_config() -> None:
        _sync_wandb_config(
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            lora_config=lora_config,
            dpo_config=dpo_config,
            grpo_config=grpo_config,
            hpo_config=hpo_config,
            benchmark_eval_config=benchmark_eval_config,
        )

    _sync_current_wandb_config()

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
    if trainer_type == "sft":
        if LoraSFTTrainer is None or SFTConfig is None:
            _raise_missing_trl_dependency(
                "TRL is required for causal LM finetuning. Install it with: pip install trl",
                TRL_SFT_IMPORT_ERROR,
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
    elif trainer_type == "dpo":
        if dpo_config is None or LoraDPOTrainer is None or TRLDPOConfig is None:
            _raise_missing_trl_dependency(
                "TRL is required for DPO finetuning. Install it with: pip install trl",
                TRL_DPO_IMPORT_ERROR,
            )
        training_args = get_dpo_training_arguments(
            training_config,
            model_config,
            dpo_config=dpo_config,
            data_config=data_config,
        )
        logger.info("Using TRL DPOTrainer for preference finetuning")
    elif trainer_type == "grpo":
        if grpo_config is None or LoraGRPOTrainer is None or TRLGRPOConfig is None:
            _raise_missing_trl_dependency(
                "TRL is required for GRPO finetuning. Install it with: pip install trl",
                TRL_GRPO_IMPORT_ERROR,
            )
        training_args = get_grpo_training_arguments(
            training_config,
            model_config,
            grpo_config=grpo_config,
        )
        logger.info("Using TRL GRPOTrainer for RL finetuning")
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

    with capture_stdout():
        if trainer_type == "sft":
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "data_collator": data_collator,
                "callbacks": trainer_callbacks,
                "compute_metrics": compute_metrics,
                "lora_config": lora_config,
                "hpo_config": hpo_config,
                "hpo_config_sections": hpo_config_sections,
                "hpo_sync_callback": _sync_current_wandb_config,
            }
            sft_signature = inspect.signature(SFTTrainer.__init__).parameters
            if "processing_class" in sft_signature:
                trainer_kwargs["processing_class"] = processing_class
            elif "tokenizer" in sft_signature:
                trainer_kwargs["tokenizer"] = processing_class
            trainer = LoraSFTTrainer(**trainer_kwargs)
        elif trainer_type == "dpo":
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "callbacks": trainer_callbacks,
                "compute_metrics": compute_metrics,
                "lora_config": lora_config,
                "hpo_config": hpo_config,
                "hpo_config_sections": hpo_config_sections,
                "hpo_sync_callback": _sync_current_wandb_config,
                "processing_class": processing_class,
                "data_collator": data_collator,
            }
            trainer = LoraDPOTrainer(**trainer_kwargs)
        elif trainer_type == "grpo":
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "callbacks": trainer_callbacks,
                "lora_config": lora_config,
                "hpo_config": hpo_config,
                "hpo_config_sections": hpo_config_sections,
                "hpo_sync_callback": _sync_current_wandb_config,
                "processing_class": processing_class,
                "reward_funcs": build_grpo_reward_functions(grpo_config),
            }
            trainer = LoraGRPOTrainer(**trainer_kwargs)
        else:
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "data_collator": data_collator,
                "callbacks": trainer_callbacks,
                "compute_metrics": compute_metrics,
                "lora_config": lora_config,
                "hpo_config": hpo_config,
                "hpo_config_sections": hpo_config_sections,
                "hpo_sync_callback": _sync_current_wandb_config,
                "processing_class": processing_class,
            }
            trainer = LoraTrainer(**trainer_kwargs)

    if hpo_enabled and training_config.report_to == "wandb":
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(HPOWandbCallback())

    # Remove default PrinterCallback (we use RichProgressCallback instead)
    trainer.remove_callback(PrinterCallback)
    if model_init is not None:
        trainer.model_init = model_init

    return trainer


def enable_gradient_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
    """Enable gradient checkpointing for memory efficiency."""
    if hasattr(model, "gradient_checkpointing_enable"):
        # Disable use_cache as it's incompatible with gradient checkpointing
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
    elif training_config.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False

    # Sync model config with tokenizer to avoid mismatch warnings (after PEFT wrapping)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()

    return model
