import copy
import logging
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

from transformers import Trainer
from transformers.integrations import WandbCallback

from .config import (
    BenchmarkEvalConfig,
    DataConfig,
    HPOConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from .config import DPOConfig as ProjectDPOConfig
from .config import GRPOConfig as ProjectGRPOConfig

logger = logging.getLogger(__name__)
_SENSITIVE_WANDB_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "auth_token",
    "hf_token",
    "hub_token",
    "password",
    "secret",
}
_WANDB_HPO_RESERVED_KEYS = {"_wandb", "assignments", "metric"}
_SUPPORTED_HPO_CONFIG_SECTIONS = {"training", "lora", "dpo", "grpo"}
_DISALLOWED_HPO_TRAINING_FIELDS = {
    "deepspeed",
    "fsdp",
    "fsdp_config",
    "hub_model_id",
    "hub_token",
    "llm_trainer",
    "local_rank",
    "output_dir",
    "push_to_hub",
    "report_to",
    "resume_from_checkpoint",
    "run_name",
    "trainer_type",
    "wandb_project",
    "wandb_run_name",
}


def _coerce_config_value(current_value: Any, new_value: Any) -> Any:
    if current_value is None:
        return new_value
    if isinstance(current_value, bool):
        if isinstance(new_value, str):
            normalized = new_value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(new_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(new_value)
    if isinstance(current_value, float):
        return float(new_value)
    if isinstance(current_value, tuple) and isinstance(new_value, list):
        return tuple(new_value)
    return type(current_value)(new_value) if current_value is not None else new_value


def apply_hpo_parameters_to_config_sections(
    config_sections: Dict[str, Any],
    parameters: Dict[str, Any],
) -> None:
    for parameter_name, value in parameters.items():
        if parameter_name in _WANDB_HPO_RESERVED_KEYS:
            continue
        if "." in parameter_name:
            section_name, field_name = parameter_name.split(".", 1)
        else:
            section_name, field_name = "training", parameter_name
        if section_name not in _SUPPORTED_HPO_CONFIG_SECTIONS:
            raise ValueError(
                f"Unsupported HPO parameter '{parameter_name}'. Supported section prefixes: "
                f"{', '.join(sorted(_SUPPORTED_HPO_CONFIG_SECTIONS))}"
            )
        section = config_sections.get(section_name)
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


def _configure_wandb_environment(
    config: TrainingConfig,
    hpo_config: Optional[HPOConfig] = None,
) -> None:
    wandb_watch = str(config.wandb_watch).strip().lower() if config.wandb_watch else "false"
    os.environ["WANDB_WATCH"] = wandb_watch
    os.environ["WANDB_LOG_MODEL"] = "true" if config.wandb_log_model else "false"
    project_name = _resolve_wandb_project_name(config, hpo_config)
    if project_name:
        os.environ["WANDB_PROJECT"] = project_name
    if hpo_config is not None and hpo_config.entity:
        os.environ["WANDB_ENTITY"] = hpo_config.entity
    else:
        os.environ.pop("WANDB_ENTITY", None)


def _resolve_wandb_project_name(
    training_config: TrainingConfig,
    hpo_config: Optional[HPOConfig] = None,
) -> str:
    if hpo_config is not None and hpo_config.project:
        return hpo_config.project
    return training_config.wandb_project or "lora-finetune"


def _normalize_hpo_wandb_metric_name(metric_name: str) -> str:
    if metric_name.startswith("eval_"):
        return "eval/" + metric_name[len("eval_") :]
    if metric_name.startswith("test_"):
        return "test/" + metric_name[len("test_") :]
    if metric_name == "train_loss":
        return "train/loss"
    return metric_name


def get_hpo_metric_names(training_config: TrainingConfig, hpo_config: HPOConfig) -> tuple[str, str]:
    metric_name = hpo_config.metric_name or training_config.metric_for_best_model or "eval_loss"
    return metric_name, _normalize_hpo_wandb_metric_name(metric_name)


def build_hpo_hp_space(hpo_config: HPOConfig) -> Dict[str, Any]:
    if hpo_config.parameters:
        return {
            "method": hpo_config.method,
            "metric": {"name": "objective", "goal": hpo_config.direction},
            "parameters": copy.deepcopy(hpo_config.parameters),
        }
    return {
        "method": hpo_config.method,
        "metric": {"name": "objective", "goal": hpo_config.direction},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 6},
            "seed": {"distribution": "int_uniform", "min": 1, "max": 40},
            "per_device_train_batch_size": {"values": [4, 8, 16, 32, 64]},
        },
    }


def _iter_hpo_parameter_names(hpo_config: HPOConfig):
    if hpo_config.parameters:
        return list(hpo_config.parameters.keys())
    return ["learning_rate", "num_train_epochs", "seed", "per_device_train_batch_size"]


def validate_hpo_parameter_support(
    trainer_type: str,
    training_config: TrainingConfig,
    lora_config: Optional[LoraConfig],
    dpo_config: Optional[ProjectDPOConfig],
    grpo_config: Optional[ProjectGRPOConfig],
    hpo_config: HPOConfig,
) -> None:
    config_sections: Dict[str, Any] = {
        "training": training_config,
        "lora": lora_config,
        "dpo": dpo_config,
        "grpo": grpo_config,
    }

    for parameter_name in _iter_hpo_parameter_names(hpo_config):
        if "." in parameter_name:
            section_name, field_name = parameter_name.split(".", 1)
        else:
            section_name, field_name = "training", parameter_name

        if section_name not in _SUPPORTED_HPO_CONFIG_SECTIONS:
            raise ValueError(
                f"Unsupported HPO parameter '{parameter_name}'. Supported section prefixes: "
                f"{', '.join(sorted(_SUPPORTED_HPO_CONFIG_SECTIONS))}"
            )

        section = config_sections.get(section_name)
        if section is None or not hasattr(section, field_name):
            raise ValueError(
                f"HPO parameter '{parameter_name}' does not match any available config field"
            )

        if section_name == "training" and field_name in _DISALLOWED_HPO_TRAINING_FIELDS:
            raise ValueError(
                f"HPO parameter '{parameter_name}' is not supported because it changes trainer wiring or run metadata"
            )

        if trainer_type != "transformers" and section_name != "training":
            raise ValueError(
                f"HPO parameter '{parameter_name}' is not supported with TRL-backed trainer_type='{trainer_type}'. "
                "For TRL trainers, only training.* parameters are currently supported."
            )

        if trainer_type != "dpo" and section_name == "dpo":
            raise ValueError(
                f"HPO parameter '{parameter_name}' is only valid for training.trainer_type='dpo'"
            )

        if trainer_type != "grpo" and section_name == "grpo":
            raise ValueError(
                f"HPO parameter '{parameter_name}' is only valid for training.trainer_type='grpo'"
            )

        if trainer_type == "transformers" and section_name not in {"training", "lora"}:
            raise ValueError(
                f"HPO parameter '{parameter_name}' is not supported for transformers-backed trainers"
            )


def build_hpo_compute_objective(
    training_config: TrainingConfig,
    hpo_config: HPOConfig,
) -> Callable[[Dict[str, float]], float]:
    metric_name, _ = get_hpo_metric_names(training_config, hpo_config)

    def _compute_objective(metrics: Dict[str, float]) -> float:
        if metric_name not in metrics:
            available = ", ".join(sorted(metrics.keys()))
            raise ValueError(
                f"HPO metric '{metric_name}' was not found in evaluation metrics. Available metrics: {available}"
            )
        return float(metrics[metric_name])

    return _compute_objective


def run_hyperparameter_search(
    trainer: Trainer,
    training_config: TrainingConfig,
    hpo_config: HPOConfig,
):
    metric_name, wandb_metric_name = get_hpo_metric_names(training_config, hpo_config)
    project_name = _resolve_wandb_project_name(training_config, hpo_config)
    logger.info(
        "Starting hyperparameter search with backend=%s, trials=%s, metric=%s, direction=%s",
        hpo_config.backend,
        hpo_config.n_trials,
        metric_name,
        hpo_config.direction,
    )
    return trainer.hyperparameter_search(
        backend=hpo_config.backend,
        hp_space=lambda _trial: build_hpo_hp_space(hpo_config),
        compute_objective=build_hpo_compute_objective(training_config, hpo_config),
        n_trials=hpo_config.n_trials,
        direction=hpo_config.direction,
        project=project_name,
        entity=hpo_config.entity,
        name=hpo_config.sweep_name,
        sweep_id=hpo_config.sweep_id,
        metric=wandb_metric_name,
    )


def _sanitize_wandb_config(value: Any, key: Optional[str] = None) -> Any:
    if isinstance(key, str) and key.lower() in _SENSITIVE_WANDB_CONFIG_KEYS:
        return "***REDACTED***" if value is not None else None
    if isinstance(value, dict):
        return {
            nested_key: _sanitize_wandb_config(
                nested_value,
                key=nested_key if isinstance(nested_key, str) else None,
            )
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_wandb_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_wandb_config(item) for item in value)
    return value


def _build_wandb_config_payload(
    training_config: TrainingConfig,
    model_config: Optional[ModelConfig] = None,
    data_config: Optional[DataConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    dpo_config: Optional[ProjectDPOConfig] = None,
    grpo_config: Optional[ProjectGRPOConfig] = None,
    hpo_config: Optional[HPOConfig] = None,
    benchmark_eval_config: Optional[BenchmarkEvalConfig] = None,
) -> Dict[str, Any]:
    config_payload: Dict[str, Any] = {"training": training_config.to_serializable_dict()}
    if model_config is not None:
        config_payload["model"] = asdict(model_config)
    if data_config is not None:
        config_payload["data"] = asdict(data_config)
    if lora_config is not None:
        config_payload["lora"] = asdict(lora_config)
    if dpo_config is not None:
        config_payload["dpo"] = asdict(dpo_config)
    if grpo_config is not None:
        config_payload["grpo"] = asdict(grpo_config)
    if hpo_config is not None:
        config_payload["hpo"] = asdict(hpo_config)
    if benchmark_eval_config is not None:
        config_payload["benchmark_eval"] = asdict(benchmark_eval_config)
    return _sanitize_wandb_config(config_payload)


def _sync_wandb_config(
    training_config: TrainingConfig,
    model_config: Optional[ModelConfig] = None,
    data_config: Optional[DataConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    dpo_config: Optional[ProjectDPOConfig] = None,
    grpo_config: Optional[ProjectGRPOConfig] = None,
    hpo_config: Optional[HPOConfig] = None,
    benchmark_eval_config: Optional[BenchmarkEvalConfig] = None,
) -> None:
    if training_config.report_to != "wandb":
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None or getattr(wandb, "config", None) is None:
        return
    config_payload = _build_wandb_config_payload(
        training_config=training_config,
        model_config=model_config,
        data_config=data_config,
        lora_config=lora_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        hpo_config=hpo_config,
        benchmark_eval_config=benchmark_eval_config,
    )
    try:
        wandb.config.update(config_payload, allow_val_change=True)
    except TypeError:
        wandb.config.update(config_payload)


def setup_wandb(
    config: TrainingConfig,
    model_config: Optional[ModelConfig] = None,
    data_config: Optional[DataConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    dpo_config: Optional[ProjectDPOConfig] = None,
    grpo_config: Optional[ProjectGRPOConfig] = None,
    hpo_config: Optional[HPOConfig] = None,
    benchmark_eval_config: Optional[BenchmarkEvalConfig] = None,
) -> Optional[str]:
    """Setup wandb logging and return run name for output directory."""
    if config.report_to != "wandb":
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        return None

    _configure_wandb_environment(config, hpo_config=hpo_config)

    config_payload = _build_wandb_config_payload(
        training_config=config,
        model_config=model_config,
        data_config=data_config,
        lora_config=lora_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        hpo_config=hpo_config,
        benchmark_eval_config=benchmark_eval_config,
    )

    if hpo_config is not None and hpo_config.enabled:
        logger.info(
            "Skipping eager wandb.init because HPO is enabled and W&B sweep runs will initialize per trial"
        )
        return None

    if wandb.run is None:
        wandb.init(
            project=_resolve_wandb_project_name(config, hpo_config),
            name=config.wandb_run_name,
            config=config_payload,
        )
    _sync_wandb_config(
        training_config=config,
        model_config=model_config,
        data_config=data_config,
        lora_config=lora_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        hpo_config=hpo_config,
        benchmark_eval_config=benchmark_eval_config,
    )

    return wandb.run.name if wandb.run else None


class HPOWandbCallback(WandbCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        if state.is_hyper_param_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
