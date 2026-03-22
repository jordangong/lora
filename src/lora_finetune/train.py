"""Main training script for LoRA finetuning."""

import logging
import sys
from typing import Any

from . import train_modes as imported_train_modes
from . import train_runtime as imported_train_runtime
from .cli import build_config, parse_args
from .config import BenchmarkEvalConfig, Config
from .utils import (
    console,
    get_method_display_name,
    get_warning_handler,
    set_seed,
    setup_logging,
    suppress_warnings,
)
from .utils import (
    print_model_size as imported_print_model_size,
)

logger = logging.getLogger(__name__)

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

print_model_size = imported_print_model_size

hf_set_seed = None
get_text_collator = None
get_text_classification_collator = None
load_text_dataset = None
prepare_grpo_dataset_for_trl = None
prepare_preference_dataset_for_trl = None
prepare_text_dataset_for_trl = None
preprocess_text_dataset = None
preprocess_text_classification_dataset = None
requires_trl_native_dataset = None
get_vision_collator = None
load_vision_dataset = None
preprocess_vision_dataset = None
LightEvalCallback = None
run_lighteval = None
get_peft_model_with_lora = None
load_model_and_tokenizer = None
get_llm_target_modules = None
get_text_target_modules = None
get_num_labels_from_dataset = None
get_id2label = None
get_label2id = None
get_vision_target_modules = None
compute_metrics_for_classification = None
compute_metrics_for_lm = None
create_trainer = None
prepare_model_for_training = None
run_hyperparameter_search = None
apply_hpo_parameters_to_config_sections = None
imported_train_support = None


def _ensure_runtime_imports():
    imported_train_runtime.ensure_runtime_imports(sys.modules[__name__])


def _get_train_support():
    global imported_train_support

    if imported_train_support is None:
        from . import train_support as imported_train_support_module

        imported_train_support = imported_train_support_module
    return imported_train_support


def run_benchmark_eval(
    model,
    model_name: str,
    eval_config: BenchmarkEvalConfig,
    trainer=None,
) -> None:
    """Run benchmark evaluation after training using lighteval."""
    _get_train_support().run_benchmark_eval(
        sys.modules[__name__],
        model,
        model_name,
        eval_config,
        trainer=trainer,
    )


def _cleanup_trainer_callbacks(trainer) -> None:
    _get_train_support().cleanup_trainer_callbacks(trainer)


def _run_final_trainer_evaluation(trainer):
    return _get_train_support().run_final_trainer_evaluation(console, trainer)


def _should_run_final_benchmark_eval(trainer, eval_config: BenchmarkEvalConfig) -> bool:
    return _get_train_support().should_run_final_benchmark_eval(trainer, eval_config)


def _run_trainer_training(
    trainer,
    resume_from_checkpoint=None,
    final_evaluation_enabled=True,
    final_evaluation_fn=None,
) -> None:
    if final_evaluation_fn is None:
        final_evaluation_fn = _run_final_trainer_evaluation

    _get_train_support().run_trainer_training(
        trainer,
        cleanup_trainer_callbacks_fn=_cleanup_trainer_callbacks,
        resume_from_checkpoint=resume_from_checkpoint,
        final_evaluation_fn=final_evaluation_fn,
        final_evaluation_enabled=final_evaluation_enabled,
    )


def _run_trainer_hpo(trainer, config: Config):
    return _get_train_support().run_trainer_hpo(
        trainer,
        config,
        run_hyperparameter_search_fn=run_hyperparameter_search,
        cleanup_trainer_callbacks_fn=_cleanup_trainer_callbacks,
    )


def _display_hpo_best_run(best_run: Any) -> None:
    _get_train_support().display_hpo_best_run(console, best_run)


def _save_hpo_best_config(config: Config, best_run: Any) -> None:
    _get_train_support().save_hpo_best_config(sys.modules[__name__], config, best_run)


def _run_with_status(status_message: str, fn):
    return _get_train_support().run_with_status(
        get_warning_handler,
        console,
        status_message,
        fn,
    )


def _resolve_default_target_modules(config: Config, resolver) -> None:
    _get_train_support().resolve_default_target_modules(config, resolver)


def _get_train_and_eval_datasets(dataset, data_config):
    return _get_train_support().get_train_and_eval_datasets(dataset, data_config)


def _get_hpo_config_sections(config: Config) -> dict[str, Any]:
    return _get_train_support().get_hpo_config_sections(config)


def _get_hpo_setup(config: Config, build_model):
    return _get_train_support().get_hpo_setup(
        config,
        build_model,
        _get_hpo_config_sections,
    )


def _run_hpo_if_enabled(trainer, config: Config) -> bool:
    return _get_train_support().run_hpo_if_enabled(
        trainer,
        config,
        _run_trainer_hpo,
        _display_hpo_best_run,
        _save_hpo_best_config,
    )


def _save_and_maybe_push_model(trainer, config: Config) -> None:
    _get_train_support().save_and_maybe_push_model(console, trainer, config)


def train_llm(config: Config) -> None:
    """Train a language model with LoRA."""
    if config.model.use_unsloth:
        from ._optional_unsloth import ensure_unsloth_imported

        ensure_unsloth_imported()
    _ensure_runtime_imports()
    imported_train_modes.train_llm(config, deps=sys.modules[__name__], logger=logger)


def train_vision(config: Config) -> None:
    """Train a vision model with LoRA."""
    _ensure_runtime_imports()
    imported_train_modes.train_vision(config, deps=sys.modules[__name__])


def train_text_classification(config: Config) -> None:
    _ensure_runtime_imports()
    imported_train_modes.train_text_classification(config, deps=sys.modules[__name__])


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

    console.print(
        _get_train_support().build_configuration_panel(
            config,
            get_method_display_name,
        )
    )

    if config.model.model_type == "vision":
        train_vision(config)
    elif config.model.model_type == "text_classification":
        train_text_classification(config)
    else:
        train_llm(config)


if __name__ == "__main__":
    main()
