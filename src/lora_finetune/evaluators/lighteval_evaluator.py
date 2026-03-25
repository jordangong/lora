"""Benchmark evaluation using HuggingFace lighteval."""

import gc
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from ..utils import (
    RichWarningHandler,
    WarningRule,
    configure_warning_loggers,
    console,
    restore_logger_configuration,
)

logger = logging.getLogger(__name__)

_MISSING = object()


LIGHTEVAL_WARNING_RULES = (
    WarningRule(
        contains_any=("max_samples WAS SET",),
        replacement="Benchmark eval using partial samples (--max_samples set)",
        logger_names=("lighteval",),
    ),
    WarningRule(
        contains_any=("cannot select the number of dataset splits",),
        replacement="Auto-selecting dataset splits for generative evaluation",
        logger_names=("lighteval",),
    ),
)


class _RichTqdmBridge:
    """A tqdm-compatible wrapper that redirects progress to a Rich progress task."""

    def __init__(self, iterable=None, *, progress=None, task_label="", **kwargs):
        self._iterable = iterable
        self._progress = progress
        self._task_label = task_label
        self._task_id = None
        self._total = kwargs.get("total") or (
            len(iterable) if iterable is not None and hasattr(iterable, "__len__") else None
        )
        # Use the caller's desc to pick an appropriate label:
        # - outer splits loop → show progress_label (e.g. "Benchmark eval (gsm8k,mmlu)")
        # - inner generation loop → show "Generating"
        self._desc = kwargs.get("desc", "")
        self._show_progress = self._progress is not None
        if "generation" in self._desc.lower():
            self._display_label = "Generating"
        else:
            self._display_label = self._task_label

    def __iter__(self):
        if self._show_progress and self._total is not None:
            self._task_id = self._progress.add_task(
                f"[yellow]{self._display_label}[/yellow]",
                total=self._total,
            )
        try:
            for item in self._iterable:
                yield item
                if self._task_id is not None:
                    self._progress.advance(self._task_id)
        finally:
            if self._task_id is not None:
                self._progress.remove_task(self._task_id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _import_lighteval_components() -> Dict[str, Any]:
    """Import lighteval modules lazily so callers can mock this in tests."""
    import lighteval.models.transformers.transformers_model as transformers_model_module
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    return {
        "EvaluationTracker": EvaluationTracker,
        "GenerationParameters": GenerationParameters,
        "TransformersModelConfig": TransformersModelConfig,
        "ParallelismManager": ParallelismManager,
        "Pipeline": Pipeline,
        "PipelineParameters": PipelineParameters,
        "transformers_model_module": transformers_model_module,
    }


def _get_module_device(module) -> Optional[torch.device]:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return None


def _normalize_unsloth_target_device(device) -> Optional[int | str]:
    if device is None:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return "cpu"
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        if device == "cuda":
            if torch.cuda.is_available():
                return torch.cuda.current_device()
            return "cpu"
        return device
    if isinstance(device, torch.device):
        if device.type == "cuda":
            if device.index is not None:
                return device.index
            if torch.cuda.is_available():
                return torch.cuda.current_device()
            return "cpu"
        if device.type == "cpu":
            return "cpu"
        return str(device)
    return device


def _normalize_unsloth_layer_device_indices(model) -> None:
    fallback_device = _normalize_unsloth_target_device(_get_module_device(model))
    for module in model.modules():
        if not hasattr(module, "_per_layer_device_index"):
            continue
        if getattr(module, "_per_layer_device_index") is not None:
            continue
        module_device = _normalize_unsloth_target_device(_get_module_device(module))
        setattr(module, "_per_layer_device_index", module_device or fallback_device)


def _capture_use_cache_state(model) -> Dict[str, Any]:
    state = {
        "config_use_cache": _MISSING,
        "generation_config_use_cache": _MISSING,
    }
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        state["config_use_cache"] = config.use_cache
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        state["generation_config_use_cache"] = generation_config.use_cache
    return state


def _restore_use_cache_state(model, state: Dict[str, Any]) -> None:
    if state is None:
        return

    config = getattr(model, "config", None)
    if config is not None and state.get("config_use_cache", _MISSING) is not _MISSING:
        config.use_cache = state["config_use_cache"]

    generation_config = getattr(model, "generation_config", None)
    if (
        generation_config is not None
        and state.get("generation_config_use_cache", _MISSING) is not _MISSING
    ):
        generation_config.use_cache = state["generation_config_use_cache"]


def _get_instance_callable(model, name: str):
    model_dict = getattr(model, "__dict__", None)
    if not isinstance(model_dict, dict):
        return None
    value = model_dict.get(name)
    if callable(value):
        return value
    return None


def _is_gradient_checkpointing_enabled(model) -> bool:
    for attr_name in ("is_gradient_checkpointing", "gradient_checkpointing"):
        try:
            value = getattr(model, attr_name)
        except Exception:
            continue
        if isinstance(value, bool):
            return value

    try:
        for module in model.modules():
            value = getattr(module, "gradient_checkpointing", None)
            if isinstance(value, bool) and value:
                return True
    except Exception:
        pass

    return False


def _restore_training_state(
    model, was_training: bool, gradient_checkpointing_enabled: bool
) -> None:
    if was_training:
        for_training = _get_instance_callable(model, "for_training")
        if for_training is not None:
            for_training(use_gradient_checkpointing=gradient_checkpointing_enabled)
            return
        model.train(True)
        return

    for_inference = _get_instance_callable(model, "for_inference")
    if for_inference is not None:
        for_inference()
        return
    model.train(False)


def _release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _capture_generation_limit_state(model) -> Dict[str, Any]:
    state = {
        "generation_config_max_length": _MISSING,
        "generation_config_max_new_tokens": _MISSING,
    }
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "max_length"):
        state["generation_config_max_length"] = generation_config.max_length
    if generation_config is not None and hasattr(generation_config, "max_new_tokens"):
        state["generation_config_max_new_tokens"] = generation_config.max_new_tokens
    return state


def _restore_generation_limit_state(model, state: Dict[str, Any]) -> None:
    if state is None:
        return

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    if state.get("generation_config_max_length", _MISSING) is not _MISSING and hasattr(
        generation_config, "max_length"
    ):
        generation_config.max_length = state["generation_config_max_length"]
    if state.get("generation_config_max_new_tokens", _MISSING) is not _MISSING and hasattr(
        generation_config, "max_new_tokens"
    ):
        generation_config.max_new_tokens = state["generation_config_max_new_tokens"]


def _apply_generation_limit_override(model, max_new_tokens: int) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    if hasattr(generation_config, "max_length"):
        generation_config.max_length = None
    if hasattr(generation_config, "max_new_tokens"):
        generation_config.max_new_tokens = max_new_tokens


def _override_pipeline_generation_limits(pipeline, max_new_tokens: int) -> None:
    pipeline_model = getattr(pipeline, "model", None)
    generation_config_dict = getattr(pipeline_model, "generation_config_dict", None)
    if isinstance(generation_config_dict, dict):
        generation_config_dict["max_new_tokens"] = max_new_tokens
        generation_config_dict.pop("max_length", None)

    tasks_dict = getattr(pipeline, "tasks_dict", None)
    if isinstance(tasks_dict, dict):
        for task in tasks_dict.values():
            if hasattr(task, "generation_size"):
                task.generation_size = max_new_tokens

    documents_dict = getattr(pipeline, "documents_dict", None)
    if isinstance(documents_dict, dict):
        for docs in documents_dict.values():
            for doc in docs:
                if hasattr(doc, "generation_size"):
                    doc.generation_size = max_new_tokens


def _patch_transformers_attention_mask_compat() -> None:
    try:
        import transformers.masking_utils as masking_utils
        import transformers.modeling_attn_mask_utils as legacy_masking_utils
        from transformers.utils.import_utils import is_tracing
    except ImportError:
        return

    if getattr(legacy_masking_utils, "_lora_finetune_compat_installed", False):
        return

    def _cache_position(query_length: int, key_value_length: int, device) -> torch.Tensor:
        return torch.arange(
            key_value_length - query_length,
            key_value_length,
            device=device,
            dtype=torch.long,
        )

    def _mask_function(is_causal: bool, sliding_window: Optional[int]):
        if not is_causal:
            return masking_utils.bidirectional_mask_function
        if sliding_window is None:
            return masking_utils.causal_mask_function
        return masking_utils.sliding_window_causal_mask_function(sliding_window)

    class AttentionMaskConverter:
        def __init__(self, is_causal: bool, sliding_window: int | None = None):
            self.is_causal = is_causal
            self.sliding_window = sliding_window
            if self.sliding_window is not None and self.sliding_window <= 0:
                raise ValueError(
                    "Make sure that when passing `sliding_window` that its value is a strictly positive integer, "
                    f"not `{self.sliding_window}`"
                )

        def to_causal_4d(
            self,
            batch_size: int,
            query_length: int,
            key_value_length: int,
            dtype: torch.dtype,
            device: torch.device | str = "cpu",
        ) -> torch.Tensor | None:
            if not self.is_causal:
                raise ValueError(
                    f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True."
                )
            if query_length <= 1 and self.sliding_window is None:
                return None
            return masking_utils.eager_mask(
                batch_size=batch_size,
                cache_position=_cache_position(query_length, key_value_length, device),
                kv_length=key_value_length,
                mask_function=_mask_function(True, self.sliding_window),
                dtype=dtype,
                allow_is_bidirectional_skip=False,
                use_vmap=False,
            )

        def to_4d(
            self,
            attention_mask_2d: torch.Tensor,
            query_length: int,
            dtype: torch.dtype,
            key_value_length: int | None = None,
        ) -> torch.Tensor:
            if self.is_causal and key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly "
                    "create a causal mask."
                )
            if not self.is_causal and self.sliding_window is not None:
                raise NotImplementedError(
                    "Sliding window is currently only implemented for causal masking"
                )
            key_value_length = (
                key_value_length if key_value_length is not None else attention_mask_2d.shape[-1]
            )
            return masking_utils.eager_mask(
                batch_size=attention_mask_2d.shape[0],
                cache_position=_cache_position(
                    query_length, key_value_length, attention_mask_2d.device
                ),
                kv_length=key_value_length,
                mask_function=_mask_function(self.is_causal, self.sliding_window),
                attention_mask=attention_mask_2d,
                dtype=dtype,
                allow_is_bidirectional_skip=False,
                use_vmap=False,
            )

        @staticmethod
        def _make_causal_mask(
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
            past_key_values_length: int = 0,
            sliding_window: int | None = None,
        ):
            batch_size, query_length = input_ids_shape
            key_value_length = query_length + past_key_values_length
            if query_length <= 1 and sliding_window is None:
                return None
            return masking_utils.eager_mask(
                batch_size=batch_size,
                cache_position=_cache_position(query_length, key_value_length, device),
                kv_length=key_value_length,
                mask_function=_mask_function(True, sliding_window),
                dtype=dtype,
                allow_is_bidirectional_skip=False,
                use_vmap=False,
            )

        @staticmethod
        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
            tgt_len = tgt_len if tgt_len is not None else mask.shape[-1]
            return masking_utils.eager_mask(
                batch_size=mask.shape[0],
                cache_position=torch.arange(tgt_len, device=mask.device, dtype=torch.long),
                kv_length=mask.shape[-1],
                mask_function=masking_utils.bidirectional_mask_function,
                attention_mask=mask,
                dtype=dtype,
                allow_is_bidirectional_skip=False,
                use_vmap=False,
            )

        @staticmethod
        def _unmask_unattended(expanded_mask: torch.FloatTensor, min_dtype: float):
            if expanded_mask.dtype == torch.bool:
                raise ValueError(
                    "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
                )
            return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))

        @staticmethod
        def _ignore_causal_mask_sdpa(
            attention_mask: torch.Tensor | None,
            inputs_embeds: torch.Tensor,
            past_key_values_length: int,
            sliding_window: int | None = None,
            is_training: bool = False,
        ) -> bool:
            del is_training
            if attention_mask is not None and attention_mask.dim() == 4:
                return False
            query_length = inputs_embeds.shape[1]
            key_value_length = query_length + past_key_values_length
            return masking_utils._ignore_causal_mask_sdpa(
                attention_mask,
                query_length,
                key_value_length,
                past_key_values_length,
                sliding_window,
            )

    def _prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask: torch.Tensor | None,
        input_shape: torch.Size | tuple | list,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: int | None = None,
    ):
        attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
        key_value_length = input_shape[-1] + past_key_values_length
        is_tracing_ = is_tracing(inputs_embeds)
        ignore_causal_mask = AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )
        if ignore_causal_mask:
            expanded_4d_mask = None
        elif attention_mask is None:
            expanded_4d_mask = attn_mask_converter.to_causal_4d(
                input_shape[0],
                input_shape[-1],
                key_value_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        else:
            if attention_mask.dim() == 4:
                expanded_4d_mask = attention_mask
            else:
                expanded_4d_mask = attn_mask_converter.to_4d(
                    attention_mask,
                    input_shape[-1],
                    dtype=inputs_embeds.dtype,
                    key_value_length=key_value_length,
                )
            if not is_tracing_ and expanded_4d_mask.device.type in ["cuda", "xpu"]:
                expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                    expanded_4d_mask,
                    min_dtype=torch.finfo(inputs_embeds.dtype).min,
                )
        return expanded_4d_mask

    def _prepare_4d_attention_mask_for_sdpa(
        mask: torch.Tensor,
        dtype: torch.dtype,
        tgt_len: int | None = None,
    ):
        if not is_tracing(mask) and torch.all(mask == 1):
            return None
        return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)

    legacy_masking_utils.AttentionMaskConverter = AttentionMaskConverter
    legacy_masking_utils._prepare_4d_causal_attention_mask_for_sdpa = (
        _prepare_4d_causal_attention_mask_for_sdpa
    )
    legacy_masking_utils._prepare_4d_attention_mask_for_sdpa = _prepare_4d_attention_mask_for_sdpa
    legacy_masking_utils._lora_finetune_compat_installed = True

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("unsloth.models"):
            continue
        if hasattr(module, "AttentionMaskConverter"):
            setattr(module, "AttentionMaskConverter", AttentionMaskConverter)
        if hasattr(module, "_prepare_4d_causal_attention_mask_for_sdpa"):
            setattr(
                module,
                "_prepare_4d_causal_attention_mask_for_sdpa",
                _prepare_4d_causal_attention_mask_for_sdpa,
            )
        if hasattr(module, "_prepare_4d_attention_mask_for_sdpa"):
            setattr(
                module, "_prepare_4d_attention_mask_for_sdpa", _prepare_4d_attention_mask_for_sdpa
            )


def run_lighteval(
    model: PreTrainedModel,
    model_name: str,
    tasks: str,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    rich_progress: Optional[Any] = None,
    progress_label: str = "Benchmark eval",
) -> Dict[str, Any]:
    """Run lighteval benchmark evaluation on a pre-loaded model.

    Args:
        model: Pre-loaded HuggingFace model to evaluate.
        model_name: HuggingFace model name or path (used for tokenizer loading).
        tasks: Comma-separated lighteval task string (e.g. "gsm8k", "gsm8k,mmlu").
        max_samples: Maximum number of samples to evaluate per task (None for all).
        batch_size: Batch size for evaluation.
        max_new_tokens: Maximum new tokens to generate.
        rich_progress: Optional Rich Progress instance to show eval progress.
        progress_label: Label for the Rich progress task.

    Returns:
        Dictionary mapping metric names to values, e.g.
        {"gsm8k_0|expr_gold_metric": 0.42, ...}
    """
    components = _import_lighteval_components()
    evaluation_tracker_cls = components["EvaluationTracker"]
    generation_parameters_cls = components["GenerationParameters"]
    transformers_model_config_cls = components["TransformersModelConfig"]
    parallelism_manager = components["ParallelismManager"]
    pipeline_cls = components["Pipeline"]
    pipeline_parameters_cls = components["PipelineParameters"]
    transformers_model_module = components["transformers_model_module"]
    model_use_cache_state = _capture_use_cache_state(model)
    model_generation_limit_state = _capture_generation_limit_state(model)

    _patch_transformers_attention_mask_compat()
    _normalize_unsloth_layer_device_indices(model)
    _apply_generation_limit_override(model, max_new_tokens)

    pipeline_params = pipeline_parameters_cls(
        launcher_type=parallelism_manager.NONE,
        max_samples=max_samples,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        lighteval_model_name = model_name
        tokenizer_name = None
        expanded_model_name = os.path.expanduser(model_name)
        if os.path.isabs(expanded_model_name) or os.path.exists(expanded_model_name):
            lighteval_model_name = (
                os.path.basename(os.path.normpath(expanded_model_name)) or "model"
            )
            tokenizer_name = model_name

        model_config = transformers_model_config_cls(
            model_name=lighteval_model_name,
            tokenizer=tokenizer_name,
            batch_size=batch_size,
            generation_parameters=generation_parameters_cls(
                max_new_tokens=max_new_tokens,
            ),
            cache_dir=tmpdir,
        )

        # Use tmpdir as cache_dir so each evaluation gets a fresh SampleCache.
        # lighteval caches predictions to disk keyed by model config (not weights),
        # which would return stale results during mid-training evaluation.
        evaluation_tracker = evaluation_tracker_cls(
            output_dir=tmpdir,
            save_details=False,
            push_to_hub=False,
        )

        # Route lighteval warnings through our Rich handler.
        # Use rich_progress.console if available so prints don't duplicate the live bar.
        warn_console = rich_progress.console if rich_progress is not None else console
        warning_handler = RichWarningHandler(warn_console, extra_rules=LIGHTEVAL_WARNING_RULES)
        saved_logging = configure_warning_loggers(["lighteval"], warning_handler)
        _orig_tqdm = None
        metrics = {}
        pipeline = None
        pipeline_model = None
        pipeline_root_model = None
        pipeline_root_model_use_cache_state = None
        pipeline_root_model_generation_limit_state = None

        try:
            pipeline = pipeline_cls(
                tasks=tasks,
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model_config=model_config,
                model=model,
            )
            pipeline_model = getattr(pipeline, "model", None)
            pipeline_root_model = getattr(pipeline_model, "model", None)
            if pipeline_root_model is not None:
                _normalize_unsloth_layer_device_indices(pipeline_root_model)
                if pipeline_root_model is not model:
                    pipeline_root_model_use_cache_state = _capture_use_cache_state(
                        pipeline_root_model
                    )
                    pipeline_root_model_generation_limit_state = _capture_generation_limit_state(
                        pipeline_root_model
                    )
                _apply_generation_limit_override(pipeline_root_model, max_new_tokens)

            _override_pipeline_generation_limits(pipeline, max_new_tokens)

            # Capture any loggers created during Pipeline init
            configure_warning_loggers(["lighteval"], warning_handler, saved=saved_logging)

            # Prevent cleanup() from deleting the model (we still need it for training)
            if callable(getattr(pipeline_model, "cleanup", None)):
                pipeline_model_ref = pipeline_model
                original_cleanup = pipeline_model.cleanup
                shared_model = getattr(pipeline_model, "model", None)

                def _cleanup_preserving_model():
                    if not hasattr(pipeline_model_ref, "_tokenizer"):
                        if shared_model is not None:
                            pipeline_model_ref.model = shared_model
                        return
                    original_cleanup()
                    if shared_model is not None:
                        pipeline_model_ref.model = shared_model

                pipeline_model_ref.cleanup = _cleanup_preserving_model

            # Replace lighteval's tqdm with our Rich bridge (or disable it)
            _orig_tqdm = transformers_model_module.tqdm
            if rich_progress is not None:
                transformers_model_module.tqdm = lambda iterable=None, **kw: _RichTqdmBridge(
                    iterable, progress=rich_progress, task_label=progress_label, **kw
                )
            else:
                transformers_model_module.tqdm = lambda iterable=None, **kw: _RichTqdmBridge(
                    iterable, **kw
                )

            pipeline.evaluate()
            raw_results = pipeline.get_results() or {}
            for task_name, task_metrics in raw_results.get("results", {}).items():
                for metric_name, value in task_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{task_name}|{metric_name}"] = value
        finally:
            restore_logger_configuration(saved_logging)
            if _orig_tqdm is not None:
                transformers_model_module.tqdm = _orig_tqdm
            if pipeline_root_model is not None and pipeline_root_model is not model:
                _restore_use_cache_state(pipeline_root_model, pipeline_root_model_use_cache_state)
                _restore_generation_limit_state(
                    pipeline_root_model, pipeline_root_model_generation_limit_state
                )
            _restore_use_cache_state(model, model_use_cache_state)
            _restore_generation_limit_state(model, model_generation_limit_state)
            pipeline_model = None
            pipeline_root_model = None
            pipeline = None
            evaluation_tracker = None
            warning_handler = None
            _release_gpu_memory()

    return metrics


class LightEvalCallback(TrainerCallback):
    """Callback to run lighteval benchmark evaluation during training."""

    def __init__(
        self,
        model_name: str,
        tasks: str = "gsm8k",
        eval_steps: int = 500,
        max_samples: Optional[int] = 100,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        rich_progress_callback: Optional[Any] = None,
    ):
        if eval_steps <= 0:
            raise ValueError("LightEvalCallback eval_steps must be greater than 0")

        self.model_name = model_name
        self.tasks = tasks
        self.eval_steps = eval_steps
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.rich_progress_callback = rich_progress_callback

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        **kwargs,
    ):
        """Run lighteval benchmark evaluation at specified intervals."""
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        if model is None:
            return

        logger.info(f"Running lighteval benchmark evaluation at step {state.global_step}")

        # Pass Rich progress so eval batches show real progress
        rich_progress = None
        if self.rich_progress_callback and self.rich_progress_callback.progress:
            rich_progress = self.rich_progress_callback.progress

        # Save training state (lighteval sets model.eval() and disables gradients)
        was_training = model.training
        grad_enabled = torch.is_grad_enabled()
        gradient_checkpointing_enabled = _is_gradient_checkpointing_enabled(model)

        try:
            metrics = run_lighteval(
                model=model,
                model_name=self.model_name,
                tasks=self.tasks,
                max_samples=self.max_samples,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                rich_progress=rich_progress,
                progress_label=f"Benchmark eval ({self.tasks})",
            )
        finally:
            _restore_training_state(model, was_training, gradient_checkpointing_enabled)
            torch.set_grad_enabled(grad_enabled)
            _release_gpu_memory()

        # Log metrics to wandb directly (trainer.log() adds unwanted "train/" prefix)
        prefixed_metrics = {f"benchmark/{k}": v for k, v in metrics.items()}
        prefixed_metrics["train/global_step"] = state.global_step
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(prefixed_metrics, step=state.global_step)
        except ImportError:
            pass

        # Print results to console
        epoch = state.epoch or 0
        metrics_str = "  ".join(f"[magenta]{k}[/magenta]={v:.4f}" for k, v in metrics.items())
        if self.rich_progress_callback and self.rich_progress_callback.progress:
            self.rich_progress_callback.progress.console.print(
                f"  [bold]Benchmark[/bold] @ epoch {epoch:.2f}: {metrics_str}"
            )

        logger.info(
            f"Benchmark eval @ epoch {epoch:.2f}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
