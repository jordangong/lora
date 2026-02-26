"""Benchmark evaluation using HuggingFace lighteval."""

import logging
import tempfile
import warnings
from typing import Any, Dict, Optional

import torch
from rich.console import Console
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)
console = Console()


class _LightEvalWarningHandler(logging.Handler):
    """Logging handler that formats lighteval warnings elegantly with Rich."""

    def __init__(self, rich_console: Console):
        super().__init__()
        self._console = rich_console

    def emit(self, record):
        msg = record.getMessage()

        # Simplify common verbose lighteval warnings
        if "max_samples WAS SET" in msg:
            msg = "Benchmark eval using partial samples (--max_samples set)"
        elif "cannot select the number of dataset splits" in msg:
            msg = "Auto-selecting dataset splits for generative evaluation"
        elif not msg.strip():
            return

        self._console.print(f"  [dim yellow]⚠ {msg}[/dim yellow]")


def _setup_lighteval_logging(handler: logging.Handler) -> dict:
    """Route lighteval loggers through the given handler. Returns saved state."""
    saved = {}
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("lighteval"):
            lg = logging.getLogger(name)
            saved[name] = (lg.handlers[:], lg.level, lg.propagate)
            lg.handlers = [handler]
            lg.setLevel(logging.WARNING)
            lg.propagate = False
    return saved


def _restore_lighteval_logging(saved: dict) -> None:
    """Restore lighteval loggers to their saved state."""
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers = handlers
        lg.setLevel(level)
        lg.propagate = propagate


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

    model_config = transformers_model_config_cls(
        model_name=model_name,
        batch_size=batch_size,
        generation_parameters=generation_parameters_cls(
            max_new_tokens=max_new_tokens,
        ),
    )

    pipeline_params = pipeline_parameters_cls(
        launcher_type=parallelism_manager.NONE,
        max_samples=max_samples,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use tmpdir as cache_dir so each evaluation gets a fresh SampleCache.
        # lighteval caches predictions to disk keyed by model config (not weights),
        # which would return stale results during mid-training evaluation.
        model_config.cache_dir = tmpdir

        evaluation_tracker = evaluation_tracker_cls(
            output_dir=tmpdir,
            save_details=False,
            push_to_hub=False,
        )

        # Route lighteval warnings through our Rich handler.
        # Use rich_progress.console if available so prints don't duplicate the live bar.
        warn_console = rich_progress.console if rich_progress is not None else console
        warning_handler = _LightEvalWarningHandler(warn_console)
        saved_logging = _setup_lighteval_logging(warning_handler)
        _orig_tqdm = None
        results = None

        try:
            pipeline = pipeline_cls(
                tasks=tasks,
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model_config=model_config,
                model=model,
            )

            # Capture any loggers created during Pipeline init
            saved_logging.update(_setup_lighteval_logging(warning_handler))

            # Prevent cleanup() from deleting the model (we still need it for training)
            pipeline.model.cleanup = lambda: None

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

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
                pipeline.evaluate()

            results = pipeline.get_results()
        finally:
            _restore_lighteval_logging(saved_logging)
            if _orig_tqdm is not None:
                transformers_model_module.tqdm = _orig_tqdm

    if results is None:
        return {}

    # Extract the metrics from the results dict
    # Format: {"results": {"task_name": {"metric_name": value, ...}}, ...}
    metrics = {}
    raw_results = results.get("results", {})
    for task_name, task_metrics in raw_results.items():
        for metric_name, value in task_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"{task_name}|{metric_name}"] = value

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
            # Restore training state
            model.train(was_training)
            torch.set_grad_enabled(grad_enabled)

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
