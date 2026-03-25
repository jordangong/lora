"""Tests for benchmark evaluators using lighteval."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from lora_finetune.evaluators import LightEvalCallback, lighteval_evaluator, run_lighteval


def _mock_lighteval_components(mock_pipeline):
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)
    mock_tracker_cls = MagicMock()
    mock_generation_parameters_cls = MagicMock(side_effect=lambda **kwargs: kwargs)
    mock_model_config_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    mock_pipeline_parameters_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    mock_transformers_module = SimpleNamespace(tqdm=MagicMock())

    components = {
        "EvaluationTracker": mock_tracker_cls,
        "GenerationParameters": mock_generation_parameters_cls,
        "TransformersModelConfig": mock_model_config_cls,
        "ParallelismManager": SimpleNamespace(NONE="none"),
        "Pipeline": mock_pipeline_cls,
        "PipelineParameters": mock_pipeline_parameters_cls,
        "transformers_model_module": mock_transformers_module,
    }

    return components, mock_pipeline_cls


class TestRunLighteval:
    """Tests for run_lighteval wrapper function."""

    def test_run_lighteval_creates_pipeline_with_correct_args(self):
        """Test that run_lighteval creates Pipeline with correct configuration."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42}}
        }
        original_cleanup = MagicMock()
        mock_pipeline.model = SimpleNamespace(
            model=MagicMock(), cleanup=original_cleanup, _tokenizer=object()
        )
        components, mock_pipeline_cls = _mock_lighteval_components(mock_pipeline)

        model = MagicMock()
        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
            return_value=components,
        ):
            metrics = run_lighteval(
                model=model,
                model_name="meta-llama/Llama-3.1-8B",
                tasks="gsm8k",
                max_samples=10,
                batch_size=2,
                max_new_tokens=256,
            )

        # Verify Pipeline was called with the model
        mock_pipeline_cls.assert_called_once()
        call_kwargs = mock_pipeline_cls.call_args
        assert call_kwargs.kwargs["tasks"] == "gsm8k"
        assert call_kwargs.kwargs["model"] is model

        # Verify evaluate was called
        mock_pipeline.evaluate.assert_called_once()

        # Verify metrics are returned
        assert metrics["gsm8k_0|expr_gold_metric"] == 0.42

        # Verify cleanup was monkey-patched to no-op
        mock_pipeline.model.cleanup()  # should not raise
        original_cleanup.assert_called_once()

    def test_run_lighteval_extracts_metrics(self):
        """Test that run_lighteval correctly extracts metrics from results."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42, "some_other_metric": 0.99}}
        }
        components, _ = _mock_lighteval_components(mock_pipeline)

        model = MagicMock()
        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
            return_value=components,
        ):
            metrics = run_lighteval(
                model=model,
                model_name="test-model",
                tasks="gsm8k",
            )

        assert "gsm8k_0|expr_gold_metric" in metrics
        assert metrics["gsm8k_0|expr_gold_metric"] == 0.42
        assert "gsm8k_0|some_other_metric" in metrics
        assert metrics["gsm8k_0|some_other_metric"] == 0.99

    def test_run_lighteval_returns_empty_on_none_results(self):
        """Test that run_lighteval returns empty dict when results are None."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = None
        components, _ = _mock_lighteval_components(mock_pipeline)

        model = MagicMock()
        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
            return_value=components,
        ):
            metrics = run_lighteval(model=model, model_name="test-model", tasks="gsm8k")

        assert metrics == {}

    def test_run_lighteval_multiple_tasks(self):
        """Test that run_lighteval handles multiple tasks."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {
                "gsm8k_0": {"expr_gold_metric": 0.42},
                "mmlu_0": {"accuracy": 0.65},
            }
        }
        components, _ = _mock_lighteval_components(mock_pipeline)

        model = MagicMock()
        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
            return_value=components,
        ):
            metrics = run_lighteval(
                model=model,
                model_name="test-model",
                tasks="gsm8k,mmlu",
            )

        assert "gsm8k_0|expr_gold_metric" in metrics
        assert "mmlu_0|accuracy" in metrics

    def test_run_lighteval_uses_safe_model_name_for_local_paths(self, tmp_path):
        """Test that local checkpoint paths do not leak into lighteval cache directories."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42}}
        }
        components, _ = _mock_lighteval_components(mock_pipeline)

        model = MagicMock()
        model_path = tmp_path / "checkpoints" / "Mistral-7B-v0.1"
        model_path.mkdir(parents=True)

        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
            return_value=components,
        ):
            run_lighteval(
                model=model,
                model_name=str(model_path),
                tasks="gsm8k",
            )

        model_config_kwargs = components["TransformersModelConfig"].call_args.kwargs
        assert model_config_kwargs["model_name"] == "Mistral-7B-v0.1"
        assert model_config_kwargs["tokenizer"] == str(model_path)
        assert model_config_kwargs["cache_dir"]

    def test_run_lighteval_normalizes_unsloth_layer_device_indices(self):
        """Test that benchmark eval repairs missing Unsloth per-layer device indices."""

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self._per_layer_device_index = None

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = FakeLayer()

        wrapped_model = FakeModel()
        input_model = FakeModel()
        mock_pipeline = MagicMock()
        mock_pipeline.model = SimpleNamespace(model=wrapped_model, cleanup=lambda: None)
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42}}
        }
        components, _ = _mock_lighteval_components(mock_pipeline)

        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
            return_value=components,
        ):
            run_lighteval(
                model=input_model,
                model_name="test-model",
                tasks="gsm8k",
            )

        assert input_model.layer._per_layer_device_index == "cpu"
        assert wrapped_model.layer._per_layer_device_index == "cpu"

    def test_run_lighteval_restores_use_cache_state_and_releases_memory(self):
        input_model = torch.nn.Linear(1, 1)
        input_model.config = SimpleNamespace(use_cache=False)
        input_model.generation_config = SimpleNamespace(use_cache=False)

        wrapped_model = torch.nn.Linear(1, 1)
        wrapped_model.config = SimpleNamespace(use_cache=False)
        wrapped_model.generation_config = SimpleNamespace(use_cache=False)

        mock_pipeline = MagicMock()
        mock_pipeline.model = SimpleNamespace(model=wrapped_model, cleanup=lambda: None)
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42}}
        }

        def mutate_cache_state():
            input_model.config.use_cache = True
            input_model.generation_config.use_cache = True
            wrapped_model.config.use_cache = True
            wrapped_model.generation_config.use_cache = True

        mock_pipeline.evaluate.side_effect = mutate_cache_state
        components, _ = _mock_lighteval_components(mock_pipeline)
        release_calls = []

        with (
            patch(
                "lora_finetune.evaluators.lighteval_evaluator._import_lighteval_components",
                return_value=components,
            ),
            patch(
                "lora_finetune.evaluators.lighteval_evaluator._release_gpu_memory",
                side_effect=lambda: release_calls.append("released"),
            ),
        ):
            metrics = run_lighteval(
                model=input_model,
                model_name="test-model",
                tasks="gsm8k",
            )

        assert metrics == {"gsm8k_0|expr_gold_metric": 0.42}
        assert input_model.config.use_cache is False
        assert input_model.generation_config.use_cache is False
        assert wrapped_model.config.use_cache is False
        assert wrapped_model.generation_config.use_cache is False
        assert release_calls == ["released"]


class TestLightEvalLoggingHelpers:
    """Tests for lighteval logging setup/restore helpers."""

    def test_restore_preserves_logger_propagate(self):
        """Ensure temporary warning routing does not mutate propagate permanently."""
        logger = logging.getLogger("lighteval.test_logger")
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate

        try:
            logger.propagate = False
            handler = logging.NullHandler()

            saved = lighteval_evaluator.configure_warning_loggers(["lighteval"], handler)
            assert logger.propagate is False

            lighteval_evaluator.restore_logger_configuration(saved)
            assert logger.propagate is False
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate


class TestLightEvalCallback:
    """Tests for LightEvalCallback class."""

    def test_callback_initialization(self):
        """Test LightEvalCallback initialization."""
        callback = LightEvalCallback(
            model_name="test-model",
            tasks="gsm8k",
            eval_steps=100,
            max_samples=50,
            max_new_tokens=256,
            batch_size=4,
        )

        assert callback.model_name == "test-model"
        assert callback.tasks == "gsm8k"
        assert callback.eval_steps == 100
        assert callback.max_samples == 50
        assert callback.max_new_tokens == 256
        assert callback.batch_size == 4

    def test_callback_default_values(self):
        """Test LightEvalCallback default values."""
        callback = LightEvalCallback(model_name="test-model")

        assert callback.tasks == "gsm8k"
        assert callback.eval_steps == 500
        assert callback.max_samples == 100
        assert callback.max_new_tokens == 512
        assert callback.batch_size == 1

    def test_callback_rejects_non_positive_eval_steps(self):
        """Test LightEvalCallback rejects invalid eval_steps."""
        with pytest.raises(ValueError, match="greater than 0"):
            LightEvalCallback(model_name="test-model", eval_steps=0)

    def test_callback_has_required_methods(self):
        """Test callback has TrainerCallback methods."""
        callback = LightEvalCallback(model_name="test-model")

        assert hasattr(callback, "on_step_end")
        assert callable(callback.on_step_end)

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_skips_non_eval_steps(self, mock_run):
        """Test that on_step_end skips when not at eval_steps interval."""
        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 50  # Not a multiple of 100
        control = MagicMock()
        model = MagicMock()

        callback.on_step_end(args, state, control, model=model)
        mock_run.assert_not_called()

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_skips_step_zero(self, mock_run):
        """Test that on_step_end skips at step 0."""
        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 0
        control = MagicMock()
        model = MagicMock()

        callback.on_step_end(args, state, control, model=model)
        mock_run.assert_not_called()

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_skips_when_no_model(self, mock_run):
        """Test that on_step_end skips when model is None."""
        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        control = MagicMock()

        callback.on_step_end(args, state, control, model=None)
        mock_run.assert_not_called()

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_runs_at_eval_steps(self, mock_run):
        """Test that on_step_end runs eval at correct intervals."""
        mock_run.return_value = {"gsm8k_0|expr_gold_metric": 0.42}

        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        model = MagicMock()
        model.training = True

        callback.on_step_end(args, state, control, model=model)
        mock_run.assert_called_once()

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_restores_training_state(self, mock_run):
        """Test that training state is restored after eval."""
        mock_run.return_value = {"gsm8k_0|expr_gold_metric": 0.42}

        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        model = MagicMock()
        model.training = True

        # Ensure grad is enabled before
        torch.set_grad_enabled(True)

        callback.on_step_end(args, state, control, model=model)

        # Training state should be restored
        model.train.assert_called_with(True)
        assert torch.is_grad_enabled()

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_restores_unsloth_training_state(self, mock_run):
        """Test that Unsloth models are restored via for_training after eval."""
        mock_run.return_value = {"gsm8k_0|expr_gold_metric": 0.42}

        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()

        model = torch.nn.Linear(1, 1)
        model.training = True
        model.gradient_checkpointing = True
        model.for_training = MagicMock()
        model.train = MagicMock()

        with patch(
            "lora_finetune.evaluators.lighteval_evaluator._release_gpu_memory"
        ) as mock_release:
            torch.set_grad_enabled(True)
            callback.on_step_end(args, state, control, model=model)

        model.for_training.assert_called_once_with(use_gradient_checkpointing=True)
        model.train.assert_not_called()
        mock_release.assert_called_once()
        assert torch.is_grad_enabled()

    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_restores_state_on_error(self, mock_run):
        """Test that training state is restored even if eval fails."""
        mock_run.side_effect = RuntimeError("eval failed")

        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        model = MagicMock()
        model.training = True

        torch.set_grad_enabled(True)

        try:
            callback.on_step_end(args, state, control, model=model)
        except RuntimeError:
            pass

        # Training state should still be restored
        model.train.assert_called_with(True)
        assert torch.is_grad_enabled()

    @patch("wandb.log")
    @patch("wandb.run", new_callable=lambda: MagicMock)
    @patch("lora_finetune.evaluators.lighteval_evaluator.run_lighteval")
    def test_on_step_end_logs_metrics_to_wandb(self, mock_run, _mock_run_attr, mock_wandb_log):
        """Test that metrics are logged directly to wandb."""
        mock_run.return_value = {"gsm8k_0|expr_gold_metric": 0.42}

        callback = LightEvalCallback(model_name="test-model", eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.0
        control = MagicMock()
        model = MagicMock()
        model.training = True

        callback.on_step_end(args, state, control, model=model)

        mock_wandb_log.assert_called_once_with(
            {"benchmark/gsm8k_0|expr_gold_metric": 0.42, "train/global_step": 100},
            step=100,
        )
