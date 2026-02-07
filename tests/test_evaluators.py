"""Tests for benchmark evaluators using lighteval."""

from unittest.mock import MagicMock, patch

import torch

from lora_finetune.evaluators import LightEvalCallback, run_lighteval


class TestRunLighteval:
    """Tests for run_lighteval wrapper function."""

    @patch("lighteval.pipeline.Pipeline")
    @patch("lighteval.logging.evaluation_tracker.EvaluationTracker")
    def test_run_lighteval_creates_pipeline_with_correct_args(
        self, mock_tracker_cls, mock_pipeline_cls
    ):
        """Test that run_lighteval creates Pipeline with correct configuration."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42}}
        }
        mock_pipeline_cls.return_value = mock_pipeline

        model = MagicMock()
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

        # Verify cleanup was monkey-patched to no-op
        mock_pipeline.model.cleanup()  # should not raise

    @patch("lighteval.pipeline.Pipeline")
    @patch("lighteval.logging.evaluation_tracker.EvaluationTracker")
    def test_run_lighteval_extracts_metrics(self, mock_tracker_cls, mock_pipeline_cls):
        """Test that run_lighteval correctly extracts metrics from results."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {"gsm8k_0": {"expr_gold_metric": 0.42, "some_other_metric": 0.99}}
        }
        mock_pipeline_cls.return_value = mock_pipeline

        model = MagicMock()
        metrics = run_lighteval(
            model=model,
            model_name="test-model",
            tasks="gsm8k",
        )

        assert "gsm8k_0|expr_gold_metric" in metrics
        assert metrics["gsm8k_0|expr_gold_metric"] == 0.42
        assert "gsm8k_0|some_other_metric" in metrics
        assert metrics["gsm8k_0|some_other_metric"] == 0.99

    @patch("lighteval.pipeline.Pipeline")
    @patch("lighteval.logging.evaluation_tracker.EvaluationTracker")
    def test_run_lighteval_returns_empty_on_none_results(self, mock_tracker_cls, mock_pipeline_cls):
        """Test that run_lighteval returns empty dict when results are None."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = None
        mock_pipeline_cls.return_value = mock_pipeline

        model = MagicMock()
        metrics = run_lighteval(model=model, model_name="test-model", tasks="gsm8k")

        assert metrics == {}

    @patch("lighteval.pipeline.Pipeline")
    @patch("lighteval.logging.evaluation_tracker.EvaluationTracker")
    def test_run_lighteval_multiple_tasks(self, mock_tracker_cls, mock_pipeline_cls):
        """Test that run_lighteval handles multiple tasks."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_results.return_value = {
            "results": {
                "gsm8k_0": {"expr_gold_metric": 0.42},
                "mmlu_0": {"accuracy": 0.65},
            }
        }
        mock_pipeline_cls.return_value = mock_pipeline

        model = MagicMock()
        metrics = run_lighteval(
            model=model,
            model_name="test-model",
            tasks="gsm8k,mmlu",
        )

        assert "gsm8k_0|expr_gold_metric" in metrics
        assert "mmlu_0|accuracy" in metrics


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
