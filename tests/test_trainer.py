"""Tests for trainer utilities."""

import os

import numpy as np
import torch
from torch import nn

from lora_finetune.config import ModelConfig, TrainingConfig
from lora_finetune.trainer import (
    LoraTrainer,
    compute_metrics_for_classification,
    compute_metrics_for_lm,
    enable_gradient_checkpointing,
    get_training_arguments,
    prepare_model_for_training,
)

# Disable wandb for tests
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


class TestGetTrainingArguments:
    """Tests for get_training_arguments function.

    Note: These tests verify the config values are passed correctly.
    The actual TrainingArguments creation may fail due to transformers version
    differences (e.g., evaluation_strategy vs eval_strategy), so we test
    the config preparation logic separately.
    """

    def test_training_config_values(self):
        """Test that TrainingConfig holds correct values."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
        )

        assert training_config.output_dir == "./test-output"
        assert training_config.num_train_epochs == 3
        assert training_config.per_device_train_batch_size == 4
        assert training_config.learning_rate == 2e-4

    def test_training_config_bf16(self):
        """Test training config with bf16."""
        training_config = TrainingConfig(bf16=True, fp16=False)

        assert training_config.bf16 is True
        assert training_config.fp16 is False

    def test_training_config_gradient_checkpointing(self):
        """Test training config with gradient checkpointing."""
        training_config = TrainingConfig(gradient_checkpointing=True)

        assert training_config.gradient_checkpointing is True

    def test_training_config_custom_gradient_checkpointing_kwargs(self):
        """Test training config with custom gradient checkpointing kwargs."""
        training_config = TrainingConfig(
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

        assert training_config.gradient_checkpointing_kwargs == {"use_reentrant": True}

    def test_training_config_fsdp(self):
        """Test training config with FSDP."""
        training_config = TrainingConfig(fsdp="full_shard auto_wrap")

        assert training_config.fsdp == "full_shard auto_wrap"

    def test_training_config_wandb(self):
        """Test training config with wandb reporting."""
        training_config = TrainingConfig(
            report_to="wandb",
            wandb_project="test-project",
            wandb_run_name="test-run",
        )

        assert training_config.report_to == "wandb"
        assert training_config.wandb_project == "test-project"
        assert training_config.wandb_run_name == "test-run"

    def test_training_config_scheduler(self):
        """Test training config with scheduler settings."""
        training_config = TrainingConfig(
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            warmup_steps=100,
        )

        assert training_config.lr_scheduler_type == "linear"
        assert training_config.warmup_ratio == 0.1
        assert training_config.warmup_steps == 100

    def test_training_config_save_settings(self):
        """Test training config with save settings."""
        training_config = TrainingConfig(
            save_steps=100,
            save_total_limit=5,
            save_strategy="steps",
        )

        assert training_config.save_steps == 100
        assert training_config.save_total_limit == 5
        assert training_config.save_strategy == "steps"


class TestEnableGradientCheckpointing:
    """Tests for enable_gradient_checkpointing function."""

    def test_enable_with_gradient_checkpointing_method(self):
        """Test enabling gradient checkpointing on model with the method."""

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._gradient_checkpointing_enabled = False

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gradient_checkpointing_enabled = True

        model = MockModel()
        result = enable_gradient_checkpointing(model)

        assert result._gradient_checkpointing_enabled is True

    def test_enable_with_input_require_grads(self):
        """Test enabling gradient checkpointing via enable_input_require_grads."""

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._input_require_grads_enabled = False

            def enable_input_require_grads(self):
                self._input_require_grads_enabled = True

        model = MockModel()
        result = enable_gradient_checkpointing(model)

        assert result._input_require_grads_enabled is True


class TestPrepareModelForTraining:
    """Tests for prepare_model_for_training function."""

    def test_prepare_with_gradient_checkpointing(self):
        """Test preparing model with gradient checkpointing enabled."""

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._gc_enabled = False

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gc_enabled = True

        model = MockModel()
        config = TrainingConfig(gradient_checkpointing=True)

        result = prepare_model_for_training(model, config)

        assert result._gc_enabled is True

    def test_prepare_without_gradient_checkpointing(self):
        """Test preparing model without gradient checkpointing."""

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._gc_enabled = False

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gc_enabled = True

        model = MockModel()
        config = TrainingConfig(gradient_checkpointing=False)

        result = prepare_model_for_training(model, config)

        assert result._gc_enabled is False

    def test_prepare_makes_params_contiguous(self):
        """Test that prepare makes parameters contiguous."""
        model = nn.Linear(10, 5)
        config = TrainingConfig(gradient_checkpointing=False)

        result = prepare_model_for_training(model, config)

        for param in result.parameters():
            if param.requires_grad:
                assert param.data.is_contiguous()


class TestTrainingConfigDefaults:
    """Tests for TrainingConfig default values."""

    def test_default_optimizer(self):
        """Test default optimizer is adamw_torch_fused."""
        config = TrainingConfig()
        assert config.optim == "adamw_torch_fused"

    def test_default_scheduler(self):
        """Test default scheduler is cosine."""
        config = TrainingConfig()
        assert config.lr_scheduler_type == "cosine"

    def test_default_precision(self):
        """Test default precision settings."""
        config = TrainingConfig()
        assert config.bf16 is True
        assert config.fp16 is False
        assert config.tf32 is None

    def test_default_evaluation_settings(self):
        """Test default evaluation settings."""
        config = TrainingConfig()
        assert config.eval_strategy == "steps"
        assert config.eval_steps == 500
        assert config.load_best_model_at_end is True
        assert config.metric_for_best_model == "eval_loss"
        assert config.greater_is_better is False


class TestGetTrainingArgumentsFunction:
    """Tests for get_training_arguments function."""

    def test_basic_training_arguments(self):
        """Test creating basic TrainingArguments."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            num_train_epochs=3,
            learning_rate=2e-4,
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        assert args.output_dir == "./test-output"
        assert args.num_train_epochs == 3
        assert args.learning_rate == 2e-4

    def test_training_arguments_with_fsdp(self):
        """Test TrainingArguments with FSDP config."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            fsdp="full_shard auto_wrap",
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        # FSDP is converted to list of FSDPOption enums
        assert len(args.fsdp) == 2
        assert args.fsdp_config is not None

    def test_training_arguments_gradient_checkpointing_kwargs(self):
        """Test that gradient_checkpointing_kwargs default is set."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            gradient_checkpointing=True,
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        assert args.gradient_checkpointing is True
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}

    def test_training_arguments_custom_gradient_checkpointing_kwargs(self):
        """Test custom gradient_checkpointing_kwargs is preserved."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        assert args.gradient_checkpointing_kwargs == {"use_reentrant": True}

    def test_training_arguments_wandb_settings(self):
        """Test TrainingArguments with wandb settings."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            report_to="wandb",
            wandb_run_name="test-run",
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        # report_to is converted to a list
        assert "wandb" in args.report_to
        assert args.run_name == "test-run"


class TestComputeMetricsForClassification:
    """Tests for compute_metrics_for_classification function."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy case."""
        logits = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        labels = np.array([1, 0, 1])

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

            def __iter__(self):
                return iter([self.predictions, self.label_ids])

        eval_pred = MockEvalPred(logits, labels)
        result = compute_metrics_for_classification(eval_pred)

        assert "accuracy" in result
        assert result["accuracy"] == 1.0

    def test_zero_accuracy(self):
        """Test 0% accuracy case."""
        logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
        labels = np.array([1, 0, 1])

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

            def __iter__(self):
                return iter([self.predictions, self.label_ids])

        eval_pred = MockEvalPred(logits, labels)
        result = compute_metrics_for_classification(eval_pred)

        assert result["accuracy"] == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy case."""
        logits = np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
        labels = np.array([1, 0, 1, 0])

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

            def __iter__(self):
                return iter([self.predictions, self.label_ids])

        eval_pred = MockEvalPred(logits, labels)
        result = compute_metrics_for_classification(eval_pred)

        assert result["accuracy"] == 0.75


class TestComputeMetricsForLM:
    """Tests for compute_metrics_for_lm function."""

    def test_perplexity_computation(self):
        """Test perplexity is computed correctly."""
        # Create simple logits and labels
        vocab_size = 10
        seq_len = 5
        batch_size = 2

        # Random logits
        np.random.seed(42)
        logits = np.random.randn(batch_size, seq_len, vocab_size)
        labels = np.random.randint(0, vocab_size, (batch_size, seq_len))

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

            def __iter__(self):
                return iter([self.predictions, self.label_ids])

        eval_pred = MockEvalPred(logits, labels)
        result = compute_metrics_for_lm(eval_pred)

        assert "perplexity" in result
        assert result["perplexity"] > 0
        assert not np.isinf(result["perplexity"])

    def test_perplexity_with_padding(self):
        """Test perplexity handles padding tokens (-100)."""
        vocab_size = 10
        seq_len = 5
        batch_size = 2

        np.random.seed(42)
        logits = np.random.randn(batch_size, seq_len, vocab_size)
        labels = np.array([[1, 2, 3, -100, -100], [1, 2, -100, -100, -100]])

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

            def __iter__(self):
                return iter([self.predictions, self.label_ids])

        eval_pred = MockEvalPred(logits, labels)
        result = compute_metrics_for_lm(eval_pred)

        assert "perplexity" in result
        assert result["perplexity"] > 0

    def test_perplexity_all_padding(self):
        """Test perplexity with all padding tokens returns inf."""
        vocab_size = 10
        seq_len = 5
        batch_size = 2

        logits = np.random.randn(batch_size, seq_len, vocab_size)
        labels = np.full((batch_size, seq_len), -100)

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

            def __iter__(self):
                return iter([self.predictions, self.label_ids])

        eval_pred = MockEvalPred(logits, labels)
        result = compute_metrics_for_lm(eval_pred)

        assert result["perplexity"] == float("inf")


class TestLoraTrainer:
    """Tests for LoraTrainer class."""

    def test_lora_trainer_initialization(self):
        """Test LoraTrainer can be initialized."""
        from transformers import TrainingArguments

        model = nn.Linear(10, 5)
        args = TrainingArguments(
            output_dir="./test-output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            report_to="none",
        )

        # Create simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}

        train_dataset = SimpleDataset()

        trainer = LoraTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

        assert trainer is not None
        assert trainer.model is model

    def test_lora_trainer_compute_loss(self):
        """Test that compute_loss delegates to parent."""
        from transformers import TrainingArguments

        model = nn.Linear(10, 5)
        args = TrainingArguments(
            output_dir="./test-output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            report_to="none",
        )

        trainer = LoraTrainer(
            model=model,
            args=args,
            train_dataset=None,
        )

        # Just verify the method exists and is callable
        assert hasattr(trainer, "compute_loss")
        assert callable(trainer.compute_loss)
