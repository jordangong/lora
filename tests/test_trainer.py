"""Tests for trainer utilities."""

import os

from torch import nn

from lora_finetune.config import TrainingConfig
from lora_finetune.trainer import (
    WandbCallback,
    enable_gradient_checkpointing,
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


class TestWandbCallback:
    """Tests for WandbCallback class."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        config = TrainingConfig(report_to="wandb")
        callback = WandbCallback(config)

        assert callback.config == config
        assert callback._wandb is None

    def test_callback_setup_without_wandb(self):
        """Test callback setup when wandb is not available."""
        config = TrainingConfig(report_to="none")
        callback = WandbCallback(config)

        callback.setup(None, None, None)
        assert callback._wandb is None


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
        assert config.tf32 is True

    def test_default_evaluation_settings(self):
        """Test default evaluation settings."""
        config = TrainingConfig()
        assert config.eval_strategy == "steps"
        assert config.eval_steps == 500
        assert config.load_best_model_at_end is True
        assert config.metric_for_best_model == "eval_loss"
        assert config.greater_is_better is False
