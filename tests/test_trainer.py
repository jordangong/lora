"""Tests for trainer utilities."""

import os
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn

import lora_finetune.trainer as trainer_module
from lora_finetune.config import (
    BenchmarkEvalConfig,
    DataConfig,
    DPOConfig,
    GRPOConfig,
    HPOConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from lora_finetune.trainer import (
    LoraTrainer,
    RichProgressCallback,
    build_grpo_reward_functions,
    compute_metrics_for_classification,
    compute_metrics_for_lm,
    create_trainer,
    enable_gradient_checkpointing,
    generate_run_id,
    get_sft_training_arguments,
    get_training_arguments,
    prepare_model_for_training,
    resolve_trainer_type,
    setup_wandb,
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

    def test_get_sft_training_arguments_marks_dataset_as_pretokenized(self, monkeypatch):
        """Test that SFT args skip TRL dataset preparation for tokenized datasets."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
        )
        model_config = ModelConfig(model_type="causal_lm")

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.dataset_kwargs = None
                self.dataset_text_field = None
                self.packing = True
                self.padding_free = None
                self.eval_packing = None

        original_sft_config = trainer_module.SFTConfig
        try:
            monkeypatch.setattr(trainer_module, "SFTConfig", FakeSFTConfig)
            training_args = get_sft_training_arguments(training_config, model_config)
        finally:
            monkeypatch.setattr(trainer_module, "SFTConfig", original_sft_config)

        assert isinstance(training_args, FakeSFTConfig)
        assert training_args.dataset_kwargs == {"skip_prepare_dataset": True}
        assert training_args.packing is False
        assert training_args.padding_free is None
        assert training_args.eval_packing is False
        assert training_args.dataset_text_field == "text"

    def test_get_sft_training_arguments_preserves_padding_free_for_unsloth(self, monkeypatch):
        """Test that the Unsloth path leaves TRL padding_free unset for Unsloth to manage."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
        )
        model_config = ModelConfig(model_type="causal_lm", use_unsloth=True)

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.dataset_kwargs = None
                self.dataset_text_field = None
                self.packing = True
                self.padding_free = None
                self.eval_packing = None

        original_sft_config = trainer_module.SFTConfig
        try:
            monkeypatch.setattr(trainer_module, "SFTConfig", FakeSFTConfig)
            training_args = get_sft_training_arguments(training_config, model_config)
        finally:
            monkeypatch.setattr(trainer_module, "SFTConfig", original_sft_config)

        assert isinstance(training_args, FakeSFTConfig)
        assert training_args.packing is False
        assert training_args.padding_free is None

    def test_get_sft_training_arguments_maps_data_config_fields(self, monkeypatch):
        """Test that SFT args reuse compatible TRL-native data settings."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
        )
        model_config = ModelConfig(model_type="causal_lm")
        data_config = DataConfig(
            text_column="body",
            preprocessing_num_workers=8,
            max_seq_length=4096,
            response_only_loss=False,
            assistant_only_loss=True,
            eos_token="<eos>",
        )

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.dataset_kwargs = None
                self.dataset_text_field = None
                self.dataset_num_proc = None
                self.max_length = None
                self.completion_only_loss = None
                self.assistant_only_loss = False
                self.eos_token = None
                self.packing = True
                self.eval_packing = None

        original_sft_config = trainer_module.SFTConfig
        try:
            monkeypatch.setattr(trainer_module, "SFTConfig", FakeSFTConfig)
            training_args = get_sft_training_arguments(
                training_config,
                model_config,
                data_config=data_config,
                skip_prepare_dataset=False,
            )
        finally:
            monkeypatch.setattr(trainer_module, "SFTConfig", original_sft_config)

        assert isinstance(training_args, FakeSFTConfig)
        assert training_args.dataset_kwargs == {"skip_prepare_dataset": False}
        assert training_args.dataset_text_field == "body"
        assert training_args.dataset_num_proc == 8
        assert training_args.max_length == 4096
        assert training_args.completion_only_loss is False
        assert training_args.assistant_only_loss is True
        assert training_args.eos_token == "<eos>"

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

        class MockConfig:
            use_cache = True

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._gradient_checkpointing_enabled = False
                self.config = MockConfig()

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gradient_checkpointing_enabled = True

        model = MockModel()
        result = enable_gradient_checkpointing(model)

        assert result._gradient_checkpointing_enabled is True
        assert result.config.use_cache is False

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

        class MockConfig:
            use_cache = True

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._gc_enabled = False
                self.config = MockConfig()

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gc_enabled = True

        model = MockModel()
        config = TrainingConfig(gradient_checkpointing=True)

        result = prepare_model_for_training(model, config)

        assert result._gc_enabled is True
        assert result.config.use_cache is False

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

    def test_prepare_skips_duplicate_gradient_checkpointing_when_unsloth_managed(self):
        """Test Unsloth-managed gradient checkpointing is not re-enabled."""

        class MockConfig:
            use_cache = True

        class MockGenerationConfig:
            use_cache = True

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._gc_enabled = False
                self.config = MockConfig()
                self.generation_config = MockGenerationConfig()
                self._lora_finetune_unsloth_managed_gradient_checkpointing = True

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gc_enabled = True

        model = MockModel()
        config = TrainingConfig(gradient_checkpointing=True)

        result = prepare_model_for_training(model, config)

        assert result._gc_enabled is False
        assert result.config.use_cache is False
        assert result.generation_config.use_cache is False

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
            bf16=False,
            fp16=False,
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
            bf16=False,
            fp16=False,
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
            bf16=False,
            fp16=False,
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
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        assert args.gradient_checkpointing_kwargs == {"use_reentrant": True}

    def test_training_arguments_unsloth_gradient_checkpointing_mode_uses_boolean_flag(self):
        """Test explicit Unsloth mode still produces a boolean TrainingArguments flag."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            gradient_checkpointing="unsloth",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()

        args = get_training_arguments(training_config, model_config)

        assert args.gradient_checkpointing is True
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}

    def test_training_arguments_wandb_settings(self):
        """Test TrainingArguments with wandb settings."""
        training_config = TrainingConfig(
            output_dir="./test-output",
            report_to="wandb",
            wandb_run_name="test-run",
            bf16=False,
            fp16=False,
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
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        train_dataset = SimpleDataset()

        trainer = LoraTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

        assert trainer is not None
        assert trainer.model is model

    def test_lora_trainer_with_lora_config(self):
        """Test LoraTrainer with lora_config parameter."""
        from transformers import TrainingArguments

        model = nn.Linear(10, 5)
        args = TrainingArguments(
            output_dir="./test-output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            report_to="none",
        )
        lora_config = LoraConfig(method="lora", r=16)

        trainer = LoraTrainer(
            model=model,
            args=args,
            train_dataset=None,
            lora_config=lora_config,
        )

        assert trainer.lora_config is lora_config
        assert trainer.lora_config.method == "lora"

    def test_lora_trainer_loraplus_config(self):
        """Test LoraTrainer with LoRA+ configuration."""
        from transformers import TrainingArguments

        model = nn.Linear(10, 5)
        args = TrainingArguments(
            output_dir="./test-output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            report_to="none",
        )
        lora_config = LoraConfig(method="loraplus", loraplus_lr_ratio=16.0)

        trainer = LoraTrainer(
            model=model,
            args=args,
            train_dataset=None,
            lora_config=lora_config,
        )

        assert trainer.lora_config.method == "loraplus"
        assert trainer.lora_config.loraplus_lr_ratio == 16.0

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

    def test_lora_trainer_create_optimizer_method_exists(self):
        """Test that create_optimizer method exists."""
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

        assert hasattr(trainer, "create_optimizer")
        assert callable(trainer.create_optimizer)

    def test_lora_trainer_loraplus_optimizer_method_exists(self):
        """Test that _create_loraplus_optimizer method exists."""
        from transformers import TrainingArguments

        model = nn.Linear(10, 5)
        args = TrainingArguments(
            output_dir="./test-output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            report_to="none",
        )
        lora_config = LoraConfig(method="loraplus")

        trainer = LoraTrainer(
            model=model,
            args=args,
            train_dataset=None,
            lora_config=lora_config,
        )

        assert hasattr(trainer, "_create_loraplus_optimizer")
        assert callable(trainer._create_loraplus_optimizer)

    def test_lora_trainer_hpo_uses_trial_specific_output_dir(self, monkeypatch):
        from transformers import TrainingArguments

        recorded = {}
        original_hp_search_setup = trainer_module.Trainer._hp_search_setup

        def fake_hp_search_setup(self, trial):
            recorded["trial"] = dict(trial)

        monkeypatch.setattr(trainer_module.Trainer, "_hp_search_setup", fake_hp_search_setup)

        try:
            model = nn.Linear(10, 5)
            args = TrainingArguments(
                output_dir="./test-output",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                report_to="none",
            )
            training_config = TrainingConfig(output_dir="./test-output", report_to="wandb")
            trainer = LoraTrainer(
                model=model,
                args=args,
                train_dataset=None,
                hpo_config=HPOConfig(enabled=True, n_trials=2),
                hpo_config_sections={"training": training_config, "lora": LoraConfig()},
            )

            trainer.state.trial_name = "trial-001"
            trainer._hp_search_setup({"learning_rate": 1e-5})

            assert recorded["trial"] == {"learning_rate": 1e-5}
            assert trainer.args.output_dir == os.path.join("./test-output", "trial-001")
            assert os.path.isdir(trainer.args.output_dir)
        finally:
            monkeypatch.setattr(
                trainer_module.Trainer, "_hp_search_setup", original_hp_search_setup
            )

    def test_lora_trainer_hpo_ignores_wandb_metadata_keys(self, monkeypatch):
        from transformers import TrainingArguments

        recorded = {}
        original_hp_search_setup = trainer_module.Trainer._hp_search_setup

        def fake_hp_search_setup(self, trial):
            recorded["trial"] = dict(trial)

        monkeypatch.setattr(trainer_module.Trainer, "_hp_search_setup", fake_hp_search_setup)

        try:
            model = nn.Linear(10, 5)
            args = TrainingArguments(
                output_dir="./test-output",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                report_to="none",
            )
            training_config = TrainingConfig(output_dir="./test-output", report_to="wandb")
            trainer = LoraTrainer(
                model=model,
                args=args,
                train_dataset=None,
                hpo_config=HPOConfig(enabled=True, n_trials=2),
                hpo_config_sections={"training": training_config, "lora": LoraConfig()},
            )

            trainer._hp_search_setup(
                {"_wandb": {"runtime": 1}, "metric": "eval/loss", "learning_rate": 1e-5}
            )

            assert recorded["trial"] == {"learning_rate": 1e-5}
        finally:
            monkeypatch.setattr(
                trainer_module.Trainer, "_hp_search_setup", original_hp_search_setup
            )

    def test_lora_trainer_hpo_sync_callback_sees_updated_effective_batch_size(self, monkeypatch):
        from transformers import TrainingArguments

        sync_payloads = []
        original_hp_search_setup = trainer_module.Trainer._hp_search_setup

        def fake_hp_search_setup(self, trial):
            return None

        monkeypatch.setattr(trainer_module.Trainer, "_hp_search_setup", fake_hp_search_setup)

        try:
            model = nn.Linear(10, 5)
            args = TrainingArguments(
                output_dir="./test-output",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                report_to="none",
            )
            training_config = TrainingConfig(
                output_dir="./test-output",
                report_to="wandb",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
            )
            trainer = LoraTrainer(
                model=model,
                args=args,
                train_dataset=None,
                hpo_config=HPOConfig(enabled=True, n_trials=2),
                hpo_config_sections={"training": training_config, "lora": LoraConfig()},
                hpo_sync_callback=lambda: sync_payloads.append(
                    training_config.to_serializable_dict()
                ),
            )

            trainer._hp_search_setup(
                {"per_device_train_batch_size": 6, "gradient_accumulation_steps": 3}
            )

            assert training_config.per_device_train_batch_size == 6
            assert training_config.gradient_accumulation_steps == 3
            assert training_config.effective_batch_size == 18
            assert sync_payloads[-1]["effective_batch_size"] == 18
        finally:
            monkeypatch.setattr(
                trainer_module.Trainer, "_hp_search_setup", original_hp_search_setup
            )

    def test_lora_trainer_save_model_method_exists(self):
        """Test that save_model method exists."""
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

        assert hasattr(trainer, "save_model")
        assert callable(trainer.save_model)

    def test_lora_trainer_evaluation_loop_method_exists(self):
        """Test that evaluation_loop method exists."""
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

        assert hasattr(trainer, "evaluation_loop")
        assert callable(trainer.evaluation_loop)


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_returns_string(self):
        """Test that generate_run_id returns a string."""
        run_id = generate_run_id()
        assert isinstance(run_id, str)

    def test_format_timestamp(self):
        """Test that run_id has timestamp format."""
        run_id = generate_run_id()
        # Format should be YYYYMMDD_HHMMSS (15 chars)
        assert len(run_id) == 15
        assert run_id[8] == "_"
        # Should be numeric except for underscore
        assert run_id[:8].isdigit()
        assert run_id[9:].isdigit()

    def test_unique_ids(self):
        """Test that consecutive calls produce different IDs (with small delay)."""
        import time

        run_id1 = generate_run_id()
        time.sleep(1.1)  # Wait for at least 1 second difference
        run_id2 = generate_run_id()
        assert run_id1 != run_id2


class TestSetupWandb:
    """Tests for setup_wandb function."""

    def test_setup_wandb_disables_watch_by_default(self, monkeypatch):
        """Test that setup_wandb clears stale watch settings by default."""
        wandb_module = ModuleType("wandb")
        wandb_module.run = None

        def init(**kwargs):
            wandb_module.run = type("Run", (), {"name": "dummy-run"})()

        wandb_module.init = init

        monkeypatch.setitem(sys.modules, "wandb", wandb_module)
        monkeypatch.setenv("WANDB_WATCH", "gradients")
        monkeypatch.setenv("WANDB_LOG_MODEL", "true")

        config = TrainingConfig(report_to="wandb")

        run_name = setup_wandb(config)

        assert run_name == "dummy-run"
        assert os.environ["WANDB_WATCH"] == "false"
        assert os.environ["WANDB_LOG_MODEL"] == "false"

    def test_setup_wandb_preserves_explicit_watch_settings(self, monkeypatch):
        """Test that setup_wandb keeps explicit watch and log-model settings."""
        wandb_module = ModuleType("wandb")
        wandb_module.run = None
        wandb_module.init_kwargs = None

        def init(**kwargs):
            wandb_module.init_kwargs = kwargs
            wandb_module.run = type("Run", (), {"name": "explicit-run"})()

        wandb_module.init = init

        monkeypatch.setitem(sys.modules, "wandb", wandb_module)

        config = TrainingConfig(
            report_to="wandb",
            wandb_run_name="test-run",
            wandb_watch="all",
            wandb_log_model=True,
        )

        run_name = setup_wandb(config)

        assert run_name == "explicit-run"
        assert wandb_module.init_kwargs["name"] == "test-run"
        assert os.environ["WANDB_WATCH"] == "all"
        assert os.environ["WANDB_LOG_MODEL"] == "true"

    def test_setup_wandb_logs_full_config_sections_and_redacts_sensitive_values(self, monkeypatch):
        wandb_module = ModuleType("wandb")
        wandb_module.run = None
        wandb_module.init_kwargs = None

        class FakeWandbConfig:
            def __init__(self):
                self.updates = []

            def update(self, payload, allow_val_change=False):
                self.updates.append((payload, allow_val_change))

        wandb_module.config = FakeWandbConfig()

        def init(**kwargs):
            wandb_module.init_kwargs = kwargs
            wandb_module.run = type("Run", (), {"name": "full-config-run"})()

        wandb_module.init = init

        monkeypatch.setitem(sys.modules, "wandb", wandb_module)

        training_config = TrainingConfig(
            report_to="wandb",
            output_dir="./outputs",
            per_device_train_batch_size=3,
            gradient_accumulation_steps=5,
            hub_token="super-secret",
        )
        model_config = ModelConfig(model_name_or_path="test-model")
        data_config = DataConfig(text_column="prompt")
        lora_config = LoraConfig(method="adalora", target_r=4)
        benchmark_eval_config = BenchmarkEvalConfig(enabled=True, tasks="gsm8k,mmlu")

        run_name = setup_wandb(
            training_config,
            model_config=model_config,
            data_config=data_config,
            lora_config=lora_config,
            benchmark_eval_config=benchmark_eval_config,
        )

        assert run_name == "full-config-run"
        assert "training" in wandb_module.init_kwargs["config"]
        assert "model" in wandb_module.init_kwargs["config"]
        assert "data" in wandb_module.init_kwargs["config"]
        assert "lora" in wandb_module.init_kwargs["config"]
        assert "benchmark_eval" in wandb_module.init_kwargs["config"]
        assert wandb_module.init_kwargs["config"]["training"]["hub_token"] == "***REDACTED***"
        assert wandb_module.init_kwargs["config"]["training"]["effective_batch_size"] == 15
        assert wandb_module.init_kwargs["config"]["model"]["model_name_or_path"] == "test-model"
        assert wandb_module.init_kwargs["config"]["data"]["text_column"] == "prompt"
        assert wandb_module.init_kwargs["config"]["lora"]["method"] == "adalora"
        assert wandb_module.init_kwargs["config"]["benchmark_eval"]["tasks"] == "gsm8k,mmlu"
        assert wandb_module.config.updates[-1][0]["training"]["hub_token"] == "***REDACTED***"
        assert wandb_module.config.updates[-1][0]["training"]["effective_batch_size"] == 15

    def test_setup_wandb_skips_eager_init_for_hpo(self, monkeypatch):
        wandb_module = ModuleType("wandb")
        wandb_module.run = None
        wandb_module.init_called = False

        def init(**kwargs):
            wandb_module.init_called = True

        wandb_module.init = init
        monkeypatch.setitem(sys.modules, "wandb", wandb_module)

        training_config = TrainingConfig(report_to="wandb", wandb_watch="all", wandb_log_model=True)
        hpo_config = HPOConfig(enabled=True, n_trials=3)

        run_name = setup_wandb(training_config, hpo_config=hpo_config)

        assert run_name is None
        assert wandb_module.init_called is False
        assert os.environ["WANDB_WATCH"] == "all"
        assert os.environ["WANDB_LOG_MODEL"] == "true"

    def test_setup_wandb_sets_project_and_entity_env_for_hpo(self, monkeypatch):
        wandb_module = ModuleType("wandb")
        wandb_module.run = None
        wandb_module.init_called = False

        def init(**kwargs):
            wandb_module.init_called = True

        wandb_module.init = init
        monkeypatch.setitem(sys.modules, "wandb", wandb_module)

        training_config = TrainingConfig(report_to="wandb", wandb_project="project-from-training")
        hpo_config = HPOConfig(enabled=True, n_trials=3, entity="entity-from-hpo")

        setup_wandb(training_config, hpo_config=hpo_config)

        assert os.environ["WANDB_PROJECT"] == "project-from-training"
        assert os.environ["WANDB_ENTITY"] == "entity-from-hpo"

    def test_setup_wandb_clears_stale_entity_env_when_hpo_entity_missing(self, monkeypatch):
        wandb_module = ModuleType("wandb")
        wandb_module.run = None
        wandb_module.init_called = False

        def init(**kwargs):
            wandb_module.init_called = True

        wandb_module.init = init
        monkeypatch.setitem(sys.modules, "wandb", wandb_module)
        monkeypatch.setenv("WANDB_ENTITY", "stale-entity")

        training_config = TrainingConfig(report_to="wandb", wandb_project="project-from-training")
        hpo_config = HPOConfig(enabled=True, n_trials=3)

        setup_wandb(training_config, hpo_config=hpo_config)

        assert os.environ["WANDB_PROJECT"] == "project-from-training"
        assert "WANDB_ENTITY" not in os.environ

    def test_run_hyperparameter_search_falls_back_to_training_wandb_project(self):
        class FakeTrainer:
            def __init__(self):
                self.kwargs = None

            def hyperparameter_search(self, **kwargs):
                self.kwargs = kwargs
                return "best-run"

        trainer = FakeTrainer()
        training_config = TrainingConfig(report_to="wandb", wandb_project="project-from-training")
        hpo_config = HPOConfig(enabled=True, n_trials=3)

        result = trainer_module.run_hyperparameter_search(trainer, training_config, hpo_config)

        assert result == "best-run"
        assert trainer.kwargs["project"] == "project-from-training"
        assert trainer.kwargs["metric"] == "eval/loss"

    def test_hpo_wandb_callback_resets_and_reinitializes_each_trial(self):
        callback = trainer_module.HPOWandbCallback.__new__(trainer_module.HPOWandbCallback)
        events = []

        class FakeWandb:
            def finish(self):
                events.append("finish")

        def fake_setup(args, state, model, **kwargs):
            events.append(("setup", args.run_name, state.trial_name, model))
            callback._initialized = True

        callback._wandb = FakeWandb()
        callback._initialized = True
        callback.setup = fake_setup

        args = type("Args", (), {"run_name": "base-run"})()
        first_state = type("State", (), {"is_hyper_param_search": True, "trial_name": "trial-1"})()
        second_state = type("State", (), {"is_hyper_param_search": True, "trial_name": "trial-2"})()

        callback.on_train_begin(args, first_state, None, model="model-1")
        args.run_name = "second-run"
        callback.on_train_begin(args, second_state, None, model="model-2")

        assert events == [
            "finish",
            ("setup", None, "trial-1", "model-1"),
            "finish",
            ("setup", None, "trial-2", "model-2"),
        ]
        assert args.run_name is None

    def test_create_trainer_replaces_default_wandb_callback_for_hpo(self, monkeypatch):
        class SimpleDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="steps",
            load_best_model_at_end=False,
            report_to="wandb",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig(model_type="causal_lm")
        model = nn.Linear(3, 3)
        train_dataset = SimpleDataset()
        eval_dataset = SimpleDataset()

        monkeypatch.setattr(trainer_module, "setup_wandb", lambda *args, **kwargs: None)
        monkeypatch.setattr(trainer_module, "_sync_wandb_config", lambda *args, **kwargs: None)

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda features: features,
            hpo_config=HPOConfig(enabled=True, n_trials=2),
            model_init=lambda _: nn.Linear(3, 3),
            hpo_config_sections={"training": training_config, "lora": LoraConfig()},
        )

        callback_types = [type(cb).__name__ for cb in trainer.callback_handler.callbacks]
        assert callback_types.count("HPOWandbCallback") == 1
        assert "WandbCallback" not in callback_types


class TestRichProgressCallback:
    """Tests for RichProgressCallback class."""

    def test_initialization(self):
        """Test RichProgressCallback initialization."""
        callback = RichProgressCallback()

        assert callback.progress is None
        assert callback.train_task is None
        assert callback.eval_task is None
        assert callback.max_epochs == 1
        assert callback.in_eval is False

    def test_has_required_methods(self):
        """Test that callback has all required TrainerCallback methods."""
        callback = RichProgressCallback()

        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_step_end")
        assert hasattr(callback, "on_log")
        assert hasattr(callback, "on_evaluate")
        assert hasattr(callback, "on_train_end")
        assert hasattr(callback, "_start_eval_progress")

    def test_start_eval_progress_without_progress(self):
        """Test _start_eval_progress when progress is None."""
        callback = RichProgressCallback()
        # Should not raise when progress is None
        callback._start_eval_progress(10)
        assert callback.eval_task is None

    def test_cleanup_stops_progress_and_restores_cursor(self):
        callback = RichProgressCallback()
        events = []

        class FakeConsole:
            def show_cursor(self, show=True):
                events.append(("show_cursor", show))

        class FakeProgress:
            def __init__(self):
                self.console = FakeConsole()

            def stop(self):
                events.append("stop")

        callback.progress = FakeProgress()
        callback.train_task = 1
        callback.eval_task = 2
        callback.in_eval = True

        callback.cleanup()

        assert events == ["stop", ("show_cursor", True)]
        assert callback.progress is None
        assert callback.train_task is None
        assert callback.eval_task is None
        assert callback.in_eval is False

    def test_format_epoch_handles_non_numeric_values(self):
        """Test _format_epoch handles missing and string epochs safely."""
        assert RichProgressCallback._format_epoch(None) == "?"
        assert RichProgressCallback._format_epoch("?") == "?"
        assert RichProgressCallback._format_epoch("1.5") == "1.50"

    def test_on_evaluate_handles_string_epoch(self):
        """Test on_evaluate does not fail when epoch is a string."""
        callback = RichProgressCallback()
        printed = []

        class FakeConsole:
            def print(self, message):
                printed.append(message)

        class FakeProgress:
            def __init__(self):
                self.console = FakeConsole()
                self.removed_tasks = []

            def remove_task(self, task_id):
                self.removed_tasks.append(task_id)

        callback.progress = FakeProgress()
        callback.eval_task = 7

        callback.on_evaluate(
            args=None,
            state=None,
            control=None,
            metrics={"epoch": "?", "eval_loss": 1.2345},
        )

        assert callback.eval_task is None
        assert callback.progress.removed_tasks == [7]
        assert printed == ["  [bold]Eval[/bold] @ epoch ?: [green]loss[/green]=1.2345"]

    def test_on_evaluate_falls_back_to_state_epoch_when_metrics_omit_epoch(self):
        """Test on_evaluate uses state.epoch when eval metrics omit epoch."""
        callback = RichProgressCallback()
        printed = []

        class FakeConsole:
            def print(self, message):
                printed.append(message)

        class FakeProgress:
            def __init__(self):
                self.console = FakeConsole()
                self.removed_tasks = []

            def remove_task(self, task_id):
                self.removed_tasks.append(task_id)

        class FakeState:
            epoch = 2.0

        callback.progress = FakeProgress()
        callback.eval_task = 9

        callback.on_evaluate(
            args=None,
            state=FakeState(),
            control=None,
            metrics={"eval_loss": 0.8464},
        )

        assert callback.eval_task is None
        assert callback.progress.removed_tasks == [9]
        assert printed == ["  [bold]Eval[/bold] @ epoch 2.00: [green]loss[/green]=0.8464"]


class TestCreateTrainer:
    """Tests for create_trainer function."""

    def test_create_trainer_returns_lora_trainer(self):
        """Test that create_trainer returns a LoraTrainer instance."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        train_dataset = SimpleDataset()

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=train_dataset,
        )

        assert isinstance(trainer, LoraTrainer)

    def test_create_trainer_attaches_model_init_for_hpo(self, monkeypatch):
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="steps",
            load_best_model_at_end=False,
            report_to="wandb",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig(model_type="text_classification")
        hpo_config = HPOConfig(enabled=True, n_trials=2)

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor(1),
                }

        monkeypatch.setattr(trainer_module, "setup_wandb", lambda *args, **kwargs: None)
        monkeypatch.setattr(trainer_module, "_sync_wandb_config", lambda *args, **kwargs: None)

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=SimpleDataset(),
            eval_dataset=SimpleDataset(),
            hpo_config=hpo_config,
            model_init=lambda trial=None: nn.Linear(10, 5),
            hpo_config_sections={"training": training_config, "lora": LoraConfig()},
        )

        assert isinstance(trainer, LoraTrainer)
        assert callable(trainer.model_init)
        assert trainer.args.output_dir == "./test-output"

    def test_create_trainer_with_lora_config(self):
        """Test create_trainer with LoRA config."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()
        lora_config = LoraConfig(method="lora", r=16)

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        train_dataset = SimpleDataset()

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=train_dataset,
            lora_config=lora_config,
        )

        assert trainer.lora_config is lora_config

    def test_create_trainer_has_rich_callback(self):
        """Test that create_trainer adds RichProgressCallback."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        train_dataset = SimpleDataset()

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=train_dataset,
        )

        # Check that RichProgressCallback is in the callbacks
        callback_types = [type(cb).__name__ for cb in trainer.callback_handler.callbacks]
        assert "RichProgressCallback" in callback_types

    def test_create_trainer_disables_tqdm(self):
        """Test that create_trainer disables tqdm."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        train_dataset = SimpleDataset()

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=train_dataset,
        )

        assert trainer.args.disable_tqdm is True

    def test_create_trainer_fails_fast_without_eval_dataset_when_eval_enabled(self):
        """Test fail-fast behavior when eval is enabled but eval dataset is missing."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="steps",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig()

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        train_dataset = SimpleDataset()

        with pytest.raises(ValueError, match="no eval_dataset was provided"):
            create_trainer(
                model=model,
                training_config=training_config,
                model_config=model_config,
                train_dataset=train_dataset,
                eval_dataset=None,
            )

    def test_create_trainer_uses_transformers_backend_when_requested(self):
        """Test that the transformers backend is used when explicitly requested."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="transformers",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig(model_type="causal_lm")

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        trainer = create_trainer(
            model=model,
            training_config=training_config,
            model_config=model_config,
            train_dataset=SimpleDataset(),
        )

        assert isinstance(trainer, LoraTrainer)

    def test_create_trainer_uses_trl_backend_for_causal_lm(self, monkeypatch):
        """Test that the TRL backend is selected for causal LM training."""
        model = nn.Linear(10, 5)
        processing_class = object()
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
        )
        model_config = ModelConfig(model_type="causal_lm")

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.dataset_kwargs = None
                self.dataset_text_field = None
                self.packing = True
                self.eval_packing = None

        class FakeSFTTrainer:
            def __init__(self, *args, processing_class=None, **kwargs):
                self.processing_class = processing_class
                self.args = kwargs.get("args")
                self.kwargs = kwargs

        class FakeLoraSFTTrainer(FakeSFTTrainer):
            def remove_callback(self, callback_cls):
                return None

        original_sft_config = trainer_module.SFTConfig
        original_sft_trainer = trainer_module.SFTTrainer
        original_lora_sft_trainer = trainer_module.LoraSFTTrainer
        try:
            monkeypatch.setattr(trainer_module, "SFTConfig", FakeSFTConfig)
            monkeypatch.setattr(trainer_module, "SFTTrainer", FakeSFTTrainer)
            monkeypatch.setattr(trainer_module, "LoraSFTTrainer", FakeLoraSFTTrainer)
            trainer = create_trainer(
                model=model,
                training_config=training_config,
                model_config=model_config,
                train_dataset=SimpleDataset(),
                processing_class=processing_class,
            )
        finally:
            monkeypatch.setattr(trainer_module, "SFTConfig", original_sft_config)
            monkeypatch.setattr(trainer_module, "SFTTrainer", original_sft_trainer)
            monkeypatch.setattr(trainer_module, "LoraSFTTrainer", original_lora_sft_trainer)

        assert isinstance(trainer, FakeLoraSFTTrainer)
        assert trainer.processing_class is processing_class
        assert isinstance(trainer.args, FakeSFTConfig)
        assert trainer.args.dataset_kwargs == {"skip_prepare_dataset": True}
        assert trainer.args.dataset_text_field == "text"
        assert trainer.args.disable_tqdm is True

    def test_create_trainer_raises_when_trl_requested_but_unavailable(self, monkeypatch):
        """Test that requesting the TRL backend fails clearly when TRL is unavailable."""
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
        )
        model_config = ModelConfig(model_type="causal_lm")

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        original_lora_sft_trainer = trainer_module.LoraSFTTrainer
        try:
            monkeypatch.setattr(trainer_module, "LoraSFTTrainer", None)
            with pytest.raises(ImportError, match="TRL is required for causal LM finetuning"):
                create_trainer(
                    model=model,
                    training_config=training_config,
                    model_config=model_config,
                    train_dataset=SimpleDataset(),
                )
        finally:
            monkeypatch.setattr(trainer_module, "LoraSFTTrainer", original_lora_sft_trainer)

    def test_create_trainer_uses_dpo_trainer_when_requested(self, monkeypatch):
        model = nn.Linear(10, 5)
        processing_class = object()
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
            trainer_type="dpo",
        )
        model_config = ModelConfig(model_type="causal_lm")

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"prompt": "hi", "chosen": "hello", "rejected": "goodbye"}

        class FakeDPOConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeDPOTrainer:
            def __init__(self, *args, processing_class=None, **kwargs):
                self.processing_class = processing_class
                self.args = kwargs.get("args")

            def remove_callback(self, callback_cls):
                return None

        original_dpo_config = trainer_module.TRLDPOConfig
        original_lora_dpo_trainer = trainer_module.LoraDPOTrainer
        try:
            monkeypatch.setattr(trainer_module, "TRLDPOConfig", FakeDPOConfig)
            monkeypatch.setattr(trainer_module, "LoraDPOTrainer", FakeDPOTrainer)
            trainer = create_trainer(
                model=model,
                training_config=training_config,
                model_config=model_config,
                train_dataset=SimpleDataset(),
                processing_class=processing_class,
                dpo_config=DPOConfig(),
                data_config=DataConfig(preprocessing_num_workers=1),
            )
        finally:
            monkeypatch.setattr(trainer_module, "TRLDPOConfig", original_dpo_config)
            monkeypatch.setattr(trainer_module, "LoraDPOTrainer", original_lora_dpo_trainer)

        assert isinstance(trainer, FakeDPOTrainer)
        assert trainer.processing_class is processing_class
        assert isinstance(trainer.args, FakeDPOConfig)
        assert trainer.args.beta == 0.1

    def test_create_trainer_rejects_non_training_hpo_params_for_trl(self):
        model = nn.Linear(10, 5)
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="steps",
            load_best_model_at_end=False,
            report_to="wandb",
            llm_trainer="trl",
            trainer_type="dpo",
            bf16=False,
            fp16=False,
        )
        model_config = ModelConfig(model_type="causal_lm")
        hpo_config = HPOConfig(
            enabled=True,
            parameters={"lora.r": {"values": [8, 16]}},
        )

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"prompt": "hi", "chosen": "hello", "rejected": "goodbye"}

        with pytest.raises(
            ValueError, match=r"only training\.\* parameters are currently supported"
        ):
            create_trainer(
                model=model,
                training_config=training_config,
                model_config=model_config,
                train_dataset=SimpleDataset(),
                eval_dataset=SimpleDataset(),
                lora_config=LoraConfig(),
                dpo_config=DPOConfig(),
                hpo_config=hpo_config,
                model_init=lambda trial=None: nn.Linear(10, 5),
                hpo_config_sections={
                    "training": training_config,
                    "lora": LoraConfig(),
                    "dpo": DPOConfig(),
                    "grpo": GRPOConfig(),
                },
            )

    def test_create_trainer_uses_grpo_trainer_when_requested(self, monkeypatch):
        model = nn.Linear(10, 5)
        processing_class = object()
        training_config = TrainingConfig(
            output_dir="./test-output",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            llm_trainer="trl",
            trainer_type="grpo",
        )
        model_config = ModelConfig(model_type="causal_lm")

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {"prompt": "Solve 1+1", "answer": "2"}

        class FakeGRPOConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeGRPOTrainer:
            def __init__(self, *args, processing_class=None, reward_funcs=None, **kwargs):
                self.processing_class = processing_class
                self.reward_funcs = reward_funcs
                self.args = kwargs.get("args")

            def remove_callback(self, callback_cls):
                return None

        original_grpo_config = trainer_module.TRLGRPOConfig
        original_lora_grpo_trainer = trainer_module.LoraGRPOTrainer
        try:
            monkeypatch.setattr(trainer_module, "TRLGRPOConfig", FakeGRPOConfig)
            monkeypatch.setattr(trainer_module, "LoraGRPOTrainer", FakeGRPOTrainer)
            trainer = create_trainer(
                model=model,
                training_config=training_config,
                model_config=model_config,
                train_dataset=SimpleDataset(),
                processing_class=processing_class,
                grpo_config=GRPOConfig(reward_funcs=["length"]),
            )
        finally:
            monkeypatch.setattr(trainer_module, "TRLGRPOConfig", original_grpo_config)
            monkeypatch.setattr(trainer_module, "LoraGRPOTrainer", original_lora_grpo_trainer)

        assert isinstance(trainer, FakeGRPOTrainer)
        assert trainer.processing_class is processing_class
        assert isinstance(trainer.args, FakeGRPOConfig)
        assert len(trainer.reward_funcs) == 1


class TestTrainerTypeResolution:
    def test_resolve_trainer_type_returns_transformers_for_non_causal_models(self):
        assert (
            resolve_trainer_type(
                TrainingConfig(trainer_type="sft"),
                ModelConfig(model_type="text_classification"),
            )
            == "transformers"
        )

    def test_resolve_trainer_type_returns_dpo_for_causal_lm(self):
        assert (
            resolve_trainer_type(
                TrainingConfig(trainer_type="dpo", llm_trainer="trl"),
                ModelConfig(model_type="causal_lm"),
            )
            == "dpo"
        )

    def test_resolve_trainer_type_rejects_non_trl_dpo(self):
        with pytest.raises(ValueError, match="requires training.llm_trainer='trl'"):
            resolve_trainer_type(
                TrainingConfig(trainer_type="dpo", llm_trainer="transformers"),
                ModelConfig(model_type="causal_lm"),
            )


class TestGrpoRewardFunctions:
    def test_build_grpo_reward_functions_exact_match(self):
        reward_func = build_grpo_reward_functions(
            GRPOConfig(reward_funcs=["exact_match"], reward_column="answer")
        )[0]

        assert reward_func(["2", "3"], answer=["2", "4"]) == [1.0, 0.0]

    def test_build_grpo_reward_functions_regex(self):
        reward_func = build_grpo_reward_functions(
            GRPOConfig(reward_funcs=["regex"], reward_regex=r"^Answer:")
        )[0]

        assert reward_func(["Answer: yes", "no"]) == [1.0, 0.0]


class TestPrepareModelWithTokenizer:
    """Tests for prepare_model_for_training with tokenizer parameter."""

    def test_prepare_with_tokenizer_sets_pad_token_id(self):
        """Test that prepare_model_for_training syncs pad_token_id with tokenizer."""

        class MockConfig:
            use_cache = True
            pad_token_id = None

        class MockGenerationConfig:
            pad_token_id = None

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.config = MockConfig()
                self.generation_config = MockGenerationConfig()

        class MockTokenizer:
            pad_token_id = 42

        model = MockModel()
        tokenizer = MockTokenizer()
        config = TrainingConfig(gradient_checkpointing=False)

        result = prepare_model_for_training(model, config, tokenizer=tokenizer)

        assert result.config.pad_token_id == 42
        assert result.generation_config.pad_token_id == 42

    def test_prepare_without_tokenizer(self):
        """Test that prepare_model_for_training works without tokenizer."""
        model = nn.Linear(10, 5)
        config = TrainingConfig(gradient_checkpointing=False)

        result = prepare_model_for_training(model, config, tokenizer=None)

        # Should not raise and return the model
        assert result is not None
