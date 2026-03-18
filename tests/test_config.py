"""Tests for configuration dataclasses."""

import os
import tempfile

import yaml

from lora_finetune.config import (
    AugmentationConfig,
    BenchmarkEvalConfig,
    Config,
    DataConfig,
    DPOConfig,
    GRPOConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)


class TestLoraConfig:
    """Tests for LoraConfig dataclass."""

    def test_default_values(self):
        """Test default LoRA configuration values."""
        config = LoraConfig()
        assert config.method == "lora"
        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert config.bias == "none"
        assert config.task_type is None
        assert config.modules_to_save is None

    def test_default_target_modules(self):
        """Test that target_modules defaults to attention projections."""
        config = LoraConfig()
        assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_custom_target_modules(self):
        """Test custom target modules are preserved."""
        custom_modules = ["query", "key", "value"]
        config = LoraConfig(target_modules=custom_modules)
        assert config.target_modules == custom_modules

    def test_custom_values(self):
        """Test custom LoRA configuration values."""
        config = LoraConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            bias="all",
            task_type="CAUSAL_LM",
        )
        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == 0.1
        assert config.bias == "all"
        assert config.task_type == "CAUSAL_LM"

    def test_method_full_finetuning(self):
        """Test full finetuning method configuration."""
        config = LoraConfig(method="full")
        assert config.method == "full"

    def test_method_dora(self):
        """Test DoRA method configuration."""
        config = LoraConfig(method="dora")
        assert config.method == "dora"
        # DoRA should auto-enable use_dora flag
        assert config.use_dora is True

    def test_method_dora_explicit_flag(self):
        """Test DoRA with explicit use_dora flag."""
        config = LoraConfig(method="lora", use_dora=True)
        assert config.use_dora is True

    def test_method_adalora(self):
        """Test AdaLoRA method configuration."""
        config = LoraConfig(
            method="adalora",
            init_r=12,
            target_r=8,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            beta1=0.85,
            beta2=0.85,
            orth_reg_weight=0.5,
        )
        assert config.method == "adalora"
        assert config.init_r == 12
        assert config.target_r == 8
        assert config.tinit == 200
        assert config.tfinal == 1000
        assert config.deltaT == 10
        assert config.beta1 == 0.85
        assert config.beta2 == 0.85
        assert config.orth_reg_weight == 0.5

    def test_method_loraplus(self):
        """Test LoRA+ method configuration."""
        config = LoraConfig(method="loraplus", loraplus_lr_ratio=16.0)
        assert config.method == "loraplus"
        assert config.loraplus_lr_ratio == 16.0

    def test_loraplus_default_lr_ratio(self):
        """Test LoRA+ default learning rate ratio."""
        config = LoraConfig(method="loraplus")
        assert config.loraplus_lr_ratio == 16.0

    def test_method_ia3(self):
        """Test IA3 method configuration."""
        config = LoraConfig(
            method="ia3",
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
        assert config.method == "ia3"
        assert config.feedforward_modules == ["down_proj"]

    def test_method_prefix_tuning(self):
        """Test prefix tuning method configuration."""
        config = LoraConfig(
            method="prefix_tuning",
            num_virtual_tokens=20,
            prefix_projection=True,
        )
        assert config.method == "prefix_tuning"
        assert config.num_virtual_tokens == 20
        assert config.prefix_projection is True

    def test_prefix_tuning_defaults(self):
        """Test prefix tuning default values."""
        config = LoraConfig(method="prefix_tuning")
        assert config.num_virtual_tokens == 20
        assert config.prefix_projection is False

    def test_rslora_flag(self):
        """Test RSLoRA scaling flag."""
        config = LoraConfig(use_rslora=True)
        assert config.use_rslora is True


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default model configuration values."""
        config = ModelConfig()
        assert config.model_name_or_path == "meta-llama/Meta-Llama-3-8B"
        assert config.model_type == "causal_lm"
        assert config.use_unsloth is False
        assert config.torch_dtype == "auto"
        assert config.trust_remote_code is False
        assert config.use_flash_attention_2 is True
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False

    def test_quantization_config(self):
        """Test quantization configuration."""
        config = ModelConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
        )
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "fp4"
        assert config.bnb_4bit_use_double_quant is False

    def test_vision_model_type(self):
        """Test vision model type configuration."""
        config = ModelConfig(
            model_name_or_path="google/vit-base-patch16-224",
            model_type="vision",
        )
        assert config.model_type == "vision"

    def test_text_classification_model_type(self):
        config = ModelConfig(
            model_name_or_path="roberta-base",
            model_type="text_classification",
        )
        assert config.model_type == "text_classification"

    def test_unsloth_flag(self):
        """Test Unsloth opt-in flag."""
        config = ModelConfig(use_unsloth=True)
        assert config.use_unsloth is True


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_values(self):
        """Test default augmentation configuration values."""
        config = AugmentationConfig()
        assert config.random_resized_crop is True
        assert config.random_horizontal_flip is True
        assert config.random_vertical_flip is False
        assert config.color_jitter is False
        assert config.random_rotation is False
        assert config.auto_augment is None
        assert config.rand_augment is False
        assert config.trivial_augment is False

    def test_custom_augmentation(self):
        """Test custom augmentation configuration."""
        config = AugmentationConfig(
            random_resized_crop=False,
            color_jitter=True,
            color_jitter_brightness=0.4,
            random_rotation=True,
            random_rotation_degrees=30.0,
        )
        assert config.random_resized_crop is False
        assert config.color_jitter is True
        assert config.color_jitter_brightness == 0.4
        assert config.random_rotation is True
        assert config.random_rotation_degrees == 30.0

    def test_normalization_values(self):
        """Test normalization configuration."""
        config = AugmentationConfig(
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
        )
        assert config.normalize_mean == [0.5, 0.5, 0.5]
        assert config.normalize_std == [0.5, 0.5, 0.5]


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test default data configuration values."""
        config = DataConfig()
        assert config.dataset_name is None
        assert config.text_column == "text"
        assert config.label_column == "label"
        assert config.image_column == "image"
        assert config.max_seq_length == 2048
        assert config.image_size == 224
        assert config.streaming is False
        assert config.response_only_loss is True
        assert config.assistant_only_loss is False
        assert config.append_eos_token is True
        assert config.eos_token is None

    def test_custom_values(self):
        """Test custom data configuration values."""
        config = DataConfig(
            dataset_name="tatsu-lab/alpaca",
            max_seq_length=4096,
            preprocessing_num_workers=8,
            response_only_loss=False,
            assistant_only_loss=True,
            append_eos_token=False,
            eos_token="<eos>",
        )
        assert config.dataset_name == "tatsu-lab/alpaca"
        assert config.max_seq_length == 4096
        assert config.preprocessing_num_workers == 8
        assert config.response_only_loss is False
        assert config.assistant_only_loss is True
        assert config.append_eos_token is False
        assert config.eos_token == "<eos>"

    def test_augmentation_nested(self):
        """Test nested augmentation config."""
        aug_config = AugmentationConfig(color_jitter=True)
        config = DataConfig(augmentation=aug_config)
        assert config.augmentation.color_jitter is True


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default training configuration values."""
        config = TrainingConfig()
        assert config.output_dir == "./outputs"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.bf16 is True
        assert config.gradient_checkpointing is True
        assert config.optim == "adamw_torch_fused"
        assert config.wandb_watch == "false"
        assert config.wandb_console == "auto"
        assert config.trainer_type == "sft"

    def test_custom_values(self):
        """Test custom training configuration values."""
        config = TrainingConfig(
            output_dir="/custom/output",
            num_train_epochs=5,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
        )
        assert config.output_dir == "/custom/output"
        assert config.num_train_epochs == 5
        assert config.learning_rate == 1e-4
        assert config.per_device_train_batch_size == 8
        assert config.gradient_accumulation_steps == 8


class TestBenchmarkEvalConfig:
    """Tests for BenchmarkEvalConfig dataclass."""

    def test_rejects_non_positive_eval_steps(self):
        """Test benchmark eval steps must be positive."""
        import pytest

        with pytest.raises(ValueError, match="eval_steps"):
            BenchmarkEvalConfig(eval_steps=0)

    def test_rejects_non_positive_num_samples(self):
        """Test num_samples must be positive when provided."""
        import pytest

        with pytest.raises(ValueError, match="num_samples"):
            BenchmarkEvalConfig(num_samples=0)

    def test_accepts_none_num_samples(self):
        """Test num_samples=None remains valid."""
        config = BenchmarkEvalConfig(num_samples=None)
        assert config.num_samples is None


class TestDPOConfig:
    def test_default_values(self):
        config = DPOConfig()
        assert config.beta == 0.1
        assert config.max_prompt_length == 512
        assert config.max_length == 1024
        assert config.reference_free is False


class TestGRPOConfig:
    def test_default_values(self):
        config = GRPOConfig()
        assert config.reward_funcs == ["non_empty"]
        assert config.reward_column == "answer"
        assert config.num_generations == 4

    def test_requires_reward_functions(self):
        import pytest

        with pytest.raises(ValueError, match="reward_funcs"):
            GRPOConfig(reward_funcs=[])


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.lora, LoraConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.dpo, DPOConfig)
        assert isinstance(config.grpo, GRPOConfig)
        assert config.training.llm_trainer == "trl"
        assert config.training.trainer_type == "sft"

    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = {
            "model": {
                "model_name_or_path": "test-model",
                "model_type": "causal_lm",
            },
            "lora": {
                "r": 8,
                "alpha": 16,
            },
            "data": {
                "dataset_name": "test-dataset",
                "max_seq_length": 1024,
            },
            "training": {
                "output_dir": "./test-output",
                "num_train_epochs": 1,
                "trainer_type": "dpo",
                "llm_trainer": "transformers",
            },
            "dpo": {"beta": 0.2},
            "grpo": {"reward_funcs": ["length"]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)
            assert config.model.model_name_or_path == "test-model"
            assert config.lora.r == 8
            assert config.lora.alpha == 16
            assert config.data.dataset_name == "test-dataset"
            assert config.data.max_seq_length == 1024
            assert config.training.output_dir == "./test-output"
            assert config.training.num_train_epochs == 1
            assert config.training.trainer_type == "dpo"
            assert config.training.llm_trainer == "transformers"
            assert config.dpo.beta == 0.2
            assert config.grpo.reward_funcs == ["length"]
        finally:
            os.unlink(temp_path)

    def test_from_yaml_with_augmentation(self):
        """Test loading configuration with nested augmentation from YAML."""
        yaml_content = {
            "model": {"model_name_or_path": "test-model"},
            "lora": {"r": 16},
            "data": {
                "dataset_name": "test-dataset",
                "augmentation": {
                    "color_jitter": True,
                    "random_rotation": True,
                    "random_rotation_degrees": 15.0,
                },
            },
            "training": {"output_dir": "./output"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)
            assert config.data.augmentation.color_jitter is True
            assert config.data.augmentation.random_rotation is True
            assert config.data.augmentation.random_rotation_degrees == 15.0
        finally:
            os.unlink(temp_path)

    def test_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = Config()
        config.model.model_name_or_path = "saved-model"
        config.lora.r = 32

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            # Read raw content to verify key values are present
            with open(temp_path, "r") as f:
                content = f.read()

            assert "saved-model" in content
            assert "r: 32" in content
        finally:
            os.unlink(temp_path)

    def test_update_from_args(self):
        """Test updating configuration from command line arguments."""
        config = Config()
        args = {
            "model_name_or_path": "updated-model",
            "r": 64,
            "learning_rate": 5e-5,
            "dataset_name": "new-dataset",
            "nonexistent_arg": "should_be_ignored",
        }

        config.update_from_args(args)

        assert config.model.model_name_or_path == "updated-model"
        assert config.lora.r == 64
        assert config.training.learning_rate == 5e-5
        assert config.data.dataset_name == "new-dataset"

    def test_update_from_args_none_values(self):
        """Test that None values in args are ignored."""
        config = Config()
        original_lr = config.training.learning_rate

        args = {
            "learning_rate": None,
        }

        config.update_from_args(args)
        assert config.training.learning_rate == original_lr

    def test_from_yaml_with_empty_file(self):
        """Test loading config from an empty YAML file uses defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)
            assert isinstance(config.model, ModelConfig)
            assert isinstance(config.training, TrainingConfig)
        finally:
            os.unlink(temp_path)
