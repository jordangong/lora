"""Tests for configuration dataclasses."""

import os
import tempfile

import yaml

from lora_finetune.config import (
    AugmentationConfig,
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)


class TestLoraConfig:
    """Tests for LoraConfig dataclass."""

    def test_default_values(self):
        """Test default LoRA configuration values."""
        config = LoraConfig()
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


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default model configuration values."""
        config = ModelConfig()
        assert config.model_name_or_path == "meta-llama/Meta-Llama-3-8B"
        assert config.model_type == "causal_lm"
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

    def test_custom_values(self):
        """Test custom data configuration values."""
        config = DataConfig(
            dataset_name="tatsu-lab/alpaca",
            max_seq_length=4096,
            preprocessing_num_workers=8,
        )
        assert config.dataset_name == "tatsu-lab/alpaca"
        assert config.max_seq_length == 4096
        assert config.preprocessing_num_workers == 8

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
        assert config.optim == "adamw_torch"

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


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.lora, LoraConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)

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
            },
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
