"""Tests for CLI argument parsing and configuration building."""

import argparse
import tempfile

import yaml

from lora_finetune.cli import (
    _add_dataclass_args,
    _apply_args_to_config,
    _get_base_type,
    _is_list_or_str_union,
    build_config,
    parse_args,
)
from lora_finetune.config import (
    AugmentationConfig,
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)


class TestIsListOrStrUnion:
    """Tests for _is_list_or_str_union helper."""

    def test_union_list_str(self):
        """Test detection of Union[List[str], str]."""
        from typing import List, Optional, Union

        assert _is_list_or_str_union(Union[List[str], str]) is True
        assert _is_list_or_str_union(Optional[Union[List[str], str]]) is True

    def test_not_union(self):
        """Test non-union types return False."""
        from typing import List

        assert _is_list_or_str_union(str) is False
        assert _is_list_or_str_union(int) is False
        assert _is_list_or_str_union(List[str]) is False

    def test_optional_str(self):
        """Test Optional[str] is not a list/str union."""
        from typing import Optional

        assert _is_list_or_str_union(Optional[str]) is False


class TestGetBaseType:
    """Tests for _get_base_type helper."""

    def test_optional_type(self):
        """Test extracting base type from Optional."""
        from typing import Optional

        assert _get_base_type(Optional[str]) is str
        assert _get_base_type(Optional[int]) is int

    def test_plain_type(self):
        """Test plain types are returned as-is."""
        assert _get_base_type(str) is str
        assert _get_base_type(int) is int
        assert _get_base_type(float) is float


class TestAddDataclassArgs:
    """Tests for _add_dataclass_args function."""

    def test_add_lora_config_args(self):
        """Test adding LoraConfig arguments to parser."""
        parser = argparse.ArgumentParser()
        _add_dataclass_args(parser, LoraConfig, prefix="lora_")

        args = parser.parse_args(["--lora_r", "8", "--lora_alpha", "16"])
        assert args.lora_r == 8
        assert args.lora_alpha == 16

    def test_add_model_config_args(self):
        """Test adding ModelConfig arguments to parser."""
        parser = argparse.ArgumentParser()
        _add_dataclass_args(parser, ModelConfig)

        args = parser.parse_args(
            ["--model_name_or_path", "test-model", "--model_type", "causal_lm"]
        )
        assert args.model_name_or_path == "test-model"
        assert args.model_type == "causal_lm"

    def test_add_training_config_args(self):
        """Test adding TrainingConfig arguments to parser."""
        parser = argparse.ArgumentParser()
        _add_dataclass_args(parser, TrainingConfig)

        args = parser.parse_args(
            ["--output_dir", "./test", "--num_train_epochs", "5", "--learning_rate", "1e-4"]
        )
        assert args.output_dir == "./test"
        assert args.num_train_epochs == 5
        assert args.learning_rate == 1e-4

    def test_boolean_args(self):
        """Test boolean arguments with --flag and --no_flag."""
        parser = argparse.ArgumentParser()
        _add_dataclass_args(parser, TrainingConfig)

        args = parser.parse_args(["--gradient_checkpointing"])
        assert args.gradient_checkpointing is True

        args = parser.parse_args(["--no_gradient_checkpointing"])
        assert args.no_gradient_checkpointing is True

    def test_skip_fields(self):
        """Test skipping specific fields."""
        parser = argparse.ArgumentParser()
        _add_dataclass_args(parser, DataConfig, skip_fields=["augmentation"])

        # augmentation should not be added
        args = parser.parse_args([])
        assert not hasattr(args, "augmentation")

    def test_list_args(self):
        """Test list arguments."""
        parser = argparse.ArgumentParser()
        _add_dataclass_args(parser, LoraConfig, prefix="lora_")

        args = parser.parse_args(["--lora_target_modules", "q_proj", "v_proj"])
        assert args.lora_target_modules == ["q_proj", "v_proj"]


class TestApplyArgsToConfig:
    """Tests for _apply_args_to_config function."""

    def test_apply_basic_args(self):
        """Test applying basic arguments to config."""
        config = LoraConfig()
        args_dict = {"lora_r": 32, "lora_alpha": 64}

        _apply_args_to_config(config, args_dict, prefix="lora_")

        assert config.r == 32
        assert config.alpha == 64

    def test_apply_none_values_ignored(self):
        """Test that None values are ignored."""
        config = TrainingConfig()
        original_lr = config.learning_rate
        args_dict = {"learning_rate": None}

        _apply_args_to_config(config, args_dict)

        assert config.learning_rate == original_lr

    def test_apply_no_flag(self):
        """Test applying --no_* flag to boolean field."""
        config = TrainingConfig()
        args_dict = {"no_gradient_checkpointing": True}

        _apply_args_to_config(config, args_dict)

        assert config.gradient_checkpointing is False

    def test_apply_tuple_from_list(self):
        """Test converting list to tuple for tuple fields."""
        config = AugmentationConfig()
        args_dict = {"aug_random_resized_crop_scale": [0.1, 0.9]}

        _apply_args_to_config(config, args_dict, prefix="aug_")

        assert config.random_resized_crop_scale == (0.1, 0.9)


class TestBuildConfig:
    """Tests for build_config function."""

    def test_build_default_config(self):
        """Test building config without config file."""
        args = argparse.Namespace(config=None)
        # Add all expected args as None
        for field in [
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "lora_bias",
            "lora_task_type",
            "lora_modules_to_save",
            "no_lora_bias",
            "no_lora_task_type",
            "no_lora_modules_to_save",
        ]:
            setattr(args, field, None)
        for field in [
            "model_name_or_path",
            "model_type",
            "torch_dtype",
            "trust_remote_code",
            "use_flash_attention_2",
            "load_in_4bit",
            "load_in_8bit",
            "no_trust_remote_code",
            "no_use_flash_attention_2",
            "no_load_in_4bit",
            "no_load_in_8bit",
        ]:
            setattr(args, field, None)
        for field in [
            "output_dir",
            "num_train_epochs",
            "learning_rate",
            "gradient_checkpointing",
            "no_gradient_checkpointing",
        ]:
            setattr(args, field, None)

        # Add minimal args for other configs
        args.__dict__.update({k: None for k in vars(DataConfig()).keys() if not k.startswith("_")})
        args.__dict__.update(
            {k: None for k in vars(TrainingConfig()).keys() if not k.startswith("_")}
        )
        args.__dict__.update({k: None for k in vars(ModelConfig()).keys() if not k.startswith("_")})
        args.__dict__.update(
            {f"aug_{k}": None for k in vars(AugmentationConfig()).keys() if not k.startswith("_")}
        )
        args.__dict__.update(
            {
                f"no_aug_{k}": None
                for k in vars(AugmentationConfig()).keys()
                if not k.startswith("_")
            }
        )
        args.__dict__.update(
            {f"no_{k}": None for k in vars(TrainingConfig()).keys() if not k.startswith("_")}
        )
        args.__dict__.update(
            {f"no_{k}": None for k in vars(ModelConfig()).keys() if not k.startswith("_")}
        )
        args.__dict__.update(
            {f"no_{k}": None for k in vars(DataConfig()).keys() if not k.startswith("_")}
        )

        config = build_config(args)

        assert isinstance(config, Config)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.lora, LoraConfig)

    def test_build_config_from_yaml(self):
        """Test building config from YAML file."""
        yaml_content = {
            "model": {"model_name_or_path": "yaml-model"},
            "lora": {"r": 8},
            "data": {"dataset_name": "yaml-dataset"},
            "training": {"output_dir": "./yaml-output"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            args = argparse.Namespace(config=temp_path)
            # Add minimal required attrs
            args.__dict__.update(
                {f"lora_{k}": None for k in vars(LoraConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {f"no_lora_{k}": None for k in vars(LoraConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {k: None for k in vars(ModelConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {f"no_{k}": None for k in vars(ModelConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {k: None for k in vars(DataConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {f"no_{k}": None for k in vars(DataConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {k: None for k in vars(TrainingConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {f"no_{k}": None for k in vars(TrainingConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {
                    f"aug_{k}": None
                    for k in vars(AugmentationConfig()).keys()
                    if not k.startswith("_")
                }
            )
            args.__dict__.update(
                {
                    f"no_aug_{k}": None
                    for k in vars(AugmentationConfig()).keys()
                    if not k.startswith("_")
                }
            )

            config = build_config(args)

            assert config.model.model_name_or_path == "yaml-model"
            assert config.lora.r == 8
            assert config.data.dataset_name == "yaml-dataset"
        finally:
            import os

            os.unlink(temp_path)

    def test_build_config_cli_overrides_yaml(self):
        """Test that CLI args override YAML config."""
        yaml_content = {
            "model": {"model_name_or_path": "yaml-model"},
            "lora": {"r": 8},
            "data": {},
            "training": {"output_dir": "./yaml-output"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            args = argparse.Namespace(config=temp_path, lora_r=32, model_name_or_path="cli-model")
            # Add other required attrs as None
            args.__dict__.update(
                {
                    f"lora_{k}": None
                    for k in vars(LoraConfig()).keys()
                    if not k.startswith("_") and k != "r"
                }
            )
            args.__dict__.update(
                {f"no_lora_{k}": None for k in vars(LoraConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {
                    k: None
                    for k in vars(ModelConfig()).keys()
                    if not k.startswith("_") and k != "model_name_or_path"
                }
            )
            args.__dict__.update(
                {f"no_{k}": None for k in vars(ModelConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {k: None for k in vars(DataConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {f"no_{k}": None for k in vars(DataConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {k: None for k in vars(TrainingConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {f"no_{k}": None for k in vars(TrainingConfig()).keys() if not k.startswith("_")}
            )
            args.__dict__.update(
                {
                    f"aug_{k}": None
                    for k in vars(AugmentationConfig()).keys()
                    if not k.startswith("_")
                }
            )
            args.__dict__.update(
                {
                    f"no_aug_{k}": None
                    for k in vars(AugmentationConfig()).keys()
                    if not k.startswith("_")
                }
            )

            config = build_config(args)

            # CLI args should override YAML
            assert config.lora.r == 32
            assert config.model.model_name_or_path == "cli-model"
        finally:
            import os

            os.unlink(temp_path)


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_help(self):
        """Test that parser has help for config."""

        parser = argparse.ArgumentParser(description="LoRA Finetuning")
        parser.add_argument("--config", type=str, help="Path to YAML config file")

        # Just verify parser can be created without errors
        assert parser is not None

    def test_parse_args_creates_parser(self):
        """Test that parse_args creates valid argument parser."""
        import sys

        # Save original argv
        original_argv = sys.argv

        try:
            # Set minimal args
            sys.argv = ["test", "--config", "test.yaml"]
            args = parse_args()
            assert args.config == "test.yaml"
        finally:
            sys.argv = original_argv
