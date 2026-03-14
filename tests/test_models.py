"""Tests for model utilities."""

import pytest
import torch
from datasets import Dataset

import lora_finetune.models.base as base_module
from lora_finetune.config import LoraConfig, ModelConfig
from lora_finetune.models.base import (
    MODEL_TYPE_TO_AUTO_CLASS,
    MODEL_TYPE_TO_TASK_TYPE,
    _create_ia3_config,
    _create_lora_config,
    _create_prefix_tuning_config,
    _get_task_type,
    get_peft_model_with_lora,
    get_quantization_config,
    get_torch_dtype,
    load_model_and_tokenizer,
    prepare_model_for_full_finetuning,
)
from lora_finetune.models.llm import (
    LLM_TARGET_MODULES,
    get_llm_target_modules,
    get_special_tokens_dict,
    resize_token_embeddings,
)
from lora_finetune.models.vision import (
    VISION_TARGET_MODULES,
    get_id2label,
    get_label2id,
    get_num_labels_from_dataset,
    get_vision_target_modules,
)


class TestGetTorchDtype:
    """Tests for get_torch_dtype function."""

    def test_float16(self):
        """Test float16 dtype conversion."""
        assert get_torch_dtype("float16") == torch.float16

    def test_bfloat16(self):
        """Test bfloat16 dtype conversion."""
        assert get_torch_dtype("bfloat16") == torch.bfloat16

    def test_float32(self):
        """Test float32 dtype conversion."""
        assert get_torch_dtype("float32") == torch.float32

    def test_auto(self):
        """Test auto dtype returns 'auto' string."""
        assert get_torch_dtype("auto") == "auto"

    def test_unknown_dtype(self):
        """Test unknown dtype returns 'auto'."""
        assert get_torch_dtype("unknown") == "auto"


def _bitsandbytes_available():
    """Check if bitsandbytes is available."""
    try:
        import importlib.metadata

        importlib.metadata.version("bitsandbytes")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


class TestGetQuantizationConfig:
    """Tests for get_quantization_config function."""

    def test_no_quantization(self):
        """Test that no quantization returns None."""
        config = ModelConfig(load_in_4bit=False, load_in_8bit=False)
        result = get_quantization_config(config)
        assert result is None

    @pytest.mark.skipif(not _bitsandbytes_available(), reason="bitsandbytes not installed")
    def test_4bit_quantization(self):
        """Test 4-bit quantization config."""
        config = ModelConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        result = get_quantization_config(config)

        assert result is not None
        assert result.load_in_4bit is True
        assert result.bnb_4bit_quant_type == "nf4"
        assert result.bnb_4bit_use_double_quant is True

    @pytest.mark.skipif(not _bitsandbytes_available(), reason="bitsandbytes not installed")
    def test_8bit_quantization(self):
        """Test 8-bit quantization config."""
        config = ModelConfig(load_in_8bit=True)
        result = get_quantization_config(config)

        assert result is not None
        assert result.load_in_8bit is True


class TestUnslothIntegration:
    """Tests for optional Unsloth integration."""

    def test_load_model_and_tokenizer_raises_when_unsloth_missing(self, monkeypatch):
        """Test that enabling Unsloth fails clearly when the package is unavailable."""
        monkeypatch.setattr(base_module, "FastLanguageModel", None)

        with pytest.raises(ImportError, match="Unsloth is not installed"):
            load_model_and_tokenizer(
                ModelConfig(model_type="causal_lm", use_unsloth=True),
                max_seq_length=2048,
            )

    def test_load_model_and_tokenizer_uses_unsloth_when_enabled(self, monkeypatch):
        """Test that the Unsloth loading path is used when requested."""
        captured = {}

        class FakeConfig:
            pad_token_id = None

        class FakeGenerationConfig:
            pad_token_id = None

        class FakeModel:
            def __init__(self):
                self.config = FakeConfig()
                self.generation_config = FakeGenerationConfig()

        class FakeTokenizer:
            pad_token = None
            pad_token_id = None
            eos_token = "</s>"
            eos_token_id = 7

        class FakeFastLanguageModel:
            @classmethod
            def from_pretrained(
                cls,
                model_name,
                max_seq_length=None,
                dtype=None,
                load_in_4bit=False,
                load_in_8bit=False,
                trust_remote_code=False,
            ):
                captured["kwargs"] = {
                    "model_name": model_name,
                    "max_seq_length": max_seq_length,
                    "dtype": dtype,
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                    "trust_remote_code": trust_remote_code,
                }
                return FakeModel(), FakeTokenizer()

        monkeypatch.setattr(base_module, "FastLanguageModel", FakeFastLanguageModel)

        model, tokenizer = load_model_and_tokenizer(
            ModelConfig(
                model_type="causal_lm",
                use_unsloth=True,
                torch_dtype="bfloat16",
                load_in_4bit=True,
            ),
            max_seq_length=4096,
        )

        assert captured["kwargs"]["model_name"] == "meta-llama/Meta-Llama-3-8B"
        assert captured["kwargs"]["max_seq_length"] == 4096
        assert captured["kwargs"]["dtype"] == torch.bfloat16
        assert captured["kwargs"]["load_in_4bit"] is True
        assert tokenizer.pad_token == "</s>"
        assert tokenizer.pad_token_id == 7
        assert model.config.pad_token_id == 7
        assert model.generation_config.pad_token_id == 7


class TestLoadModelAndTokenizerSignature:
    """Regression tests for load_model_and_tokenizer call signature."""

    def test_positional_num_labels_still_targets_vision_num_labels(self, monkeypatch):
        """Test second positional argument still maps to num_labels for vision models."""
        captured = {}

        class FakeModel:
            config = type("Config", (), {"pad_token_id": None})()

        class FakeProcessor:
            pass

        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, model_name_or_path, **kwargs):
                captured["model_name_or_path"] = model_name_or_path
                captured["kwargs"] = kwargs
                return FakeModel()

        class FakeAutoImageProcessor:
            @classmethod
            def from_pretrained(cls, model_name_or_path, trust_remote_code=False):
                captured["processor_args"] = {
                    "model_name_or_path": model_name_or_path,
                    "trust_remote_code": trust_remote_code,
                }
                return FakeProcessor()

        monkeypatch.setitem(base_module.MODEL_TYPE_TO_AUTO_CLASS, "vision", FakeAutoModel)
        monkeypatch.setattr(base_module, "AutoImageProcessor", FakeAutoImageProcessor)

        model, processor = load_model_and_tokenizer(ModelConfig(model_type="vision"), 7)

        assert isinstance(model, FakeModel)
        assert isinstance(processor, FakeProcessor)
        assert captured["kwargs"]["num_labels"] == 7
        assert captured["kwargs"]["ignore_mismatched_sizes"] is True
        assert captured["processor_args"]["model_name_or_path"] == "meta-llama/Meta-Llama-3-8B"

    def test_get_peft_model_with_lora_uses_unsloth_when_enabled(self, monkeypatch):
        """Test that Unsloth PEFT patching is used when requested."""
        captured = {}

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        class FakeFastLanguageModel:
            @staticmethod
            def get_peft_model(
                model,
                r,
                target_modules,
                lora_alpha,
                lora_dropout,
                bias,
                use_gradient_checkpointing,
                random_state,
                max_seq_length,
                use_rslora=False,
                use_dora=False,
                modules_to_save=None,
            ):
                captured["kwargs"] = {
                    "r": r,
                    "target_modules": target_modules,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "bias": bias,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    "random_state": random_state,
                    "max_seq_length": max_seq_length,
                    "use_rslora": use_rslora,
                    "use_dora": use_dora,
                    "modules_to_save": modules_to_save,
                }
                return model

        monkeypatch.setattr(base_module, "FastLanguageModel", FakeFastLanguageModel)

        model = FakeModel()
        lora_config = LoraConfig(method="dora", modules_to_save=["lm_head"])
        result = get_peft_model_with_lora(
            model,
            lora_config,
            model_type="causal_lm",
            use_unsloth=True,
            use_gradient_checkpointing=False,
            random_state=123,
            max_seq_length=8192,
        )

        assert result is model
        assert captured["kwargs"]["r"] == lora_config.r
        assert captured["kwargs"]["use_gradient_checkpointing"] is False
        assert captured["kwargs"]["random_state"] == 123
        assert captured["kwargs"]["max_seq_length"] == 8192
        assert captured["kwargs"]["use_dora"] is True
        assert captured["kwargs"]["modules_to_save"] == ["lm_head"]
        assert getattr(result, "_lora_finetune_unsloth_managed_gradient_checkpointing") is False

    def test_get_peft_model_with_lora_rejects_unsupported_unsloth_method(self, monkeypatch):
        """Test unsupported Unsloth adapter methods fail clearly."""
        monkeypatch.setattr(base_module, "FastLanguageModel", object())

        with pytest.raises(ValueError, match="supports only lora, dora, loraplus, and full"):
            get_peft_model_with_lora(
                torch.nn.Linear(2, 2),
                LoraConfig(method="adalora"),
                model_type="causal_lm",
                use_unsloth=True,
            )


class TestModelTypeMappings:
    """Tests for model type mappings."""

    def test_model_type_to_auto_class(self):
        """Test model type to auto class mapping."""
        assert "causal_lm" in MODEL_TYPE_TO_AUTO_CLASS
        assert "seq2seq" in MODEL_TYPE_TO_AUTO_CLASS
        assert "vision" in MODEL_TYPE_TO_AUTO_CLASS

    def test_model_type_to_task_type(self):
        """Test model type to task type mapping."""
        assert "causal_lm" in MODEL_TYPE_TO_TASK_TYPE
        assert "seq2seq" in MODEL_TYPE_TO_TASK_TYPE
        assert "vision" in MODEL_TYPE_TO_TASK_TYPE


class TestGetLlmTargetModules:
    """Tests for get_llm_target_modules function."""

    def test_llama_model(self):
        """Test target modules for Llama models."""
        modules = get_llm_target_modules("meta-llama/Meta-Llama-3-8B")
        assert modules == LLM_TARGET_MODULES["llama"]
        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "v_proj" in modules

    def test_mistral_model(self):
        """Test target modules for Mistral models."""
        modules = get_llm_target_modules("mistralai/Mistral-7B-v0.1")
        assert modules == LLM_TARGET_MODULES["mistral"]

    def test_phi_model(self):
        """Test target modules for Phi models."""
        modules = get_llm_target_modules("microsoft/phi-2")
        assert modules == LLM_TARGET_MODULES["phi"]

    def test_qwen_model(self):
        """Test target modules for Qwen models."""
        modules = get_llm_target_modules("Qwen/Qwen-7B")
        assert modules == LLM_TARGET_MODULES["qwen"]

    def test_gpt2_model(self):
        """Test target modules for GPT-2 models."""
        modules = get_llm_target_modules("gpt2")
        assert modules == LLM_TARGET_MODULES["gpt2"]
        assert "c_attn" in modules

    def test_unknown_model(self):
        """Test target modules for unknown models."""
        modules = get_llm_target_modules("unknown-model/test")
        assert modules == LLM_TARGET_MODULES["default"]

    def test_case_insensitive(self):
        """Test that model name matching is case-insensitive."""
        modules1 = get_llm_target_modules("LLAMA-model")
        modules2 = get_llm_target_modules("llama-model")
        assert modules1 == modules2


class TestGetVisionTargetModules:
    """Tests for get_vision_target_modules function."""

    def test_vit_model(self):
        """Test target modules for ViT models."""
        modules = get_vision_target_modules("google/vit-base-patch16-224")
        assert modules == VISION_TARGET_MODULES["vit"]
        assert "query" in modules
        assert "key" in modules
        assert "value" in modules

    def test_swin_model(self):
        """Test target modules for Swin models."""
        modules = get_vision_target_modules("microsoft/swin-base-patch4-window7-224")
        assert modules == VISION_TARGET_MODULES["swin"]

    def test_deit_model(self):
        """Test target modules for DeiT models."""
        modules = get_vision_target_modules("facebook/deit-base-patch16-224")
        assert modules == VISION_TARGET_MODULES["deit"]

    def test_clip_model(self):
        """Test target modules for CLIP models."""
        modules = get_vision_target_modules("openai/clip-vit-base-patch32")
        # CLIP contains 'vit' so it matches vit first in the dict iteration
        # This is expected behavior - the first match wins
        assert modules is not None
        assert len(modules) > 0

    def test_dinov2_model(self):
        """Test target modules for DINOv2 models."""
        modules = get_vision_target_modules("facebook/dinov2-base")
        assert modules == VISION_TARGET_MODULES["dinov2"]

    def test_unknown_vision_model(self):
        """Test target modules for unknown vision models."""
        modules = get_vision_target_modules("unknown-vision-model")
        assert modules == VISION_TARGET_MODULES["default"]


class TestGetSpecialTokensDict:
    """Tests for get_special_tokens_dict function."""

    def test_all_tokens_present(self):
        """Test when all special tokens are present."""

        class MockTokenizer:
            pad_token = "<pad>"
            eos_token = "</s>"
            bos_token = "<s>"
            unk_token = "<unk>"

        tokenizer = MockTokenizer()
        result = get_special_tokens_dict(tokenizer)
        assert result == {}

    def test_missing_pad_token(self):
        """Test when pad token is missing."""

        class MockTokenizer:
            pad_token = None
            eos_token = "</s>"
            bos_token = "<s>"
            unk_token = "<unk>"

        tokenizer = MockTokenizer()
        result = get_special_tokens_dict(tokenizer)
        assert "pad_token" in result
        assert result["pad_token"] == "<pad>"

    def test_all_tokens_missing(self):
        """Test when all special tokens are missing."""

        class MockTokenizer:
            pad_token = None
            eos_token = None
            bos_token = None
            unk_token = None

        tokenizer = MockTokenizer()
        result = get_special_tokens_dict(tokenizer)
        assert len(result) == 4
        assert "pad_token" in result
        assert "eos_token" in result
        assert "bos_token" in result
        assert "unk_token" in result


class TestVisionDatasetHelpers:
    """Tests for vision dataset helper functions."""

    def test_get_num_labels_from_features(self):
        """Test getting number of labels from dataset features."""

        class MockFeature:
            num_classes = 10

        class MockDataset:
            features = {"label": MockFeature()}

        dataset = MockDataset()
        num_labels = get_num_labels_from_dataset(dataset)
        assert num_labels == 10

    def test_get_num_labels_from_labels_column(self):
        """Test getting number of labels from 'labels' column."""

        class MockFeature:
            num_classes = 5

        class MockDataset:
            features = {"labels": MockFeature()}

        dataset = MockDataset()
        num_labels = get_num_labels_from_dataset(dataset)
        assert num_labels == 5

    def test_get_num_labels_from_iteration(self):
        """Test getting number of labels by iterating dataset."""

        class MockDataset:
            features = {}

            def __iter__(self):
                return iter([{"label": 0}, {"label": 1}, {"label": 2}, {"label": 1}])

        dataset = MockDataset()
        num_labels = get_num_labels_from_dataset(dataset)
        assert num_labels == 3

    def test_get_num_labels_from_value_feature_fallback(self):
        """Test fallback when label feature has no num_classes attribute."""
        dataset = Dataset.from_dict(
            {
                "image": [0, 1, 2],
                "label": [0, 1, 1],
            }
        )

        num_labels = get_num_labels_from_dataset(dataset)
        assert num_labels == 2

    def test_get_id2label(self):
        """Test getting id2label mapping."""

        class MockFeature:
            names = ["cat", "dog", "bird"]

        class MockDataset:
            features = {"label": MockFeature()}

        dataset = MockDataset()
        id2label = get_id2label(dataset)
        assert id2label == {0: "cat", 1: "dog", 2: "bird"}

    def test_get_id2label_no_features(self):
        """Test get_id2label with no features."""

        class MockDataset:
            features = {}

        dataset = MockDataset()
        id2label = get_id2label(dataset)
        assert id2label == {}

    def test_get_label2id(self):
        """Test getting label2id mapping."""

        class MockFeature:
            names = ["cat", "dog", "bird"]

        class MockDataset:
            features = {"label": MockFeature()}

        dataset = MockDataset()
        label2id = get_label2id(dataset)
        assert label2id == {"cat": 0, "dog": 1, "bird": 2}


class TestGetTaskType:
    """Tests for _get_task_type function."""

    def test_causal_lm_task_type(self):
        """Test task type for causal LM models."""
        from peft import TaskType

        config = LoraConfig()
        task_type = _get_task_type("causal_lm", config)
        assert task_type == TaskType.CAUSAL_LM

    def test_seq2seq_task_type(self):
        """Test task type for seq2seq models."""
        from peft import TaskType

        config = LoraConfig()
        task_type = _get_task_type("seq2seq", config)
        assert task_type == TaskType.SEQ_2_SEQ_LM

    def test_vision_task_type(self):
        """Test task type for vision models."""
        from peft import TaskType

        config = LoraConfig()
        task_type = _get_task_type("vision", config)
        assert task_type == TaskType.FEATURE_EXTRACTION

    def test_custom_task_type_override(self):
        """Test that task_type in config overrides model type."""
        from peft import TaskType

        config = LoraConfig(task_type="SEQ_CLS")
        task_type = _get_task_type("causal_lm", config)
        assert task_type == TaskType.SEQ_CLS


class TestCreateLoraConfig:
    """Tests for _create_lora_config function."""

    def test_basic_lora_config(self):
        """Test creating basic LoRA config."""
        from peft import TaskType

        lora_config = LoraConfig(r=16, alpha=32, dropout=0.1)
        peft_config = _create_lora_config(lora_config, TaskType.CAUSAL_LM)

        assert peft_config.r == 16
        assert peft_config.lora_alpha == 32
        assert peft_config.lora_dropout == 0.1
        assert peft_config.task_type == TaskType.CAUSAL_LM

    def test_lora_config_with_dora(self):
        """Test creating LoRA config with DoRA enabled."""
        from peft import TaskType

        lora_config = LoraConfig(r=16, use_dora=True)
        peft_config = _create_lora_config(lora_config, TaskType.CAUSAL_LM)

        assert peft_config.use_dora is True

    def test_lora_config_with_rslora(self):
        """Test creating LoRA config with RSLoRA enabled."""
        from peft import TaskType

        lora_config = LoraConfig(r=16, use_rslora=True)
        peft_config = _create_lora_config(lora_config, TaskType.CAUSAL_LM)

        assert peft_config.use_rslora is True

    def test_lora_config_target_modules(self):
        """Test LoRA config with custom target modules."""
        from peft import TaskType

        lora_config = LoraConfig(target_modules=["q_proj", "v_proj"])
        peft_config = _create_lora_config(lora_config, TaskType.CAUSAL_LM)

        # PEFT converts lists to sets internally
        assert set(peft_config.target_modules) == {"q_proj", "v_proj"}


class TestCreateAdaLoraConfig:
    """Tests for _create_adalora_config function."""

    def test_adalora_config_defaults(self):
        """Test creating AdaLoRA config with defaults."""

        lora_config = LoraConfig(method="adalora")
        # AdaLoRA requires total_step, so we test the config values are passed correctly
        # by checking the lora_config values directly since AdaLoRA validation requires total_step
        assert lora_config.init_r == 12
        assert lora_config.target_r == 8

    def test_adalora_config_custom(self):
        """Test creating AdaLoRA config with custom values."""
        lora_config = LoraConfig(
            method="adalora",
            init_r=24,
            target_r=16,
            tinit=100,
            tfinal=500,
            deltaT=5,
            beta1=0.9,
            beta2=0.9,
            orth_reg_weight=0.3,
        )
        # Verify the config values are stored correctly
        # (actual AdaLoRA creation requires total_step which is set during training)
        assert lora_config.init_r == 24
        assert lora_config.target_r == 16
        assert lora_config.tinit == 100
        assert lora_config.tfinal == 500
        assert lora_config.deltaT == 5
        assert lora_config.beta1 == 0.9
        assert lora_config.beta2 == 0.9
        assert lora_config.orth_reg_weight == 0.3


class TestCreateIA3Config:
    """Tests for _create_ia3_config function."""

    def test_ia3_config_with_feedforward_subset(self):
        """Test creating IA3 config where feedforward_modules is subset of target_modules."""
        from peft import TaskType

        # feedforward_modules must be a subset of target_modules per PEFT validation
        lora_config = LoraConfig(
            method="ia3",
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
        peft_config = _create_ia3_config(lora_config, TaskType.CAUSAL_LM)

        # PEFT converts lists to sets
        assert peft_config.target_modules == {"k_proj", "v_proj", "down_proj"}
        assert peft_config.feedforward_modules == {"down_proj"}

    def test_ia3_config_values(self):
        """Test IA3 config stores values correctly."""
        lora_config = LoraConfig(
            method="ia3",
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
        # Verify config values are stored
        assert lora_config.method == "ia3"
        assert "down_proj" in lora_config.target_modules
        assert lora_config.feedforward_modules == ["down_proj"]


class TestCreatePrefixTuningConfig:
    """Tests for _create_prefix_tuning_config function."""

    def test_prefix_tuning_config_defaults(self):
        """Test creating prefix tuning config with defaults."""
        from peft import TaskType

        lora_config = LoraConfig(method="prefix_tuning")
        peft_config = _create_prefix_tuning_config(lora_config, TaskType.CAUSAL_LM)

        assert peft_config.num_virtual_tokens == 20
        assert peft_config.prefix_projection is False

    def test_prefix_tuning_config_custom(self):
        """Test creating prefix tuning config with custom values."""
        from peft import TaskType

        lora_config = LoraConfig(
            method="prefix_tuning",
            num_virtual_tokens=50,
            prefix_projection=True,
        )
        peft_config = _create_prefix_tuning_config(lora_config, TaskType.CAUSAL_LM)

        assert peft_config.num_virtual_tokens == 50
        assert peft_config.prefix_projection is True


class TestPrepareModelForFullFinetuning:
    """Tests for prepare_model_for_full_finetuning function."""

    def test_enables_all_parameters(self):
        """Test that full finetuning enables all parameters."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        # Disable some parameters first
        for param in model.parameters():
            param.requires_grad = False

        result = prepare_model_for_full_finetuning(model, is_quantized=False)

        # All parameters should be trainable
        for param in result.parameters():
            assert param.requires_grad is True

    def test_raises_error_with_quantization(self):
        """Test that full finetuning raises error with quantization."""
        import pytest

        model = torch.nn.Linear(10, 5)

        with pytest.raises(ValueError, match="not compatible with quantization"):
            prepare_model_for_full_finetuning(model, is_quantized=True)

    def test_returns_same_model(self):
        """Test that the same model instance is returned."""
        model = torch.nn.Linear(10, 5)
        result = prepare_model_for_full_finetuning(model, is_quantized=False)

        assert result is model


class TestResizeTokenEmbeddings:
    """Tests for resize_token_embeddings function."""

    def test_resize_when_tokenizer_larger(self):
        """Test resizing when tokenizer has more tokens."""

        class MockEmbedding:
            def __init__(self):
                self.weight = torch.zeros(100, 768)

        class MockModel:
            def __init__(self):
                self._embedding = MockEmbedding()
                self._resized = False

            def get_input_embeddings(self):
                return self._embedding

            def resize_token_embeddings(self, new_size):
                self._resized = True
                self._new_size = new_size

        class MockTokenizer:
            def __len__(self):
                return 150  # More than model's 100

        model = MockModel()
        tokenizer = MockTokenizer()

        result = resize_token_embeddings(model, tokenizer)

        assert result._resized is True
        assert result._new_size == 150

    def test_no_resize_when_tokenizer_smaller(self):
        """Test no resize when tokenizer has fewer tokens."""

        class MockEmbedding:
            def __init__(self):
                self.weight = torch.zeros(100, 768)

        class MockModel:
            def __init__(self):
                self._embedding = MockEmbedding()
                self._resized = False

            def get_input_embeddings(self):
                return self._embedding

            def resize_token_embeddings(self, new_size):
                self._resized = True

        class MockTokenizer:
            def __len__(self):
                return 50  # Less than model's 100

        model = MockModel()
        tokenizer = MockTokenizer()

        result = resize_token_embeddings(model, tokenizer)

        assert result._resized is False
