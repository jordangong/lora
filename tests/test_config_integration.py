"""Opt-in integration coverage for shipped training configs."""

import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from lora_finetune.config import Config
from lora_finetune.train import train_llm, train_text_classification, train_vision

_RUN_ROBERTA_INTEGRATION = os.getenv("LORA_RUN_ROBERTA_INTEGRATION") == "1"
_RUN_VIT_INTEGRATION = os.getenv("LORA_RUN_VIT_INTEGRATION") == "1"
_RUN_LLAMA3_INTEGRATION = os.getenv("LORA_RUN_LLAMA3_INTEGRATION") == "1"

_DEFAULT_ROBERTA_MODEL_NAME_OR_PATH = "hf-internal-testing/tiny-random-roberta"
_DEFAULT_VIT_MODEL_NAME_OR_PATH = "hf-internal-testing/tiny-random-ViTForImageClassification"
_DEFAULT_LLAMA3_MODEL_NAME_OR_PATH = "hf-internal-testing/tiny-random-LlamaForCausalLM"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBERTA_CONFIG_PATH = _REPO_ROOT / "configs" / "roberta_text_classification_lora.yaml"
_VIT_CONFIG_PATH = _REPO_ROOT / "configs" / "vit_lora.yaml"
_LLAMA3_CONFIG_PATH = _REPO_ROOT / "configs" / "llama3_lora.yaml"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=color).save(path)


def _disable_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


def _assert_saved_files(output_dir: str, expected_names: list[str]) -> None:
    output_path = Path(output_dir)
    for expected_name in expected_names:
        matches = list(output_path.rglob(expected_name))
        assert matches, f"expected '{expected_name}' to be saved under {output_path}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_roberta_text_classification_config_trains_end_to_end(tmp_path, monkeypatch):
    """Run a short real training loop with the shipped RoBERTa config."""
    if not _RUN_ROBERTA_INTEGRATION:
        pytest.skip("Set LORA_RUN_ROBERTA_INTEGRATION=1 to run the RoBERTa integration test.")

    train_rows = [
        {"sentence": "a charming and funny film", "label": 1},
        {"sentence": "an excellent performance by the cast", "label": 1},
        {"sentence": "warm, witty, and consistently engaging", "label": 1},
        {"sentence": "a thoughtful drama with real emotional weight", "label": 1},
        {"sentence": "dull, predictable, and badly paced", "label": 0},
        {"sentence": "the script is messy and the jokes do not land", "label": 0},
        {"sentence": "a flat sequel that never finds momentum", "label": 0},
        {"sentence": "tedious and overlong from start to finish", "label": 0},
    ]
    eval_rows = [
        {"sentence": "smart, lively, and entertaining", "label": 1},
        {"sentence": "a complete waste of time", "label": 0},
        {"sentence": "surprisingly heartfelt and well acted", "label": 1},
        {"sentence": "clumsy, noisy, and unconvincing", "label": 0},
    ]

    train_file = tmp_path / "roberta_train.jsonl"
    validation_file = tmp_path / "roberta_validation.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(validation_file, eval_rows)

    config = Config.from_yaml(str(_ROBERTA_CONFIG_PATH))
    config.model.model_name_or_path = os.getenv(
        "LORA_ROBERTA_MODEL_NAME_OR_PATH",
        _DEFAULT_ROBERTA_MODEL_NAME_OR_PATH,
    )
    config.data.dataset_name = None
    config.data.dataset_config_name = None
    config.data.train_file = str(train_file)
    config.data.validation_file = str(validation_file)
    config.data.preprocessing_num_workers = 1
    config.data.max_seq_length = 128
    config.data.max_train_samples = len(train_rows)
    config.data.max_eval_samples = len(eval_rows)
    config.training.output_dir = str(tmp_path / "outputs")
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 2
    config.training.per_device_eval_batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.eval_strategy = "epoch"
    config.training.save_strategy = "no"
    config.training.load_best_model_at_end = False
    config.training.logging_steps = 1
    config.training.report_to = "none"
    config.training.optim = "adamw_torch"
    config.training.dataloader_num_workers = 0
    config.training.dataloader_pin_memory = False
    config.training.gradient_checkpointing = True
    config.training.warmup_ratio = 0.0

    _disable_tracking(monkeypatch)

    try:
        train_text_classification(config)
    finally:
        torch.cuda.empty_cache()

    _assert_saved_files(
        config.training.output_dir,
        ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vit_image_classification_config_trains_end_to_end(tmp_path, monkeypatch):
    """Run a short real training loop with the shipped ViT config."""
    if not _RUN_VIT_INTEGRATION:
        pytest.skip("Set LORA_RUN_VIT_INTEGRATION=1 to run the ViT integration test.")

    train_dir = tmp_path / "vit_dataset" / "train"
    validation_dir = tmp_path / "vit_dataset" / "validation"

    for index, color in enumerate(((220, 40, 40), (190, 60, 60), (160, 30, 30), (240, 80, 80))):
        _write_rgb_image(train_dir / "red" / f"red_{index}.png", color)
    for index, color in enumerate(((40, 120, 220), (60, 140, 240), (30, 90, 180), (70, 160, 250))):
        _write_rgb_image(train_dir / "blue" / f"blue_{index}.png", color)
    _write_rgb_image(validation_dir / "red" / "red_eval.png", (200, 50, 50))
    _write_rgb_image(validation_dir / "blue" / "blue_eval.png", (50, 130, 220))

    config = Config.from_yaml(str(_VIT_CONFIG_PATH))
    config.model.model_name_or_path = os.getenv(
        "LORA_VIT_MODEL_NAME_OR_PATH",
        _DEFAULT_VIT_MODEL_NAME_OR_PATH,
    )
    config.data.dataset_name = None
    config.data.dataset_config_name = None
    config.data.train_file = str(train_dir / "**" / "*.png")
    config.data.validation_file = str(validation_dir / "**" / "*.png")
    config.data.image_column = "image"
    config.data.label_column = "label"
    config.data.validation_split = "validation"
    config.data.max_train_samples = 8
    config.data.max_eval_samples = 2
    config.training.output_dir = str(tmp_path / "outputs")
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 2
    config.training.per_device_eval_batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.eval_strategy = "epoch"
    config.training.save_strategy = "no"
    config.training.load_best_model_at_end = False
    config.training.logging_steps = 1
    config.training.report_to = "none"
    config.training.optim = "adamw_torch"
    config.training.fp16 = False
    config.training.bf16 = False
    config.training.dataloader_num_workers = 0
    config.training.dataloader_pin_memory = False
    config.training.warmup_ratio = 0.0

    _disable_tracking(monkeypatch)

    try:
        train_vision(config)
    finally:
        torch.cuda.empty_cache()

    _assert_saved_files(
        config.training.output_dir,
        ["adapter_config.json", "adapter_model.safetensors"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama3_sft_config_trains_end_to_end(tmp_path, monkeypatch):
    """Run a short real SFT loop with the shipped Llama3 config and a tiny public checkpoint."""
    if not _RUN_LLAMA3_INTEGRATION:
        pytest.skip("Set LORA_RUN_LLAMA3_INTEGRATION=1 to run the Llama3 integration test.")

    train_rows = [
        {"prompt": "Translate to French: cat", "completion": " chat"},
        {"prompt": "Translate to French: dog", "completion": " chien"},
        {"prompt": "Translate to French: house", "completion": " maison"},
        {"prompt": "Translate to French: red", "completion": " rouge"},
        {"prompt": "Translate to French: blue", "completion": " bleu"},
        {"prompt": "Translate to French: green", "completion": " vert"},
    ]
    eval_rows = [
        {"prompt": "Translate to French: bird", "completion": " oiseau"},
        {"prompt": "Translate to French: black", "completion": " noir"},
    ]

    train_file = tmp_path / "llama3_train.jsonl"
    validation_file = tmp_path / "llama3_validation.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(validation_file, eval_rows)

    config = Config.from_yaml(str(_LLAMA3_CONFIG_PATH))
    config.model.model_name_or_path = os.getenv(
        "LORA_LLAMA3_MODEL_NAME_OR_PATH",
        _DEFAULT_LLAMA3_MODEL_NAME_OR_PATH,
    )
    config.model.use_flash_attention_2 = False
    config.model.torch_dtype = "float16"
    config.data.dataset_name = None
    config.data.dataset_config_name = None
    config.data.train_file = str(train_file)
    config.data.validation_file = str(validation_file)
    config.data.eval_split_ratio = None
    config.data.preprocessing_num_workers = 1
    config.data.max_seq_length = 128
    config.data.max_train_samples = len(train_rows)
    config.data.max_eval_samples = len(eval_rows)
    config.benchmark_eval.enabled = False
    config.training.output_dir = str(tmp_path / "outputs")
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 1
    config.training.per_device_eval_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.eval_strategy = "epoch"
    config.training.save_strategy = "no"
    config.training.load_best_model_at_end = False
    config.training.logging_steps = 1
    config.training.report_to = "none"
    config.training.optim = "adamw_torch"
    config.training.fp16 = True
    config.training.bf16 = False
    config.training.dataloader_num_workers = 0
    config.training.dataloader_pin_memory = False
    config.training.gradient_checkpointing = True
    config.training.warmup_ratio = 0.0

    _disable_tracking(monkeypatch)

    try:
        train_llm(config)
    finally:
        torch.cuda.empty_cache()

    _assert_saved_files(
        config.training.output_dir,
        ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"],
    )
