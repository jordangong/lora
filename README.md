# LoRA Finetuning

A flexible framework for finetuning vision and language models using LoRA (Low-Rank Adaptation) with HuggingFace libraries.

## Features

- **Multi-model support**: ViT, Mistral, Llama3, and other HuggingFace models
- **TRL-backed LLM finetuning**: causal language models use TRL `SFTTrainer` by default, with a fallback to the standard Transformers trainer
- **Performance optimizations**:
  - Flash Attention 2
  - Gradient checkpointing
  - Fully Sharded Data Parallel (FSDP)
  - Mixed precision training (bf16/fp16)
  - 4-bit/8-bit quantization via bitsandbytes
- **Experiment tracking**: Weights & Biases integration
- **Flexible configuration**: YAML-based configs with CLI overrides

## Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Basic installation
uv sync

# With Flash Attention 2
uv sync --extra flash

# With bitsandbytes quantization
uv sync --extra bnb

# Multiple extras (uv sync is declarative - specify all desired extras together)
uv sync --extra flash --extra bnb --extra dev

# Full installation (all extras)
uv sync --all-extras
```

### CUDA Version Selection

By default, PyTorch is installed with CUDA 12.6. To use a different CUDA version, edit `pyproject.toml`:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }      # Change to desired version
torchvision = { index = "pytorch-cu128" }
```

Available options: `pytorch-cu126`, `pytorch-cu128`, `pytorch-cu130`

Then re-sync:

```bash
uv sync --upgrade-package torch --upgrade-package torchvision
```

## Quick Start

### Finetune Llama3 on text data

```bash
uv run python -m lora_finetune.train \
    --config configs/llama3_lora.yaml \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./outputs/llama3-lora
```

The shipped causal-LM configs set `training.llm_trainer: trl`, which uses TRL's `SFTTrainer` for supervised finetuning.

### Finetune ViT on image classification

```bash
uv run python -m lora_finetune.train \
    --config configs/vit_lora.yaml \
    --model_name_or_path google/vit-base-patch16-224 \
    --dataset_name cifar10 \
    --output_dir ./outputs/vit-lora
```

### Finetune Mistral

```bash
uv run python -m lora_finetune.train \
    --config configs/mistral_lora.yaml \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./outputs/mistral-lora
```

## Configuration

See `configs/` for example configurations. Key parameters:

```yaml
model:
  model_name_or_path: meta-llama/Meta-Llama-3-8B
  model_type: causal_lm  # causal_lm, seq2seq, vision

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  learning_rate: 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  llm_trainer: trl  # trl or transformers; only used for causal_lm
  gradient_checkpointing: true
  bf16: true
  fsdp: "full_shard auto_wrap"
```

For causal language models, `llm_trainer` defaults to `trl`. Set `training.llm_trainer: transformers` if you want to force the legacy Hugging Face `Trainer` path instead.

You can also override it from the CLI:

```bash
uv run python -m lora_finetune.train \
    --config configs/llama3_lora.yaml \
    --llm_trainer transformers
```

## Multi-GPU Training with FSDP

```bash
uv run accelerate launch --config_file configs/fsdp_config.yaml \
    -m lora_finetune.train \
    --config configs/llama3_lora.yaml
```

## Weights & Biases Logging

```bash
# Login to wandb
uv run wandb login

# Enable logging
uv run python -m lora_finetune.train \
    --config configs/llama3_lora.yaml \
    --report_to wandb \
    --wandb_project my-lora-project
```

## Testing

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_config.py

# Run specific test function
uv run pytest tests/test_config.py::test_function_name

# Run with coverage
uv run pytest --cov=lora_finetune
```

## Project Structure

```text
lora-finetune/
├── src/lora_finetune/
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses
│   ├── data/
│   │   ├── __init__.py
│   │   ├── text_data.py   # Text dataset utilities
│   │   └── vision_data.py # Vision dataset utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py        # Base model loading
│   │   ├── llm.py         # LLM-specific utilities
│   │   └── vision.py      # Vision model utilities
│   ├── trainer.py         # Custom trainer with optimizations
│   ├── utils.py           # Utility functions
│   └── train.py           # Main training script
├── configs/               # Example configurations
├── scripts/               # Helper scripts
└── examples/              # Example notebooks
```

## License

MIT
