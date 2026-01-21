# LoRA Finetuning

A flexible framework for finetuning vision and language models using LoRA (Low-Rank Adaptation) with HuggingFace libraries.

## Features

- **Multi-model support**: ViT, Mistral, Llama3, and other HuggingFace models
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
# Basic installation
pip install -e .

# With Flash Attention 2
pip install -e ".[flash]"

# With bitsandbytes quantization
pip install -e ".[bnb]"

# Full installation
pip install -e ".[all]"
```

## Quick Start

### Finetune Llama3 on text data

```bash
python -m lora_finetune.train \
    --config configs/llama3_lora.yaml \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./outputs/llama3-lora
```

### Finetune ViT on image classification

```bash
python -m lora_finetune.train \
    --config configs/vit_lora.yaml \
    --model_name_or_path google/vit-base-patch16-224 \
    --dataset_name cifar10 \
    --output_dir ./outputs/vit-lora
```

### Finetune Mistral

```bash
python -m lora_finetune.train \
    --config configs/mistral_lora.yaml \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./outputs/mistral-lora
```

## Configuration

See `configs/` for example configurations. Key parameters:

```yaml
# Model
model_name_or_path: meta-llama/Meta-Llama-3-8B
model_type: causal_lm  # causal_lm, seq2seq, vision

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
learning_rate: 2e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4

# Performance
use_flash_attention_2: true
gradient_checkpointing: true
bf16: true
fsdp: "full_shard auto_wrap"
```

## Multi-GPU Training with FSDP

```bash
accelerate launch --config_file configs/fsdp_config.yaml \
    -m lora_finetune.train \
    --config configs/llama3_lora.yaml
```

## Weights & Biases Logging

```bash
# Login to wandb
wandb login

# Enable logging
python -m lora_finetune.train \
    --config configs/llama3_lora.yaml \
    --report_to wandb \
    --wandb_project my-lora-project
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
