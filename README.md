# LoRA Finetuning

A flexible Hugging Face-based framework for finetuning language, vision, and text classification models with LoRA-style adapters, optional quantization, and configurable training backends.

## Features

- **Supported model types**
  - `causal_lm`
  - `seq2seq`
  - `vision`
  - `text_classification`

- **Supported finetuning methods**
  - `lora`
  - `dora`
  - `adalora`
  - `loraplus`
  - `ia3`
  - `prefix_tuning`
  - `full`

- **Training backends**
  - TRL `SFTTrainer`, `DPOTrainer`, and `GRPOTrainer` for `causal_lm`
  - Transformers `Trainer` for vision models, text classification models, and non-TRL paths

- **Performance options**
  - Flash Attention 2
  - Gradient checkpointing
  - Mixed precision (`bf16` / `fp16`)
  - 4-bit / 8-bit loading via bitsandbytes
  - FSDP and DeepSpeed-compatible training arguments
  - Optional Unsloth integration for supported causal LM setups

- **Data handling**
  - Hugging Face datasets or local `json`, `jsonl`, and `csv`
  - Instruction-style, Q&A, plain text, text classification, prompt-completion, and conversation-style data
  - Optional response-only / assistant-only loss masking
  - Optional holdout evaluation datasets and benchmark evaluation with `lighteval`

- **Experiment tracking**
  - Weights & Biases integration
  - Rich console tables and progress output

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Base install
uv sync

# Optional extras
uv sync --extra flash
uv sync --extra bnb
uv sync --extra unsloth
uv sync --extra eval
uv sync --extra dev

# Combine extras as needed
uv sync --extra flash --extra bnb --extra eval --extra dev

# Everything
uv sync --all-extras
```

### CUDA index selection

By default, `torch` and `torchvision` are resolved from the `pytorch-cu126` index. To switch CUDA wheels, set `UV_INDEX` when syncing:

```bash
UV_INDEX=pytorch-cu128 uv sync
```

Available indices:

- `pytorch-cu126`
- `pytorch-cu128`
- `pytorch-cu130`

## Quick start

### Llama 3 LoRA finetuning

The shipped `llama3_lora.yaml` config enables:

- Flash Attention 2
- TRL training
- W&B logging
- benchmark evaluation on `gsm8k`

If you want the example to run from a base install, disable the optional features from the CLI:

```bash
uv run lora-train \
  --config configs/llama3_lora.yaml \
  --no_use_flash_attention_2 \
  --report_to none \
  --no_bench_enabled
```

If you want the full config as-shipped, install the required extras first:

```bash
uv sync --extra flash --extra eval
uv run wandb login
uv run lora-train --config configs/llama3_lora.yaml
```

You may also need Hugging Face access for gated checkpoints such as Meta Llama 3.

### Mistral LoRA finetuning

```bash
uv run lora-train \
  --config configs/mistral_lora.yaml \
  --no_use_flash_attention_2 \
  --report_to none \
  --no_bench_enabled
```

### ViT image classification

```bash
uv run lora-train \
  --config configs/vit_lora.yaml \
  --report_to none
```

### RoBERTa text classification

```bash
uv run lora-train \
  --config configs/roberta_text_classification_lora.yaml \
  --report_to none
```

## Included example configs

The `configs/` directory currently includes:

- `llama3_lora.yaml`
- `llama3_lora_hpo.yaml`
- `llama3_dpo.yaml`
- `llama3_grpo.yaml`
- `llama3_4bit_lora.yaml`
- `llama3_dora.yaml`
- `llama3_adalora.yaml`
- `llama3_loraplus.yaml`
- `llama3_ia3.yaml`
- `llama3_prefix_tuning.yaml`
- `llama3_full.yaml`
- `mistral_lora.yaml`
- `roberta_text_classification_lora.yaml`
- `vit_lora.yaml`
- `fsdp_config.yaml`

## Running training

The package installs a `lora-train` entry point:

```bash
uv run lora-train --config configs/llama3_lora.yaml
```

Equivalent module invocation:

```bash
uv run python -m lora_finetune.train --config configs/llama3_lora.yaml
```

### CLI overrides

Most non-dict config fields are exposed as CLI flags, including nested benchmark and augmentation settings via prefixes.

- **Model fields**
  - `--model_name_or_path`
  - `--model_type`
  - `--use_unsloth`
  - `--load_in_4bit`

- **LoRA fields**
  - `--lora_method`
  - `--lora_r`
  - `--lora_alpha`
  - `--lora_target_modules`

- **Training fields**
  - `--learning_rate`
  - `--trainer_type`
  - `--llm_trainer`
  - `--report_to`

- **Trainer-mode fields**
  - `--dpo_beta`
  - `--grpo_reward_funcs`
  - `--grpo_reward_column`

- **HPO fields**
  - `--hpo_enabled`
  - `--hpo_n_trials`
  - `--hpo_metric_name`
  - `--hpo_direction`
  - `--hpo_project`

- **Benchmark eval fields**
  - `--bench_enabled`
  - `--bench_tasks`
  - `--bench_eval_steps`

Boolean flags support both enable and disable forms:

- `--use_unsloth`
- `--no_use_unsloth`
- `--bench_enabled`
- `--no_bench_enabled`

Example:

```bash
uv run lora-train \
  --config configs/llama3_lora.yaml \
  --llm_trainer transformers \
  --report_to none \
  --no_bench_enabled
```

## Configuration overview

The runtime config is composed from:

- `model`
- `lora`
- `data`
- `training`
- `dpo`
- `grpo`
- `hpo`
- `benchmark_eval`

Example:

```yaml
model:
  model_name_or_path: meta-llama/Meta-Llama-3-8B
  model_type: causal_lm
  use_unsloth: false
  use_flash_attention_2: true
  load_in_4bit: false

lora:
  method: lora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

data:
  dataset_name: tatsu-lab/alpaca
  max_seq_length: 2048
  eval_split_ratio: 0.05
  response_only_loss: true
  append_eos_token: true

training:
  output_dir: ./outputs/llama3-lora
  trainer_type: sft
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  gradient_checkpointing: true
  llm_trainer: trl
  report_to: wandb
  wandb_watch: false

dpo:
  beta: 0.1

grpo:
  reward_funcs:
    - non_empty

hpo:
  enabled: false
  backend: wandb
  method: random
  n_trials: 8

benchmark_eval:
  enabled: true
  tasks: gsm8k
  eval_steps: 500
```

## Data formats

Text and text classification training support several dataset layouts.

- **Plain text**
  - A column matching `data.text_column` (default: `text`)

- **Instruction data**
  - `instruction`
  - optional `input`
  - `output` or `response`

- **Question answering**
  - `question`
  - `answer`

- **Text classification**
  - A text column matching `data.text_column`
  - A label column matching `data.label_column`

- **Prompt-completion**
  - `prompt`
  - `completion`

- **Preference data for DPO**
  - `prompt`
  - `chosen`
  - `rejected`
  - or `instruction` / optional `input` together with `chosen` / `rejected`

- **Conversational data**
  - `messages`
  - or `conversations`, which is normalized to `messages`

`prompt` / `completion`, `messages`, and `conversations` are intended for the TRL-backed causal LM path.

For GRPO, the training split must provide a `prompt` column or a source column that can be normalized into `prompt`, such as:

- `data.text_column`
- `instruction` with optional `input`
- `question`

Additional columns are preserved and can be consumed by custom or built-in GRPO reward functions.

For local files, set `data.train_file` and optionally `data.validation_file`.

For evaluation splits, the code supports:

- `data.validation_file`
- `data.eval_split_ratio`
- `data.eval_dataset_name` with optional config and split

When `data.max_train_samples` or `data.max_eval_samples` is set, the split is shuffled deterministically with `training.data_seed` before sample selection.

## Trainer behavior

- **Causal LM supervised path**
  - `training.trainer_type: sft`
  - Uses TRL `SFTTrainer` when `training.llm_trainer: trl`
  - Uses Transformers `Trainer` when `training.llm_trainer: transformers`

- **Causal LM preference path**
  - `training.trainer_type: dpo`
  - Requires `training.llm_trainer: trl`
  - Uses TRL `DPOTrainer`

- **Causal LM RL path**
  - `training.trainer_type: grpo`
  - Requires `training.llm_trainer: trl`
  - Uses TRL `GRPOTrainer`
  - Built-in reward functions include `non_empty`, `length`, `exact_match`, and `regex`

- **Non-TRL path**
  - Uses Transformers `Trainer`

- **Output directories**
  - The configured `training.output_dir` is used as a base directory
  - Each run writes to a generated subdirectory based on the W&B run name or a timestamp

- **Evaluation guardrail**
  - If `training.eval_strategy` is not `no`, an eval dataset must exist

## Optional integrations

### Unsloth

Install:

```bash
uv sync --extra unsloth
```

Enable with:

```yaml
model:
  use_unsloth: true
```

Current Unsloth integration is limited to `causal_lm` and supports:

- `lora`
- `dora`
- `loraplus`
- `full`

Unsloth currently supports only `training.trainer_type: sft`.

### Bitsandbytes

Install:

```bash
uv sync --extra bnb
```

Then enable:

```yaml
model:
  load_in_4bit: true
```

### Benchmark evaluation with lighteval

Install:

```bash
uv sync --extra eval
```

Configure with:

```yaml
benchmark_eval:
  enabled: true
  tasks: gsm8k,mmlu
  eval_steps: 500
  num_samples: 100
  max_new_tokens: 512
  batch_size: 1
```

This runs generation-based benchmark evaluation during training and after training completes.

Benchmark evaluation currently supports only `training.trainer_type: sft`.

### Weights & Biases

Install and login if you want W&B logging:

```bash
uv run wandb login
```

Useful settings:

```yaml
training:
  report_to: wandb
  wandb_project: my-project
  wandb_run_name: my-run
  wandb_watch: false
  wandb_log_model: false
```

`wandb_watch` defaults to `false`.

### Hyperparameter search with W&B

HPO uses Hugging Face `Trainer.hyperparameter_search()` with the W&B backend.

Start from the included example config:

```bash
uv run lora-train --config configs/llama3_lora_hpo.yaml
```

Minimal YAML shape:

```yaml
training:
  report_to: wandb
  output_dir: ./outputs/llama3-lora-hpo
  eval_strategy: steps
  wandb_project: llama3-lora

data:
  eval_split_ratio: 0.05

hpo:
  enabled: true
  backend: wandb
  method: random
  metric_name: eval_loss
  direction: minimize
  n_trials: 8
  parameters:
    learning_rate:
      distribution: uniform
      min: 1.0e-5
      max: 3.0e-4
    per_device_train_batch_size:
      values: [2, 4, 8]
```

Scalar HPO settings can be overridden from the CLI:

```bash
uv run lora-train \
  --config configs/llama3_lora_hpo.yaml \
  --hpo_n_trials 12 \
  --hpo_metric_name eval_accuracy \
  --hpo_direction maximize
```

Sweep parameter specs can also be overridden with repeatable `--hpo_param NAME=SPEC` flags, where `SPEC` is a YAML/JSON mapping:

```bash
uv run lora-train \
  --config configs/llama3_lora_hpo.yaml \
  --hpo_param "learning_rate={values: [1.0e-5, 2.0e-5, 3.0e-5]}" \
  --hpo_param "warmup_ratio={distribution: uniform, min: 0.0, max: 0.1}"
```

Notes:

- **W&B project resolution**
  - `hpo.project` overrides the sweep project when set.
  - Otherwise HPO falls back to `training.wandb_project`, then `lora-finetune`.

- **Output layout**
  - Each HPO trial gets its own subdirectory under `training.output_dir`.
  - After the sweep completes, the best merged config is written to `best_hpo_config.yaml` in the output directory.

- **Supported sweep parameters**
  - Transformers-backed trainers support `training.*` and `lora.*` sweep parameters.
  - TRL-backed `sft`, `dpo`, and `grpo` currently support `training.*` sweep parameters only.
  - Non-training parameters should use dotted keys such as `lora.r` when supported.

- **Current restrictions**
  - HPO requires `training.report_to: wandb`.
  - HPO requires evaluation data so each trial can be scored.
  - `training.push_to_hub`, `training.resume_from_checkpoint`, and `benchmark_eval.enabled` are not supported during HPO.

## Multi-GPU training

### FSDP with Accelerate

```bash
uv run accelerate launch --config_file configs/fsdp_config.yaml \
  -m lora_finetune.train \
  --config configs/llama3_lora.yaml
```

You can also set `training.fsdp` and `training.fsdp_config` directly in YAML or via CLI.

## Testing

```bash
uv sync --extra dev
uv run pytest
uv run pytest -v
uv run pytest tests/test_config.py
uv run pytest tests/test_config.py::test_function_name
uv run pytest --cov=lora_finetune
```

## Project structure

```text
lora/
├── configs/
├── src/lora_finetune/
│   ├── data/
│   │   ├── text_classification.py
│   │   ├── text_data.py
│   │   ├── text_formatting.py
│   │   ├── text_preprocessing.py
│   │   └── vision_data.py
│   ├── evaluators/
│   ├── models/
│   ├── _optional_unsloth.py
│   ├── cli.py
│   ├── config.py
│   ├── train.py
│   ├── train_modes.py
│   ├── train_runtime.py
│   ├── train_support.py
│   ├── trainer.py
│   ├── trainer_hpo.py
│   ├── trainer_metrics.py
│   ├── trainer_progress.py
│   └── utils.py
├── tests/
├── pyproject.toml
└── README.md
```

`train.py` remains the CLI and module entrypoint, while `train_modes.py`, `train_runtime.py`, and `train_support.py` hold the extracted implementation details.

`trainer.py` and `data/text_data.py` remain stable public import surfaces, with focused helpers split into sibling modules.

## License

MIT
