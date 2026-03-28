# Intruder Dimension Analysis

This repository includes a standalone analysis script for measuring "intruder dimensions" from the paper `LoRA vs Full Fine-tuning: An Illusion of Equivalence` (`arXiv:2410.21228`).

An intruder dimension is a high-ranking singular vector in a fine-tuned weight matrix whose maximum cosine similarity to all singular vectors of the corresponding base weight matrix is below a threshold `epsilon`.

## What the implementation does

The implementation in this repo:

- loads base and tuned weights directly from `safetensors` without instantiating a full Transformers model
- supports both full fine-tuning checkpoints and LoRA adapter checkpoints
- reconstructs LoRA-tuned weights as `W_tuned = W0 + (alpha / r) * (B @ A)`
- runs thin SVD with `torch.linalg.svd`
- compares left singular vectors only, using absolute cosine similarity to account for sign ambiguity

## Supported checkpoint layouts

- **Base model**
  - sharded Hugging Face checkpoints with `model.safetensors.index.json`
  - single-file checkpoints with `model.safetensors`

- **Full fine-tuning checkpoint**
  - `model.safetensors`
  - or `model.safetensors.index.json`

- **LoRA checkpoint**
  - `adapter_config.json`
  - `adapter_model.safetensors`

You can point the script at either:

- a concrete checkpoint directory like `checkpoint-3000`
- a run directory containing `checkpoint-*` subdirectories

If a run directory is provided, the script resolves the latest checkpoint by step number.

## Default module scope

By default the script analyzes the standard LLaMA projection weights:

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

This keeps the first version practical for LLaMA-family checkpoints while staying close to the paper's method.

## CLI usage

Run the script with:

```bash
uv run python scripts/find_intruder_dimensions.py \
  --base-model checkpoints/meta-llama/Meta-Llama-3-8B \
  --tuned outputs/llama3-lora-hpo/llama3_full/amber-sweep-1/run-82lduunn \
  --epsilon 0.5 \
  --k 10 \
  --device auto \
  --output /tmp/llama3_full_intruders.json
```

For a LoRA checkpoint:

```bash
uv run python scripts/find_intruder_dimensions.py \
  --base-model checkpoints/meta-llama/Meta-Llama-3-8B \
  --tuned outputs/llama3-lora-hpo/llama3_lora_r128/dainty-sweep-1/run-xq8s45mw \
  --epsilon 0.5 \
  --k 10 \
  --device auto \
  --output /tmp/llama3_lora_r128_intruders.json
```

For a faster smoke test on a small subset of matrices:

```bash
uv run python scripts/find_intruder_dimensions.py \
  --base-model checkpoints/meta-llama/Meta-Llama-3-8B \
  --tuned outputs/llama3-lora-hpo/llama3_lora_r128/dainty-sweep-1/run-xq8s45mw \
  --module-regex 'model\.layers\.\d+\.self_attn\.q_proj\.weight$' \
  --limit 1 \
  --device auto
```

## CLI arguments

- `--base-model`: path to the original pretrained checkpoint directory
- `--tuned`: path to a fine-tuned checkpoint directory or run directory
- `--epsilon`: intruder threshold for maximum absolute cosine similarity, default `0.5`
- `--k`: number of top singular vectors in the tuned matrix to inspect, default `10`
- `--device`: `auto`, `cpu`, or `cuda`
- `--module-regex`: repeatable regex filter for weight names
- `--output`: optional path to write a JSON report
- `--limit`: optional cap on the number of matrices analyzed

## Output

The script prints a short summary including:

- resolved checkpoint path
- detected checkpoint type (`full` or `lora`)
- number of matrices analyzed
- total number of intruder dimensions
- top matrices ranked by intruder count

If `--output` is provided, it also writes a JSON report with:

- top-level metadata for the run
- total intruder count across analyzed matrices
- one result object per matrix
- per-matrix `max_abs_cosines` and `intruder_indices`

## Python API

The core implementation lives in [src/lora_finetune/intruder_dimensions.py](/home/yizheng/repos/lora/src/lora_finetune/intruder_dimensions.py).

Main entry points:

- `AnalysisConfig`
- `analyze_model(config)`

## Runtime notes

- Exact SVD on LLaMA 3 8B projection matrices is expensive, especially for MLP projections.
- Full fine-tuning checkpoints can be much slower to read than LoRA adapter checkpoints because they store dense model weights.
- `--device auto` prefers CUDA when available, but the implementation falls back to CPU if CUDA SVD fails at runtime.
- The script currently analyzes selected LLaMA projection weights by default, not every 2D parameter matrix in the model.
