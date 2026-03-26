# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `src/lora_finetune/`. Keep feature-specific logic grouped by area: `models/` for model wrappers, `data/` for dataset loading and preprocessing, `evaluators/` for benchmark integrations, and top-level modules such as `cli.py`, `config.py`, `trainer.py`, and `train.py` for entry points and orchestration. Example training configs live in `configs/*.yaml`. Tests mirror package behavior under `tests/` and should stay close to the module they validate.

## Build, Test, and Development Commands
Use `uv` for local setup and execution.

- `uv sync --extra dev`: install the project with test and lint tooling.
- `uv sync --all-extras`: install optional integrations such as Flash Attention, bitsandbytes, Unsloth, and LightEval.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_cli.py`: run a focused test module while iterating.
- `uv run black src tests`: format Python sources and tests.
- `uv run ruff check src tests`: run lint checks.
- `uv run lora-train --config configs/llama3_lora.yaml --report_to none`: launch a local training run from a shipped config.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow the existing style in `pyproject.toml`: 4-space indentation and a 100-character line limit. Format with Black and lint with Ruff before opening a PR. Use `snake_case` for functions, variables, and module names; `PascalCase` for classes; keep helper functions private with a leading underscore when they are internal-only. Prefer small, focused modules over adding unrelated logic to `train.py` or `trainer.py`.

## Testing Guidelines
Tests use `pytest`. Name files `test_*.py`, classes `Test*`, and test functions `test_*`. Add or update tests whenever behavior changes in CLI parsing, config handling, trainers, data pipelines, or model integration paths. Favor targeted unit tests first, then add broader training-path coverage only where it catches regressions.

## Commit & Pull Request Guidelines
Recent history uses imperative, descriptive commit subjects, for example: `Add effective_batch_size property to TrainingConfig`. Keep commits focused and explain the observable behavior change. PRs should include a short summary, affected configs or modules, test results from `uv run pytest`, and relevant logs or screenshots when a CLI/UI-facing change alters output.

## Configuration & Environment Tips
Keep secrets out of YAML configs. Use environment-backed auth for services such as Hugging Face and W&B. If CUDA wheels need to change, set `UV_INDEX` during sync, for example `UV_INDEX=pytorch-cu128 uv sync`.
