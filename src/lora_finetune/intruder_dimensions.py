"""Intruder dimension analysis for fine-tuned weight matrices."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from safetensors import safe_open

DEFAULT_LLAMA_MODULE_REGEXES = [
    r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
]


@dataclass
class AnalysisConfig:
    """Configuration for intruder dimension analysis."""

    base_model_path: Path
    tuned_path: Path
    epsilon: float = 0.5
    k: int = 10
    module_regexes: list[str] = field(default_factory=lambda: list(DEFAULT_LLAMA_MODULE_REGEXES))
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    svd_dtype: torch.dtype = torch.float32
    checkpoint_strategy: str = "latest"
    limit: Optional[int] = None


@dataclass
class MatrixIntruderResult:
    """Per-matrix intruder measurement."""

    weight_name: str
    shape: tuple[int, int]
    intruder_count: int
    examined_k: int
    max_abs_cosines: list[float]
    intruder_indices: list[int]


@dataclass
class ModelIntruderReport:
    """Aggregate model-level intruder analysis report."""

    base_model_path: str
    tuned_path: str
    resolved_checkpoint_path: str
    tuned_type: str
    epsilon: float
    k: int
    device: str
    svd_dtype: str
    total_intruders: int
    num_matrices: int
    results: list[MatrixIntruderResult]

    def to_dict(self) -> dict:
        return asdict(self)


ProgressCallback = Callable[[dict], None]


def resolve_checkpoint_dir(path: Path) -> Path:
    """Resolve a checkpoint directory from a checkpoint or run directory."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if _is_checkpoint_dir(path):
        return path

    checkpoints = []
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if _is_checkpoint_dir(child):
            checkpoints.append((step, child))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {path}")

    return max(checkpoints, key=lambda item: item[0])[1]


def detect_tuned_type(path: Path) -> Literal["full", "lora"]:
    """Detect whether a checkpoint is a full fine-tuning or LoRA checkpoint."""
    path = path.expanduser().resolve()
    if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
        return "lora"
    if (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists():
        return "full"
    raise FileNotFoundError(f"Could not detect checkpoint type under {path}")


def iter_target_weight_names(base_model_path: Path, module_regexes: list[str]) -> list[str]:
    """Return weight names matching the configured module regex filters."""
    index = _load_weight_map(base_model_path)
    return filter_weight_names(index.keys(), module_regexes)


def filter_weight_names(weight_names: list[str] | tuple[str, ...], module_regexes: list[str]) -> list[str]:
    """Filter weight names down to matching 2D weight matrices."""
    patterns = [re.compile(pattern) for pattern in module_regexes]
    filtered = []
    for weight_name in sorted(weight_names):
        if not weight_name.endswith(".weight"):
            continue
        if any(pattern.search(weight_name) for pattern in patterns):
            filtered.append(weight_name)
    return filtered


def load_base_tensor(base_model_path: Path, weight_name: str) -> torch.Tensor:
    """Load a weight tensor from a sharded or unsharded base model directory."""
    return _load_tensor_from_model_dir(base_model_path, weight_name)


def load_full_ft_tensor(tuned_checkpoint_path: Path, weight_name: str) -> torch.Tensor:
    """Load a dense fine-tuned tensor from a full fine-tuning checkpoint."""
    return _load_tensor_from_model_dir(tuned_checkpoint_path, weight_name)


def load_lora_delta_tensor(tuned_checkpoint_path: Path, weight_name: str) -> torch.Tensor:
    """Load and reconstruct the LoRA update for a base weight tensor."""
    tuned_checkpoint_path = tuned_checkpoint_path.expanduser().resolve()
    adapter_path = tuned_checkpoint_path / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {adapter_path}")

    adapter_config_path = tuned_checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {adapter_config_path}")

    adapter_config = json.loads(adapter_config_path.read_text())
    rank = adapter_config["r"]
    alpha = adapter_config["lora_alpha"]
    prefix = _weight_name_to_lora_prefix(weight_name)
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"

    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [key for key in (key_a, key_b) if key not in keys]
        if missing:
            raise KeyError(
                f"Missing LoRA adapter keys for {weight_name}: {', '.join(missing)}"
            )
        lora_a = handle.get_tensor(key_a)
        lora_b = handle.get_tensor(key_b)

    scale = alpha / rank
    delta = (lora_b @ lora_a) * scale
    return delta


def build_tuned_tensor(
    base_tensor: torch.Tensor,
    tuned_checkpoint_path: Path,
    tuned_type: Literal["full", "lora"],
    weight_name: str,
) -> torch.Tensor:
    """Build the post-fine-tuning weight tensor for a specific matrix."""
    if tuned_type == "full":
        return load_full_ft_tensor(tuned_checkpoint_path, weight_name)
    if tuned_type == "lora":
        return base_tensor + load_lora_delta_tensor(tuned_checkpoint_path, weight_name)
    raise ValueError(f"Unsupported tuned type: {tuned_type}")


def compute_intruders(
    base_tensor: torch.Tensor,
    tuned_tensor: torch.Tensor,
    epsilon: float,
    k: int,
    *,
    weight_name: str = "<unknown>",
    device: str = "cpu",
    svd_dtype: torch.dtype = torch.float32,
    progress_callback: Optional[ProgressCallback] = None,
    matrix_index: Optional[int] = None,
    total_matrices: Optional[int] = None,
) -> MatrixIntruderResult:
    """Compute intruder dimensions for a pair of weight matrices."""
    if base_tensor.ndim != 2 or tuned_tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensors for {weight_name}")
    if base_tensor.shape != tuned_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {weight_name}: {tuple(base_tensor.shape)} vs {tuple(tuned_tensor.shape)}"
        )

    _emit_progress(
        progress_callback,
        phase="svd_base",
        weight_name=weight_name,
        matrix_index=matrix_index,
        total_matrices=total_matrices,
    )
    base_u = _compute_left_singular_vectors(base_tensor, device=device, svd_dtype=svd_dtype)
    _emit_progress(
        progress_callback,
        phase="svd_tuned",
        weight_name=weight_name,
        matrix_index=matrix_index,
        total_matrices=total_matrices,
    )
    tuned_u = _compute_left_singular_vectors(tuned_tensor, device=device, svd_dtype=svd_dtype)

    max_rank = min(base_u.shape[1], tuned_u.shape[1])
    examined_k = min(k, max_rank)
    if examined_k == 0:
        return MatrixIntruderResult(
            weight_name=weight_name,
            shape=tuple(base_tensor.shape),
            intruder_count=0,
            examined_k=0,
            max_abs_cosines=[],
            intruder_indices=[],
        )

    _emit_progress(
        progress_callback,
        phase="compare",
        weight_name=weight_name,
        matrix_index=matrix_index,
        total_matrices=total_matrices,
    )
    similarities = torch.abs(base_u.transpose(0, 1) @ tuned_u[:, :examined_k])
    max_abs_cosines = similarities.max(dim=0).values.cpu()
    intruder_indices = [
        index for index, value in enumerate(max_abs_cosines.tolist()) if value < epsilon
    ]

    return MatrixIntruderResult(
        weight_name=weight_name,
        shape=tuple(base_tensor.shape),
        intruder_count=len(intruder_indices),
        examined_k=examined_k,
        max_abs_cosines=[float(value) for value in max_abs_cosines.tolist()],
        intruder_indices=intruder_indices,
    )


def analyze_model(
    config: AnalysisConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> ModelIntruderReport:
    """Analyze all selected weight matrices for intruder dimensions."""
    base_model_path = config.base_model_path.expanduser().resolve()
    tuned_path = config.tuned_path.expanduser().resolve()
    resolved_checkpoint = resolve_checkpoint_dir(tuned_path)
    tuned_type = detect_tuned_type(resolved_checkpoint)

    weight_names = iter_target_weight_names(base_model_path, config.module_regexes)
    if config.limit is not None:
        weight_names = weight_names[: config.limit]

    results = []
    total_matrices = len(weight_names)
    _emit_progress(
        progress_callback,
        phase="start",
        total_matrices=total_matrices,
        tuned_type=tuned_type,
        resolved_checkpoint_path=str(resolved_checkpoint),
    )
    for matrix_index, weight_name in enumerate(weight_names, start=1):
        _emit_progress(
            progress_callback,
            phase="load_base",
            weight_name=weight_name,
            matrix_index=matrix_index,
            total_matrices=total_matrices,
        )
        base_tensor = load_base_tensor(base_model_path, weight_name)
        _emit_progress(
            progress_callback,
            phase="build_tuned",
            weight_name=weight_name,
            matrix_index=matrix_index,
            total_matrices=total_matrices,
        )
        tuned_tensor = build_tuned_tensor(base_tensor, resolved_checkpoint, tuned_type, weight_name)
        result = compute_intruders(
            base_tensor,
            tuned_tensor,
            config.epsilon,
            config.k,
            weight_name=weight_name,
            device=config.device,
            svd_dtype=config.svd_dtype,
            progress_callback=progress_callback,
            matrix_index=matrix_index,
            total_matrices=total_matrices,
        )
        results.append(result)
        _emit_progress(
            progress_callback,
            phase="matrix_complete",
            weight_name=weight_name,
            matrix_index=matrix_index,
            total_matrices=total_matrices,
            intruder_count=result.intruder_count,
        )

    total_intruders = sum(result.intruder_count for result in results)
    _emit_progress(
        progress_callback,
        phase="complete",
        total_matrices=total_matrices,
        total_intruders=total_intruders,
    )
    return ModelIntruderReport(
        base_model_path=str(base_model_path),
        tuned_path=str(tuned_path),
        resolved_checkpoint_path=str(resolved_checkpoint),
        tuned_type=tuned_type,
        epsilon=config.epsilon,
        k=config.k,
        device=config.device,
        svd_dtype=str(config.svd_dtype).replace("torch.", ""),
        total_intruders=total_intruders,
        num_matrices=len(results),
        results=results,
    )


def _is_checkpoint_dir(path: Path) -> bool:
    return (
        (path / "adapter_model.safetensors").exists()
        or (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
    )


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    model_dir = model_dir.expanduser().resolve()
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return index["weight_map"]

    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        with safe_open(str(single_file), framework="pt", device="cpu") as handle:
            return {key: single_file.name for key in handle.keys()}

    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json found under {model_dir}")


def _load_tensor_from_model_dir(model_dir: Path, weight_name: str) -> torch.Tensor:
    weight_map = _load_weight_map(model_dir)
    if weight_name not in weight_map:
        raise KeyError(f"Missing weight {weight_name} under {model_dir}")

    tensor_file = model_dir / weight_map[weight_name]
    with safe_open(str(tensor_file), framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_name)


def _weight_name_to_lora_prefix(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Unexpected base weight name: {weight_name}")
    return f"base_model.model.{weight_name[:-7]}"


def _compute_left_singular_vectors(
    tensor: torch.Tensor,
    *,
    device: str,
    svd_dtype: torch.dtype,
) -> torch.Tensor:
    working_tensor = tensor.to(dtype=svd_dtype)
    selected_device = _resolve_device(device)
    try:
        working_tensor = working_tensor.to(selected_device)
        u, _, _ = torch.linalg.svd(working_tensor, full_matrices=False)
        return u.cpu()
    except RuntimeError:
        if selected_device == "cpu":
            raise
        working_tensor = tensor.to(dtype=svd_dtype, device="cpu")
        u, _, _ = torch.linalg.svd(working_tensor, full_matrices=False)
        return u.cpu()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    *,
    phase: str,
    weight_name: Optional[str] = None,
    matrix_index: Optional[int] = None,
    total_matrices: Optional[int] = None,
    tuned_type: Optional[str] = None,
    resolved_checkpoint_path: Optional[str] = None,
    intruder_count: Optional[int] = None,
    total_intruders: Optional[int] = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "phase": phase,
            "weight_name": weight_name,
            "matrix_index": matrix_index,
            "total_matrices": total_matrices,
            "tuned_type": tuned_type,
            "resolved_checkpoint_path": resolved_checkpoint_path,
            "intruder_count": intruder_count,
            "total_intruders": total_intruders,
        }
    )
