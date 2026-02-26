"""Utility functions for LoRA finetuning."""

import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Display names for finetuning methods
METHOD_DISPLAY_NAMES = {
    "lora": "LoRA",
    "dora": "DoRA",
    "adalora": "AdaLoRA",
    "loraplus": "LoRA+",
    "ia3": "IA³",
    "prefix_tuning": "Prefix Tuning",
    "full": "Full Finetuning",
}


def get_method_display_name(method: str) -> str:
    """Get the display name for a finetuning method."""
    return METHOD_DISPLAY_NAMES.get(method, method.upper())


class RichWarningHandler(logging.Handler):
    """Custom logging handler that formats warnings elegantly with Rich."""

    def emit(self, record):
        msg = record.getMessage()

        # Simplify common verbose warnings
        if "not initialized from the model checkpoint" in msg:
            msg = "Some model weights were randomly initialized (expected for fine-tuning)"
        elif "Fast image processor" in msg or "slow image processor" in msg:
            msg = "Using standard image processor (fast processor available with use_fast=True)"
        elif "You should probably TRAIN" in msg:
            return  # Skip this one, it's obvious
        elif "generation flags are not valid" in msg:
            return  # Triggered by lighteval's default temperature=0, not actionable
        elif not msg.strip():
            return

        console.print(f"  [dim yellow]⚠ {msg}[/dim yellow]")


def suppress_warnings() -> None:
    """Configure warnings to be captured for elegant display."""
    # Suppress tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Disable HuggingFace tqdm progress bars (we use Rich instead)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Disable transformers/datasets tqdm bars via their logging API
    try:
        from transformers.utils.logging import disable_progress_bar as tf_disable_progress_bar

        tf_disable_progress_bar()
    except ImportError:
        pass

    try:
        from datasets.utils.logging import disable_progress_bar as ds_disable_progress_bar

        ds_disable_progress_bar()
    except ImportError:
        pass

    # Replace transformers logger handlers with our Rich handler
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.handlers = [RichWarningHandler()]
    transformers_logger.setLevel(logging.WARNING)
    transformers_logger.propagate = False

    # Also handle Python warnings module
    def _rich_showwarning(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if (
            "not initialized" in msg
            or "You should probably TRAIN" in msg
            or "generation flags are not valid" in msg
            or not msg.strip()
        ):
            return
        console.print(f"  [dim yellow]⚠ {msg}[/dim yellow]")

    warnings.showwarning = _rich_showwarning


def setup_logging(level: str = "WARNING") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, level.upper()),
    )

    # Suppress verbose transformers/datasets logs
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_memory_usage() -> dict:
    """Get GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        stats[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }
    return stats


def print_gpu_memory_usage() -> None:
    """Print GPU memory usage."""
    stats = get_gpu_memory_usage()
    if not stats:
        return

    table = Table(title="GPU Memory", show_header=True, header_style="bold cyan")
    table.add_column("GPU", style="dim")
    table.add_column("Allocated", justify="right")
    table.add_column("Reserved", justify="right")
    table.add_column("Free", justify="right", style="green")
    table.add_column("Total", justify="right")

    for gpu, info in stats.items():
        table.add_row(
            gpu,
            f"{info['allocated_gb']:.2f} GB",
            f"{info['reserved_gb']:.2f} GB",
            f"{info['free_gb']:.2f} GB",
            f"{info['total_gb']:.2f} GB",
        )
    console.print(table)


def check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


def check_bitsandbytes_available() -> bool:
    """Check if bitsandbytes is available."""
    import importlib.util

    return importlib.util.find_spec("bitsandbytes") is not None


def get_model_size(model: torch.nn.Module) -> dict:
    """Get model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
        "param_size_mb": param_size_bytes / 1024**2,
    }


def print_model_size(model: torch.nn.Module) -> None:
    """Print model size statistics."""
    stats = get_model_size(model)

    table = Table(title="Model Statistics", show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Total parameters", f"{stats['total_params']:,}")
    table.add_row("Trainable parameters", f"{stats['trainable_params']:,}")
    table.add_row("Trainable %", f"{stats['trainable_percent']:.2f}%")
    table.add_row("Model size", f"{stats['param_size_mb']:.2f} MB")

    console.print(Panel(table, border_style="blue"))


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters.

    Convenience wrapper around get_model_size for simple parameter counting.
    """
    stats = get_model_size(model)
    return stats["total_params"], stats["trainable_params"]


def ensure_dir(path: str) -> str:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def find_all_linear_names(model: torch.nn.Module) -> list:
    """Find all linear layer names in model for LoRA targeting."""
    linear_cls = torch.nn.Linear

    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt

        linear_cls = (torch.nn.Linear, Linear4bit, Linear8bitLt)
    except ImportError:
        pass

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            names = name.split(".")
            lora_module_names.add(names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)
