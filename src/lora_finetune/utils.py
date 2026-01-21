"""Utility functions for LoRA finetuning."""

import logging
import os
import random
import sys

import numpy as np
import torch


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, level.upper()),
    )
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
    for gpu, info in stats.items():
        print(
            f"{gpu}: {info['allocated_gb']:.2f}GB allocated, "
            f"{info['reserved_gb']:.2f}GB reserved, "
            f"{info['free_gb']:.2f}GB free / {info['total_gb']:.2f}GB total"
        )


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
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Trainable %: {stats['trainable_percent']:.4f}%")
    print(f"Model size: {stats['param_size_mb']:.2f} MB")


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


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
