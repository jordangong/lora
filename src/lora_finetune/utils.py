"""Utility functions for LoRA finetuning."""

import logging
import os
import random
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
_VERBOSE_LOGGING_ENABLED = False

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ORPHAN_SGR_RE = re.compile(r"\[(?:\d{1,3}(?:;\d{1,3})*)m")

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


@dataclass(frozen=True)
class WarningRule:
    contains_any: tuple[str, ...] = ()
    contains_all: tuple[str, ...] = ()
    logger_names: tuple[str, ...] = ()
    replacement: Optional[str] = None
    suppress: bool = False


WARNING_RULES = (
    WarningRule(
        contains_any=("not initialized from the model checkpoint",),
        replacement="Some model weights were randomly initialized (expected for fine-tuning)",
    ),
    WarningRule(
        contains_any=("Fast image processor", "slow image processor"),
        replacement="Using standard image processor (fast processor available with use_fast=True)",
    ),
    WarningRule(contains_any=("You should probably TRAIN",), suppress=True),
    WarningRule(contains_any=("generation flags are not valid",), suppress=True),
    WarningRule(
        contains_any=("unauthenticated requests to the HF Hub",),
        replacement="HF Hub requests are unauthenticated; set HF_TOKEN for higher rate limits and faster downloads",
    ),
    WarningRule(
        contains_all=("Detected kernel version", "below the recommended minimum of"),
        logger_names=("accelerate",),
        replacement="Kernel version is below Accelerate's recommended minimum and may cause hangs",
    ),
)

CAPTURED_OUTPUT_RULES = (
    WarningRule(
        contains_any=("Unsloth: Will patch your computer to enable 2x faster free finetuning.",),
        suppress=True,
    ),
    WarningRule(
        contains_any=("Unsloth Zoo will now patch everything to make training faster!",),
        suppress=True,
    ),
    WarningRule(
        contains_any=(
            "Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!",
        ),
        suppress=True,
    ),
    WarningRule(
        contains_any=("Unsloth: Padding-free auto-enabled, enabling faster training.",),
        suppress=True,
    ),
    WarningRule(
        contains_all=("==((====))==", "Unsloth", "Transformers:"),
        suppress=True,
    ),
    WarningRule(
        contains_any=(
            "Num GPUs =",
            "CUDA Toolkit:",
            "Bfloat16 =",
            "Free license: http://github.com/unslothai/unsloth",
        ),
        suppress=True,
    ),
    WarningRule(
        contains_any=("Unsloth - 2x faster free finetuning |",),
        suppress=True,
    ),
    WarningRule(
        contains_all=("Unsloth", "patched", "QKV layers"),
        suppress=True,
    ),
    WarningRule(
        contains_all=("Unsloth: Will load", "as a legacy tokenizer"),
        replacement="Unsloth is using the legacy tokenizer path",
    ),
    WarningRule(
        contains_any=("Dropout = 0 is supported for fast patching",),
        replacement=(
            "Unsloth fast patching is partially disabled because LoRA dropout is non-zero; "
            "expect lower performance. Set LoRA dropout to 0 for full fast patching."
        ),
    ),
)

CapturedOutputAction = Literal["render", "suppress", "passthrough"]


def get_method_display_name(method: str) -> str:
    """Get the display name for a finetuning method."""
    return METHOD_DISPLAY_NAMES.get(method, method.upper())


def _normalize_warning_message(msg: str) -> str:
    """Normalize warning message text for clean terminal rendering."""
    cleaned = ANSI_ESCAPE_RE.sub("", msg)
    cleaned = ORPHAN_SGR_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    # Transformers can emit a multiline LOAD REPORT with mismatched checkpoint keys.
    # Summarize it to avoid awkward line wrapping and duplicated noise.
    if "LOAD REPORT" in cleaned:
        mismatched_keys = []
        for line in cleaned.splitlines():
            if "MISMATCH" not in line or "|" not in line:
                continue
            key = line.split("|", 1)[0].strip()
            if key and key.lower() != "key":
                mismatched_keys.append(key)

        if mismatched_keys:
            joined_keys = ", ".join(mismatched_keys)
            return f"Checkpoint shape mismatch detected; reinitialized: {joined_keys}"
        return "Checkpoint shape mismatch detected; some weights were reinitialized"

    # Keep warnings compact and single-line.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return " ".join(lines)


def _warning_rule_matches(msg: str, logger_name: str, rule: WarningRule) -> bool:
    if rule.logger_names and not any(
        logger_name == name or logger_name.startswith(f"{name}.") for name in rule.logger_names
    ):
        return False
    if rule.contains_all and not all(text in msg for text in rule.contains_all):
        return False
    if rule.contains_any and not any(text in msg for text in rule.contains_any):
        return False
    return bool(rule.contains_any or rule.contains_all)


def format_warning_message(
    msg: str,
    *,
    logger_name: str = "",
    extra_rules: Sequence[WarningRule] = (),
) -> Optional[str]:
    formatted = _normalize_warning_message(msg)
    if not formatted:
        return None

    for rule in (*extra_rules, *WARNING_RULES):
        if not _warning_rule_matches(formatted, logger_name, rule):
            continue
        if rule.suppress:
            return None
        if rule.replacement is not None:
            return rule.replacement

    return formatted


def format_captured_output_message(msg: str) -> Optional[str]:
    """Normalize and classify captured third-party stdout/stderr for normal mode."""
    action, formatted = _classify_captured_output_message(msg)
    if action != "render":
        return None
    return formatted


def _classify_captured_output_message(msg: str) -> tuple[CapturedOutputAction, Optional[str]]:
    """Classify captured stdout/stderr for rendering, suppression, or passthrough."""
    formatted = _normalize_warning_message(msg)
    if not formatted:
        return "suppress", None

    if formatted.startswith("The following layers were not sharded:"):
        return "render", _summarize_sharded_layers(_parse_sharded_layer_tokens(formatted))

    for rule in CAPTURED_OUTPUT_RULES:
        if not _warning_rule_matches(formatted, "", rule):
            continue
        if rule.suppress:
            return "suppress", None
        if rule.replacement is not None:
            return "render", rule.replacement
        return "render", formatted

    return "passthrough", None


def _print_warning_message(msg: str, rich_console: Optional[Console] = None) -> None:
    """Render warning message with Rich without interpreting message markup."""
    if not msg:
        return
    target_console = rich_console or console
    target_console.print(Text(f"  ⚠ {msg}", style="dim yellow"))


def _parse_sharded_layer_tokens(msg: str) -> list[str]:
    raw_layers = msg.split(":", 1)[1] if ":" in msg else msg
    return [layer.strip() for layer in raw_layers.split(",") if layer.strip()]


def _summarize_sharded_layers(layers: Sequence[str]) -> str:
    preview = ", ".join(layers[:4])
    if len(layers) > 4:
        preview += ", ..."
    return f"Some layers were not sharded: {preview}"


def _looks_like_sharded_layer_continuation(msg: str) -> bool:
    if not msg:
        return False
    tokens = [token.strip() for token in msg.split(",") if token.strip()]
    if not tokens:
        return False
    return all(
        re.fullmatch(r"[\w.*\[\]-]+", token) is not None
        and (".weight" in token or ".bias" in token)
        for token in tokens
    )


def _unwrap_passthrough_stream(stream):
    unwrapped = stream
    seen: set[int] = set()
    while hasattr(unwrapped, "rich_proxied_file") and id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        unwrapped = unwrapped.rich_proxied_file
    return unwrapped


def verbose_logging_enabled() -> bool:
    """Return whether runtime logging is configured for verbose output."""
    return _VERBOSE_LOGGING_ENABLED


class _RuntimeOutputCapture:
    """Line-buffered stdout/stderr capture that re-renders actionable messages via Rich."""

    def __init__(self, emitter: Callable[[str, bool], None], fallback_stream):
        self._emitter = emitter
        self._fallback_stream = fallback_stream
        self._buffer = ""
        self.encoding = getattr(fallback_stream, "encoding", "utf-8")

    def write(self, data: str) -> int:
        if not data:
            return 0
        normalized = data.replace("\r\n", "\n")
        start = 0
        for index, char in enumerate(normalized):
            if char not in {"\n", "\r"}:
                continue
            self._buffer += normalized[start:index]
            if char == "\n":
                self._emitter(self._buffer, True)
            self._buffer = ""
            start = index + 1
        self._buffer += normalized[start:]
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._emitter(self._buffer, False)
            self._buffer = ""
        if hasattr(self._fallback_stream, "flush"):
            self._fallback_stream.flush()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


@contextmanager
def capture_runtime_output(
    *,
    enabled: bool = True,
    rich_console: Optional[Console] = None,
):
    """Capture raw third-party stdout/stderr and re-render actionable messages cleanly."""
    if not enabled:
        yield
        return

    target_console = rich_console or console
    seen_messages: set[str] = set()
    pending_sharded_layers: list[str] = []

    def _flush_pending_sharded_layers() -> None:
        nonlocal pending_sharded_layers
        if not pending_sharded_layers:
            return
        message = _summarize_sharded_layers(pending_sharded_layers)
        pending_sharded_layers = []
        if message in seen_messages:
            return
        seen_messages.add(message)
        _print_warning_message(message, target_console)

    def _emit(line: str, terminated: bool, fallback_stream) -> None:
        nonlocal pending_sharded_layers
        formatted = _normalize_warning_message(line)
        if pending_sharded_layers:
            if formatted and _looks_like_sharded_layer_continuation(formatted):
                pending_sharded_layers.extend(_parse_sharded_layer_tokens(formatted))
                if terminated and not formatted.endswith(","):
                    _flush_pending_sharded_layers()
                return
            _flush_pending_sharded_layers()
        if formatted and formatted.startswith("The following layers were not sharded:"):
            pending_sharded_layers = _parse_sharded_layer_tokens(formatted)
            if terminated and not formatted.endswith(","):
                _flush_pending_sharded_layers()
            return
        action, message = _classify_captured_output_message(line)
        if action == "passthrough":
            passthrough_stream = _unwrap_passthrough_stream(fallback_stream)
            passthrough_stream.write(line)
            if terminated:
                passthrough_stream.write("\n")
            return
        if action != "render" or message is None or message in seen_messages:
            return
        seen_messages.add(message)
        _print_warning_message(message, target_console)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_stdout = _RuntimeOutputCapture(
        lambda line, terminated: _emit(line, terminated, original_stdout),
        original_stdout,
    )
    captured_stderr = _RuntimeOutputCapture(
        lambda line, terminated: _emit(line, terminated, original_stderr),
        original_stderr,
    )
    sys.stdout = captured_stdout
    sys.stderr = captured_stderr
    try:
        yield
    finally:
        captured_stdout.flush()
        captured_stderr.flush()
        _flush_pending_sharded_layers()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _format_log_record_message(record: logging.LogRecord) -> str:
    try:
        return record.getMessage()
    except TypeError:
        message = str(record.msg)
        if not record.args:
            return message

        extra_parts = []
        for arg in record.args:
            if isinstance(arg, type) and issubclass(arg, Warning):
                continue
            extra_parts.append(str(arg))

        if extra_parts:
            return " ".join([message, *extra_parts])
        return message


class RichWarningHandler(logging.Handler):
    """Custom logging handler that formats warnings elegantly with Rich."""

    def __init__(
        self, rich_console: Optional[Console] = None, extra_rules: Sequence[WarningRule] = ()
    ):
        super().__init__()
        self._console = rich_console or console
        self._extra_rules = tuple(extra_rules)

    def emit(self, record):
        msg = format_warning_message(
            _format_log_record_message(record),
            logger_name=record.name,
            extra_rules=self._extra_rules,
        )
        if msg is None:
            return

        _print_warning_message(msg, self._console)


def configure_warning_loggers(
    logger_names: Sequence[str],
    handler: logging.Handler,
    saved: Optional[dict] = None,
) -> dict:
    saved = {} if saved is None else saved
    names_to_configure = set(logger_names)

    for name in list(logging.Logger.manager.loggerDict):
        if any(
            name == logger_name or name.startswith(f"{logger_name}.")
            for logger_name in logger_names
        ):
            names_to_configure.add(name)

    for name in sorted(names_to_configure):
        lg = logging.getLogger(name)
        if name not in saved:
            saved[name] = (lg.handlers[:], lg.level, lg.propagate)
        lg.handlers = [handler]
        lg.setLevel(logging.WARNING)
        lg.propagate = False

    return saved


def restore_logger_configuration(saved: dict) -> None:
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers = handlers
        lg.setLevel(level)
        lg.propagate = propagate


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
    handler = RichWarningHandler()
    configure_warning_loggers(
        ["transformers", "datasets", "accelerate", "huggingface_hub", "py.warnings"],
        handler,
    )

    # Also handle Python warnings module
    def _rich_showwarning(message, category, filename, lineno, file=None, line=None):
        logging.getLogger("py.warnings").warning(str(message))

    warnings.showwarning = _rich_showwarning


def setup_logging(level: str = "WARNING") -> logging.Logger:
    """Setup logging configuration."""
    global _VERBOSE_LOGGING_ENABLED
    requested_level = getattr(logging, level.upper())
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=requested_level,
    )
    logging.getLogger().setLevel(requested_level)
    _VERBOSE_LOGGING_ENABLED = requested_level <= logging.INFO

    # Suppress verbose transformers/datasets logs
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

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
