"""Optional Unsloth bootstrap helpers."""

import io
import os
from contextlib import redirect_stderr, redirect_stdout


def _startup_verbose_requested() -> bool:
    return os.environ.get("LORA_FINETUNE_VERBOSE_STARTUP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


try:
    if _startup_verbose_requested():
        import unsloth
    else:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            import unsloth
except ImportError:
    unsloth = None

UNSLOTH_AVAILABLE = unsloth is not None
FastLanguageModel = unsloth.FastLanguageModel if unsloth is not None else None
