"""Optional Unsloth bootstrap helpers."""

from .utils import capture_stdout

try:
    with capture_stdout():
        import unsloth
except ImportError:
    unsloth = None

UNSLOTH_AVAILABLE = unsloth is not None
FastLanguageModel = unsloth.FastLanguageModel if unsloth is not None else None
