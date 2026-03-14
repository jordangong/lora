"""Optional Unsloth bootstrap helpers."""

try:
    import unsloth
except ImportError:
    unsloth = None

UNSLOTH_AVAILABLE = unsloth is not None
FastLanguageModel = unsloth.FastLanguageModel if unsloth is not None else None
