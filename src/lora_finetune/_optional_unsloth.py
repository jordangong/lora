"""Optional Unsloth bootstrap helpers."""

from .utils import capture_stdout

unsloth = None
UNSLOTH_AVAILABLE = False
FastLanguageModel = None
_bootstrap_attempted = False


def ensure_unsloth_imported():
    global unsloth, UNSLOTH_AVAILABLE, FastLanguageModel, _bootstrap_attempted

    if _bootstrap_attempted:
        return FastLanguageModel

    _bootstrap_attempted = True
    try:
        with capture_stdout():
            import unsloth as imported_unsloth
    except ImportError:
        imported_unsloth = None

    unsloth = imported_unsloth
    UNSLOTH_AVAILABLE = imported_unsloth is not None
    FastLanguageModel = imported_unsloth.FastLanguageModel if imported_unsloth is not None else None
    return FastLanguageModel
