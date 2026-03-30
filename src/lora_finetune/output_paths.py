"""Helpers for resolving training output directory layouts."""

import os
import re


def resolve_hpo_sweep_output_dir(output_dir: str, sweep_name: str | None) -> str:
    """Nest HPO artifacts under a sweep-specific subdirectory when available."""
    safe_sweep_name = _sanitize_sweep_name(sweep_name)
    if safe_sweep_name is None:
        return output_dir

    return os.path.join(output_dir, safe_sweep_name)


def ensure_hpo_sweep_output_dir(output_dir: str, sweep_name: str | None) -> str:
    """Return an HPO output path that includes the sweep segment exactly once."""
    safe_sweep_name = _sanitize_sweep_name(sweep_name)
    if safe_sweep_name is None:
        return output_dir

    if os.path.basename(os.path.normpath(output_dir)) == safe_sweep_name:
        return output_dir

    return os.path.join(output_dir, safe_sweep_name)


def _sanitize_sweep_name(sweep_name: str | None) -> str | None:
    if not sweep_name:
        return None

    safe_sweep_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sweep_name)).strip("._")
    if not safe_sweep_name:
        return None

    return safe_sweep_name
