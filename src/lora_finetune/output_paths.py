"""Helpers for resolving training output directory layouts."""

import os
import re


def resolve_hpo_sweep_output_dir(output_dir: str, sweep_name: str | None) -> str:
    """Nest HPO artifacts under a sweep-specific subdirectory when available."""
    if not sweep_name:
        return output_dir

    safe_sweep_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sweep_name)).strip("._")
    if not safe_sweep_name:
        return output_dir

    return os.path.join(output_dir, safe_sweep_name)
