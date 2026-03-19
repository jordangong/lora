"""Command-line interface and argument parsing for LoRA finetuning."""

import argparse
import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

import yaml

from .config import (
    AugmentationConfig,
    BenchmarkEvalConfig,
    Config,
    DataConfig,
    DPOConfig,
    GRPOConfig,
    HPOConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)


def _is_list_or_str_union(type_hint: Type) -> bool:
    """Check if type is Union[List[str], str] (possibly Optional)."""
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        non_none_args = [a for a in args if a is not type(None)]
        # Check if we have both list and str types
        has_list = any(get_origin(a) is list or a is list for a in non_none_args)
        has_str = any(a is str for a in non_none_args)
        return has_list and has_str
    return False


def _get_base_type(type_hint: Type) -> Type:
    """Extract the base type from Optional or other generic types."""
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return non_none_args[0]
    return type_hint


def _parse_hpo_param_override(value: str) -> Tuple[str, Dict[str, Any]]:
    name, separator, raw_spec = value.partition("=")
    name = name.strip()
    raw_spec = raw_spec.strip()

    if not separator or not name or not raw_spec:
        raise argparse.ArgumentTypeError(
            "--hpo_param must be provided as NAME=SPEC, where SPEC is a YAML/JSON mapping"
        )

    try:
        spec = yaml.safe_load(raw_spec)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --hpo_param spec for '{name}': {exc}") from exc

    if not isinstance(spec, dict):
        raise argparse.ArgumentTypeError(
            f"Invalid --hpo_param spec for '{name}': expected a YAML/JSON mapping"
        )

    return name, spec


def _merge_hpo_param_overrides(
    existing_parameters: Optional[Dict[str, Dict[str, Any]]],
    overrides: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not overrides:
        return existing_parameters

    merged_parameters = (
        copy.deepcopy(existing_parameters) if existing_parameters is not None else {}
    )
    for override in overrides:
        if isinstance(override, str):
            parameter_name, spec = _parse_hpo_param_override(override)
        else:
            parameter_name, spec = override
        merged_parameters[parameter_name] = spec
    return merged_parameters


def _add_dataclass_args(
    parser: argparse.ArgumentParser,
    dataclass_type: Type,
    prefix: str = "",
    skip_fields: Optional[List[str]] = None,
) -> None:
    """Add arguments from a dataclass to the parser."""
    skip_fields = skip_fields or []

    for field_info in dataclasses.fields(dataclass_type):
        field_name = field_info.name
        if field_name in skip_fields:
            continue

        # Skip nested dataclass fields (handled separately)
        if dataclasses.is_dataclass(field_info.type):
            continue

        arg_name = f"--{prefix}{field_name}" if prefix else f"--{field_name}"
        field_type = _get_base_type(field_info.type)

        # Extract help text from field metadata
        help_text = field_info.metadata.get("help") if field_info.metadata else None

        # Get default value
        default_val = field_info.default
        if default_val is dataclasses.MISSING:
            default_val = None

        # Build help text with default value
        def _build_help(base_help: Optional[str], default: Any) -> str:
            if base_help and default is not None:
                return f"{base_help} (default: {default})"
            elif base_help:
                return base_help
            elif default is not None:
                return f"(default: {default})"
            return ""

        # Handle boolean fields - add both --flag and --no_flag (no_ hidden from help)
        if field_type is bool:
            if default_val is None:
                bool_help = help_text if help_text else ""
            else:
                default_state = "enabled" if default_val else "disabled"
                bool_help = (
                    f"{help_text} (default: {default_state})"
                    if help_text
                    else f"(default: {default_state})"
                )
            parser.add_argument(arg_name, action="store_true", default=None, help=bool_help)
            no_arg_name = f"--no_{prefix}{field_name}" if prefix else f"--no_{field_name}"
            parser.add_argument(
                no_arg_name, action="store_true", default=None, help=argparse.SUPPRESS
            )
        # Handle Union[List[str], str] fields (e.g., target_modules supporting regex)
        elif _is_list_or_str_union(field_info.type):
            parser.add_argument(
                arg_name,
                type=str,
                nargs="*",
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle list fields
        elif get_origin(field_type) is list or field_type is list:
            inner_type = get_args(field_type)
            item_type = inner_type[0] if inner_type else str
            parser.add_argument(
                arg_name,
                type=item_type,
                nargs="+",
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle tuple fields (as space-separated values)
        elif get_origin(field_type) is tuple or field_type is tuple:
            parser.add_argument(
                arg_name,
                type=float,
                nargs="+",
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle Literal types by extracting choices
        elif get_origin(field_type) is type(None):
            parser.add_argument(
                arg_name, type=str, default=None, help=_build_help(help_text, default_val)
            )
        elif hasattr(field_type, "__origin__") and str(field_type.__origin__) == "typing.Literal":
            choices = get_args(field_type)
            parser.add_argument(
                arg_name,
                type=str,
                choices=choices,
                default=None,
                help=_build_help(help_text, default_val),
            )
        # Handle dict fields (skip - too complex for CLI)
        elif field_type is dict or get_origin(field_type) is dict:
            continue
        # Handle basic types
        elif field_type in (int, float, str):
            parser.add_argument(
                arg_name, type=field_type, default=None, help=_build_help(help_text, default_val)
            )
        else:
            # Try to use the type directly, fallback to str
            try:
                parser.add_argument(
                    arg_name, type=str, default=None, help=_build_help(help_text, default_val)
                )
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA Finetuning")

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (INFO) logging for debugging"
    )

    # Add arguments from all config dataclasses
    # LoRA config with "lora_" prefix for r, alpha, dropout to avoid conflicts
    _add_dataclass_args(parser, LoraConfig, prefix="lora_")

    # Model config
    _add_dataclass_args(parser, ModelConfig)

    # Data config (skip augmentation - handled separately)
    _add_dataclass_args(parser, DataConfig, skip_fields=["augmentation"])

    # Augmentation config with "aug_" prefix
    _add_dataclass_args(parser, AugmentationConfig, prefix="aug_")

    # Training config
    _add_dataclass_args(parser, TrainingConfig)

    # DPO config with "dpo_" prefix
    _add_dataclass_args(parser, DPOConfig, prefix="dpo_")

    # GRPO config with "grpo_" prefix
    _add_dataclass_args(parser, GRPOConfig, prefix="grpo_")

    # HPO config with "hpo_" prefix
    _add_dataclass_args(parser, HPOConfig, prefix="hpo_")
    parser.add_argument(
        "--hpo_param",
        action="append",
        type=_parse_hpo_param_override,
        default=None,
        metavar="NAME=SPEC",
        help="Repeatable HPO parameter override, e.g. --hpo_param learning_rate='{values: [1.0e-5, 2.0e-5]}'",
    )

    # Benchmark evaluation config with "bench_" prefix
    _add_dataclass_args(parser, BenchmarkEvalConfig, prefix="bench_")

    return parser.parse_args()


def _apply_args_to_config(
    config_obj: Any,
    args_dict: Dict[str, Any],
    prefix: str = "",
) -> None:
    """Apply CLI arguments to a config object."""
    for field_info in dataclasses.fields(config_obj):
        field_name = field_info.name

        # Skip nested dataclass fields
        if dataclasses.is_dataclass(field_info.type):
            continue

        arg_name = f"{prefix}{field_name}" if prefix else field_name
        field_type = _get_base_type(field_info.type)

        # Handle --no_* flags for boolean fields
        if field_type is bool:
            no_arg_name = f"no_{arg_name}"
            if args_dict.get(no_arg_name):
                setattr(config_obj, field_name, False)
                continue

        if arg_name in args_dict and args_dict[arg_name] is not None:
            value = args_dict[arg_name]
            # Convert list to tuple if needed
            if (get_origin(field_type) is tuple or field_type is tuple) and isinstance(value, list):
                value = tuple(value)
            # For Union[List[str], str] fields, keep single value as string (for regex)
            if (
                _is_list_or_str_union(field_info.type)
                and isinstance(value, list)
                and len(value) == 1
            ):
                value = value[0]
            setattr(config_obj, field_name, value)


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from args and config file."""
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    args_dict = vars(args)

    # Apply args to each config section
    _apply_args_to_config(config.lora, args_dict, prefix="lora_")
    _apply_args_to_config(config.model, args_dict)
    _apply_args_to_config(config.data, args_dict)
    _apply_args_to_config(config.data.augmentation, args_dict, prefix="aug_")
    _apply_args_to_config(config.training, args_dict)
    _apply_args_to_config(config.dpo, args_dict, prefix="dpo_")
    _apply_args_to_config(config.grpo, args_dict, prefix="grpo_")
    _apply_args_to_config(config.hpo, args_dict, prefix="hpo_")
    config.hpo.parameters = _merge_hpo_param_overrides(
        config.hpo.parameters,
        args_dict.get("hpo_param"),
    )
    _apply_args_to_config(config.benchmark_eval, args_dict, prefix="bench_")

    return config
