"""Base model loading utilities."""

import inspect
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
from peft import (
    AdaLoraConfig,
    IA3Config,
    PeftModel,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft import LoraConfig as PeftLoraConfig
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .. import _optional_unsloth
from ..config import GradientCheckpointingConfig, LoraConfig, ModelConfig
from ..utils import capture_stdout, get_method_display_name

logger = logging.getLogger(__name__)

_FAST_LANGUAGE_MODEL_UNSET = object()
FastLanguageModel = _FAST_LANGUAGE_MODEL_UNSET


MODEL_TYPE_TO_AUTO_CLASS = {
    "causal_lm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "vision": AutoModelForImageClassification,
    "text_classification": AutoModelForSequenceClassification,
}

MODEL_TYPE_TO_TASK_TYPE = {
    "causal_lm": TaskType.CAUSAL_LM,
    "seq2seq": TaskType.SEQ_2_SEQ_LM,
    "vision": TaskType.FEATURE_EXTRACTION,
    "text_classification": TaskType.SEQ_CLS,
}

UNSLOTH_SUPPORTED_METHODS = {"lora", "dora", "loraplus", "full"}


def _get_fast_language_model():
    fast_language_model = FastLanguageModel
    if fast_language_model is not _FAST_LANGUAGE_MODEL_UNSET:
        return fast_language_model
    return _optional_unsloth.FastLanguageModel


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    return dtype_map.get(dtype_str, "auto")


def get_quantization_config(config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Get quantization config if needed."""
    if config.load_in_4bit:
        logger.info("Using 4-bit quantization")
        compute_dtype = get_torch_dtype(config.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    elif config.load_in_8bit:
        logger.info("Using 8-bit quantization")
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _filter_supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only those accepted by the callable signature when available."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _set_tokenizer_padding(tokenizer: Any) -> bool:
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token is None or eos_token_id is None:
        return False

    pad_token = getattr(tokenizer, "pad_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    should_use_eos_padding = (
        pad_token is None
        or pad_token_id is None
        or (unk_token is not None and pad_token == unk_token)
        or (unk_token_id is not None and pad_token_id == unk_token_id)
    )
    if not should_use_eos_padding:
        return False

    tokenizer.pad_token = eos_token
    tokenizer.pad_token_id = eos_token_id
    logger.info("Set pad_token to eos_token")
    return True


def _set_unsloth_tokenizer_padding(tokenizer: Any) -> bool:
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if (
        pad_token is not None
        and pad_token_id is not None
        and pad_token != eos_token
        and pad_token_id != eos_token_id
    ):
        return False

    unk_token = getattr(tokenizer, "unk_token", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if (
        unk_token is not None
        and unk_token_id is not None
        and unk_token != eos_token
        and unk_token_id != eos_token_id
    ):
        tokenizer.pad_token = unk_token
        tokenizer.pad_token_id = unk_token_id
        logger.info("Set pad_token to unk_token for Unsloth")
        return True

    return _set_tokenizer_padding(tokenizer)


def _is_local_model_path(model_name_or_path: str) -> bool:
    return Path(model_name_or_path).expanduser().exists()


def _populate_local_model_override_dir(source_dir: Path, target_dir: Path) -> None:
    for item in source_dir.iterdir():
        target = target_dir / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def _create_unsloth_tokenizer_override(
    config: ModelConfig,
    max_seq_length: Optional[int] = None,
) -> Optional[tempfile.TemporaryDirectory]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
        model_max_length=max_seq_length,
    )
    if not _set_unsloth_tokenizer_padding(tokenizer):
        return None

    temp_dir = tempfile.TemporaryDirectory()
    tokenizer.save_pretrained(temp_dir.name)
    if _is_local_model_path(config.model_name_or_path):
        _populate_local_model_override_dir(
            Path(config.model_name_or_path).expanduser(),
            Path(temp_dir.name),
        )
    return temp_dir


def _load_unsloth_model_and_tokenizer(
    config: ModelConfig,
    max_seq_length: Optional[int] = None,
) -> Tuple[PreTrainedModel, Any]:
    """Load a causal LM through Unsloth when enabled."""
    fast_language_model = _get_fast_language_model()
    if fast_language_model is None:
        raise ImportError(
            "Unsloth is not installed. Install it with: pip install unsloth "
            "or uv sync --extra unsloth"
        )
    if config.model_type != "causal_lm":
        raise ValueError("Unsloth integration is only supported for causal_lm models")

    torch_dtype = get_torch_dtype(config.torch_dtype)
    load_kwargs = {
        "model_name": config.model_name_or_path,
        "max_seq_length": max_seq_length,
        "dtype": torch_dtype,
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
        "device_map": None,
        "trust_remote_code": config.trust_remote_code,
    }
    if torch_dtype == "auto":
        load_kwargs["dtype"] = "auto"

    tokenizer_override_dir = _create_unsloth_tokenizer_override(
        config,
        max_seq_length=max_seq_length,
    )
    if tokenizer_override_dir is not None:
        if _is_local_model_path(config.model_name_or_path):
            load_kwargs["model_name"] = tokenizer_override_dir.name
        else:
            load_kwargs["tokenizer_name"] = tokenizer_override_dir.name

    load_kwargs = {
        key: value
        for key, value in load_kwargs.items()
        if (value is not None and value is not False)
        or key in {"load_in_4bit", "load_in_8bit", "device_map"}
    } | {
        key: value
        for key, value in load_kwargs.items()
        if key in {"load_in_4bit", "load_in_8bit", "device_map"}
    }
    load_kwargs = _filter_supported_kwargs(fast_language_model.from_pretrained, load_kwargs)

    logger.info(f"Loading model from {config.model_name_or_path} with Unsloth")
    try:
        with capture_stdout():
            model, tokenizer = fast_language_model.from_pretrained(**load_kwargs)
    finally:
        if tokenizer_override_dir is not None:
            tokenizer_override_dir.cleanup()

    _set_unsloth_tokenizer_padding(tokenizer)

    model.name_or_path = config.model_name_or_path
    if getattr(model, "config", None) is not None:
        model.config._name_or_path = config.model_name_or_path
    if tokenizer is not None:
        tokenizer.name_or_path = config.model_name_or_path

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def _apply_unsloth_peft_model(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    is_quantized: bool = False,
    use_gradient_checkpointing: GradientCheckpointingConfig = True,
    random_state: int = 42,
    max_seq_length: Optional[int] = None,
) -> Union[PreTrainedModel, PeftModel]:
    """Apply Unsloth LoRA patching when supported by the current config."""
    fast_language_model = _get_fast_language_model()
    if fast_language_model is None:
        raise ImportError(
            "Unsloth is not installed. Install it with: pip install unsloth "
            "or uv sync --extra unsloth"
        )

    method = lora_config.method
    if method == "full":
        return prepare_model_for_full_finetuning(model, is_quantized)
    if method not in UNSLOTH_SUPPORTED_METHODS:
        raise ValueError(
            "Unsloth integration currently supports only lora, dora, loraplus, and full methods"
        )

    unsloth_gradient_checkpointing = (
        "unsloth" if use_gradient_checkpointing == "unsloth" else bool(use_gradient_checkpointing)
    )
    unsloth_kwargs = {
        "r": lora_config.r,
        "target_modules": lora_config.target_modules,
        "lora_alpha": lora_config.alpha,
        "lora_dropout": lora_config.dropout,
        "bias": lora_config.bias,
        "use_gradient_checkpointing": unsloth_gradient_checkpointing,
        "random_state": random_state,
        "max_seq_length": max_seq_length,
        "use_rslora": lora_config.use_rslora,
        "use_dora": lora_config.use_dora,
        "modules_to_save": lora_config.modules_to_save,
    }
    unsloth_kwargs = {key: value for key, value in unsloth_kwargs.items() if value is not None}
    unsloth_kwargs = _filter_supported_kwargs(fast_language_model.get_peft_model, unsloth_kwargs)

    logger.info(
        f"Applying {get_method_display_name(method)} with Unsloth: r={lora_config.r}, alpha={lora_config.alpha}"
    )
    model = fast_language_model.get_peft_model(model, **unsloth_kwargs)
    setattr(
        model,
        "_lora_finetune_unsloth_managed_gradient_checkpointing",
        bool(use_gradient_checkpointing),
    )
    return model


def load_model_and_tokenizer(
    config: ModelConfig,
    num_labels: Optional[int] = None,
    *,
    max_seq_length: Optional[int] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
) -> Tuple[PreTrainedModel, Any]:
    """Load model and tokenizer/processor based on model type."""
    if config.use_unsloth:
        if (
            config.model_type != "causal_lm"
            or num_labels is not None
            or id2label is not None
            or label2id is not None
        ):
            raise ValueError("Unsloth integration is only supported for causal_lm models")
        return _load_unsloth_model_and_tokenizer(config, max_seq_length=max_seq_length)

    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }

    torch_dtype = get_torch_dtype(config.torch_dtype)
    if torch_dtype != "auto":
        model_kwargs["dtype"] = torch_dtype

    if config.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    quantization_config = get_quantization_config(config)
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    auto_class = MODEL_TYPE_TO_AUTO_CLASS[config.model_type]

    if config.model_type in {"vision", "text_classification"} and num_labels is not None:
        model_kwargs["num_labels"] = num_labels
        model_kwargs["ignore_mismatched_sizes"] = True
        if id2label:
            model_kwargs["id2label"] = id2label
        if label2id:
            model_kwargs["label2id"] = label2id

    logger.info(f"Loading model from {config.model_name_or_path}")
    model = auto_class.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    if config.model_type == "vision":
        logger.info("Loading image processor")
        processor = AutoImageProcessor.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        return model, processor
    else:
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
        )
        _set_tokenizer_padding(tokenizer)

        # Sync model config and generation config with tokenizer to avoid mismatch warnings
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer


def _get_task_type(model_type: str, lora_config: LoraConfig) -> TaskType:
    """Get PEFT task type from model type and config."""
    task_type = MODEL_TYPE_TO_TASK_TYPE.get(model_type, TaskType.CAUSAL_LM)
    if lora_config.task_type:
        task_type = TaskType[lora_config.task_type.upper()]
    return task_type


def _create_lora_config(
    lora_config: LoraConfig,
    task_type: TaskType,
) -> PeftLoraConfig:
    """Create standard LoRA or DoRA config."""
    return PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=task_type,
        modules_to_save=lora_config.modules_to_save,
        use_dora=lora_config.use_dora,
        use_rslora=lora_config.use_rslora,
    )


def _create_adalora_config(
    lora_config: LoraConfig,
    task_type: TaskType,
) -> AdaLoraConfig:
    """Create AdaLoRA config for adaptive rank allocation."""
    return AdaLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=task_type,
        modules_to_save=lora_config.modules_to_save,
        init_r=lora_config.init_r,
        target_r=lora_config.target_r,
        tinit=lora_config.tinit,
        tfinal=lora_config.tfinal,
        deltaT=lora_config.deltaT,
        beta1=lora_config.beta1,
        beta2=lora_config.beta2,
        orth_reg_weight=lora_config.orth_reg_weight,
    )


def _create_ia3_config(
    lora_config: LoraConfig,
    task_type: TaskType,
) -> IA3Config:
    """Create IA3 config for few-shot learning."""
    feedforward_modules = lora_config.feedforward_modules
    if feedforward_modules is None:
        feedforward_modules = ["down_proj", "mlp.fc2", "dense_4h_to_h"]

    return IA3Config(
        target_modules=lora_config.target_modules,
        feedforward_modules=feedforward_modules,
        task_type=task_type,
        modules_to_save=lora_config.modules_to_save,
    )


def _create_prefix_tuning_config(
    lora_config: LoraConfig,
    task_type: TaskType,
) -> PrefixTuningConfig:
    """Create prefix tuning config."""
    return PrefixTuningConfig(
        task_type=task_type,
        num_virtual_tokens=lora_config.num_virtual_tokens,
        prefix_projection=lora_config.prefix_projection,
    )


def prepare_model_for_full_finetuning(
    model: PreTrainedModel,
    is_quantized: bool = False,
) -> PreTrainedModel:
    """Prepare model for full finetuning (all parameters trainable)."""
    if is_quantized:
        raise ValueError(
            "Full finetuning is not compatible with quantization. "
            "Use LoRA or other PEFT methods for quantized models."
        )

    # Enable all parameters for training
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Full finetuning: {trainable_params:,} trainable parameters "
        f"({100 * trainable_params / total_params:.2f}% of {total_params:,} total)"
    )

    return model


def get_peft_model_with_adapter(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    model_type: str = "causal_lm",
    is_quantized: bool = False,
    use_unsloth: bool = False,
    use_gradient_checkpointing: GradientCheckpointingConfig = True,
    random_state: int = 42,
    max_seq_length: Optional[int] = None,
) -> Union[PreTrainedModel, PeftModel]:
    """Apply PEFT adapter to model based on config method.

    Supported methods:
    - lora: Standard LoRA
    - dora: Weight-Decomposed LoRA (DoRA)
    - adalora: Adaptive LoRA with rank allocation
    - loraplus: LoRA with different learning rates for A and B matrices
    - ia3: Infused Adapter by Inhibiting and Amplifying Inner Activations
    - prefix_tuning: Prefix tuning with virtual tokens
    - full: Full finetuning (no adapter)
    """
    if use_unsloth:
        if model_type != "causal_lm":
            raise ValueError("Unsloth integration is only supported for causal_lm models")
        return _apply_unsloth_peft_model(
            model,
            lora_config,
            is_quantized=is_quantized,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            max_seq_length=max_seq_length,
        )

    method = lora_config.method

    # Full finetuning - no PEFT adapter
    if method == "full":
        return prepare_model_for_full_finetuning(model, is_quantized)

    # Prepare quantized model for adapter training
    if is_quantized:
        logger.info("Preparing quantized model for k-bit training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    task_type = _get_task_type(model_type, lora_config)

    # Create appropriate PEFT config based on method
    if method in ("lora", "dora", "loraplus"):
        peft_config = _create_lora_config(lora_config, task_type)
        logger.info(
            f"Applying {get_method_display_name(method)} with config: r={lora_config.r}, alpha={lora_config.alpha}"
        )
    elif method == "adalora":
        peft_config = _create_adalora_config(lora_config, task_type)
        logger.info(
            f"Applying {get_method_display_name(method)} with init_r={lora_config.init_r}, target_r={lora_config.target_r}"
        )
    elif method == "ia3":
        peft_config = _create_ia3_config(lora_config, task_type)
        logger.info(
            f"Applying {get_method_display_name(method)} to modules: {lora_config.target_modules}"
        )
    elif method == "prefix_tuning":
        peft_config = _create_prefix_tuning_config(lora_config, task_type)
        logger.info(
            f"Applying {get_method_display_name(method)} with {lora_config.num_virtual_tokens} virtual tokens"
        )
    else:
        raise ValueError(f"Unknown finetuning method: {method}")

    model = get_peft_model(model, peft_config)

    return model


def get_peft_model_with_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    model_type: str = "causal_lm",
    is_quantized: bool = False,
    use_unsloth: bool = False,
    use_gradient_checkpointing: GradientCheckpointingConfig = True,
    random_state: int = 42,
    max_seq_length: Optional[int] = None,
) -> Union[PreTrainedModel, PeftModel]:
    """Apply adapter to model using PEFT. Wrapper for backward compatibility."""
    return get_peft_model_with_adapter(
        model,
        lora_config,
        model_type,
        is_quantized,
        use_unsloth,
        use_gradient_checkpointing,
        random_state,
        max_seq_length,
    )
