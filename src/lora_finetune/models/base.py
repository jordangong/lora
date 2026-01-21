"""Base model loading utilities."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from peft import (
    LoraConfig as PeftLoraConfig,
)
from peft import (
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from ..config import LoraConfig, ModelConfig

logger = logging.getLogger(__name__)


MODEL_TYPE_TO_AUTO_CLASS = {
    "causal_lm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "vision": AutoModelForImageClassification,
}

MODEL_TYPE_TO_TASK_TYPE = {
    "causal_lm": TaskType.CAUSAL_LM,
    "seq2seq": TaskType.SEQ_2_SEQ_LM,
    "vision": TaskType.FEATURE_EXTRACTION,
}


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
        compute_dtype = get_torch_dtype(config.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    elif config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(
    config: ModelConfig,
    num_labels: Optional[int] = None,
) -> Tuple[PreTrainedModel, Any]:
    """Load model and tokenizer/processor based on model type."""
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

    if config.model_type == "vision" and num_labels:
        model_kwargs["num_labels"] = num_labels
        model_kwargs["ignore_mismatched_sizes"] = True

    logger.info(f"Loading model from {config.model_name_or_path}")
    model = auto_class.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    if config.model_type == "vision":
        processor = AutoImageProcessor.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        return model, processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer


def get_peft_model_with_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    model_type: str = "causal_lm",
    is_quantized: bool = False,
) -> PreTrainedModel:
    """Apply LoRA to model using PEFT."""
    if is_quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    task_type = MODEL_TYPE_TO_TASK_TYPE.get(model_type, TaskType.CAUSAL_LM)
    if lora_config.task_type:
        task_type = TaskType[lora_config.task_type.upper()]

    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=task_type,
        modules_to_save=lora_config.modules_to_save,
    )

    logger.info(f"Applying LoRA with config: {lora_config}")
    model = get_peft_model(model, peft_config)

    return model
