"""LLM-specific model utilities."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..config import ModelConfig

logger = logging.getLogger(__name__)


LLM_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "mpt": ["Wqkv", "out_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def get_llm_target_modules(model_name_or_path: str) -> List[str]:
    """Get target modules for LoRA based on model architecture."""
    model_name_lower = model_name_or_path.lower()

    for key in LLM_TARGET_MODULES:
        if key in model_name_lower:
            return LLM_TARGET_MODULES[key]

    return LLM_TARGET_MODULES["default"]


def load_causal_lm(
    config: ModelConfig,
    device_map: Optional[str] = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load causal language model with optimizations."""
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": device_map,
    }

    if config.torch_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config.torch_dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif config.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch.float32

    if config.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")

    if config.load_in_4bit:
        compute_dtype = (
            torch.bfloat16 if config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        logger.info("Loading model in 4-bit quantization")
    elif config.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Loading model in 8-bit quantization")

    logger.info(f"Loading model: {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_special_tokens_dict(tokenizer: PreTrainedTokenizer) -> Dict[str, str]:
    """Get special tokens that may need to be added."""
    special_tokens = {}

    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<pad>"
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = "<unk>"

    return special_tokens


def resize_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedModel:
    """Resize model embeddings if tokenizer has more tokens."""
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")
    return model
