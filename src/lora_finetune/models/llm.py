"""LLM-specific model utilities."""

import logging
from typing import Dict, List

from transformers import PreTrainedModel, PreTrainedTokenizer

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
