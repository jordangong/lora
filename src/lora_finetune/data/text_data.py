"""Text dataset utilities for LLM finetuning."""

from datasets import load_dataset

from . import text_classification as imported_text_classification
from . import text_preprocessing as imported_text_preprocessing
from .text_classification import (
    get_text_classification_collator,
    tokenize_classification_function,
)
from .text_classification import (
    preprocess_text_classification_dataset as imported_preprocess_text_classification_dataset,
)
from .text_formatting import (
    CHAT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    SOURCE_TEXT_COLUMN,
    format_instruction,
    format_instruction_as_prompt,
    format_instruction_as_prompt_completion,
    format_instruction_with_source,
    format_qa,
    format_qa_as_prompt,
    format_qa_as_prompt_completion,
    format_qa_with_source,
    normalize_conversations_to_messages,
)
from .text_preprocessing import (
    get_text_collator,
    maybe_append_eos,
    prepare_dataset_for_causal_lm,
    requires_trl_native_dataset,
    shuffle_dataset_split,
    tokenize_function,
)
from .text_preprocessing import (
    load_text_dataset as imported_load_text_dataset,
)
from .text_preprocessing import (
    prepare_grpo_dataset_for_trl as imported_prepare_grpo_dataset_for_trl,
)
from .text_preprocessing import (
    prepare_preference_dataset_for_trl as imported_prepare_preference_dataset_for_trl,
)
from .text_preprocessing import (
    prepare_text_dataset_for_trl as imported_prepare_text_dataset_for_trl,
)
from .text_preprocessing import (
    preprocess_text_dataset as imported_preprocess_text_dataset,
)


def _sync_patch_points() -> None:
    imported_text_preprocessing.load_dataset = load_dataset
    imported_text_preprocessing.shuffle_dataset_split = shuffle_dataset_split
    imported_text_classification.shuffle_dataset_split = shuffle_dataset_split


def load_text_dataset(config):
    _sync_patch_points()
    return imported_load_text_dataset(config)


def prepare_text_dataset_for_trl(dataset, config, shuffle_seed=None):
    _sync_patch_points()
    return imported_prepare_text_dataset_for_trl(dataset, config, shuffle_seed=shuffle_seed)


def prepare_preference_dataset_for_trl(dataset, config, shuffle_seed=None):
    _sync_patch_points()
    return imported_prepare_preference_dataset_for_trl(dataset, config, shuffle_seed=shuffle_seed)


def prepare_grpo_dataset_for_trl(dataset, config, shuffle_seed=None):
    _sync_patch_points()
    return imported_prepare_grpo_dataset_for_trl(dataset, config, shuffle_seed=shuffle_seed)


def preprocess_text_dataset(dataset, tokenizer, config, shuffle_seed=None):
    _sync_patch_points()
    return imported_preprocess_text_dataset(
        dataset,
        tokenizer,
        config,
        shuffle_seed=shuffle_seed,
    )


def preprocess_text_classification_dataset(dataset, tokenizer, config, shuffle_seed=None):
    _sync_patch_points()
    return imported_preprocess_text_classification_dataset(
        dataset,
        tokenizer,
        config,
        shuffle_seed=shuffle_seed,
    )


__all__ = [
    "CHAT_TEMPLATE",
    "DEFAULT_PROMPT_TEMPLATE",
    "SOURCE_TEXT_COLUMN",
    "format_instruction",
    "format_qa",
    "format_instruction_with_source",
    "format_qa_with_source",
    "format_instruction_as_prompt_completion",
    "format_qa_as_prompt_completion",
    "format_instruction_as_prompt",
    "format_qa_as_prompt",
    "normalize_conversations_to_messages",
    "load_text_dataset",
    "maybe_append_eos",
    "tokenize_function",
    "shuffle_dataset_split",
    "requires_trl_native_dataset",
    "prepare_text_dataset_for_trl",
    "prepare_preference_dataset_for_trl",
    "prepare_grpo_dataset_for_trl",
    "preprocess_text_dataset",
    "tokenize_classification_function",
    "preprocess_text_classification_dataset",
    "get_text_collator",
    "get_text_classification_collator",
    "prepare_dataset_for_causal_lm",
]
